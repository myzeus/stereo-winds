"""Unit-ish tests for the multi-time inference helpers in
``scripts/infer_student_global.py``.

We construct a randomly-initialized ``StudentWindsModel`` plus synthetic
radiance cubes + a stub RAFT (``_run_pair`` returning small random flows),
and confirm ``infer_one_time`` returns the stereo-cache variable schema
with correct shapes, finite values inside a known-valid region, and a
``quality_flag`` of 2 where outputs are finite.
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE))

from stereo_winds.config import SatelliteConfig
from stereo_winds.student_zeus_model import StudentWindsModel
from stereo_winds.student_dataset import DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS

# Import the helpers from the refactored script
from infer_student_global import infer_one_time, time_to_yyyymm


class _StubDisp:
    """Stand-in for ``StereoDisparity`` — returns a small random (2, H, W) flow."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _run_pair(self, img1, img2):
        H, W = img1.shape
        return self.rng.normal(0, 1.0, size=(2, H, W)).astype(np.float32)


def _tiny_sat() -> SatelliteConfig:
    """A small fixed-grid sat config so the synthetic test runs in <1 s."""
    return SatelliteConfig(
        satellite_id="testsat", sub_lon_deg=-75.0, sweep="x",
        # GOES-19 scale/offset, but only a small grid
        scale_x=5.6e-05, scale_y=-5.6e-05,
        x_offset=-0.05, y_offset=0.05,
        n_rows=128, n_cols=128,
    )


def _make_cubes(sat, t0, bands, valid_mask):
    """Build a dict[band -> xr.Dataset] with `Rad` at t0-10/0/+10."""
    delta = np.timedelta64(10, "m")
    times = np.array([t0 - delta, t0, t0 + delta])
    cubes = {}
    rng = np.random.default_rng(1)
    for b in bands:
        data = rng.normal(250, 30, size=(3, sat.n_rows, sat.n_cols)).astype(np.float32)
        # Inject NaNs outside the synthetic valid region so finite_mask exercises
        data[:, ~valid_mask] = np.nan
        cubes[b] = xr.Dataset(
            {"Rad": (("time", "y", "x"), data)},
            coords={"time": times},
        )
    return cubes


class TestInferOneTime:
    def test_shapes_and_quality_flag(self):
        sat = _tiny_sat()
        # Disk-ish valid region: an oval centered on the grid
        yy, xx = np.ogrid[:sat.n_rows, :sat.n_cols]
        cy, cx = sat.n_rows / 2, sat.n_cols / 2
        valid = ((yy - cy) ** 2 / (sat.n_rows / 2) ** 2
                 + (xx - cx) ** 2 / (sat.n_cols / 2) ** 2) < 0.9

        flow_bands = ["C14", "C08"]
        rad_bands = ["C14", "C08"]
        t0 = np.datetime64("2025-07-01T12:00")
        cubes = _make_cubes(sat, t0, set(flow_bands) | set(rad_bands), valid)
        disp = _StubDisp()

        model = StudentWindsModel(
            n_flow_bands=len(flow_bands), n_rad_bands=len(rad_bands),
            hidden=16, n_layers=2, context=False, learning_rate=1e-3,
        )
        model.eval()
        out = infer_one_time(
            model, disp, cubes, sat, t0, flow_bands, rad_bands,
            row_strip=64, halo=2, device="cpu",
        )

        assert set(out) == {"u_wind", "v_wind", "cloud_top_height", "quality_flag",
                             "chi_squared", "sigma_u", "sigma_v", "sigma_h"}
        for k, v in out.items():
            assert v.shape == (sat.n_rows, sat.n_cols), f"{k} shape {v.shape}"
            assert v.dtype == np.float32

        qf = out["quality_flag"]
        # Inside the valid region the model should produce finite outputs (qf=2)
        assert (qf[valid] == 2).all() or (qf[valid] >= 2).mean() > 0.95
        # Outside the valid region the radiance was NaN -> finite_mask False -> qf=0
        assert (qf[~valid] == 0).all()
        # u/v in meters/s should be finite where qf==2
        m = qf == 2
        for k in ("u_wind", "v_wind", "cloud_top_height", "sigma_u", "sigma_v", "sigma_h"):
            assert np.isfinite(out[k][m]).all(), f"{k} has NaN in valid region"

    def test_height_units_meters(self):
        """``infer_one_time`` returns cloud_top_height in METERS (not km).

        This matches the stereo-cache contract so the IGRA evaluator works
        as a drop-in.
        """
        sat = _tiny_sat()
        valid = np.ones((sat.n_rows, sat.n_cols), bool)
        flow_bands = ["C14"]
        rad_bands = ["C14"]
        t0 = np.datetime64("2025-07-01T12:00")
        cubes = _make_cubes(sat, t0, set(flow_bands) | set(rad_bands), valid)
        disp = _StubDisp(seed=7)
        # Seed the model init: the assertion below is on the *magnitude* of the
        # denormalized height, and an unseeded head lands under the 1 km
        # threshold for ~4.5% of initializations (min observed 737 m).
        import torch
        torch.manual_seed(0)
        model = StudentWindsModel(
            n_flow_bands=1, n_rad_bands=1, hidden=8, n_layers=2, context=False,
        )
        # Fake a target_sd / target_mu so h_mean is denormalized to a known scale
        model.target_mu.copy_(torch.tensor([0.0, 0.0, 8.0]))  # h_km mean
        model.target_sd.copy_(torch.tensor([1.0, 1.0, 1.0]))
        model.eval()
        out = infer_one_time(
            model, disp, cubes, sat, t0, flow_bands, rad_bands,
            row_strip=64, halo=0, device="cpu",
        )
        h = out["cloud_top_height"]
        m = np.isfinite(h)
        # h is returned in METERS (not km).  With target_mu_h=8 km and a
        # randomly-initialized unbounded head, denormalized predictions sit
        # near +/- a few km — absolute scale should be thousands of meters.
        assert np.abs(h[m]).mean() > 1000.0, (
            f"|h|.mean()={np.abs(h[m]).mean()} too small — looks like km not m")


def test_time_to_yyyymm():
    assert time_to_yyyymm(np.datetime64("2025-07-01T12:00")) == "202507"
    assert time_to_yyyymm(np.datetime64("2025-10-15T00:30")) == "202510"


class TestCacheTimeOrdering:
    """Regression for the append-mode time-coord scramble we hit.

    `to_zarr(append_dim="time")` does not preserve write order; the new
    template + region-write pattern in ``cache_student_inference`` should
    put each scene at its pre-assigned chronological slot, regardless of
    the order writes happen.
    """

    def test_preserves_chronological_time_coord(self, tmp_path):
        from cache_student_inference import (
            prealloc_template, write_scene, VAR_KEYS,
        )

        eval_times = np.asarray(
            [np.datetime64("2025-07-01T12:00"),
             np.datetime64("2025-07-13T12:00"),
             np.datetime64("2025-10-15T00:00")],
            dtype="datetime64[ns]",
        )
        H, W = 32, 32
        out = str(tmp_path / "cache.zarr")
        prealloc_template(out, eval_times, H, W, {"satellite_id": "test"})

        def _scene(idx):
            # Put the scene index into every var so we can read it back per slot.
            return {k: np.full((H, W), float(idx), dtype=np.float32) for k in VAR_KEYS}

        # Write OUT of chronological order on purpose: slot 2, then 0, then 1.
        write_scene(out, 2, _scene(2))
        write_scene(out, 0, _scene(0))
        write_scene(out, 1, _scene(1))

        ds = xr.open_zarr(out, chunks=None)
        # Time coord stays chronological (template-controlled).
        assert (np.asarray(ds.time.values) == eval_times).all()
        # Data lands at the right slot, not the write order.
        for i in (0, 1, 2):
            assert float(ds["u_wind"].values[i].mean()) == float(i)
        # All three flags marked written.
        assert (ds["_written"].values == 1).all()

    def test_resume_skips_filled_slots(self, tmp_path):
        from cache_student_inference import (
            prealloc_template, write_scene, _resume_todo, VAR_KEYS,
        )

        eval_times = np.asarray(
            [np.datetime64("2025-07-01T12:00"),
             np.datetime64("2025-07-02T00:00"),
             np.datetime64("2025-07-02T12:00")],
            dtype="datetime64[ns]",
        )
        out = str(tmp_path / "cache.zarr")
        prealloc_template(out, eval_times, 16, 16, {"satellite_id": "test"})
        # Initially all slots are TODO.
        assert _resume_todo(out).tolist() == [0, 1, 2]
        # After writing slot 0 and slot 2, only slot 1 remains.
        scene = {k: np.zeros((16, 16), np.float32) for k in VAR_KEYS}
        scene["quality_flag"] = np.full((16, 16), 2.0, np.float32)
        write_scene(out, 0, scene)
        write_scene(out, 2, scene)
        assert _resume_todo(out).tolist() == [1]
