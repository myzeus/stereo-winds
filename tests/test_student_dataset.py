"""Tests for StudentWindsDataset — synthetic feature/label Zarr alignment."""

import numpy as np
import xarray as xr

from stereo_winds.student_dataset import (
    StudentWindsDataset, flow_var, rad_var, FLOW_STUBS, GEOM_VARS, FLOW_SCALE,
)

# Decoupled bands: flow on a subset, radiance on a (different) superset.
FLOW_BANDS = ["C14", "C08"]
RAD_BANDS = ["C07", "C14", "C08"]   # C07 is radiance-only (no flow)
N_GRID = 64          # full-disk grid (small)
ROW_OFF, COL_OFF = 10, 12
FEAT_H, FEAT_W = 40, 40
N_TIMES = 10


def _build(tmp_path):
    feat_path = tmp_path / "student_feat.zarr"
    lab_path = tmp_path / "labels.zarr"
    mask_path = tmp_path / "valid_mask.npy"

    # 10-min spaced times so the train/val slot split is exercised
    times = np.array([np.datetime64("2026-01-01T00:00") + np.timedelta64(10 * i, "m")
                      for i in range(N_TIMES)])

    # Feature flows encode feature-local position: value = i*1000 + j
    ii, jj = np.meshgrid(np.arange(FEAT_H), np.arange(FEAT_W), indexing="ij")
    pos = (ii * 1000 + jj).astype(np.float32)
    fvars = {}
    for b in FLOW_BANDS:
        for s in FLOW_STUBS:
            fvars[flow_var(s, b)] = (("time", "y", "x"),
                                     np.broadcast_to(pos, (N_TIMES, FEAT_H, FEAT_W)).copy())
    for b in RAD_BANDS:  # varying raw radiance so per-band z-score is meaningful
        fvars[rad_var(b)] = (("time", "y", "x"),
                             np.broadcast_to(pos, (N_TIMES, FEAT_H, FEAT_W)).copy())
    for v in GEOM_VARS:
        fvars[v] = (("y", "x"), np.full((FEAT_H, FEAT_W), 2000.0, np.float32))
    feat = xr.Dataset(fvars, coords={"time": times})
    feat.attrs.update(row_offset=ROW_OFF, col_offset=COL_OFF,
                      flow_bands=",".join(FLOW_BANDS), rad_bands=",".join(RAD_BANDS))
    feat.to_zarr(feat_path, mode="w")

    # Labels encode FULL-DISK position: u_wind = r*1000 + c
    rr, cc = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")
    fullpos = (rr * 1000 + cc).astype(np.float32)
    lvars = {
        "u_wind": (("time", "y", "x"), np.broadcast_to(fullpos, (N_TIMES, N_GRID, N_GRID)).copy()),
        "v_wind": (("time", "y", "x"), np.zeros((N_TIMES, N_GRID, N_GRID), np.float32)),
        "cloud_top_height": (("time", "y", "x"), np.full((N_TIMES, N_GRID, N_GRID), 8000.0, np.float32)),
        "quality_flag": (("time", "y", "x"), np.full((N_TIMES, N_GRID, N_GRID), 2.0, np.float32)),
        "sigma_u": (("time", "y", "x"), np.ones((N_TIMES, N_GRID, N_GRID), np.float32)),
        "sigma_v": (("time", "y", "x"), np.ones((N_TIMES, N_GRID, N_GRID), np.float32)),
        "sigma_h": (("time", "y", "x"), np.full((N_TIMES, N_GRID, N_GRID), 500.0, np.float32)),
    }
    xr.Dataset(lvars, coords={"time": times}).to_zarr(lab_path, mode="w")

    mask = np.zeros((N_GRID, N_GRID), bool)
    mask[ROW_OFF:ROW_OFF + FEAT_H, COL_OFF:COL_OFF + FEAT_W] = True
    np.save(mask_path, mask)
    return str(feat_path), str(lab_path), str(mask_path)


def _make(tmp_path, **kw):
    feat, lab, mask = _build(tmp_path)
    defaults = dict(feature_zarr=feat, label_zarr=lab, valid_mask_path=mask,
                    flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS, patch_size=16, seed=0)
    defaults.update(kw)
    return StudentWindsDataset(**defaults)


class TestStudentWindsDataset:
    def test_in_channels(self, tmp_path):
        ds = _make(tmp_path)
        # 4 flow * 2 flow-bands + 3 rad-bands + 3 geom
        assert ds.in_channels == 4 * 2 + 3 + 3
        ds_no_rad = _make(tmp_path, rad_bands=[])
        assert ds_no_rad.in_channels == 4 * 2 + 3

    def test_getitem_shapes(self, tmp_path):
        ds = _make(tmp_path)
        s = ds[0]
        assert s["x"].shape == (ds.in_channels, 16, 16)
        for k in ("u_target", "v_target", "h_target_km", "weight"):
            assert s[k].shape == (16, 16)
        assert s["mask"].dtype == __import__("torch").bool
        assert s["mask"].all()  # qf == 2 everywhere

    def test_height_target_in_km(self, tmp_path):
        ds = _make(tmp_path)
        s = ds[0]
        assert np.allclose(s["h_target_km"].numpy(), 8.0)

    def test_feature_label_alignment(self, tmp_path):
        """A patch at feature-local (r0,c0) must read labels at the offset
        full-disk location and features at the local location."""
        ds = _make(tmp_path)
        r0, c0 = 5, 7
        s = ds._load_patch(ds.valid_times[0], r0, c0)
        # Label target top-left = full-disk (r0+ROW_OFF, c0+COL_OFF)
        exp_u = (r0 + ROW_OFF) * 1000 + (c0 + COL_OFF)
        assert np.isclose(s["u_target"][0, 0].item(), exp_u)
        # Feature channel 0 (flow_back_u of first flow band) top-left = local (r0,c0)/scale
        exp_flow = (r0 * 1000 + c0) / FLOW_SCALE
        assert np.isclose(s["x"][0, 0, 0].item(), exp_flow)

    def test_train_val_disjoint(self, tmp_path):
        feat, lab, mask = _build(tmp_path)
        common = dict(feature_zarr=feat, label_zarr=lab, valid_mask_path=mask,
                      flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS, patch_size=16, seed=0)
        tr = StudentWindsDataset(train=True, **common)
        va = StudentWindsDataset(train=False, **common)
        assert set(map(str, tr.valid_times)).isdisjoint(set(map(str, va.valid_times)))
        assert len(tr.valid_times) + len(va.valid_times) == N_TIMES

    def test_rad_standardization(self, tmp_path):
        ds = _make(tmp_path)
        # one (mean, std) per radiance band, std > 0
        assert ds.rad_stats.shape == (len(RAD_BANDS), 2)
        assert (ds.rad_stats[:, 1] > 0).all()
        # The standardized radiance channels (after the flow channels) should be
        # roughly zero-mean / unit-variance over the full feature extent.
        n_flow = 4 * len(FLOW_BANDS)
        s = ds._load_patch(ds.valid_times[0], 0, 0)  # full-extent patch (ps=16<40)
        rad0 = s["x"][n_flow].numpy()
        assert abs(float(rad0.mean())) < 2.0  # standardized, not raw (~1e4)

    def test_rad_stats_reuse(self, tmp_path):
        """Passing train stats to a second dataset reproduces normalization."""
        ds = _make(tmp_path)
        ds2 = _make(tmp_path, rad_stats=ds.rad_stats)
        n_flow = 4 * len(FLOW_BANDS)
        a = ds._load_patch(ds.valid_times[0], 3, 4)["x"][n_flow:n_flow + len(RAD_BANDS)]
        b = ds2._load_patch(ds2.valid_times[0], 3, 4)["x"][n_flow:n_flow + len(RAD_BANDS)]
        assert np.allclose(a.numpy(), b.numpy())

    def test_teacher_sigma_optional(self, tmp_path):
        ds = _make(tmp_path, use_teacher_sigma=True)
        s = ds[0]
        # sigma_u (pixel/s) * dx_m(2000) -> m/s
        assert np.allclose(s["sigma_u_ms"].numpy(), 2000.0)
