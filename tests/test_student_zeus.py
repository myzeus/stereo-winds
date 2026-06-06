"""Tests for the zeus-tooling student path: xbatcher dataset + BaseLightningModule."""

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from stereo_winds.student_xbatcher import StudentXBatchDataset
from stereo_winds.student_zeus_model import StudentWindsModel
from stereo_winds.student_dataset import (
    flow_var, rad_var, rad_tminus_var, rad_tplus_var, FLOW_STUBS, GEOM_VARS,
)

FLOW_BANDS = ["C14", "C08"]
RAD_BANDS = ["C07", "C14", "C08"]
N_GRID = 96
ROW_OFF, COL_OFF = 8, 10
FH, FW = 64, 64
N_TIMES = 10


def _build(tmp_path):
    feat_p, lab_p = tmp_path / "feat.zarr", tmp_path / "lab.zarr"
    times = np.array([np.datetime64("2026-01-01T00:00") + np.timedelta64(10 * i, "m")
                      for i in range(N_TIMES)])
    rng = np.random.default_rng(0)
    fvars = {}
    for b in FLOW_BANDS:
        for s in FLOW_STUBS:
            fvars[flow_var(s, b)] = (("time", "y", "x"),
                                     rng.normal(0, 3, (N_TIMES, FH, FW)).astype("f4"))
    for b in RAD_BANDS:  # raw radiance with per-band offset/scale
        fvars[rad_var(b)] = (("time", "y", "x"),
                             rng.normal(250, 20, (N_TIMES, FH, FW)).astype("f4"))
    for v in GEOM_VARS:
        fvars[v] = (("y", "x"), np.full((FH, FW), 2000.0, "f4"))
    feat = xr.Dataset(fvars, coords={"time": times})
    feat.attrs.update(row_offset=ROW_OFF, col_offset=COL_OFF)
    feat.to_zarr(feat_p, mode="w")

    lv = {
        "u_wind": (("time", "y", "x"), rng.normal(0, 15, (N_TIMES, N_GRID, N_GRID)).astype("f4")),
        "v_wind": (("time", "y", "x"), rng.normal(0, 15, (N_TIMES, N_GRID, N_GRID)).astype("f4")),
        "cloud_top_height": (("time", "y", "x"), rng.uniform(0, 12000, (N_TIMES, N_GRID, N_GRID)).astype("f4")),
        "quality_flag": (("time", "y", "x"), np.full((N_TIMES, N_GRID, N_GRID), 2.0, "f4")),
    }
    xr.Dataset(lv, coords={"time": times}).to_zarr(lab_p, mode="w")
    return str(feat_p), str(lab_p)


def _ds(tmp_path, **kw):
    f, l = _build(tmp_path)
    d = dict(feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS,
             patch_size=16, seed=0)
    d.update(kw)
    return StudentXBatchDataset(**d)


def _build_3t(tmp_path):
    """Same as _build but adds rad_tminus_<b> and rad_tplus_<b> per rad band."""
    feat_p, lab_p = tmp_path / "feat.zarr", tmp_path / "lab.zarr"
    times = np.array([np.datetime64("2026-01-01T00:00") + np.timedelta64(10 * i, "m")
                      for i in range(N_TIMES)])
    rng = np.random.default_rng(0)
    fvars = {}
    for b in FLOW_BANDS:
        for s in FLOW_STUBS:
            fvars[flow_var(s, b)] = (("time", "y", "x"),
                                     rng.normal(0, 3, (N_TIMES, FH, FW)).astype("f4"))
    for b in RAD_BANDS:
        fvars[rad_var(b)] = (("time", "y", "x"),
                             rng.normal(250, 20, (N_TIMES, FH, FW)).astype("f4"))
        fvars[rad_tminus_var(b)] = (("time", "y", "x"),
                                    rng.normal(250, 20, (N_TIMES, FH, FW)).astype("f4"))
        fvars[rad_tplus_var(b)] = (("time", "y", "x"),
                                   rng.normal(250, 20, (N_TIMES, FH, FW)).astype("f4"))
    for v in GEOM_VARS:
        fvars[v] = (("y", "x"), np.full((FH, FW), 2000.0, "f4"))
    feat = xr.Dataset(fvars, coords={"time": times})
    feat.attrs.update(row_offset=ROW_OFF, col_offset=COL_OFF, rad_time_frames=3)
    feat.to_zarr(feat_p, mode="w")

    lv = {
        "u_wind": (("time", "y", "x"), rng.normal(0, 15, (N_TIMES, N_GRID, N_GRID)).astype("f4")),
        "v_wind": (("time", "y", "x"), rng.normal(0, 15, (N_TIMES, N_GRID, N_GRID)).astype("f4")),
        "cloud_top_height": (("time", "y", "x"), rng.uniform(0, 12000, (N_TIMES, N_GRID, N_GRID)).astype("f4")),
        "quality_flag": (("time", "y", "x"), np.full((N_TIMES, N_GRID, N_GRID), 2.0, "f4")),
    }
    xr.Dataset(lv, coords={"time": times}).to_zarr(lab_p, mode="w")
    return str(feat_p), str(lab_p)


class TestStudentXBatch:
    def test_item_shapes(self, tmp_path):
        ds = _ds(tmp_path)
        s = ds[0]
        assert s["flow"].shape == (4 * len(FLOW_BANDS), 16, 16)
        assert s["rad"].shape == (len(RAD_BANDS), 16, 16)
        assert s["geom"].shape == (3, 16, 16)
        for k in ("u", "v", "h_km", "weight"):
            assert s[k].shape == (16, 16)
        assert s["mask"].dtype == torch.bool and s["mask"].all()
        # radiance is RAW (not standardized in the dataset)
        assert float(s["rad"].mean()) > 100

    def test_train_val_split(self, tmp_path):
        f, l = _build(tmp_path)
        common = dict(feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
                      rad_bands=RAD_BANDS, patch_size=16)
        tr = StudentXBatchDataset(train=True, **common)
        va = StudentXBatchDataset(train=False, **common)
        assert len(tr) > 0 and len(va) > 0


class TestMonthSplit:
    """Held-out-month CV — actually tests generalization, unlike val_mod=5
    which decimates within training months and overfits."""

    def test_in_month_split_static(self):
        t_feb = np.datetime64("2025-02-15T12:00")
        t_mar = np.datetime64("2025-03-01T00:00")
        # train=True, val_months=[Feb] → Feb is val, so train should EXCLUDE Feb
        assert StudentXBatchDataset._in_month_split(t_feb, train=True,  val_months=["2025-02"]) is False
        assert StudentXBatchDataset._in_month_split(t_mar, train=True,  val_months=["2025-02"]) is True
        # train=False (val) → only Feb passes
        assert StudentXBatchDataset._in_month_split(t_feb, train=False, val_months=["2025-02"]) is True
        assert StudentXBatchDataset._in_month_split(t_mar, train=False, val_months=["2025-02"]) is False
        # multi-month val
        t_oct = np.datetime64("2024-10-12T18:00")
        assert StudentXBatchDataset._in_month_split(t_oct, train=False, val_months=["2024-10", "2025-02"]) is True
        assert StudentXBatchDataset._in_month_split(t_oct, train=True,  val_months=["2024-10", "2025-02"]) is False

    def test_dataset_holds_out_month(self, tmp_path):
        """Build a multi-month dataset, hold out one month, verify partition."""
        feat_p, lab_p = tmp_path / "feat.zarr", tmp_path / "lab.zarr"
        # 4 times in Jan 2025, 4 times in Feb 2025 — Feb is the held-out month
        times = np.array(
            [np.datetime64("2025-01-05T00:00") + np.timedelta64(6 * i, "h") for i in range(4)]
            + [np.datetime64("2025-02-10T00:00") + np.timedelta64(6 * i, "h") for i in range(4)]
        )
        n = len(times)
        rng = np.random.default_rng(0)
        fvars = {}
        for b in FLOW_BANDS:
            for s in FLOW_STUBS:
                fvars[flow_var(s, b)] = (("time", "y", "x"),
                                         rng.normal(0, 3, (n, FH, FW)).astype("f4"))
        for b in RAD_BANDS:
            fvars[rad_var(b)] = (("time", "y", "x"),
                                 rng.normal(250, 20, (n, FH, FW)).astype("f4"))
        for v in GEOM_VARS:
            fvars[v] = (("y", "x"), np.full((FH, FW), 2000.0, "f4"))
        feat = xr.Dataset(fvars, coords={"time": times})
        feat.attrs.update(row_offset=ROW_OFF, col_offset=COL_OFF)
        feat.to_zarr(feat_p, mode="w")
        lv = {
            "u_wind": (("time", "y", "x"), rng.normal(0, 15, (n, N_GRID, N_GRID)).astype("f4")),
            "v_wind": (("time", "y", "x"), rng.normal(0, 15, (n, N_GRID, N_GRID)).astype("f4")),
            "cloud_top_height": (("time", "y", "x"), rng.uniform(0, 12000, (n, N_GRID, N_GRID)).astype("f4")),
            "quality_flag": (("time", "y", "x"), np.full((n, N_GRID, N_GRID), 2.0, "f4")),
        }
        xr.Dataset(lv, coords={"time": times}).to_zarr(lab_p, mode="w")

        common = dict(feature_zarr=str(feat_p), label_zarr=str(lab_p),
                      flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS, patch_size=16,
                      val_months=["2025-02"])
        tr = StudentXBatchDataset(train=True, **common)
        va = StudentXBatchDataset(train=False, **common)
        tr_times = np.unique(np.asarray(tr.ds.time.values).astype("datetime64[M]"))
        va_times = np.unique(np.asarray(va.ds.time.values).astype("datetime64[M]"))
        assert tr_times.tolist() == [np.datetime64("2025-01")], tr_times
        assert va_times.tolist() == [np.datetime64("2025-02")], va_times

    def test_val_mod_path_unchanged_when_val_months_none(self, tmp_path):
        """Default behavior (val_mod=5, no val_months) is preserved."""
        f, l = _build(tmp_path)
        common = dict(feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
                      rad_bands=RAD_BANDS, patch_size=16)
        tr = StudentXBatchDataset(train=True, val_months=None, **common)
        va = StudentXBatchDataset(train=False, val_months=None, **common)
        # Same as the legacy split test — both partitions are non-empty.
        assert len(tr) > 0 and len(va) > 0


class TestStudentZeusModel:
    def test_forward_and_transform(self, tmp_path):
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = StudentWindsModel(n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
                                  hidden=16, n_layers=2)
        # fit the radiance StandardScalar
        model.prepare_data_transformation(loader, n_batches=2)
        assert model.transform.mu["rad"].shape == (len(RAD_BANDS),)
        assert (model.transform.sd["rad"] > 0).all()
        batch = next(iter(loader))
        out = model(batch["flow"], batch["rad"], batch["geom"])
        assert out["u_mean"].shape == (4, 16, 16)
        # predictions are standardized (no hard bounds); just finite.
        assert torch.isfinite(out["h_mean"]).all()
        assert model.target_mu.shape == (3,) and model.target_sd.shape == (3,)

    def test_two_step_fit(self, tmp_path):
        import pytorch_lightning as pl
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = StudentWindsModel(n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
                                  hidden=16, n_layers=2)
        trainer = pl.Trainer(accelerator="cpu", max_steps=2, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        assert trainer.global_step == 2


class TestVectorWindLoss:
    """Joint-logvar / vector NLL path: forward keys, predict() shape, two-step fit."""

    def _model(self, **kw):
        return StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="vector",
            logvar_init_offset=5.0, **kw,
        )

    def test_forward_returns_joint_logvar(self, tmp_path):
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = self._model()
        model.prepare_data_transformation(loader, n_batches=2)
        batch = next(iter(loader))
        out = model(batch["flow"], batch["rad"], batch["geom"])
        # Joint mode: uv_logvar instead of separate u_logvar / v_logvar.
        assert "uv_logvar" in out
        assert "u_logvar" not in out and "v_logvar" not in out
        assert out["uv_logvar"].shape == out["u_mean"].shape
        assert "h_logvar" in out  # height still per-target

    def test_predict_surfaces_uv_sigma_symmetrically(self, tmp_path):
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = self._model()
        model.prepare_data_transformation(loader, n_batches=2)
        batch = next(iter(loader))
        out = model.predict(batch["flow"], batch["rad"], batch["geom"])
        # predict() guarantees downstream consumers get a stable dict shape.
        for k in ("u_mean", "v_mean", "h_mean", "u_logvar", "v_logvar", "h_logvar"):
            assert k in out
        # In vector mode u_logvar == v_logvar pixel-wise (both = joint physical logvar).
        assert torch.allclose(out["u_logvar"], out["v_logvar"])

    def test_two_step_fit_vector(self, tmp_path):
        import pytorch_lightning as pl
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = self._model()
        trainer = pl.Trainer(accelerator="cpu", max_steps=2, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        assert trainer.global_step == 2

    def test_head_channel_count(self, tmp_path):
        """Vector mode → 5-channel head; gaussian mode → 6-channel head."""
        m_vec = self._model()
        m_gauss = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="gaussian",
        )
        assert m_vec.net.n_out == 5
        assert m_gauss.net.n_out == 6


class TestUNetTrunk:
    """U-Net trunk wired into StudentWindsModel preserves the same step contract."""

    def test_construct_with_unet(self):
        m = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            trunk="unet", unet_base_channels=16, unet_n_levels=2,
            wind_loss="vector",
        )
        assert m.trunk_name == "unet"
        # The U-Net has 5-channel output in joint mode.
        assert m.net.n_out == 5

    def test_two_step_fit_unet(self, tmp_path):
        import pytorch_lightning as pl
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            trunk="unet", unet_base_channels=16, unet_n_levels=2,
            wind_loss="vector",
        )
        trainer = pl.Trainer(accelerator="cpu", max_steps=2, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        assert trainer.global_step == 2


class TestRadTimeFrames:
    """3-frame radiance input: t₋ / t₀ / t₊ stacked per band."""

    def test_dataset_rad_channels_3x(self, tmp_path):
        f, l = _build_3t(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS,
            patch_size=16, seed=0, rad_time_frames=3,
        )
        s = ds[0]
        # 3 frames × 3 rad bands = 9 channels
        assert s["rad"].shape == (3 * len(RAD_BANDS), 16, 16)
        # flow + geom unchanged
        assert s["flow"].shape == (4 * len(FLOW_BANDS), 16, 16)
        assert s["geom"].shape == (3, 16, 16)

    def test_dataset_frame_interleave_order(self, tmp_path):
        """Channel order is [t-,t0,t+] per band, bands in outer loop."""
        f, l = _build_3t(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS,
            patch_size=16, seed=0, rad_time_frames=3,
        )
        s = ds[0]
        # The first 3 channels are band 0's [t-, t0, t+]
        # By construction the three frames are drawn from different rng states
        # so they're statistically distinct; assert no two are bit-identical.
        rad = s["rad"].numpy()
        assert not np.array_equal(rad[0], rad[1])
        assert not np.array_equal(rad[1], rad[2])

    def test_default_1frame_unchanged(self, tmp_path):
        """rad_time_frames=1 (default) still emits n_rad channels."""
        ds = _ds(tmp_path)
        s = ds[0]
        assert s["rad"].shape == (len(RAD_BANDS), 16, 16)

    def test_model_input_channels_3x(self, tmp_path):
        m = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, rad_time_frames=3,
        )
        # in_channels = 4*n_flow + 3*n_rad + 3
        expected = 4 * len(FLOW_BANDS) + 3 * len(RAD_BANDS) + 3
        assert m.net.in_channels == expected
        assert m.rad_time_frames == 3
        # Transform width matches the 3x rad channels.
        assert m.transform.mu["rad"].shape == (3 * len(RAD_BANDS),)

    def test_two_step_fit_3t(self, tmp_path):
        import pytorch_lightning as pl
        f, l = _build_3t(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS, rad_bands=RAD_BANDS,
            patch_size=16, seed=0, rad_time_frames=3,
        )
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, rad_time_frames=3, wind_loss="vector",
        )
        trainer = pl.Trainer(accelerator="cpu", max_steps=2, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        assert trainer.global_step == 2

    def test_rejects_invalid_n_frames(self):
        with pytest.raises(ValueError):
            StudentWindsModel(n_flow_bands=2, n_rad_bands=3, rad_time_frames=2)
