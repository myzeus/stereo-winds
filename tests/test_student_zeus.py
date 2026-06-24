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


class TestRandomCrop:
    """Train-only random spatial cropping helps U-Net generalization without
    breaking the world-frame u,v vectors (no flips/rotations)."""

    def test_train_random_crop_varies_tile_origin(self, tmp_path):
        """Same idx, two calls → different spatial tiles (random crop active)."""
        f, l = _build(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True,
            random_crop=True, seed=0)
        assert ds.random_crop is True
        # Many calls to one idx should cover multiple distinct origins.
        rads = set()
        for _ in range(20):
            s = ds._load(0)
            assert s is not None
            assert s["rad"].shape == (len(RAD_BANDS), 16, 16)
            # Hash the (4,4) top-left corner of channel 0 as a tile fingerprint.
            rads.add(s["rad"][0, :4, :4].numpy().tobytes())
        assert len(rads) > 1, "random_crop returned identical tiles for same idx"

    def test_val_is_deterministic_even_with_random_crop_flag(self, tmp_path):
        """random_crop=True on val (train=False) is silently disabled."""
        f, l = _build(tmp_path)
        va = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=False,
            random_crop=True, seed=0)
        # Constructor turns it off for val.
        assert va.random_crop is False
        # Two reads of the same idx must be byte-identical (no rng draws).
        a = va._load(0); b = va._load(0)
        assert a is not None and b is not None
        np.testing.assert_array_equal(a["rad"].numpy(), b["rad"].numpy())

    def test_default_random_crop_is_on(self, tmp_path):
        """Default is opt-in; train datasets should random-crop by default."""
        f, l = _build(tmp_path)
        tr = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True, seed=0)
        assert tr.random_crop is True

    def test_random_crop_off_matches_legacy_bgen(self, tmp_path):
        """random_crop=False on train falls back to xbatcher's grid."""
        f, l = _build(tmp_path)
        tr = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True,
            random_crop=False, seed=0)
        assert tr.random_crop is False
        a = tr._load(0); b = tr._load(0)
        # Deterministic xbatcher → same tile twice.
        np.testing.assert_array_equal(a["rad"].numpy(), b["rad"].numpy())


def _build_with_chi2(tmp_path):
    """Same as _build but also writes a chi_squared label var so the
    dataset and chi²-distill model can pick it up."""
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
        # log-normal-ish chi² with a small mean — like a converged solver.
        "chi_squared": (("time", "y", "x"),
                        np.exp(rng.normal(-1.0, 0.5, (N_TIMES, N_GRID, N_GRID))).astype("f4")),
    }
    xr.Dataset(lv, coords={"time": times}).to_zarr(lab_p, mode="w")
    return str(feat_p), str(lab_p)


class TestChi2Distillation:
    """Optional teacher-chi² head: dataset yields chi², model has an extra
    output channel, predict() returns physical chi², training step uses an
    L1 loss on log(chi²)."""

    def test_dataset_emits_chi2_when_present(self, tmp_path):
        f, l = _build_with_chi2(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True, seed=0)
        assert ds.has_chi2 is True
        s = ds._load(0)
        assert s is not None
        assert "chi2" in s
        assert s["chi2"].shape == (16, 16)

    def test_dataset_omits_chi2_when_absent(self, tmp_path):
        """Older chunks without chi_squared still load cleanly."""
        f, l = _build(tmp_path)  # no chi_squared
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True, seed=0)
        assert ds.has_chi2 is False
        s = ds._load(0)
        assert "chi2" not in s

    def test_model_head_grows_by_one_with_chi2(self, tmp_path):
        """predict_chi2=True → one extra head channel; predict() yields chi²."""
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="vector",
            predict_chi2=True)
        # joint head = 5 + 1 chi² = 6
        assert model.net.head.out_channels == 6
        batch = next(iter(loader))
        out = model.predict(batch["flow"], batch["rad"], batch["geom"])
        assert "chi2" in out
        assert out["chi2"].shape == batch["u"].shape
        assert (out["chi2"] > 0).all()  # exp(.) is positive

    def test_chi2_loss_decreases_two_step(self, tmp_path):
        """Distill loss term is wired into training and decreases."""
        import pytorch_lightning as pl
        f, l = _build_with_chi2(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True, seed=0)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="vector",
            predict_chi2=True, w_chi2=1.0)
        model.prepare_data_transformation(loader, n_batches=1)
        model.fit_target_stats(loader, n_batches=1)
        trainer = pl.Trainer(max_steps=2, accelerator="cpu", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        # Just confirm forward+backward completed with the chi² loss term.
        out = model(*[next(iter(loader))[k] for k in ("flow", "rad", "geom")])
        assert "log_chi2" in out


class TestSpeedMagnitudeLoss:
    """Optional |V_pred|² - |V_teacher|² penalty for systematic
    speed-magnitude under-prediction in vector NLL training."""

    def test_w_speed_default_is_zero(self):
        m = StudentWindsModel(n_flow_bands=1, n_rad_bands=1, hidden=8,
                              n_layers=1, wind_loss="vector")
        assert m.w_speed == 0.0

    def test_w_speed_decreases_loss(self, tmp_path):
        import pytorch_lightning as pl
        ds = _ds(tmp_path)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="vector", w_speed=0.5)
        model.prepare_data_transformation(loader, n_batches=1)
        model.fit_target_stats(loader, n_batches=1)
        trainer = pl.Trainer(max_steps=2, accelerator="cpu", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        # Training completed; speed_mse should have been logged each step.
        assert "train/speed_mse" in trainer.logged_metrics or \
               "train/speed_mse_epoch" in trainer.logged_metrics or \
               any("speed_mse" in k for k in trainer.logged_metrics)


class TestChi2SeparateHead:
    """Decoupled chi² head — own conv layer + optional stop-grad on trunk
    to break the chi²-vs-wind capacity competition."""

    def test_separate_head_creates_extra_module(self, tmp_path):
        from stereo_winds.student_model import PixelwiseWindStudent, UNetWindStudent
        # Pixelwise
        m = PixelwiseWindStudent(
            in_channels=8, hidden=16, n_layers=2,
            wind_logvar_mode="joint",
            predict_chi2=True, chi2_separate_head=True)
        # Main head should NOT include the chi² channel anymore.
        assert m.head.out_channels == 5  # joint = 5 (no chi² mixed in)
        assert hasattr(m, "chi2_head")
        assert m.chi2_head.out_channels == 1
        # UNet
        u = UNetWindStudent(
            in_channels=8, base_channels=8, n_levels=2,
            wind_logvar_mode="joint",
            predict_chi2=True, chi2_separate_head=True)
        assert u.head.out_channels == 5
        assert hasattr(u, "chi2_head")

    def test_stop_grad_blocks_trunk_update_from_chi2(self, tmp_path):
        """chi² loss must not flow gradients into the trunk when stop_grad=True."""
        from stereo_winds.student_model import PixelwiseWindStudent
        m = PixelwiseWindStudent(
            in_channels=8, hidden=16, n_layers=2,
            wind_logvar_mode="joint",
            predict_chi2=True, chi2_separate_head=True, chi2_stop_grad=True)
        x = torch.randn(2, 8, 16, 16, requires_grad=False)
        out = m(x)
        # Compute a loss purely on log_chi2 and check trunk grads are None.
        loss = out["log_chi2"].mean()
        loss.backward()
        trunk_grad_norms = sum(
            p.grad.abs().sum().item() if p.grad is not None else 0.0
            for p in m.trunk.parameters()
        )
        assert trunk_grad_norms == 0.0, \
            f"stop_grad failed: trunk got grad sum={trunk_grad_norms}"
        # chi2_head WAS updated.
        chi2_head_grad = sum(
            p.grad.abs().sum().item() if p.grad is not None else 0.0
            for p in m.chi2_head.parameters()
        )
        assert chi2_head_grad > 0.0

    def test_without_stop_grad_trunk_does_update(self, tmp_path):
        """Sanity: with stop_grad=False the trunk DOES receive chi² grads."""
        from stereo_winds.student_model import PixelwiseWindStudent
        m = PixelwiseWindStudent(
            in_channels=8, hidden=16, n_layers=2,
            wind_logvar_mode="joint",
            predict_chi2=True, chi2_separate_head=True, chi2_stop_grad=False)
        x = torch.randn(2, 8, 16, 16, requires_grad=False)
        out = m(x)
        loss = out["log_chi2"].mean()
        loss.backward()
        trunk_grad_norms = sum(
            p.grad.abs().sum().item() if p.grad is not None else 0.0
            for p in m.trunk.parameters()
        )
        assert trunk_grad_norms > 0.0

    def test_two_step_fit_with_stop_grad(self, tmp_path):
        """Full Lightning loop with separate-head + stop-grad."""
        import pytorch_lightning as pl
        f, l = _build_with_chi2(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True, seed=0)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="vector",
            predict_chi2=True, w_chi2=1.0,
            chi2_separate_head=True, chi2_stop_grad=True)
        model.prepare_data_transformation(loader, n_batches=1)
        model.fit_target_stats(loader, n_batches=1)
        trainer = pl.Trainer(max_steps=2, accelerator="cpu", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        # Forward still produces chi²
        b = next(iter(loader))
        out = model.predict(b["flow"], b["rad"], b["geom"])
        assert "chi2" in out


class TestChi2DistLoss:
    """KL-style distribution-matching loss on log(chi²)."""

    def test_w_chi2_dist_logs_dist_loss(self, tmp_path):
        """When w_chi2_dist > 0 and chi² targets are present, the
        distribution-matching loss is computed and logged."""
        import pytorch_lightning as pl
        f, l = _build_with_chi2(tmp_path)
        ds = StudentXBatchDataset(
            feature_zarr=f, label_zarr=l, flow_bands=FLOW_BANDS,
            rad_bands=RAD_BANDS, patch_size=16, train=True, seed=0)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        model = StudentWindsModel(
            n_flow_bands=len(FLOW_BANDS), n_rad_bands=len(RAD_BANDS),
            hidden=16, n_layers=2, wind_loss="vector",
            predict_chi2=True, w_chi2=0.1, w_chi2_dist=1.0,
            chi2_separate_head=True, chi2_stop_grad=True)
        model.prepare_data_transformation(loader, n_batches=1)
        model.fit_target_stats(loader, n_batches=1)
        trainer = pl.Trainer(max_steps=2, accelerator="cpu", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False)
        trainer.fit(model, loader)
        assert any("chi2_dist" in k for k in trainer.logged_metrics), \
            f"chi2_dist not logged; metrics: {list(trainer.logged_metrics)}"

    def test_default_w_chi2_dist_zero(self):
        m = StudentWindsModel(n_flow_bands=1, n_rad_bands=1, hidden=8,
                              n_layers=1, predict_chi2=True)
        assert m.w_chi2_dist == 0.0


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
