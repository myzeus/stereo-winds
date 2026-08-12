"""PyTorch Dataset for the single-satellite wind student.

Pairs single-satellite *input features* (per-channel temporal optical flows +
histogram-equalized radiances + viewing geometry, produced by
``scripts/cache_student_features.py``) with *teacher targets* (stereo u/v/h +
quality flags, produced by ``scripts/cache_stereo_retrievals.py``).

The feature store is cropped to the satellite-overlap bounding box (where
teacher labels are valid); the label store is full-disk on the same fixed
grid.  ``row_offset``/``col_offset`` attrs on the feature store map between the
two.  At construction the needed variables are cropped to the overlap and
**loaded into RAM as plain numpy** — random patch reads are then pure slicing,
which is fast and fork-safe (dask-backed lazy reads deadlock under forked
DataLoader workers).

Input channel order (must match ``cache_student_features.py``)::

    per flow-band:  flow_back_u, flow_back_v, flow_fwd_u, flow_fwd_v  (4 * n_flow)
    per rad-band:   rad                                              (n_rad)
    shared:         dx_m, dy_m, sat_zenith                           (3)

Flow bands and radiance bands are decoupled: optical flow is only needed on a
handful of well-textured bands, while the full IR radiance set (C07-C16) is fed
so the model can disambiguate cloud-top height from multi-band brightness
temperatures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

logger = logging.getLogger(__name__)

PATCH_SIZE = 512
# Bands we run optical flow on (well-textured WV + window channels).
DEFAULT_FLOW_BANDS = ["C08", "C09", "C10", "C12", "C14"]
# Full ABI IR radiance set (C07-C16) — multi-band brightness temperatures give
# the height information that lets a single satellite resolve cloud-top height.
DEFAULT_RAD_BANDS = ["C07", "C08", "C09", "C10", "C11",
                     "C12", "C13", "C14", "C15", "C16"]

# Feature-store variable naming (single source of truth, shared with the
# feature-generation script).
FLOW_STUBS = ["flow_back_u", "flow_back_v", "flow_fwd_u", "flow_fwd_v"]
RAD_STUB = "rad"
# Temporal rad-frame stubs: t-Δ, t+Δ (Δ=10 min, the flow window).  Stored
# alongside ``rad_<band>`` (t₀) so downstream can opt into 3-frame inputs.
RAD_TMINUS_STUB = "rad_tminus"
RAD_TPLUS_STUB = "rad_tplus"
GEOM_VARS = ["dx_m", "dy_m", "sat_zenith"]


def flow_var(stub: str, band: str) -> str:
    return f"{stub}_{band}"


def rad_var(band: str) -> str:
    return f"{RAD_STUB}_{band}"


def rad_tminus_var(band: str) -> str:
    return f"{RAD_TMINUS_STUB}_{band}"


def rad_tplus_var(band: str) -> str:
    return f"{RAD_TPLUS_STUB}_{band}"


# Input normalization constants (applied here; the model assumes normalized
# inputs).  Flows are pixels; FLOW_SCALE ~ a typical 10-min displacement.
# Radiances are standardized PER BAND (z-score) with stats fit from the
# training split — the zeus/earthnetv2 ``StandardScalar`` convention — rather
# than histogram-equalized (which would erase the height-bearing brightness).
FLOW_SCALE = 10.0
PIXEL_SCALE_NORM = 2000.0  # ~nadir pixel footprint (m); dx_m/dy_m grow to limb
ZENITH_NORM = 90.0


def _band_zscore_stats(a: np.ndarray) -> tuple[float, float]:
    """Per-band (mean, std) over finite pixels; std floored to avoid /0."""
    if not np.isfinite(a).any():
        return 0.0, 1.0
    mu = float(np.nanmean(a))
    sd = float(np.nanstd(a))
    return mu, (sd if sd > 1e-6 else 1.0)


class StudentWindsDataset(Dataset):
    """Single-satellite student training patches.

    Parameters
    ----------
    feature_zarr : path(s) to ``student_feat_*.zarr`` (flows + radiance +
        geometry, overlap-cropped).
    label_zarr : path(s) to teacher cache zarr(s) (full-disk, same grid).
    valid_mask_path : full-disk boolean overlap mask (.npy).
    flow_bands : bands whose temporal optical flow is fed (4 channels each).
    rad_bands : bands whose RAW A0 radiance is fed (1 channel each), standardized
        per band; defaults to the full ABI IR set C07-C16.  Pass ``[]`` for none.
    rad_stats : optional (n_rad_bands, 2) array of per-band (mean, std).  If None
        (typical for the train split) they are fit from this dataset and exposed
        as ``self.rad_stats``; pass the train split's stats to the val/inference
        dataset so normalization matches.
    patch_size : spatial crop (default 512).
    qa_min : minimum teacher quality_flag to supervise a pixel (default 1).
    qa_high : quality_flag treated as high-confidence (extra loss weight).
    high_weight : per-pixel loss weight for qa>=qa_high pixels (default 3).
    train, val_mod : deterministic time-based split — a center time is held out
        for validation when ``(slot_index % val_mod) == 0``.  ``train=True``
        keeps the rest; ``train=False`` keeps only the held-out times.
    samples_per_epoch : random samples per epoch (default: #valid times).
    use_teacher_sigma : also emit teacher sigma (converted to m/s) for optional
        uncertainty weighting in the loss.
    seed : RNG seed.

    Notes
    -----
    All required variables are cropped to the overlap and loaded into RAM at
    construction.  Memory ~ ``n_times * (in_channels + ~4) * H * W * 4 B``; for
    many months this can be large — fine for the GH200 (480 GB) at the current
    scale, but a future zarr-direct reader would be needed for very large sets.
    """

    def __init__(
        self,
        feature_zarr: str | Path | list[str | Path],
        label_zarr: str | Path | list[str | Path],
        valid_mask_path: str | Path,
        flow_bands: list[str] = DEFAULT_FLOW_BANDS,
        rad_bands: list[str] = DEFAULT_RAD_BANDS,
        rad_stats: np.ndarray | None = None,
        patch_size: int = PATCH_SIZE,
        qa_min: int = 1,
        qa_high: int = 2,
        high_weight: float = 3.0,
        train: bool = True,
        val_mod: int = 5,
        samples_per_epoch: int | None = None,
        use_teacher_sigma: bool = False,
        min_label_frac: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__()
        self.flow_bands = list(flow_bands)
        self.rad_bands = list(rad_bands)
        self.patch_size = patch_size
        self.qa_min = qa_min
        self.qa_high = qa_high
        self.high_weight = high_weight
        self.use_teacher_sigma = use_teacher_sigma
        self.min_label_frac = min_label_frac

        if isinstance(feature_zarr, (str, Path)):
            feature_zarr = [feature_zarr]
        if isinstance(label_zarr, (str, Path)):
            label_zarr = [label_zarr]

        feat = xr.open_mfdataset(
            [str(p) for p in feature_zarr], engine="zarr", concat_dim="time",
            combine="nested", chunks={}, data_vars="minimal",
        )
        lab = xr.open_mfdataset(
            [str(p) for p in label_zarr], engine="zarr", concat_dim="time",
            combine="nested", chunks={}, data_vars="minimal",
        )

        # Crop offsets + (time-independent) geometry from the first store.
        first = xr.open_zarr(str(feature_zarr[0]))
        self.row_offset = int(first.attrs.get("row_offset", 0))
        self.col_offset = int(first.attrs.get("col_offset", 0))
        self.feat_h = int(first.sizes["y"])
        self.feat_w = int(first.sizes["x"])
        geom_raw = np.stack(
            [first[v].values.astype(np.float32) for v in GEOM_VARS], axis=0,
        )  # (3, feat_h, feat_w)
        first.close()

        # Validate requested vars exist
        missing = [flow_var(s, b) for b in self.flow_bands for s in FLOW_STUBS
                   if flow_var(s, b) not in feat]
        if missing:
            raise ValueError(f"Feature store missing flow vars: {missing[:6]}...")
        rad_missing = [rad_var(b) for b in self.rad_bands if rad_var(b) not in feat]
        if rad_missing:
            raise ValueError(f"Feature store missing radiance vars: {rad_missing}")

        # Common times present in BOTH stores, after the train/val split.
        ft = set(np.asarray(feat.time.values))
        lt = set(np.asarray(lab.time.values))
        common = sorted(ft & lt)
        self.valid_times = [t for t in common if self._in_split(t, train, val_mod)]
        if not self.valid_times:
            raise ValueError("No common feature/label times after the train/val split")
        self._ti = {t: i for i, t in enumerate(self.valid_times)}
        logger.info(
            "StudentWindsDataset: %d %s times (%d common)",
            len(self.valid_times), "train" if train else "val", len(common),
        )

        # ---- Preload into RAM (numpy), cropped to the overlap ----
        times = self.valid_times

        def _feat(var):
            return feat[var].sel(time=times).values.astype(np.float32)

        flow = [_feat(flow_var(s, b)) / FLOW_SCALE
                for b in self.flow_bands for s in FLOW_STUBS]

        # Raw radiance -> per-band z-score (fit here, or reuse passed-in stats).
        rad_raw = [_feat(rad_var(b)) for b in self.rad_bands]
        if rad_stats is None:
            self.rad_stats = np.array(
                [_band_zscore_stats(a) for a in rad_raw], np.float32
            ).reshape(-1, 2)
        else:
            self.rad_stats = np.asarray(rad_stats, np.float32).reshape(-1, 2)
            if len(self.rad_stats) != len(rad_raw):
                raise ValueError("rad_stats length != number of rad_bands")
        rad = [(a - self.rad_stats[i, 0]) / self.rad_stats[i, 1]
               for i, a in enumerate(rad_raw)]

        # (n_times, C_fr, feat_h, feat_w)
        self._Xfr = np.stack(flow + rad, axis=1) if (flow or rad) else \
            np.zeros((len(times), 0, self.feat_h, self.feat_w), np.float32)

        ly = slice(self.row_offset, self.row_offset + self.feat_h)
        lx = slice(self.col_offset, self.col_offset + self.feat_w)

        def _lab(name):
            return lab[name].sel(time=times).isel(y=ly, x=lx).values.astype(np.float32)

        self._u = _lab("u_wind")
        self._v = _lab("v_wind")
        self._h_km = _lab("cloud_top_height") / 1000.0
        self._qf = _lab("quality_flag")

        self._geomn = geom_raw.copy()
        self._geomn[0] /= PIXEL_SCALE_NORM
        self._geomn[1] /= PIXEL_SCALE_NORM
        self._geomn[2] /= ZENITH_NORM

        if use_teacher_sigma:
            self._su = _lab("sigma_u") * geom_raw[0]   # pixel/s -> m/s
            self._sv = _lab("sigma_v") * geom_raw[1]
            self._sh_km = _lab("sigma_h") / 1000.0

        feat.close()
        lab.close()

        # Valid patch origins (feature-local coords) with enough overlap.
        self._build_patch_origins(valid_mask_path)

        self.samples_per_epoch = samples_per_epoch or len(self.valid_times)
        self.rng = np.random.default_rng(seed)

    @property
    def in_channels(self) -> int:
        return 4 * len(self.flow_bands) + len(self.rad_bands) + len(GEOM_VARS)

    @staticmethod
    def _in_split(t: np.datetime64, train: bool, val_mod: int) -> bool:
        slot = int(np.asarray(t).astype("datetime64[m]").astype("int64") // 10)
        is_val = (slot % val_mod) == 0
        return is_val != train  # val times when ~train, train times otherwise

    def _build_patch_origins(self, valid_mask_path, min_valid_frac: float = 0.3):
        mask = np.load(valid_mask_path)
        crop = mask[self.row_offset:self.row_offset + self.feat_h,
                    self.col_offset:self.col_offset + self.feat_w]
        ps = self.patch_size
        step = ps // 2
        origins = []
        for r in range(0, self.feat_h - ps + 1, step):
            for c in range(0, self.feat_w - ps + 1, step):
                if crop[r:r + ps, c:c + ps].mean() >= min_valid_frac:
                    origins.append((r, c))
        if not origins:
            raise ValueError("No valid patch origins within the feature crop")
        self.patch_origins = origins
        logger.info("StudentWindsDataset: %d patch origins", len(origins))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ps = self.patch_size
        for _ in range(50):
            t0 = self.valid_times[self.rng.integers(len(self.valid_times))]
            r0, c0 = self.patch_origins[self.rng.integers(len(self.patch_origins))]
            sample = self._load_patch(t0, r0, c0)
            if sample is not None:
                return sample
        raise RuntimeError("No labeled patch after 50 attempts")

    def _load_patch(self, t0, r0, c0):
        ti = self._ti[t0]
        ps = self.patch_size
        rs, cs = slice(r0, r0 + ps), slice(c0, c0 + ps)

        u = self._u[ti, rs, cs]
        v = self._v[ti, rs, cs]
        h_km = self._h_km[ti, rs, cs]
        qf = self._qf[ti, rs, cs]

        mask = (qf >= self.qa_min) & np.isfinite(u) & np.isfinite(v) & np.isfinite(h_km)
        if mask.mean() < self.min_label_frac:
            return None
        weight = np.where(qf >= self.qa_high, self.high_weight, 1.0).astype(np.float32) * mask

        x = np.concatenate([self._Xfr[ti, :, rs, cs], self._geomn[:, rs, cs]], axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        out = {
            "x": torch.from_numpy(x),
            "u_target": torch.from_numpy(np.nan_to_num(u)),
            "v_target": torch.from_numpy(np.nan_to_num(v)),
            "h_target_km": torch.from_numpy(np.nan_to_num(h_km)),
            "mask": torch.from_numpy(mask),
            "weight": torch.from_numpy(weight),
        }
        if self.use_teacher_sigma:
            out["sigma_u_ms"] = torch.from_numpy(np.nan_to_num(self._su[ti, rs, cs], nan=1.0))
            out["sigma_v_ms"] = torch.from_numpy(np.nan_to_num(self._sv[ti, rs, cs], nan=1.0))
            out["sigma_h_km"] = torch.from_numpy(np.nan_to_num(self._sh_km[ti, rs, cs], nan=1.0))
        return out
