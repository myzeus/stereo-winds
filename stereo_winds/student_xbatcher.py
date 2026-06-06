"""xbatcher-based dataset for the single-satellite wind student.

Streams (C, H, W) patches from the cropped feature Zarr (per-channel temporal
flows + RAW radiance + geometry) merged with the teacher label Zarr (stereo
u/v/h + quality), following the zeus ``xbatcher.BatchGenerator`` pattern used
across the codebase.

Each item is a dict of per-modality tensors:
    flow   : (4*n_flow, ps, ps)  temporal flows, scaled by FLOW_SCALE
    rad    : (n_rad,    ps, ps)  RAW radiance  -- standardized in the model by
             a zeus ``StandardScalar`` transform (task="rad"), NOT here
    geom   : (3,        ps, ps)  dx_m/dy_m/sat_zenith, fixed-scaled
    u,v    : (ps, ps)            teacher wind targets (m/s)
    h_km   : (ps, ps)            teacher cloud-top height (km)
    mask   : (ps, ps) bool       supervised pixels (quality_flag >= qa_min)
    weight : (ps, ps)            per-pixel loss weight (qa_high upweighted)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import xbatcher
from torch.utils.data import Dataset

from .student_dataset import (
    flow_var, rad_var, rad_tminus_var, rad_tplus_var, FLOW_STUBS, GEOM_VARS,
    DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
    FLOW_SCALE, PIXEL_SCALE_NORM, ZENITH_NORM,
)

logger = logging.getLogger(__name__)

PATCH_SIZE = 256
_LABEL_VARS = ["u_wind", "v_wind", "cloud_top_height", "quality_flag"]


class StudentXBatchDataset(Dataset):
    """xbatcher patches pairing single-sat features with teacher labels.

    Parameters
    ----------
    feature_zarr : str | Path
        Cropped feature store (one chunk).  For multiple chunks, build one
        dataset per chunk and combine via ``torch.utils.data.ConcatDataset`` —
        cross-chunk ``xr.concat`` materializes the full crop into RAM (~190 GB
        for 118 scenes) and gets OOM-killed.
    label_zarr : str | Path | None
        Teacher label store.  Pass ``None`` when the feature store is a
        *combined* store that already contains the label vars.
    valid_mask_path : full-disk overlap mask (.npy), used only to skip empty tiles.
    flow_bands, rad_bands : input band lists (order fixed, must match the store).
    patch_size : spatial tile (default 256).
    overlap : xbatcher tile overlap in pixels (default patch_size//2).
    qa_min, qa_high, high_weight : QA masking / weighting (see StudentWindsDataset).
    train, val_mod : deterministic time-based split (val = slot % val_mod == 0).
        Used when ``val_months`` is None.
    val_months : list of "YYYY-MM" strings. When provided, **overrides
        ``val_mod``**: val = times whose year-month is in this list, train =
        complement. Tests true generalization, unlike val_mod which decimates
        within training months.
    min_label_frac : reject (and resample) tiles with fewer supervised pixels.
    seed : RNG seed for resampling.
    """

    def __init__(
        self,
        feature_zarr,
        label_zarr=None,
        valid_mask_path: str | Path | None = None,
        flow_bands: list[str] = DEFAULT_FLOW_BANDS,
        rad_bands: list[str] = DEFAULT_RAD_BANDS,
        patch_size: int = PATCH_SIZE,
        overlap: int | None = None,
        qa_min: int = 1,
        qa_high: int = 2,
        high_weight: float = 3.0,
        train: bool = True,
        val_mod: int = 5,
        val_months: list[str] | None = None,
        min_label_frac: float = 0.05,
        preload: bool = True,
        seed: int | None = None,
        rad_time_frames: int = 1,
        random_crop: bool = True,
    ):
        super().__init__()
        if rad_time_frames not in (1, 3):
            raise ValueError(
                f"rad_time_frames must be 1 (t₀ only) or 3 (t₋,t₀,t₊); "
                f"got {rad_time_frames}")
        self.flow_bands = list(flow_bands)
        self.rad_bands = list(rad_bands)
        self.rad_time_frames = rad_time_frames
        self.patch_size = patch_size
        self.qa_min = qa_min
        self.qa_high = qa_high
        self.high_weight = high_weight
        self.min_label_frac = min_label_frac
        # Random spatial cropping is train-only — keeping val tiles
        # deterministic makes eval/rmsvd a stable signal for EarlyStopping
        # and the best-val checkpoint. Flips/rotations are NOT applied:
        # u,v are world-frame vectors and naive symmetry breaks them.
        self.random_crop = bool(random_crop) and train
        self.train = train
        self.rng = np.random.default_rng(seed)

        # chunks=None returns numpy-lazy arrays (direct zarr reads, no dask) —
        # fast random tile access and fork-safe under DataLoader workers.
        feat = xr.open_zarr(str(feature_zarr), chunks=None)
        # zarr v3 / xarray may serialize scalar int attrs as 1-element lists,
        # nested lists, or numpy arrays on round-trip; unwrap until scalar.
        def _scalar_int(v, default=0):
            if v is None:
                return default
            # Iteratively peel container types (list, tuple, ndarray) down to scalar.
            for _ in range(6):
                if hasattr(v, "tolist"):  # numpy array → python value(s)
                    v = v.tolist()
                if isinstance(v, (list, tuple)):
                    if not v:
                        return default
                    v = v[0]
                else:
                    break
            return int(v)
        ro = _scalar_int(feat.attrs.get("row_offset"), 0)
        co = _scalar_int(feat.attrs.get("col_offset"), 0)
        fh, fw = int(feat.sizes["y"]), int(feat.sizes["x"])
        lab = feat if label_zarr is None else xr.open_zarr(str(label_zarr), chunks=None)

        # Common times, after the train/val split.
        times = sorted(set(np.asarray(feat.time.values)) & set(np.asarray(lab.time.values)))
        if val_months:
            times = [t for t in times if self._in_month_split(t, train, val_months)]
        else:
            times = [t for t in times if self._in_split(t, train, val_mod)]
        if not times:
            raise ValueError("No common feature/label times after the train/val split")
        feat = feat.sel(time=times)

        # Crop the (possibly full-disk) labels onto the feature grid, then graft
        # any missing label vars onto the feature dataset by position (lazy).
        if lab.sizes["y"] != fh or lab.sizes["x"] != fw:
            lab = lab.isel(y=slice(ro, ro + fh), x=slice(co, co + fw))
        lab = lab.sel(time=times)
        ds = feat
        for v in _LABEL_VARS:
            if v not in ds:
                ds = ds.assign({v: (("time", "y", "x"), lab[v].data)})
        # Load the overlap crop into RAM (numpy) so xbatcher tiles are pure
        # slicing — fast and fork-safe (dask-backed reads deadlock under forked
        # DataLoader workers). Fine at this scale; stream (preload=False) for
        # very large multi-month sets.
        if preload:
            ds = ds.load()
        self.ds = ds

        step = overlap if overlap is not None else patch_size // 2
        self.bgen = xbatcher.BatchGenerator(
            ds,
            input_dims={"time": 1, "y": patch_size, "x": patch_size},
            input_overlap={"y": step, "x": step},
            preload_batch=False,
        )
        # Bounds for random_crop: spatial origin in [0, fh-ps] × [0, fw-ps].
        # The bgen length controls __len__ either way (so per-epoch step count
        # stays comparable to the deterministic-tiling baseline).
        self._n_times = len(times)
        self._fh = int(ds.sizes["y"])
        self._fw = int(ds.sizes["x"])
        logger.info(
            "StudentXBatchDataset: %d %s times, %d tiles%s",
            len(times), "train" if train else "val", len(self.bgen),
            " (random_crop)" if self.random_crop else "")

    @staticmethod
    def _in_split(t, train: bool, val_mod: int) -> bool:
        slot = int(np.asarray(t).astype("datetime64[m]").astype("int64") // 10)
        return ((slot % val_mod) == 0) != train

    @staticmethod
    def _in_month_split(t, train: bool, val_months: list[str]) -> bool:
        """Month-out split: val = times in ``val_months`` (YYYY-MM), train = rest.

        Tests true generalization, unlike ``_in_split`` which only decimates
        within the same months and lets the model overfit to that season.
        """
        ym = str(np.asarray(t).astype("datetime64[M]"))  # "YYYY-MM"
        is_val = ym in val_months
        return is_val != train

    def __len__(self) -> int:
        return len(self.bgen)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        for _ in range(50):
            sample = self._load(idx)
            if sample is not None:
                return sample
            idx = int(self.rng.integers(len(self.bgen)))
        raise RuntimeError("No labeled tile after 50 attempts")

    def _load(self, idx):
        # Train: random-crop a patch_size² tile at a random spatial origin
        # within the chunk's overlap window, round-robin over times to keep
        # temporal coverage roughly uniform across the epoch.
        # Val: deterministic bgen tiling so eval/rmsvd is comparable
        # epoch-to-epoch (EarlyStopping + best-val depend on this).
        # Flips/rotations are NOT applied — would break the world-frame u,v.
        if self.random_crop:
            t_idx = idx % self._n_times
            max_y = self._fh - self.patch_size
            max_x = self._fw - self.patch_size
            if max_y < 0 or max_x < 0:
                raise ValueError(
                    f"patch_size={self.patch_size} larger than chunk "
                    f"extent ({self._fh}, {self._fw}); cannot crop")
            y0 = int(self.rng.integers(0, max_y + 1)) if max_y > 0 else 0
            x0 = int(self.rng.integers(0, max_x + 1)) if max_x > 0 else 0
            x = self.ds.isel(
                time=t_idx,
                y=slice(y0, y0 + self.patch_size),
                x=slice(x0, x0 + self.patch_size),
            )
        else:
            x = self.bgen[idx].squeeze("time", drop=True)  # dims (y, x[, ...])

        def arr(name):
            return x[name].values.astype(np.float32)

        u = arr("u_wind"); v = arr("v_wind")
        h_km = arr("cloud_top_height") / 1000.0
        qf = arr("quality_flag")
        mask = (qf >= self.qa_min) & np.isfinite(u) & np.isfinite(v) & np.isfinite(h_km)
        if mask.mean() < self.min_label_frac:
            return None
        weight = np.where(qf >= self.qa_high, self.high_weight, 1.0).astype(np.float32) * mask

        flow = np.stack([arr(flow_var(s, b)) / FLOW_SCALE
                         for b in self.flow_bands for s in FLOW_STUBS], 0)
        # rad layout when rad_time_frames=3: per-band [t-, t0, t+] interleaved
        # to keep same-band frames adjacent in channel order.  The model
        # treats this as a single (3*n_rad, H, W) feature; StandardScalar is
        # fit per channel so it adapts to each (band, frame) slot.
        if self.rad_time_frames == 3:
            rad_stubs = [rad_tminus_var, rad_var, rad_tplus_var]
            rad = np.stack([arr(stub(b)) for b in self.rad_bands
                            for stub in rad_stubs], 0)
        else:
            rad = np.stack([arr(rad_var(b)) for b in self.rad_bands], 0)
        geom = np.stack([arr("dx_m") / PIXEL_SCALE_NORM,
                         arr("dy_m") / PIXEL_SCALE_NORM,
                         arr("sat_zenith") / ZENITH_NORM], 0)

        nan = lambda a: np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return {
            "flow": torch.from_numpy(nan(flow)),
            "rad": torch.from_numpy(nan(rad)),
            "geom": torch.from_numpy(nan(geom)),
            "u": torch.from_numpy(np.nan_to_num(u)),
            "v": torch.from_numpy(np.nan_to_num(v)),
            "h_km": torch.from_numpy(np.nan_to_num(h_km)),
            "mask": torch.from_numpy(mask),
            "weight": torch.from_numpy(weight),
        }
