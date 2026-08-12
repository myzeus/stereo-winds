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

from .student_dataset import (  # noqa: E501  (import kept together)
    flow_var, rad_var, rad_tminus_var, rad_tplus_var, FLOW_STUBS, GEOM_VARS,
    DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
    FLOW_SCALE, PIXEL_SCALE_NORM, ZENITH_NORM,
)

logger = logging.getLogger(__name__)

PATCH_SIZE = 256
# chi_squared is OPTIONAL — older chunks won't have it. The dataset assigns it
# from the label store only if present; consumers (e.g. chi²-distill training)
# check the returned key.
_LABEL_VARS = ["u_wind", "v_wind", "cloud_top_height", "quality_flag"]
_OPTIONAL_LABEL_VARS = ["chi_squared"]

# Physics-aware dihedral augmentation (train-only). The winds (u, v) and the
# input optical flows are VECTOR fields, so a geometric transform of the patch
# must ALSO rotate/reflect the vectors — otherwise the (input, label) physics
# is broken. `_VEC_M[op]` = (a, b, c, d): new_u = a*u + b*v, new_v = c*u + d*v.
_DIHEDRAL_OPS = ["id", "hflip", "vflip", "rot180", "rot90", "rot270"]
_VEC_M = {
    "hflip":  (-1, 0, 0, 1),    # mirror x  → u→−u
    "vflip":  (1, 0, 0, -1),    # mirror y  → v→−v
    "rot180": (-1, 0, 0, -1),
    "rot90":  (0, -1, 1, 0),    # CCW 90°   → (u,v)→(−v, u)
    "rot270": (0, 1, -1, 0),    # CCW 270°  → (u,v)→(v, −u)
}


def _dihedral_spatial(a, op):
    if op == "hflip":
        return np.flip(a, -1)
    if op == "vflip":
        return np.flip(a, -2)
    if op == "rot90":
        return np.rot90(a, 1, axes=(-2, -1))
    if op == "rot180":
        return np.rot90(a, 2, axes=(-2, -1))
    if op == "rot270":
        return np.rot90(a, 3, axes=(-2, -1))
    return a


def _dihedral_vec(u, v, op):
    a, b, c, d = _VEC_M[op]
    return a * u + b * v, c * u + d * v


def augment_dihedral(flow, rad, geom, u, v, h_km, mask, weight, chi2,
                     n_flow_bands, rng, op=None):
    """Apply one random dihedral op (D4: flips + 90/180/270 rotations) to a
    sample, transforming vector fields correctly.

    - Spatial transform applied to every array's last two dims (square patch).
    - Flow channels are per band ``[back_u, back_v, fwd_u, fwd_v]``: the (u, v)
      pairs are rotated by the op's vector matrix. Label winds (u, v) likewise.
    - `geom` dx_m/dy_m (channels 0, 1) SWAP under 90°/270° rotation (the grid x
      and y axes swap). Radiance/height/mask/weight/chi² are scalars → spatial
      only.

    Consistency (physics baseline of augmented flow == augmented wind) is
    covered by ``tests/test_student_zeus.py::TestDihedralAug``.
    """
    op = op or _DIHEDRAL_OPS[int(rng.integers(len(_DIHEDRAL_OPS)))]
    if op == "id":
        return flow, rad, geom, u, v, h_km, mask, weight, chi2
    ac = np.ascontiguousarray
    flow = ac(_dihedral_spatial(flow, op))
    rad = ac(_dihedral_spatial(rad, op))
    geom = ac(_dihedral_spatial(geom, op))
    h_km = ac(_dihedral_spatial(h_km, op))
    mask = ac(_dihedral_spatial(mask, op))
    weight = ac(_dihedral_spatial(weight, op))
    u = ac(_dihedral_spatial(u, op))
    v = ac(_dihedral_spatial(v, op))
    if chi2 is not None:
        chi2 = ac(_dihedral_spatial(chi2, op))
    # Rotate the flow vectors (per band: (back_u,back_v) and (fwd_u,fwd_v)).
    for bi in range(n_flow_bands):
        for off in (0, 2):
            cu, cv = bi * 4 + off, bi * 4 + off + 1
            flow[cu], flow[cv] = _dihedral_vec(flow[cu], flow[cv], op)
    # Rotate the label winds.
    u, v = _dihedral_vec(u, v, op)
    # dx_m/dy_m swap under 90°/270° rotation.
    if op in ("rot90", "rot270"):
        geom[[0, 1]] = geom[[1, 0]]
    return flow, rad, geom, u, v, h_km, mask, weight, chi2


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
        aug_dihedral: bool = False,
    ):
        super().__init__()
        # Train-only (val stays deterministic for a stable eval/rmsvd signal).
        self.aug_dihedral = bool(aug_dihedral) and train
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
        # Multi-band: the combined store carries per-band labels u_wind_<b> etc.
        # for each band in the `wind_bands` attr.  Absent → legacy single field.
        wb = str(feat.attrs.get("wind_bands", ""))
        self.wind_bands = [b for b in wb.split(",") if b]
        self.n_bands = len(self.wind_bands) if self.wind_bands else 1
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
        if self.n_bands > 1:
            # Per-band labels live in the combined store already (u_wind_<b> …);
            # no single-name grafting. chi² present iff the first band has it.
            self.has_chi2 = f"chi_squared_{self.wind_bands[0]}" in ds
        else:
            for v in _LABEL_VARS:
                if v not in ds:
                    ds = ds.assign({v: (("time", "y", "x"), lab[v].data)})
            # chi_squared is OPTIONAL — graft only if the label store actually
            # has it.  Lets older chunks still load.
            self.has_chi2 = False
            for v in _OPTIONAL_LABEL_VARS:
                if v not in ds and v in lab:
                    ds = ds.assign({v: (("time", "y", "x"), lab[v].data)})
                    self.has_chi2 = self.has_chi2 or (v == "chi_squared")
                elif v in ds:
                    self.has_chi2 = self.has_chi2 or (v == "chi_squared")
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

        if self.n_bands > 1:
            # Stack per-band labels onto a leading band axis → (nb, ps, ps).
            def arr_b(v):
                return np.stack([x[f"{v}_{b}"].values.astype(np.float32)
                                 for b in self.wind_bands], 0)
            u = arr_b("u_wind"); v = arr_b("v_wind")
            h_km = arr_b("cloud_top_height") / 1000.0
            qf = arr_b("quality_flag")
        else:
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

        # chi² loaded BEFORE augmentation so the same op is applied to it.
        # Clamp lower (non-negative) and upper (outliers > 1e6 blow up log()).
        chi2 = None
        if self.has_chi2:
            raw_chi2 = arr_b("chi_squared") if self.n_bands > 1 else arr("chi_squared")
            chi2 = np.clip(raw_chi2, 1e-6, 1e4).astype(np.float32)

        # Train-only physics-aware dihedral augmentation (flips + rotations with
        # matching u,v / flow vector transforms). Composes with random_crop.
        if self.aug_dihedral:
            flow, rad, geom, u, v, h_km, mask, weight, chi2 = augment_dihedral(
                flow, rad, geom, u, v, h_km, mask, weight, chi2,
                len(self.flow_bands), self.rng)

        nan = lambda a: np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        sample = {
            "flow": torch.from_numpy(nan(flow)),
            "rad": torch.from_numpy(nan(rad)),
            "geom": torch.from_numpy(nan(geom)),
            "u": torch.from_numpy(np.nan_to_num(u)),
            "v": torch.from_numpy(np.nan_to_num(v)),
            "h_km": torch.from_numpy(np.nan_to_num(h_km)),
            "mask": torch.from_numpy(mask),
            "weight": torch.from_numpy(weight),
        }
        if chi2 is not None:
            sample["chi2"] = torch.from_numpy(nan(chi2))
        return sample
