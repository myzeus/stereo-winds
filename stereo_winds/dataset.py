"""PyTorch Dataset for stereo wind training from Zarr cubes.

Wraps the GOES-19 + GOES-18 (remapped) Zarr stores produced by
``training_data.py``.  Each sample is a 5-scene stereo set with a random
512x512 spatial patch, histogram-equalized for RAFT input.

Usage
-----
    from stereo_winds.dataset import StereoWindsDataset

    ds = StereoWindsDataset(
        sat_a_zarr=["data/stereo_training/goes19_C14_202601.zarr"],
        sat_b_zarr=["data/stereo_training/goes18_remap_goes19_C14_202601.zarr"],
        parallax_path="cache/parallax_goes19_goes18.npz",
        valid_mask_path="cache/valid_mask_g19_g18.npy",
    )
    sample = ds[0]
    # sample["A0"]       : (1, 512, 512) float32 tensor
    # sample["H_matrix"] : (512, 512, 8, 5) float32 tensor
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from .solver import build_design_matrix
from .training_data import build_triplet_index

logger = logging.getLogger(__name__)

# Default patch size and time offset
PATCH_SIZE = 512
DT_MINUTES = 10.0
DT_SECONDS = DT_MINUTES * 60.0


def _histogram_equalize(image: np.ndarray, n_bins: int = 500) -> np.ndarray:
    """CDF-based histogram equalization to [0, 1].

    Same algorithm as ``zeus.utils.preprocess.image_histogram_equalization``
    but inlined to avoid the zeus import dependency.
    """
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image)
    vals = image[finite]
    hist, bins = np.histogram(vals, n_bins, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    out = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    out[~finite] = 0.0
    return out.astype(np.float32)


class StereoWindsDataset(Dataset):
    """Training dataset: 5-scene stereo patches from Zarr cubes.

    Parameters
    ----------
    sat_a_zarr : path(s) to reference satellite Zarr store(s)
    sat_b_zarr : path(s) to partner satellite (remapped) Zarr store(s)
    parallax_path : path to precomputed parallax vectors (.npz with w_u, w_v)
    valid_mask_path : path to overlap mask (.npy boolean array), or None
    patch_size : spatial crop size (default 512)
    dt_minutes : temporal offset between scenes (default 10)
    samples_per_epoch : number of random samples per epoch (default: all valid triplets)
    seed : random seed for reproducibility (None = random)
    """

    def __init__(
        self,
        sat_a_zarr: str | Path | list[str | Path],
        sat_b_zarr: str | Path | list[str | Path],
        parallax_path: str | Path,
        valid_mask_path: str | Path | None = None,
        ec_collocation_path: str | Path | None = None,
        patch_size: int = PATCH_SIZE,
        dt_minutes: float = DT_MINUTES,
        samples_per_epoch: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dt_minutes = dt_minutes

        # Open Zarr stores (kept open for fast random access)
        if isinstance(sat_a_zarr, (str, Path)):
            sat_a_zarr = [sat_a_zarr]
        if isinstance(sat_b_zarr, (str, Path)):
            sat_b_zarr = [sat_b_zarr]

        self.ds_a = xr.open_mfdataset(
            [str(p) for p in sat_a_zarr], engine="zarr", concat_dim="time",
            combine="nested", chunks={}, data_vars="minimal",
        )
        self.ds_b = xr.open_mfdataset(
            [str(p) for p in sat_b_zarr], engine="zarr", concat_dim="time",
            combine="nested", chunks={}, data_vars="minimal",
        )

        # Build valid triplet index
        self.valid_times = build_triplet_index(sat_a_zarr, sat_b_zarr, dt_minutes)
        if len(self.valid_times) == 0:
            raise ValueError("No valid triplets found in the provided Zarr stores")
        logger.info("Dataset: %d valid triplet center times", len(self.valid_times))

        # Load parallax vectors (full-disk, will be sliced per patch)
        par = np.load(parallax_path)
        self.w_u = par["w_u"]  # (n_rows, n_cols)
        self.w_v = par["w_v"]

        # Load valid overlap mask
        if valid_mask_path is not None:
            self.valid_mask = np.load(valid_mask_path)
        else:
            self.valid_mask = np.isfinite(self.w_u) & np.isfinite(self.w_v)

        # Load EarthCARE collocation labels (optional)
        self._ec_by_time: dict[np.datetime64, np.ndarray] = {}
        if ec_collocation_path is not None:
            import pandas as pd
            ec_df = pd.read_parquet(ec_collocation_path)
            for t, grp in ec_df.groupby("goes_time"):
                self._ec_by_time[np.datetime64(t)] = grp[["row_19", "col_19", "ec_cth_m"]].values
            logger.info("EarthCARE: %d labels across %d GOES times",
                        len(ec_df), len(self._ec_by_time))

        # Precompute valid patch origins (top-left corners where the patch
        # has sufficient valid pixels)
        self._build_patch_origins()

        self.samples_per_epoch = samples_per_epoch or len(self.valid_times)
        self.rng = np.random.default_rng(seed)

    def _build_patch_origins(self, min_valid_frac: float = 0.3):
        """Find (row, col) origins where patches have enough valid pixels."""
        ps = self.patch_size
        n_rows, n_cols = self.valid_mask.shape

        origins = []
        # Step by half patch for overlap
        step = ps // 2
        for r in range(0, n_rows - ps + 1, step):
            for c in range(0, n_cols - ps + 1, step):
                patch_mask = self.valid_mask[r:r + ps, c:c + ps]
                if patch_mask.mean() >= min_valid_frac:
                    origins.append((r, c))

        self.patch_origins = origins
        logger.info("Dataset: %d valid patch origins (%.0f%% of grid)",
                    len(origins),
                    100 * len(origins) / max(1, ((n_rows - ps) // step + 1) * ((n_cols - ps) // step + 1)))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_patch(self, t0, r0, c0):
        """Load 5-scene stereo patch.  Returns (scenes, H_matrix) or None if incomplete."""
        ps = self.patch_size
        delta = np.timedelta64(int(self.dt_minutes), "m")
        t_minus = t0 - delta
        t_plus = t0 + delta

        y_sl = slice(r0, r0 + ps)
        x_sl = slice(c0, c0 + ps)

        def _load(ds, t):
            arr = ds.Rad.sel(time=t).isel(y=y_sl, x=x_sl).values.astype(np.float32)
            return arr

        scenes_raw = [
            _load(self.ds_a, t_minus),
            _load(self.ds_a, t0),
            _load(self.ds_a, t_plus),
            _load(self.ds_b, t_minus),
            _load(self.ds_b, t_plus),
        ]

        # Check completeness: all 5 scenes must have finite data where the
        # valid mask is True.  Reject patches where any scene is all-NaN in
        # the overlap region.
        mask_patch = self.valid_mask[r0:r0 + ps, c0:c0 + ps]
        for arr in scenes_raw:
            if not np.isfinite(arr[mask_patch]).any():
                return None

        # Histogram-equalize each scene
        scenes = [_histogram_equalize(arr) for arr in scenes_raw]

        # Build design matrix for this patch
        w_u_patch = self.w_u[r0:r0 + ps, c0:c0 + ps]
        w_v_patch = self.w_v[r0:r0 + ps, c0:c0 + ps]
        H_matrix = build_design_matrix(
            w_u_patch, w_v_patch,
            dt_a_minus=-DT_SECONDS,
            dt_a_plus=DT_SECONDS,
            dt_b_minus=-DT_SECONDS,
            dt_b_plus=DT_SECONDS,
        ).astype(np.float32)

        return scenes, H_matrix

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Sample until we get a patch with complete data in all 5 scenes
        max_attempts = 50
        for _ in range(max_attempts):
            t0 = self.valid_times[self.rng.integers(len(self.valid_times))]
            r0, c0 = self.patch_origins[self.rng.integers(len(self.patch_origins))]
            result = self._load_patch(t0, r0, c0)
            if result is not None:
                break
        else:
            raise RuntimeError(
                f"Could not find a complete patch after {max_attempts} attempts"
            )

        scenes, H_matrix = result
        a_minus, a0, a_plus, b_minus, b_plus = scenes

        ps = self.patch_size
        batch = {
            "A_minus": torch.from_numpy(a_minus).unsqueeze(0),
            "A0":      torch.from_numpy(a0).unsqueeze(0),
            "A_plus":  torch.from_numpy(a_plus).unsqueeze(0),
            "B_minus": torch.from_numpy(b_minus).unsqueeze(0),
            "B_plus":  torch.from_numpy(b_plus).unsqueeze(0),
            "H_matrix": torch.from_numpy(H_matrix),
        }

        # EarthCARE sparse height labels
        ec_height = np.zeros((ps, ps), dtype=np.float32)
        ec_mask = np.zeros((ps, ps), dtype=bool)
        ec_data = self._ec_by_time.get(t0)
        if ec_data is not None:
            rows, cols, cth = ec_data[:, 0], ec_data[:, 1], ec_data[:, 2]
            in_patch = ((rows >= r0) & (rows < r0 + ps)
                        & (cols >= c0) & (cols < c0 + ps))
            if in_patch.any():
                lr = (rows[in_patch] - r0).astype(int)
                lc = (cols[in_patch] - c0).astype(int)
                ec_height[lr, lc] = cth[in_patch]
                ec_mask[lr, lc] = True

        batch["ec_height"] = torch.from_numpy(ec_height)
        batch["ec_mask"] = torch.from_numpy(ec_mask)
        return batch


class EarthCAREDataset(Dataset):
    """Training dataset driven by EarthCARE collocation labels.

    Every sample is guaranteed to contain EarthCARE height labels.
    Patches are centered on the EarthCARE track for each GOES scan time,
    tiled along the track with overlap.

    Parameters
    ----------
    sat_a_zarr, sat_b_zarr : Zarr store paths (same as StereoWindsDataset)
    parallax_path : precomputed parallax vectors
    valid_mask_path : overlap mask
    ec_collocation_path : parquet with EarthCARE labels (required)
    patch_size : spatial crop size (default 512)
    dt_minutes : temporal offset (default 10)
    seed : random seed
    """

    def __init__(
        self,
        sat_a_zarr: str | Path | list[str | Path],
        sat_b_zarr: str | Path | list[str | Path],
        parallax_path: str | Path,
        valid_mask_path: str | Path,
        ec_collocation_path: str | Path,
        patch_size: int = PATCH_SIZE,
        dt_minutes: float = DT_MINUTES,
        seed: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dt_minutes = dt_minutes

        if isinstance(sat_a_zarr, (str, Path)):
            sat_a_zarr = [sat_a_zarr]
        if isinstance(sat_b_zarr, (str, Path)):
            sat_b_zarr = [sat_b_zarr]

        self.ds_a = xr.open_mfdataset(
            [str(p) for p in sat_a_zarr], engine="zarr", concat_dim="time",
            combine="nested", chunks={}, data_vars="minimal",
        )
        self.ds_b = xr.open_mfdataset(
            [str(p) for p in sat_b_zarr], engine="zarr", concat_dim="time",
            combine="nested", chunks={}, data_vars="minimal",
        )

        # Valid triplet times
        valid_times_set = set(build_triplet_index(sat_a_zarr, sat_b_zarr, dt_minutes))

        # Parallax and valid mask
        par = np.load(parallax_path)
        self.w_u = par["w_u"]
        self.w_v = par["w_v"]
        self.valid_mask = np.load(valid_mask_path)
        n_rows, n_cols = self.valid_mask.shape

        # Load EarthCARE collocation and build patch index
        import pandas as pd
        ec_df = pd.read_parquet(ec_collocation_path)
        logger.info("EarthCARE: %d labels, %d GOES times",
                    len(ec_df), ec_df["goes_time"].nunique())

        # Detect multi-layer columns
        self._has_layers = "ec_layer1_top_m" in ec_df.columns
        if self._has_layers:
            layer_cols = []
            k = 1
            while f"ec_layer{k}_top_m" in ec_df.columns:
                layer_cols.append(f"ec_layer{k}_top_m")
                k += 1
            self._n_max_layers = len(layer_cols)
            logger.info("EarthCARE: %d cloud layers available per pixel", self._n_max_layers)
        else:
            self._n_max_layers = 1
            logger.info("EarthCARE: single CTH labels (no multi-layer columns)")

        # Build (time, r0, c0) samples: tile 512x512 patches along each track
        self._samples = []  # list of (np.datetime64, int r0, int c0)
        self._ec_by_time: dict[np.datetime64, np.ndarray] = {}
        ps = patch_size
        step = ps // 2  # 50% overlap along the track

        # Columns: row, col, cth, layer1_top, ..., layerN_top, layer1_base, ..., layerN_base
        value_cols = ["row_19", "col_19", "ec_cth_m"]
        if self._has_layers:
            for k in range(1, self._n_max_layers + 1):
                value_cols.append(f"ec_layer{k}_top_m")
            for k in range(1, self._n_max_layers + 1):
                value_cols.append(f"ec_layer{k}_base_m")

        for t, grp in ec_df.groupby("goes_time"):
            t64 = np.datetime64(t)
            if t64 not in valid_times_set:
                continue
            arr = grp[value_cols].values
            self._ec_by_time[t64] = arr
            rows, cols = arr[:, 0], arr[:, 1]

            # Tile patches covering the track extent
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())

            for r0 in range(max(0, r_min - ps // 4), min(n_rows - ps, r_max), step):
                for c0 in range(max(0, c_min - ps // 4), min(n_cols - ps, c_max), step):
                    # Check this patch actually has labels
                    in_patch = ((rows >= r0) & (rows < r0 + ps)
                                & (cols >= c0) & (cols < c0 + ps))
                    if in_patch.sum() >= 5:  # at least 5 labels
                        self._samples.append((t64, r0, c0))

        logger.info("EarthCARE dataset: %d patches with labels", len(self._samples))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # On corrupt chunks, fall back to a random other sample
        try:
            return self._getitem_inner(idx)
        except RuntimeError as e:
            if "decompression" in str(e).lower():
                logger.warning("Corrupt chunk at idx %d, sampling replacement", idx)
                alt = self.rng.integers(len(self._samples))
                return self._getitem_inner(alt)
            raise

    def _getitem_inner(self, idx: int) -> dict[str, torch.Tensor]:
        t0, r0, c0 = self._samples[idx]
        ps = self.patch_size
        delta = np.timedelta64(int(self.dt_minutes), "m")

        y_sl = slice(r0, r0 + ps)
        x_sl = slice(c0, c0 + ps)

        def _load(ds, t):
            return ds.Rad.sel(time=t).isel(y=y_sl, x=x_sl).values.astype(np.float32)

        a_minus = _histogram_equalize(_load(self.ds_a, t0 - delta))
        a0      = _histogram_equalize(_load(self.ds_a, t0))
        a_plus  = _histogram_equalize(_load(self.ds_a, t0 + delta))
        b_minus = _histogram_equalize(_load(self.ds_b, t0 - delta))
        b_plus  = _histogram_equalize(_load(self.ds_b, t0 + delta))

        w_u_patch = self.w_u[r0:r0 + ps, c0:c0 + ps]
        w_v_patch = self.w_v[r0:r0 + ps, c0:c0 + ps]
        H_matrix = build_design_matrix(
            w_u_patch, w_v_patch,
            dt_a_minus=-DT_SECONDS, dt_a_plus=DT_SECONDS,
            dt_b_minus=-DT_SECONDS, dt_b_plus=DT_SECONDS,
        ).astype(np.float32)

        # Scatter EarthCARE labels into patch
        ec_height = np.zeros((ps, ps), dtype=np.float32)
        ec_mask = np.zeros((ps, ps), dtype=bool)
        ec_data = self._ec_by_time[t0]
        rows, cols, cth = ec_data[:, 0], ec_data[:, 1], ec_data[:, 2]
        in_patch = ((rows >= r0) & (rows < r0 + ps)
                    & (cols >= c0) & (cols < c0 + ps))
        lr = (rows[in_patch] - r0).astype(int)
        lc = (cols[in_patch] - c0).astype(int)
        ec_height[lr, lc] = cth[in_patch]
        ec_mask[lr, lc] = True

        # Multi-layer heights (tops and bases)
        K = self._n_max_layers
        ec_layer_tops = np.full((ps, ps, K), np.nan, dtype=np.float32)
        ec_layer_bases = np.full((ps, ps, K), np.nan, dtype=np.float32)
        ec_layer_mask = np.zeros((ps, ps, K), dtype=bool)
        if self._has_layers:
            # Columns: 0=row, 1=col, 2=cth, 3..3+K=tops, 3+K..3+2K=bases
            for k in range(K):
                top_idx = 3 + k
                base_idx = 3 + K + k
                top_h = ec_data[in_patch, top_idx]
                base_h = ec_data[in_patch, base_idx]
                valid_layer = np.isfinite(top_h)
                ec_layer_tops[lr[valid_layer], lc[valid_layer], k] = top_h[valid_layer]
                ec_layer_bases[lr[valid_layer], lc[valid_layer], k] = base_h[valid_layer]
                ec_layer_mask[lr[valid_layer], lc[valid_layer], k] = True
        else:
            # Fall back to CTH as a zero-thickness layer
            ec_layer_tops[lr, lc, 0] = cth[in_patch]
            ec_layer_bases[lr, lc, 0] = cth[in_patch]
            ec_layer_mask[lr, lc, 0] = True

        return {
            "A_minus": torch.from_numpy(a_minus).unsqueeze(0),
            "A0":      torch.from_numpy(a0).unsqueeze(0),
            "A_plus":  torch.from_numpy(a_plus).unsqueeze(0),
            "B_minus": torch.from_numpy(b_minus).unsqueeze(0),
            "B_plus":  torch.from_numpy(b_plus).unsqueeze(0),
            "H_matrix": torch.from_numpy(H_matrix),
            "ec_height": torch.from_numpy(ec_height),
            "ec_mask": torch.from_numpy(ec_mask),
            "ec_layer_tops": torch.from_numpy(ec_layer_tops),
            "ec_layer_bases": torch.from_numpy(ec_layer_bases),
            "ec_layer_mask": torch.from_numpy(ec_layer_mask),
        }
