# stereo-winds

Cross-satellite stereo wind retrieval from geostationary imagery. Uses RAFT optical flow to compute disparity fields between satellite image pairs, then solves a per-pixel 5-state weighted least squares system ([Carr et al. 2020](https://doi.org/10.5194/amt-13-3195-2020)) to recover **feature-tracked height** and **wind vectors** simultaneously.

## How it works

Two geostationary satellites (e.g., GOES-19 East + GOES-18 West) view the same clouds from different angles. High clouds appear shifted between the two views — this **parallax** encodes the height of the tracked feature. Combined with temporal image pairs, the system solves for height and wind at every pixel.

```
GOES-19 (75°W)  ──┐
                   ├──→  RAFT optical flow  ──→  5-state WLS solver  ──→  winds + heights
GOES-18 (137°W) ──┘
```

**State vector per pixel:** `[h, p_u, p_v, V_u, V_v]` — feature-tracked height, position corrections, and pixel velocity components.

> **On "height":** `h` is the height of the radiometric feature RAFT tracks, which validation against EarthCARE places roughly 1–2 km *below* the active-sensor cloud top. We call it the feature-tracked height for that reason. The NetCDF variable keeps the name `cloud_top_height` for CF and downstream-consumer compatibility.

**Output:** CF-1.8 compliant NetCDF with wind vectors (m/s), feature-tracked heights (m), formal uncertainties, and quality flags at the native satellite resolution (2 km for IR/WV bands, 500 m for visible).

## Installation

```bash
git clone https://github.com/myzeus/stereo-winds.git
cd stereo-winds

pip install -e .                 # core + inference (GOES-R ABI)
pip install -e ".[viz]"          # + matplotlib/cartopy for plotting
pip install -e ".[train]"        # + pytorch-lightning/wandb/xbatcher for training
pip install -e ".[validation]"   # + h5py/pystac-client for EarthCARE and MAAP
pip install -e ".[dev]"          # + pytest
```

This is a self-contained build — no `zeus` submodule. GOES ABI data is read
directly from NOAA's public S3 buckets (via `s3fs`), so **no account or
credentials are required**.

### Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU recommended for RAFT inference)
- NumPy, SciPy, xarray, h5netcdf, netCDF4, s3fs, zarr
- Cartopy/matplotlib (optional, `[viz]` extra)

### Checkpoints

Four checkpoints ship under `checkpoints/`:

| File | What it is | Provenance |
|------|-----------|------------|
| `windflow.raft.init-ep254.ckpt` | The exact init the sonde-tuned model was fine-tuned from — **use this as the baseline** | epoch 254 / step 255000 |
| `windflow.raft.sonde-tuned.ckpt` | RAFT fine-tuned against IGRA radiosonde winds through the stereo solver | epoch 11 / step 75000, `height_reg_weight=1.0` |
| `windflow.raft.pretrained.ckpt` | A separate general-purpose WindFlow RAFT for geostationary imagery ([Vandal et al. 2022](https://doi.org/10.1145/3534678.3539345)) | epoch 1434 |
| `student.abi.mb-v3.ep21.ckpt` | Single-satellite multi-band "student" distilled from the stereo retrieval | epoch 19 / step 108482 |

> **Which checkpoint is the "pre-trained" baseline? `init-ep254` — not
> `pretrained`.** Despite its name, `windflow.raft.pretrained.ckpt` is a
> different WindFlow lineage point (epoch 1434) and was **not** the starting
> point for the sonde-tuned model, so tuned-vs-`pretrained` does not measure
> the effect of fine-tuning.
>
> `init-ep254` is the "pre-trained WindFlow" of the paper — the baseline
> behind the 7.25 → 5.88 m s⁻¹ (18.9%) fine-tuning result and the pre-trained
> panels of the Carr-consistency comparison. It is verified byte-for-byte
> against the frozen `raft_ref.*` weights embedded in the sonde-tuned
> checkpoint, so the lineage is provable from the shipped files alone.
> Reproducing the paper's ablation means `init-ep254` vs `sonde-tuned`.

Point `--model-ckpt` / `model_ckpt` at one of these (or your own). Lightning
checkpoints are converted automatically.

The fine-tuning methodology and the QA definition used throughout are
documented in [`SONDE_TUNING_SUMMARY.md`](SONDE_TUNING_SUMMARY.md). Note that
its result tables are a historical record of an earlier training run, not of
the checkpoint shipped here — see the status note at the top of that file.

## Quick start

### Command line

```bash
python scripts/run_stereo.py "2024-01-15T12:00" \
    --sat-a goes19 --sat-b goes18 --band C14 \
    --model-ckpt checkpoints/windflow.raft.sonde-tuned.ckpt \
    --device cuda --output-dir output/
```

Data is fetched from NOAA's public S3 buckets on demand and cached under
`cache/`. No console script is installed; invoke the script directly.

### Python API

```python
from datetime import datetime
from stereo_winds.config import StereoPairConfig
from stereo_winds.pipeline import StereoWindPipeline

config = StereoPairConfig.from_satellites(
    sat_a="goes19",
    sat_b="goes18",
    band="C14",
    device="cuda",
    n_iter=3,
)
pipeline = StereoWindPipeline(config)
ds = pipeline.run(datetime(2026, 3, 10, 23, 0))

# ds contains: u_wind, v_wind, cloud_top_height, sigma_u, sigma_v, sigma_h, chi_squared, quality_flag
```

### Standalone run script (multi-band)

For more control, use a standalone script that downloads data directly from NOAA S3:

```python
import datetime as dt
from stereo_winds.config import SatelliteConfig, GOES18_CONFIG
from stereo_winds.data_loading import load_native_abi
from stereo_winds.remap import load_remap_lut, remap_image
from stereo_winds.solver import (
    build_design_matrix, compute_parallax_vectors,
    solve_stereo_winds, pixels_to_wind_ms,
)
from stereo_winds.time_model import compute_scene_times
from stereo_winds.output import create_output_dataset, write_netcdf

# Configure satellites
GOES19 = SatelliteConfig(satellite_id="goes19", sub_lon_deg=-75.0, sweep="x")

# Load 5 scenes: A_minus, A0, A_plus, B_minus, B_plus
a0, _ = load_native_abi("path/to/goes19_C14_t0.nc", satellite_id="goes19")
# ... load remaining scenes, remap B scenes to A's grid ...

# Build geometry and solve
w_u, w_v = compute_parallax_vectors(GOES19, GOES18_CONFIG)
scene_times = compute_scene_times(t0, dt_minutes=10.0, sat_a=GOES19, sat_b=GOES18_CONFIG)
H_matrix = build_design_matrix(w_u, w_v, **scene_times)

solution = solve_stereo_winds(disparities, H_matrix, sat_a=GOES19, sat_b=GOES18_CONFIG,
                               n_iter=3, device="cuda")

# Convert pixel velocity to m/s and write output
u_ms, v_ms = pixels_to_wind_ms(solution["V_u"], solution["V_v"], GOES19, dt_seconds=1.0)
solution["u_wind"] = u_ms
solution["v_wind"] = v_ms
ds = create_output_dataset(solution, GOES19, t0)
write_netcdf(ds, "output/stereo_winds.nc")
```

## Supported satellite pairs

| Pair | Separation | Coverage |
|------|-----------|----------|
| GOES-19 (East) + GOES-18 (West) | 62° | Americas |
| GOES-16 (Test) + GOES-18 (West) | 62° | Americas |
| Himawari-8/9 + GOES-18 | ~82° | Pacific (AHI reader not yet ported — see below) |

## ABI bands

| Band | Wavelength | Sensing level | Resolution |
|------|-----------|---------------|------------|
| C02 | 0.64 µm | Visible (cloud/aerosol) | 500 m |
| C04 | 1.38 µm | Cirrus | 2 km |
| C08 | 6.2 µm | Upper-level water vapor | 2 km |
| C09 | 6.9 µm | Mid-level water vapor | 2 km |
| C10 | 7.3 µm | Low-level water vapor | 2 km |
| C12 | 9.6 µm | Ozone | 2 km |
| C14 | 11.2 µm | IR window (cloud tops) | 2 km |

Each band senses a different atmospheric level. Running multiple bands produces a 3D wind profile.

## Pipeline architecture

```
1. Data loading ──→ 5 scenes in native fixed-grid coordinates
         │              (A_minus, A0, A_plus, B_minus, B_plus)
         ▼
2. Remapping ────→ Remap satellite B onto A's grid via precomputed LUT
         │
         ▼
3. RAFT optical flow ──→ 4 disparity fields (D1..D4)
         │                  D1: A0→A_minus  (temporal)
         │                  D2: A0→A_plus   (temporal)
         │                  D3: A0→B_minus  (cross-satellite)
         │                  D4: A0→B_plus   (cross-satellite)
         ▼
4. Solver ───────→ Per-pixel 8×5 weighted least squares
         │           - Design matrix from parallax vectors + time offsets
         │           - Iterative: recomputes parallax at current height
         │           - GPU-accelerated via PyTorch
         ▼
5. Output ───────→ CF-1.8 NetCDF with winds, heights, uncertainties
```

## Tests

```bash
# Run all tests (216 tests, ~40s)
python -m pytest tests/ -v

# Key test categories:
#   test_solver.py     — exact recovery, iterative convergence, GPU/CPU agreement
#   test_navigation.py — round-trip geodetic ↔ fixed grid transformations
#   test_pipeline.py   — end-to-end pipeline with synthetic data
#   test_time_model.py — ABI/AHI scan time models
```

## Standalone build notes

This dependency-light release fully supports **GOES-R ABI**. A few sensors and
paths are not yet ported and raise a clear error if called: **Himawari AHI**,
**MTG-I FCI** (needs satpy + eumdac), the bundled **ERA5** reader, and
**arraylake** batch output. The single-satellite full-disk "student" model
(`nn/`, `student_*`) is included and trains/runs standalone (`[train]` extra).

## License & attribution

MIT — see [`LICENSE`](LICENSE). The optical-flow network under
`stereo_winds/flow/raft/` is a single-channel adaptation of **RAFT** (Teed &
Deng, ECCV 2020), redistributed under BSD-3-Clause (see
`stereo_winds/flow/raft/LICENSE-RAFT` and [`NOTICE`](NOTICE)). The retrieval
solver follows **Carr et al. (2020)**.

## Data sources

The pipeline reads **GOES-R ABI L1b** from the NOAA public S3 buckets (public
domain, no credentials). Small derived subsets used to regenerate the figures
are committed under `figures/` and carry their own upstream terms:

| Source | Used for | Terms |
|--------|----------|-------|
| GOES-R ABI L1b (NOAA/NESDIS) | retrieval input | public domain |
| IGRA v2 radiosondes (NOAA/NCEI) | validation, sonde fine-tuning | public domain |
| NOAA/NESDIS derived-motion AMVs | validation baseline | public domain |
| ERA5 (Copernicus/ECMWF) | curtain comparison (`figures/era5_curtain.npz`) | [Copernicus licence](https://cds.climate.copernicus.eu) — attribution required |
| EarthCARE CPR/ATLID (ESA/JAXA) | height validation (`figures/curtain_bundle.npz`) | [ESA EarthCARE terms](https://earth.esa.int/eogateway/missions/earthcare) |

Contains modified Copernicus Climate Change Service information (ERA5). Neither
the European Commission nor ECMWF is responsible for any use of this data.

## References

- Carr, J. L., Wu, D. L., Daniels, J., Friberg, M. D., Bresky, W., & Madani, H. (2020). GEO–GEO stereo-tracking of Atmospheric Motion Vectors (AMVs) from the geostationary ring. *Atmospheric Measurement Techniques*, 13, 3195–3215.
- Teed, Z., & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. *ECCV 2020*.
- Vandal, T. J., Duffy, K., McCarty, W., Sewnath, A., & Nemani, R. (2022). Dense feature tracking of atmospheric winds with deep optical flow. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 1807–1815). https://doi.org/10.1145/3534678.3539345
