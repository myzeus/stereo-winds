# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-satellite stereo wind retrieval from geostationary imagery. Uses RAFT optical flow to compute disparity fields between satellite image pairs, then solves a per-pixel 5-state weighted least squares system (Carr et al. 2020) to recover feature-tracked height and wind vectors.

State vector: `[h, p_u, p_v, V_u, V_v]` — height, position corrections, and pixel velocity components.

**Terminology:** in prose, call `h` the *feature-tracked height*, not the cloud-top height — validation against EarthCARE shows the retrieval sits ~1–2 km below the active-sensor cloud top. The NetCDF variable stays named `cloud_top_height` for CF/consumer compatibility; do not rename it.

## Commands

```bash
# Install (core + inference)
pip install -e .

# Optional extras
pip install -e ".[viz]"          # matplotlib/cartopy plotting
pip install -e ".[train]"        # lightning/wandb/xbatcher training
pip install -e ".[validation]"   # h5py/pystac-client EarthCARE + MAAP
pip install -e ".[dev]"          # pytest, pyproj

# Run all tests (216 tests, ~40s)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_solver.py -v

# Run a specific test
python -m pytest tests/test_solver.py::TestSolveStereoWinds::test_exact_recovery -v

# Run the pipeline (no console script is installed; invoke the script directly)
python scripts/run_stereo.py "2024-01-15T12:00" \
    --sat-a goes19 --sat-b goes18 --band C14 \
    --model-ckpt checkpoints/windflow.raft.sonde-tuned.ckpt \
    --device cuda
```

No linter or formatter is configured.

## Architecture

**Pipeline flow** (`pipeline.py` orchestrates):
1. **Data loading** (`data_loading.py`) — Load 5 scenes (A±, A0, B±, B_plus) in native fixed-grid coordinates; GOES ABI comes from NOAA's public S3 buckets via `readers/goes.py` (no credentials needed)
2. **Remapping** (`remap.py`) — Remap satellite B scenes onto satellite A's grid via precomputed LUT
3. **Disparity** (`disparity.py`) — Run RAFT optical flow on 4 image pairs (cross-sat and temporal)
4. **Solver** (`solver.py`) — Build 8×5 design matrix from parallax vectors and time offsets; solve per-pixel WLS system
5. **Output** (`output.py`) — Write CF-1.8 compliant NetCDF with winds, heights, quality flags, and formal uncertainties

**Key supporting modules:**
- `navigation.py` — Geostationary projection math (pixel↔scanning angle, geodetic↔fixed grid, ECEF). Handles both x-sweep (ABI) and y-sweep (AHI) conventions.
- `checkpoints/` — `windflow.raft.init-ep254.ckpt` is the fine-tuning **baseline** (verified identical to the `raft_ref.*` weights inside the tuned checkpoint); `windflow.raft.sonde-tuned.ckpt` is the deliverable. `windflow.raft.pretrained.ckpt` (epoch 1434) is a *different* lineage point — never use it as the fine-tuning baseline.
- `config.py` — `SatelliteConfig` and `StereoPairConfig` dataclasses. Presets: `GOES16_CONFIG`, `GOES18_CONFIG`, `GOES19_CONFIG`, `HIMAWARI8_CONFIG`, `HIMAWARI9_CONFIG`, `MTG_I1_CONFIG`, plus the `SATELLITE_CONFIGS` lookup used by the CLI.
- `time_model.py` — Per-pixel scan time offsets for ABI/AHI instruments
- `flow/` — Vendored single-channel RAFT (BSD-3, see `flow/raft/LICENSE-RAFT`) and the `FlowRunner` wrapper
- `readers/goes.py` — Standalone public-S3 GOES ABI reader
- `visualize.py` — Debug plotting with cartopy geostationary projections
- `validation/` — Metrics (RMSVD, speed bias), comparison against AMVs, radiosondes, EarthCARE curtains, ground points
- `nn/`, `student_*` — Single-satellite "student" model distilled from the stereo retrieval

## Code Conventions

- Python 3.10+, NumPy-style docstrings, type hints on public APIs
- Vectorized numpy operations preferred over loops
- Dataclasses for configuration; per-module `logging.getLogger(__name__)`
- Tests use pytest with class-based organization; common patterns include round-trip tests (navigation) and synthetic-problem exact-recovery tests (solver)
- Scripts resolve the repo root as `BASE = Path(__file__).resolve().parent.parent`; host-specific paths belong in env vars (`STEREO_WINDS_DATA_DIR`, `STEREO_WINDS_RAFT_CKPT`, `STEREO_WINDS_TRAIN_DIR`, `IGRA_CACHE_DIR`) or CLI flags, never hardcoded

## Not ported to the standalone build

These paths raise a clear `NotImplementedError` rather than silently failing. Do not assume they work:

- **Himawari AHI** loading (navigation and time model support AHI; the reader does not)
- **MTG-I FCI** loading — needs satpy (`fci_l1c_nc`) + eumdac
- **ERA5** reader in `validation/era5.py`
- **Arraylake** batch output in `batch.py`

Several scripts under `scripts/` still `sys.path`-insert a sibling `zeus/` checkout for internal data-source integrations. That checkout is not part of this repo, is gitignored, and is not required for the pipeline, the tests, or anything documented above.
