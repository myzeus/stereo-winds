# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-satellite stereo wind retrieval from geostationary imagery. Uses RAFT optical flow to compute disparity fields between satellite image pairs, then solves a per-pixel 5-state weighted least squares system (Carr et al. 2020) to recover cloud-top height and wind vectors.

State vector: `[h, p_u, p_v, V_u, V_v]` — height, position corrections, and pixel velocity components.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run all tests (~40 tests, ~4s)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_solver.py -v

# Run a specific test
python -m pytest tests/test_solver.py::TestSolveStereoWinds::test_exact_recovery -v

# Run the pipeline
stereo-winds "2024-01-15T12:00" --sat-a goes16 --sat-b goes18 --band C14 --model-ckpt /path/to/raft.pt --device cuda
```

No linter or formatter is configured.

## Architecture

**Pipeline flow** (`pipeline.py` orchestrates):
1. **Data loading** (`data_loading.py`) — Load 5 scenes (A±, A0, B±, B_plus) in native fixed-grid coordinates; integrates with `zeus` submodule for downloads
2. **Remapping** (`remap.py`) — Remap satellite B scenes onto satellite A's grid via precomputed LUT
3. **Disparity** (`disparity.py`) — Run RAFT optical flow on 4 image pairs (cross-sat and temporal)
4. **Solver** (`solver.py`) — Build 8×5 design matrix from parallax vectors and time offsets; solve per-pixel WLS system
5. **Output** (`output.py`) — Write CF-1.8 compliant NetCDF with winds, heights, quality flags, and formal uncertainties

**Key supporting modules:**
- `navigation.py` — Geostationary projection math (pixel↔scanning angle, geodetic↔fixed grid, ECEF). Handles both x-sweep (ABI) and y-sweep (AHI) conventions.
- `config.py` — `SatelliteConfig` and `StereoPairConfig` dataclasses. Presets: `GOES16_CONFIG`, `GOES18_CONFIG`, `HIMAWARI8_CONFIG`.
- `time_model.py` — Per-pixel scan time offsets for ABI/AHI instruments
- `visualize.py` — Debug plotting with cartopy geostationary projections
- `validation/` — Metrics (RMSVD, speed bias), comparison against AMVs, radiosondes, ground points

## Code Conventions

- Python 3.10+, NumPy-style docstrings, type hints on public APIs
- Vectorized numpy operations preferred over loops
- Dataclasses for configuration; per-module `logging.getLogger(__name__)`
- Tests use pytest with class-based organization; common patterns include round-trip tests (navigation) and synthetic-problem exact-recovery tests (solver)

## External Dependency

The `zeus` git submodule provides RAFT FlowRunner and satellite data source integrations. It must be initialized for full pipeline runs but is not needed for unit tests.
