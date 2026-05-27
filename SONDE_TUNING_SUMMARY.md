# Stereo-Winds RAFT Sonde-Tuning — Summary

Fine-tuning the windflow RAFT optical-flow model so cross-satellite
(GOES-16/18) stereo winds match held-out radiosondes, run on the NASA
ADAPT cluster. Work spanned May 2026.

## 1. Objective & outcome

**Goal:** fine-tune RAFT so the stereo-wind retrieval matches held-out IGRA
radiosondes better than the pretrained model — *not* to match the Carr et
al. AMV baseline (Carr is a cross-check only).

**Outcome:** success. The tuned model (`b82e`, step-77500) cuts held-out
sonde RMSVD by **17%** (8.67 → 7.21 m/s), raises wind-speed correlation
**0.42 → 0.92**, eliminates the speed bias (−3.25 → +0.31 m/s), drops
median direction error **45° → 9°**, and recovers full-troposphere
coverage (3.5× more valid matches, now spanning low/mid/high vs almost
all-low for the pretrained model). It keeps Carr-level height/direction
skill and generalizes zero-shot to the C04 band it never trained on.

## 2. Deliverable model

```
/explore/nobackup/people/tvandal/data/stereo-winds/weights/windflow.raft.sonde-tuned.b82e.step77500.ckpt
```
- md5 `0db214448ee76399f188682a0ea90b60`
- W&B run `sonde-all-512-rmsvd-ep254` (`jbi0b82e`), project `zeusai/stereo-winds-v2`; step 77500 = epoch 5.
- Init from `windflow.raft.202508.epoch254.ckpt`.
- Compute: grace partition (NVIDIA GH200), batch 4, ~15 h.

Train command (recipe matches the prior good run `sonde-sw1.0-d5`):
```bash
python scripts/train_stereo_raft.py \
    --mode sonde --band all --sonde-loss-type rmsvd \
    --sonde-weight 1.0 --sonde-delta 5.0 --rss-weight 1.0 \
    --raft-ckpt   $DATA/weights/windflow.raft.202508.epoch254.ckpt \
    --data-dir    $DATA/zarrs \
    --parallax    $DATA/zarrs/parallax_goes19_goes18.npz \
    --valid-mask  $DATA/zarrs/valid_mask_g19_g18.npy \
    --igra-parquet $DATA/labels/igra/igra_all_collocation.parquet \
    --patch-size 512 --batch-size 4 --max-epochs 5 --ckpt-every-steps 500
```
> The original good run inited from windflow `epoch=203-step=204000.ckpt`,
> which is not on ADAPT; `epoch254` was the chosen substitute.

## 3. Held-out radiosonde evaluation

Val split = `station_idx % 5 == 0` (matches `IGRADataset`): 85 val
stations, 32 inside the GOES-16/18 overlap. 48 scenes (202412/202501/202502,
C14, 00/12 UTC), strict QA, bracketing height match.

| metric | init ep254 | tuned b82e |
|---|---|---|
| N matched | 56 | 195 |
| RMSVD (m/s) | 8.67 | **7.21** |
| speed bias (m/s) | −3.25 | **+0.31** |
| speed corr | 0.423 | **0.922** |
| u / v corr | 0.539 / 0.663 | **0.939 / 0.912** |
| height RMSE / bias (m) | 267 / +51 | 298 / +29 |
| \|dir err\| median | 45.3° | **9.4°** |

Per-layer tuned RMSVD: low (≥700 hPa) 6.43 m/s (N=96), mid (400–700) 8.69
(N=63), high (<400) 6.28 (N=36). The pretrained model produced almost only
low-cloud matches (52/56) — tuning fixed a wind-speed scale under-bias and
extended useful winds through the jet level.

Plots: `output/sonde_eval_largeN/` (`eval_scatter_{init_ep254,tuned_b82e}_val.png`)
and `output/sonde_eval_largeN/analysis/` (`sonde_analysis.png` 12-panel, `stats.txt`).

## 4. Carr et al. cross-check

GOES-16/18, 2025-01-08 19:00 UTC, post strict QA. The tuned model **diverges
from Carr on wind speed by design** (sonde-correcting the under-speed makes
it read faster than Carr's similarly slow AMVs) while preserving height and
directional structure.

| vs Carr (post-QA) | C14 tuned | C14 ep254 (init) | C14 raft-128.1434 | C04 tuned* |
|---|---|---|---|---|
| N matched | 20,297 | 20,717 | 19,117 | 7,105 |
| RMSVD (m/s) | 5.24 | 3.86 | 4.44 | 6.97 |
| speed corr | 0.927 | — | — | 0.861 |
| height RMSE (m) | 1236 | 1214 | 922 | 1822 |
| height bias (m) | +19 | — | +8 | −666 |
| height corr | 0.939 | 0.914 | 0.942 | 0.910 |

\* **C04 was not in the training set** (trained on C08/C09/C10/C12/C14) —
the r=0.91 height / 0.86 speed agreement is zero-shot generalization to an
unseen band.

Plots: `output/carr_tuned/C{14,04}_tuned_compare_scatter.png` (+ full suite
on ADAPT under `stereo-winds-runs/carr_tuned/`).

**Pretrained baseline reproduction** (before tuning, to validate the
pipeline): `raft-128.202509.epoch1434` reproduced historical numbers —
C14 hRMSE 922 m / r 0.94 / RMSVD 4.44; C04 r 0.987.

## 5. Spatial wind maps (all trained bands)

Pretrained vs tuned barb maps (full overlap + CONUS/Gulf zoom),
geostationary projection, inverted-IR background, barbs colored by
cloud-top height — style from `stereo-winds-lambda/.../plot_ai_vs_carr_barbs.py`.

`output/sonde_eval_largeN/analysis/spatial_winds_C{08,09,10,12,14}_202501{,_conus}.png`

Median cloud-top height by band (pretrained → tuned), reflecting each
band's sensing level:

| band | sensing level | pretrained | tuned |
|---|---|---|---|
| C08 (6.2 µm) | upper-trop WV | 10.6 km | 10.4 km |
| C09 (6.9 µm) | mid-upper WV | 10.1 km | 10.0 km |
| C10 (7.3 µm) | low-mid WV | 7.4 km | 7.8 km |
| C12 (9.6 µm) | ozone/upper | 1.7 km | **3.9 km** |
| C14 (11.2 µm) | IR window / cloud-top | 1.8 km | **4.2 km** |

WV bands already sit high in both (they only see upper moisture); tuning
mainly densifies coverage there. The window bands (C12/C14) are where
tuning transforms the result — recovering the full vertical wind structure
the pretrained model collapsed to ~1.7 km. All bands gain ~25–130% more
valid retrievals.

## 6. Bug fixes (root causes of early bad numbers)

1. **`scripts/cache_scene_from_s3.py`** — build the goes18→goes16 remap LUT
   from canonical `SATELLITE_CONFIGS` (sub_lon −75.0, matching
   `training_data.py`), not satpy per-scene metadata (−75.20). The 0.20°
   drift was a ~10 px constant cross-sat shift the solver turned into a
   bogus +12 km height bias.
2. **`zeus/zeus/inference/inference_flows.py::reassemble_split_array`** —
   size accumulators to the full image (`upperleft.max()+tile`), not the
   tile shape; non-lowmem inference crashed on 5424² disks.
3. **same file `get_model_outputs`** — return the full batch (it indexed
   `[0]`, silently dropping 7/8 tiles in non-lowmem); lowmem call site now
   indexes `[0]` itself. Non-lowmem ≈ lowmem within ~1%, ~2.3× faster.

## 7. Scripts

New (in `scripts/`):
- `cache_scene_from_s3.py` — pull 5 raw scenes via zeus S3, remap goes18→goes16, write `{A0,A_minus,A_plus,B_minus,B_plus}.npy`.
- `infer_and_compare_carr.py` — RAFT+WLS on a cached scene → 4 stage-diagnostic + 5 Carr-comparison PNGs + CF-1.8 NetCDF.
- `eval_from_parquet.py` — held-out sonde evaluator; `--split {all,train,val}`, 5×5 neighborhood-median QA, bracketing match, per-layer stats, multi-store aggregation.
- `analyze_sonde_eval.py` — 12-panel init-vs-tuned diagnostic + expanded stats.
- `make_spatial_winds.py` — spatial pretrained-vs-tuned barb maps (`--time-index`, `--stride`, `--extent`).

Modified:
- `cache_stereo_retrievals.py` — `--data-dir/--sat-a-tag/--sat-b-tag/--band/--parallax-cache`; also writes sigma_u/sigma_v.
- `train_stereo_raft.py` — `--data-dir/--parallax/--valid-mask/--igra-parquet/--output-dir/--max-steps/--run-name/--no-wandb` + batch/worker/lr knobs; robust skip on corrupt zarr.
- `stereo_winds/lightning_module.py` — `_log_images` guarded for non-W&B loggers.

All synced to the ADAPT clone `/explore/nobackup/people/tvandal/stereo-winds/`.

## 8. QA definition (identical for plots and eval)

Two layers:
- **Solver `quality_flag`** (`solver.py`): 1 everywhere, zeroed where height
  is non-finite or outside [0, 20000] m.
- **Post-hoc mask** (`eval_from_parquet.py` / `make_spatial_winds.py`): keep a
  pixel only if all hold — `quality_flag>0`, `chi2 ≤ 0.2`, `sigma_h ≤ 5000 m`,
  Sobel height-gradient `≤ 3000 m/px`, speed `≤ 100 m/s`, `1000 ≤ h ≤ 20000 m`,
  all of u/v/h/chi2/sigma_h finite.

`chi2 ≤ 0.2` is the dominant cut (~83% of solver-good pixels rejected on the
Carr scene). Because the tuned model's flows fit the stereo geometry better,
more pixels survive it — the higher barb density / match counts are a real
quality gain, not a looser filter. Plots and the sonde eval use the same QA.

## 9. Key paths & environment (ADAPT)

- Data: `/explore/nobackup/people/tvandal/data/stereo-winds/`
  (`zarrs/`, `weights/`, `labels/igra/igra_all_collocation.parquet`, `carr_data/`, `cache/`).
- Runs/outputs: `/explore/nobackup/people/tvandal/stereo-winds-runs/`.
- Repo clone: `/explore/nobackup/people/tvandal/stereo-winds/`.
- Env: `module load miniforge && conda activate $NOBACKUP/envs/zeus-h100`
  (built for GH200 — runs on grace compute nodes, not the login node).
- Scheduler: SLURM, partition `grace` (GH200), account `j1115`; `sbatch`
  only on compute nodes. Submit self-contained sbatch jobs — interactive
  `nohup` processes die when the borrowing allocation ends.
- W&B: project `zeusai/stereo-winds-v2`.
- Local mirror of outputs: `output/{2025-01-08T1900_C14*, sonde_eval_largeN, carr_tuned}/`.

## 10. Open follow-ups (not done)

- **Checkpoint sweep** across b82e steps (20k/40k/60k/77.5k) to confirm
  77500 is optimal vs an earlier plateau.
- **Live held-out `val_dataloader`** in `train_stereo_raft.py` so held-out
  sonde RMSVD is a tracked W&B metric on future runs (avoids manual eval).
- **Backfill missing data** — `goes19_C14_202511.zarr` is corrupt
  (no zarr group); a few goes19 months are partial (disk-space trimmed).
- A second tuning run from the true `epoch=203` init, if it can be located,
  to match the original recipe exactly.
