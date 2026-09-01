# RAFT sonde fine-tuning — development log (superseded run)

How the RAFT optical-flow network was fine-tuned so that cross-satellite
(GOES-16/18) stereo winds match held-out IGRA radiosondes. Work carried out on
an HPC cluster (NVIDIA GH200 partition) in May 2026.

**Status: historical.** This log documents training run `b82e`, which was
later superseded. It is kept for the methodology — the recipe, the QA
definition (§7), and the root-cause notes (§6) all still apply — but its
*numbers* describe a model that is not the one shipped here. The validation of
the shipped checkpoint is the published evaluation, reproduced by
`scripts/fig_finetuning_improvement.py` and `scripts/fig_student_teacher_igra.py`.

> **Checkpoint provenance — read before quoting these numbers.**
> The results below are from training run **`b82e`, step 77500 (epoch 5)**.
> The checkpoint shipped in this repository,
> `checkpoints/windflow.raft.sonde-tuned.ckpt`, is a **different run** and is
> *not* the artifact these tables measure. Three independent facts establish
> this:
>
> - The shipped file records `epoch 11 / step 75000`, against b82e's
>   `step 77500 / epoch 5` — ~6.3k vs ~12.9k steps per epoch, so the two runs
>   did not even see the same amount of data per epoch.
> - The shipped file has `height_reg_weight = 1.0`. The `--height-reg-weight`
>   default is `0.0` and the §2 command below does not pass the flag, so b82e
>   trained without height regularization.
> - Height regularization instantiates a frozen `raft_ref` copy
>   (`lightning_module.py`). The shipped checkpoint carries 358 state-dict
>   keys (`raft.*` + `raft_ref.*`); a b82e checkpoint would carry 179.
>
> Both runs start from the same `epoch254` WindFlow init and share the loss
> recipe, but the tables in §3–§5 should not be attributed to the shipped
> file. The shipped checkpoint's validation lives in the published results,
> generated with cache label `hreg1s75` / `hreg1_step75` by
> `scripts/fig_finetuning_improvement.py`, `scripts/fig_student_teacher_igra.py`
> and `scripts/compare_quad_collocation.py`.

## 1. Objective & outcome

**Goal:** fine-tune RAFT so the stereo-wind retrieval matches held-out IGRA
radiosondes better than the pretrained model — *not* to match the Carr et
al. AMV baseline (Carr is a cross-check only).

**Outcome:** success. The tuned model cuts held-out sonde RMSVD by **17%**
(8.67 → 7.21 m/s), raises wind-speed correlation **0.42 → 0.92**, eliminates
the speed bias (−3.25 → +0.31 m/s), drops median direction error
**45° → 9°**, and recovers full-troposphere coverage (3.5× more valid
matches, now spanning low/mid/high vs almost all-low for the pretrained
model). It keeps Carr-level height/direction skill and generalizes zero-shot
to the C04 band it never trained on.

## 2. Training recipe

Initialized from `checkpoints/windflow.raft.init-ep254.ckpt` (epoch 254 /
step 255000); batch 4, ~15 h on a single GH200.

```bash
export STEREO_WINDS_DATA_DIR=/path/to/data       # zarrs/, weights/, labels/
export STEREO_WINDS_RAFT_CKPT=checkpoints/windflow.raft.init-ep254.ckpt

python scripts/train_stereo_raft.py \
    --mode sonde --band all --sonde-loss-type rmsvd \
    --sonde-weight 1.0 --sonde-delta 5.0 --rss-weight 1.0 \
    --raft-ckpt    "$STEREO_WINDS_RAFT_CKPT" \
    --data-dir     "$STEREO_WINDS_DATA_DIR/zarrs" \
    --parallax     "$STEREO_WINDS_DATA_DIR/zarrs/parallax_goes19_goes18.npz" \
    --valid-mask   "$STEREO_WINDS_DATA_DIR/zarrs/valid_mask_g19_g18.npy" \
    --igra-parquet "$STEREO_WINDS_DATA_DIR/labels/igra/igra_all_collocation.parquet" \
    --patch-size 512 --batch-size 4 --max-epochs 5 --ckpt-every-steps 500
```

## 3. Held-out radiosonde evaluation

Val split = `station_idx % 5 == 0` (matches `IGRADataset`): 85 val
stations, 32 inside the GOES-16/18 overlap. 48 scenes (202412/202501/202502,
C14, 00/12 UTC), strict QA, bracketing height match.

| metric | init ep254 | tuned |
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

## 4. Carr et al. cross-check

GOES-16/18, 2025-01-08 19:00 UTC, post strict QA. The tuned model **diverges
from Carr on wind speed by design** (sonde-correcting the under-speed makes
it read faster than Carr's similarly slow AMVs) while preserving height and
directional structure.

| vs Carr (post-QA) | C14 tuned | C14 ep254 (init) | C14 pretrained | C04 tuned* |
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

## 5. Per-band sensing level (all trained bands)

Median feature-tracked height by band (pretrained → tuned), reflecting each
band's sensing level:

| band | sensing level | pretrained | tuned |
|---|---|---|---|
| C08 (6.2 µm) | upper-trop WV | 10.6 km | 10.4 km |
| C09 (6.9 µm) | mid-upper WV | 10.1 km | 10.0 km |
| C10 (7.3 µm) | low-mid WV | 7.4 km | 7.8 km |
| C12 (9.6 µm) | ozone/upper | 1.7 km | **3.9 km** |
| C14 (11.2 µm) | IR window | 1.8 km | **4.2 km** |

WV bands already sit high in both (they only see upper moisture); tuning
mainly densifies coverage there. The window bands (C12/C14) are where
tuning transforms the result — recovering the full vertical wind structure
the pretrained model collapsed to ~1.7 km. All bands gain ~25–130% more
valid retrievals.

## 6. Root causes of early bad numbers

Recorded because each produced plausible-looking but wrong output:

1. **Remap LUT built from per-scene metadata.** `scripts/cache_scene_from_s3.py`
   must build the goes18→goes16 LUT from the canonical `SATELLITE_CONFIGS`
   sub-longitude (−75.0, matching `training_data.py`), not from satpy
   per-scene metadata (−75.20). The 0.20° drift was a ~10 px constant
   cross-satellite shift that the solver turned into a bogus +12 km height
   bias.
2. **Tiled-inference reassembly sized to the tile, not the image.** Size the
   accumulators to the full image (`upperleft.max() + tile`); otherwise
   non-lowmem inference crashes on 5424² full disks.
3. **Tiled inference dropping all but the first tile.** The batch path
   indexed `[0]` before reassembly, silently discarding 7 of 8 tiles. The
   fix returns the full batch and lets the low-memory call site index. Once
   corrected, non-lowmem agrees with lowmem to ~1% and is ~2.3× faster.

## 7. QA definition (identical for plots and eval)

Two layers:

- **Solver `quality_flag`** (`solver.py`): 1 everywhere, zeroed where height
  is non-finite or outside [0, 20000] m.
- **Post-hoc mask** (`eval_from_parquet.py` / `make_spatial_winds.py`): keep a
  pixel only if all hold — `quality_flag > 0`, `chi2 ≤ 0.2`, `sigma_h ≤ 5000 m`,
  Sobel height-gradient `≤ 3000 m/px`, speed `≤ 100 m/s`, `1000 ≤ h ≤ 20000 m`,
  all of u/v/h/chi2/sigma_h finite.

`chi2 ≤ 0.2` is the dominant cut (~83% of solver-good pixels rejected on the
Carr scene). Because the tuned model's flows fit the stereo geometry better,
more pixels survive it — the higher barb density and match counts are a real
quality gain, not a looser filter. Plots and the sonde eval use the same QA.

## 8. Supporting scripts

| Script | Role |
|--------|------|
| `scripts/cache_scene_from_s3.py` | Pull the 5 raw scenes, remap goes18→goes16, write `{A0,A_minus,A_plus,B_minus,B_plus}.npy` |
| `scripts/eval_from_parquet.py` | Held-out sonde evaluator; `--split {all,train,val}`, 5×5 neighborhood-median QA, bracketing match, per-layer stats |
| `scripts/make_spatial_winds.py` | Spatial pretrained-vs-tuned barb maps (`--time-index`, `--stride`, `--extent`) |
| `scripts/cache_stereo_retrievals.py` | Batch retrieval caching; also writes `sigma_u`/`sigma_v` |
| `scripts/train_stereo_raft.py` | The fine-tuning entry point used above |

Two scripts referenced by earlier revisions of this document,
`infer_and_compare_carr.py` and `analyze_sonde_eval.py`, were removed as
one-off analysis code and are no longer in the repository.

## 9. Open follow-ups

- **Checkpoint sweep** across training steps to confirm the deliverable step
  is optimal rather than an earlier plateau.
- **Live held-out `val_dataloader`** in `train_stereo_raft.py` so held-out
  sonde RMSVD is a tracked metric during training, avoiding manual eval.
- Nothing here needs re-running: the shipped checkpoint is already validated
  by the published evaluation. Re-running §3–4 against it would only restate
  those results in this log's format.
