"""Train RAFT for stereo wind retrieval (EarthCARE or IGRA sonde supervision)."""

import argparse
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset

from stereo_winds.dataset import StereoWindsDataset, EarthCAREDataset, IGRADataset
from stereo_winds.lightning_module import StereoWindsModule


# --- Defaults: env var, then repo-relative. Override with CLI flags. ---
# The published sonde-tuned model was fine-tuned from init-ep254, NOT from
# windflow.raft.pretrained.ckpt (epoch 1434) — those are different models.
DEFAULT_RAFT_CKPT = os.environ.get(
    "STEREO_WINDS_RAFT_CKPT", str(BASE / "checkpoints" / "windflow.raft.init-ep254.ckpt")
)
DEFAULT_DATA_DIR = Path(
    os.environ.get("STEREO_WINDS_TRAIN_DIR", str(BASE / "data" / "stereo_training"))
)


# GOES-16 (Jan 2017–Apr 2025) and GOES-19 (Apr 2025–present) share the same
# grid (sub_lon=-75°), so cubes from both are interchangeable for training.
def _gather_pairs(data_dir: Path, band: str = "C14"):
    """Find matched (sat_a, sat_b) Zarr pairs for a single band."""
    pairs = []
    for sat_a_pat, sat_b_pat in [
        (f"goes19_{band}_*.zarr", f"goes18_remap_goes19_{band}_*.zarr"),
        (f"goes16_{band}_*.zarr", f"goes18_remap_goes16_{band}_*.zarr"),
    ]:
        a_files = sorted(data_dir.glob(sat_a_pat))
        b_files = sorted(data_dir.glob(sat_b_pat))
        for a, b in zip(a_files, b_files):
            pairs.append((a, b))
    return pairs


def _gather_multiband_by_month(data_dir: Path, bands: list[str]):
    """Group (sat_a, sat_b) pairs by month across bands.

    Returns dict mapping month_key → list of (sat_a, sat_b) pairs across bands.
    Each IGRADataset instance gets all band pairs for its month, enabling
    random band selection per sample.
    """
    import re
    month_bands: dict[str, list[tuple]] = {}
    for band in bands:
        for a, b in _gather_pairs(data_dir, band):
            # Extract month key from filename (e.g., "goes19_C14_202601.zarr" → "goes19_202601")
            m = re.search(r'(goes\d+)_\w+_(\d{6})', a.name)
            if m:
                month_key = f"{m.group(1)}_{m.group(2)}"
                month_bands.setdefault(month_key, []).append((a, b))
    return month_bands


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["earthcare", "sonde"], default="earthcare")
    parser.add_argument("--ec-weight", type=float, default=100.0)
    parser.add_argument("--sonde-weight", type=float, default=1.0,
                        help="Weight for IGRA wind loss (m/s units)")
    parser.add_argument("--sonde-delta", type=float, default=5.0,
                        help="Huber threshold for IGRA wind loss (m/s)")
    parser.add_argument("--sonde-loss-type", choices=["huber", "rmsvd"], default="huber",
                        help="Loss type for IGRA wind supervision")
    parser.add_argument("--sonde-match-mode", choices=["closest", "interp"], default="interp",
                        help="How to match h_pred to sonde levels (interp = differentiable through h)")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override max training steps (default: epoch-based)")
    parser.add_argument("--ckpt-every-steps", type=int, default=200)
    parser.add_argument("--rss-weight", type=float, default=1.0,
                        help="Weight for self-supervised RSS loss (0 disables)")
    parser.add_argument("--band", default="C14",
                        help="ABI band (C14, C08, etc.) or 'all' for multi-band random selection")
    parser.add_argument("--label-radius", type=int, default=0,
                        help="Spread sonde labels to a (2R+1)x(2R+1) neighborhood (default 0 = single pixel)")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Training patch size (default 128, matching pretrained RAFT)")
    parser.add_argument("--raft-ckpt", default=DEFAULT_RAFT_CKPT,
                        help="RAFT init checkpoint")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing the goes*_{band}_*.zarr monthly stores "
                             "(also the default location for parallax / valid_mask)")
    parser.add_argument("--parallax", default=None,
                        help="Override path to parallax_goes19_goes18.npz")
    parser.add_argument("--valid-mask", default=None,
                        help="Override path to valid_mask_g19_g18.npy")
    parser.add_argument("--igra-parquet", default=None,
                        help="Override path to igra_all_collocation.parquet")
    parser.add_argument("--ec-parquet", default=None,
                        help="Override path to EarthCARE all_months_collocation.parquet")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Trainer default_root_dir (default: BASE/output/training)")
    parser.add_argument("--wandb-project", default="stereo-winds-v2")
    parser.add_argument("--run-name", default=None,
                        help="Override the W&B run name (default: auto from mode + weights)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (use CSVLogger instead)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--freeze-fnet", action="store_true")
    parser.add_argument("--height-reg-weight", type=float, default=0.0)
    parser.add_argument("--wind-reg-weight", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=12,
                        help="RAFT refinement iterations per forward pass")
    parser.add_argument("--val-months", default=None,
                        help="Comma-separated months (YYYYMM or YYYY-MM) held out "
                             "as a temporal test set (sonde mode only). These months "
                             "are excluded from training; combined with the held-out "
                             "stations (IGRADataset val split) they form a test set "
                             "independent in both space and time. Default: none "
                             "(station-only split, original behavior).")
    parser.add_argument("--resume-from-ckpt", default=None,
                        help="Path to a Lightning checkpoint to RESUME from "
                             "(restores optimizer, LR schedule, and global step "
                             "via trainer.fit(ckpt_path=...)). Use to continue an "
                             "interrupted run to --max-steps. Note: --raft-ckpt "
                             "init weights are overridden by the resume state.")
    args = parser.parse_args()

    RAFT_CKPT = args.raft_ckpt
    DATA_DIR = args.data_dir
    PARALLAX = args.parallax or str(DATA_DIR / "parallax_goes19_goes18.npz")
    VALID_MASK = args.valid_mask or str(DATA_DIR / "valid_mask_g19_g18.npy")
    EC_COLLOCATION = args.ec_parquet or str(BASE / "data" / "earthcare" / "all_months_collocation.parquet")
    IGRA_COLLOCATION = args.igra_parquet or str(BASE / "data" / "igra" / "igra_all_collocation.parquet")
    OUTPUT_DIR = args.output_dir or (BASE / "output" / "training")

    EC_WEIGHT = args.ec_weight if args.mode == "earthcare" else 0.0
    SONDE_WEIGHT = args.sonde_weight if args.mode == "sonde" else 0.0
    MAX_EPOCHS = args.max_epochs

    ALL_BANDS = ["C08", "C09", "C10", "C12", "C14"]
    multiband = args.band == "all"
    bands = ALL_BANDS if multiband else [args.band]

    # Months held out as a temporal test set (normalized to YYYYMM).
    VAL_MONTHS = {
        tok.strip().replace("-", "")
        for tok in (args.val_months or "").split(",")
        if tok.strip()
    }

    print(f"MODE={args.mode}, BAND={args.band}, EC_WEIGHT={EC_WEIGHT}, "
          f"SONDE_WEIGHT={SONDE_WEIGHT}, MAX_EPOCHS={MAX_EPOCHS}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"RAFT_CKPT={RAFT_CKPT}")
    print(f"PARALLAX={PARALLAX}")
    print(f"VALID_MASK={VALID_MASK}")
    print(f"IGRA_PARQUET={IGRA_COLLOCATION}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")

    def _ym(s: str) -> str | None:
        """Extract the YYYYMM token from a month_key or zarr filename."""
        m = re.search(r"(\d{6})", str(s))
        return m.group(1) if m else None

    def _build_igra(entries, train_flag):
        """Build IGRADatasets from (ym, label, a_list, b_list) entries.

        train_flag controls the *station* split inside IGRADataset (held-out
        stations when False). Combined with the month-level partition below,
        the val set is held out in both station (space) and month (time).
        """
        out = []
        for ym, label, a_list, b_list in entries:
            tag = "TRAIN" if train_flag else "VAL"
            print(f"  [{tag}] Loading {label} (ym={ym}): {len(a_list)} band(s)")
            try:
                ds = IGRADataset(
                    sat_a_zarr=a_list,
                    sat_b_zarr=b_list,
                    parallax_path=PARALLAX,
                    valid_mask_path=VALID_MASK,
                    igra_collocation_path=IGRA_COLLOCATION,
                    patch_size=args.patch_size,
                    train=train_flag,
                    label_radius=args.label_radius,
                    seed=42,
                )
            except Exception as e:
                print(f"    SKIP — {type(e).__name__}: {e}")
                continue
            if len(ds) > 0:
                out.append(ds)
                print(f"    → {len(ds)} patches")
            else:
                print(f"    → 0 patches, skipping")
        return out

    # Build per-month datasets, partitioned into train and held-out-month val.
    per_month = []
    val_loader = None
    if args.mode == "sonde":
        # Gather month entries as (ym, label, a_list, b_list).
        month_entries = []
        if multiband:
            month_bands = _gather_multiband_by_month(DATA_DIR, bands)
            for month_key, band_pairs in sorted(month_bands.items()):
                month_entries.append((
                    _ym(month_key), month_key,
                    [str(p[0]) for p in band_pairs],
                    [str(p[1]) for p in band_pairs],
                ))
        else:
            for a_zarr, b_zarr in _gather_pairs(DATA_DIR, args.band):
                month_entries.append((
                    _ym(a_zarr.name), a_zarr.name, [str(a_zarr)], [str(b_zarr)],
                ))

        train_entries = [e for e in month_entries if e[0] not in VAL_MONTHS]
        val_entries = [e for e in month_entries if e[0] in VAL_MONTHS]
        if VAL_MONTHS:
            print(f"Temporal split: holding out months {sorted(VAL_MONTHS)} "
                  f"(matched {len({e[0] for e in val_entries})} of them in data); "
                  f"training on months {sorted({e[0] for e in train_entries})}")
            missing = VAL_MONTHS - {e[0] for e in month_entries}
            if missing:
                print(f"  WARNING: --val-months {sorted(missing)} not present in DATA_DIR")

        print("Building TRAIN datasets...")
        per_month = _build_igra(train_entries, train_flag=True)
        if val_entries:
            print("Building VAL datasets (held-out months × held-out stations)...")
            val_ds_list = _build_igra(val_entries, train_flag=False)
            if val_ds_list:
                val_dataset = ConcatDataset(val_ds_list)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    persistent_workers=args.num_workers > 0,
                )
                print(f"Val: {len(val_dataset)} patches across "
                      f"{len(val_ds_list)} held-out month(s)")
    else:
        # EarthCARE single-band path (no temporal split)
        for a_zarr, b_zarr in _gather_pairs(DATA_DIR, args.band):
            print(f"  Loading {a_zarr.name} + {b_zarr.name}")
            try:
                ds = EarthCAREDataset(
                    sat_a_zarr=str(a_zarr),
                    sat_b_zarr=str(b_zarr),
                    parallax_path=PARALLAX,
                    valid_mask_path=VALID_MASK,
                    ec_collocation_path=EC_COLLOCATION,
                    seed=42,
                )
            except Exception as e:
                print(f"    SKIP — {type(e).__name__}: {e}")
                continue
            if len(ds) > 0:
                per_month.append(ds)
                print(f"    → {len(ds)} patches")
            else:
                print(f"    → 0 patches, skipping")

    dataset = ConcatDataset(per_month)
    print(f"Total (train): {len(dataset)} patches across {len(per_month)} months")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    module = StereoWindsModule(
        raft_ckpt=RAFT_CKPT,
        learning_rate=args.lr,
        freeze_fnet=args.freeze_fnet,
        height_reg_weight=args.height_reg_weight,
        wind_reg_weight=args.wind_reg_weight,
        ec_weight=EC_WEIGHT,
        sonde_weight=SONDE_WEIGHT,
        sonde_huber_delta=args.sonde_delta,
        sonde_loss_type=args.sonde_loss_type,
        sonde_match_mode=args.sonde_match_mode,
        rss_weight=args.rss_weight,
        iters=args.iters,
    )

    if args.run_name is not None:
        run_name = args.run_name
    elif args.mode == "earthcare":
        run_name = f"ec-{args.band}-ecw{EC_WEIGHT:.0f}"
    else:
        run_name = f"sonde-{args.band}-sw{SONDE_WEIGHT:.1f}-d{args.sonde_delta:.0f}"

    if args.no_wandb:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(OUTPUT_DIR), name=run_name)
    else:
        logger = WandbLogger(project=args.wandb_project, name=run_name)

    every_epoch_ckpt = ModelCheckpoint(
        save_top_k=-1,  # keep all
        every_n_train_steps=args.ckpt_every_steps,
        filename="step{step:05d}",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[every_epoch_ckpt],
        logger=logger,
        default_root_dir=str(OUTPUT_DIR),
    )

    if args.resume_from_ckpt:
        print(f"RESUMING from checkpoint: {args.resume_from_ckpt}")
    trainer.fit(module, loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_from_ckpt)
