"""Train RAFT for stereo wind retrieval (EarthCARE or IGRA sonde supervision)."""

import argparse
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


# --- Defaults (AWS-style; override with CLI flags on other hosts) ---
DEFAULT_RAFT_CKPT = "/home/ubuntu/earthnet-us-east-3/cache/windflow.raft.202508.epoch254.ckpt"
DEFAULT_DATA_DIR = Path("/home/ubuntu/earthnet-us-east-3/data/stereo_training")


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

    print(f"MODE={args.mode}, BAND={args.band}, EC_WEIGHT={EC_WEIGHT}, "
          f"SONDE_WEIGHT={SONDE_WEIGHT}, MAX_EPOCHS={MAX_EPOCHS}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"RAFT_CKPT={RAFT_CKPT}")
    print(f"PARALLAX={PARALLAX}")
    print(f"VALID_MASK={VALID_MASK}")
    print(f"IGRA_PARQUET={IGRA_COLLOCATION}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")

    # Build per-month datasets and concatenate
    per_month = []
    if multiband and args.mode == "sonde":
        # Multi-band: group pairs by month, each dataset randomly selects a band
        month_bands = _gather_multiband_by_month(DATA_DIR, bands)
        for month_key, band_pairs in sorted(month_bands.items()):
            a_list = [str(p[0]) for p in band_pairs]
            b_list = [str(p[1]) for p in band_pairs]
            print(f"  Loading {month_key}: {len(band_pairs)} bands")
            try:
                ds = IGRADataset(
                    sat_a_zarr=a_list,
                    sat_b_zarr=b_list,
                    parallax_path=PARALLAX,
                    valid_mask_path=VALID_MASK,
                    igra_collocation_path=IGRA_COLLOCATION,
                    patch_size=args.patch_size,
                    train=True,
                    label_radius=args.label_radius,
                    seed=42,
                )
            except Exception as e:
                print(f"    SKIP — {type(e).__name__}: {e}")
                continue
            if len(ds) > 0:
                per_month.append(ds)
                print(f"    → {len(ds)} patches ({len(band_pairs)} bands per sample)")
            else:
                print(f"    → 0 patches, skipping")
    else:
        # Single band
        SAT_PAIRS = _gather_pairs(DATA_DIR, args.band)
        SAT_A_ZARR = [p[0] for p in SAT_PAIRS]
        SAT_B_ZARR = [p[1] for p in SAT_PAIRS]
        print(f"Sat A stores: {[p.name for p in SAT_A_ZARR]}")
        print(f"Sat B stores: {[p.name for p in SAT_B_ZARR]}")
        for a_zarr, b_zarr in zip(SAT_A_ZARR, SAT_B_ZARR):
            print(f"  Loading {a_zarr.name} + {b_zarr.name}")
            try:
                if args.mode == "earthcare":
                    ds = EarthCAREDataset(
                        sat_a_zarr=str(a_zarr),
                        sat_b_zarr=str(b_zarr),
                        parallax_path=PARALLAX,
                        valid_mask_path=VALID_MASK,
                        ec_collocation_path=EC_COLLOCATION,
                        seed=42,
                    )
                else:
                    ds = IGRADataset(
                        sat_a_zarr=str(a_zarr),
                        sat_b_zarr=str(b_zarr),
                        parallax_path=PARALLAX,
                        valid_mask_path=VALID_MASK,
                        igra_collocation_path=IGRA_COLLOCATION,
                        patch_size=args.patch_size,
                        train=True,
                        label_radius=args.label_radius,
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
    print(f"Total: {len(dataset)} patches across {len(per_month)} months")

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
        callbacks=[every_epoch_ckpt],
        logger=logger,
        default_root_dir=str(OUTPUT_DIR),
    )

    trainer.fit(module, loader)
