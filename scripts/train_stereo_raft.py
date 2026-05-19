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

# --- Config ---
RAFT_CKPT = "/home/ubuntu/earthnet-us-east-3/cache/windflow.raft.202508.epoch254.ckpt"
DATA_DIR = Path("/home/ubuntu/earthnet-us-east-3/data/stereo_training")
CACHE_DIR = Path("/home/ubuntu/earthnet-us-east-3/cache")

# GOES-16 (Jan 2017–Apr 2025) and GOES-19 (Apr 2025–present) share the same
# grid (sub_lon=-75°), so cubes from both are interchangeable for training.
def _gather_pairs(band: str = "C14"):
    """Find matched (sat_a, sat_b) Zarr pairs for a single band."""
    pairs = []
    for sat_a_pat, sat_b_pat in [
        (f"goes19_{band}_*.zarr", f"goes18_remap_goes19_{band}_*.zarr"),
        (f"goes16_{band}_*.zarr", f"goes18_remap_goes16_{band}_*.zarr"),
    ]:
        a_files = sorted(DATA_DIR.glob(sat_a_pat))
        b_files = sorted(DATA_DIR.glob(sat_b_pat))
        for a, b in zip(a_files, b_files):
            pairs.append((a, b))
    return pairs


def _gather_multiband_by_month(bands: list[str]):
    """Group (sat_a, sat_b) pairs by month across bands.

    Returns dict mapping month_key → list of (sat_a, sat_b) pairs across bands.
    Each IGRADataset instance gets all band pairs for its month, enabling
    random band selection per sample.
    """
    import re
    month_bands: dict[str, list[tuple]] = {}
    for band in bands:
        for a, b in _gather_pairs(band):
            # Extract month key from filename (e.g., "goes19_C14_202601.zarr" → "goes19_202601")
            m = re.search(r'(goes\d+)_\w+_(\d{6})', a.name)
            if m:
                month_key = f"{m.group(1)}_{m.group(2)}"
                month_bands.setdefault(month_key, []).append((a, b))
    return month_bands


PARALLAX = str(DATA_DIR / "parallax_goes19_goes18.npz")
VALID_MASK = str(DATA_DIR / "valid_mask_g19_g18.npy")

EC_COLLOCATION = str(BASE / "data" / "earthcare" / "all_months_collocation.parquet")
IGRA_COLLOCATION = str(BASE / "data" / "igra" / "igra_all_collocation.parquet")

BATCH_SIZE = 4
NUM_WORKERS = 8
LR = 1e-5
FREEZE_FNET = False
HEIGHT_REG_WEIGHT = 0.0
WIND_REG_WEIGHT = 0.0
ITERS = 12

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
    parser.add_argument("--ckpt-every-steps", type=int, default=200)
    parser.add_argument("--rss-weight", type=float, default=1.0,
                        help="Weight for self-supervised RSS loss (0 disables)")
    parser.add_argument("--band", default="C14",
                        help="ABI band (C14, C08, etc.) or 'all' for multi-band random selection")
    parser.add_argument("--label-radius", type=int, default=0,
                        help="Spread sonde labels to a (2R+1)x(2R+1) neighborhood (default 0 = single pixel)")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Training patch size (default 128, matching pretrained RAFT)")
    parser.add_argument("--raft-ckpt", default=None,
                        help="Override the RAFT init checkpoint (default: windflow.raft.202508.epoch254)")
    args = parser.parse_args()
    if args.raft_ckpt:
        RAFT_CKPT = args.raft_ckpt

    EC_WEIGHT = args.ec_weight if args.mode == "earthcare" else 0.0
    SONDE_WEIGHT = args.sonde_weight if args.mode == "sonde" else 0.0
    MAX_EPOCHS = args.max_epochs

    ALL_BANDS = ["C08", "C09", "C10", "C12", "C14"]
    multiband = args.band == "all"
    bands = ALL_BANDS if multiband else [args.band]

    print(f"MODE={args.mode}, BAND={args.band}, EC_WEIGHT={EC_WEIGHT}, SONDE_WEIGHT={SONDE_WEIGHT}, MAX_EPOCHS={MAX_EPOCHS}")

    # Build per-month datasets and concatenate
    per_month = []
    if multiband and args.mode == "sonde":
        # Multi-band: group pairs by month, each dataset randomly selects a band
        month_bands = _gather_multiband_by_month(bands)
        for month_key, band_pairs in sorted(month_bands.items()):
            a_list = [str(p[0]) for p in band_pairs]
            b_list = [str(p[1]) for p in band_pairs]
            print(f"  Loading {month_key}: {len(band_pairs)} bands")
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
            if len(ds) > 0:
                per_month.append(ds)
                print(f"    → {len(ds)} patches ({len(band_pairs)} bands per sample)")
            else:
                print(f"    → 0 patches, skipping")
    else:
        # Single band
        SAT_PAIRS = _gather_pairs(args.band)
        SAT_A_ZARR = [p[0] for p in SAT_PAIRS]
        SAT_B_ZARR = [p[1] for p in SAT_PAIRS]
        print(f"Sat A stores: {[p.name for p in SAT_A_ZARR]}")
        print(f"Sat B stores: {[p.name for p in SAT_B_ZARR]}")
        for a_zarr, b_zarr in zip(SAT_A_ZARR, SAT_B_ZARR):
            print(f"  Loading {a_zarr.name} + {b_zarr.name}")
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
            if len(ds) > 0:
                per_month.append(ds)
                print(f"    → {len(ds)} patches")
            else:
                print(f"    → 0 patches, skipping")

    dataset = ConcatDataset(per_month)
    print(f"Total: {len(dataset)} patches across {len(per_month)} months")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    module = StereoWindsModule(
        raft_ckpt=RAFT_CKPT,
        learning_rate=LR,
        freeze_fnet=FREEZE_FNET,
        height_reg_weight=HEIGHT_REG_WEIGHT,
        wind_reg_weight=WIND_REG_WEIGHT,
        ec_weight=EC_WEIGHT,
        sonde_weight=SONDE_WEIGHT,
        sonde_huber_delta=args.sonde_delta,
        sonde_loss_type=args.sonde_loss_type,
        sonde_match_mode=args.sonde_match_mode,
        rss_weight=args.rss_weight,
        iters=ITERS,
    )

    if args.mode == "earthcare":
        run_name = f"ec-4month-ecw{EC_WEIGHT:.0f}"
    else:
        run_name = f"sonde-sw{SONDE_WEIGHT:.1f}-d{args.sonde_delta:.0f}"
    wandb_logger = WandbLogger(project="stereo-winds-v2", name=run_name)

    every_epoch_ckpt = ModelCheckpoint(
        save_top_k=-1,  # keep all
        every_n_train_steps=args.ckpt_every_steps,
        filename="step{step:05d}",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[every_epoch_ckpt],
        logger=wandb_logger,
        default_root_dir=str(BASE / "output" / "training"),
    )

    trainer.fit(module, loader)
