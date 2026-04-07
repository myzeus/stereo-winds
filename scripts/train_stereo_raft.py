"""Train RAFT for stereo wind retrieval (Phase 1: self-supervised RSS)."""

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

from stereo_winds.dataset import StereoWindsDataset, EarthCAREDataset
from stereo_winds.lightning_module import StereoWindsModule

# --- Config ---
RAFT_CKPT = str(BASE / "zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt")
DATA_DIR = BASE / "data" / "stereo_training"
CACHE_DIR = Path("/home/ubuntu/earthnet-us-east-3/cache")

SAT_A_ZARR = sorted(DATA_DIR.glob("goes19_C14_*.zarr"))
SAT_B_ZARR = sorted(DATA_DIR.glob("goes18_remap_goes19_C14_*.zarr"))
PARALLAX = str(DATA_DIR / "parallax_goes19_goes18.npz")
VALID_MASK = str(DATA_DIR / "valid_mask_g19_g18.npy")

EC_COLLOCATION = str(BASE / "data" / "earthcare" / "all_months_collocation.parquet")

BATCH_SIZE = 4
NUM_WORKERS = 8
LR = 1e-5
FREEZE_FNET = False
HEIGHT_REG_WEIGHT = 0.0
WIND_REG_WEIGHT = 0.0
ITERS = 12

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ec-weight", type=float, default=100.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()

    EC_WEIGHT = args.ec_weight
    MAX_EPOCHS = args.max_epochs

    print(f"EC_WEIGHT={EC_WEIGHT}, MAX_EPOCHS={MAX_EPOCHS}")
    print(f"Sat A stores: {[p.name for p in SAT_A_ZARR]}")
    print(f"Sat B stores: {[p.name for p in SAT_B_ZARR]}")

    # Build per-month datasets and concatenate — avoids slow xr.open_mfdataset
    per_month = []
    for a_zarr, b_zarr in zip(SAT_A_ZARR, SAT_B_ZARR):
        print(f"  Loading {a_zarr.name} + {b_zarr.name}")
        ds = EarthCAREDataset(
            sat_a_zarr=str(a_zarr),
            sat_b_zarr=str(b_zarr),
            parallax_path=PARALLAX,
            valid_mask_path=VALID_MASK,
            ec_collocation_path=EC_COLLOCATION,
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
        iters=ITERS,
    )

    wandb_logger = WandbLogger(project="stereo-winds-v2", name=f"ec-4month-ecw{EC_WEIGHT:.0f}")

    every_epoch_ckpt = ModelCheckpoint(
        save_top_k=-1,  # keep all
        every_n_train_steps=200,
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
