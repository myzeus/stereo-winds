"""Train the single-satellite wind student on distilled stereo labels.

Pairs ``student_feat_*.zarr`` (inputs) with the teacher ``*_iter*.zarr`` cache
(targets) and trains a lightweight pixelwise model.  Mirrors
``train_stereo_raft.py`` for arg/logging/trainer conventions.
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from stereo_winds.student_dataset import (
    StudentWindsDataset, DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS,
)
from stereo_winds.student_module import StudentWindsModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-zarr", nargs="+", required=True,
                   help="student_feat_*.zarr path(s)")
    p.add_argument("--label-zarr", nargs="+", required=True,
                   help="teacher cache zarr path(s)")
    p.add_argument("--valid-mask", required=True)
    p.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS),
                   help="Bands with temporal optical flow inputs")
    p.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS),
                   help="Bands with radiance inputs (default: all IR C07-C16)")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--use-teacher-sigma", action="store_true")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--nll-mode", choices=["gaussian", "huber_learned"], default="gaussian")
    p.add_argument("--w-u", type=float, default=1.0)
    p.add_argument("--w-v", type=float, default=1.0)
    p.add_argument("--w-h", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--ckpt-every-steps", type=int, default=200)
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--output-dir", default=str(BASE / "output" / "student"))
    p.add_argument("--wandb-project", default="stereo-winds-student")
    p.add_argument("--run-name", default="student")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="CPU, 5 steps, patch 128 — quick wiring check")
    args = p.parse_args()

    if args.smoke:
        args.accelerator = "cpu"
        args.max_steps = 5
        args.patch_size = 128
        args.num_workers = 0
        args.batch_size = 1
        args.no_wandb = True

    flow_bands = [b.strip() for b in args.flow_bands.split(",") if b.strip()]
    rad_bands = [b.strip() for b in args.rad_bands.split(",") if b.strip()]
    common = dict(
        feature_zarr=args.feature_zarr, label_zarr=args.label_zarr,
        valid_mask_path=args.valid_mask, flow_bands=flow_bands, rad_bands=rad_bands,
        patch_size=args.patch_size,
        use_teacher_sigma=args.use_teacher_sigma, seed=42,
    )
    train_ds = StudentWindsDataset(train=True, samples_per_epoch=args.samples_per_epoch, **common)
    # Reuse the train split's per-band radiance z-score stats for validation.
    try:
        val_ds = StudentWindsDataset(train=False, rad_stats=train_ds.rad_stats, **common)
    except ValueError:
        val_ds = None
        print("No validation times available; training without validation.")

    print(f"Train samples/epoch: {len(train_ds)}  in_channels: {train_ds.in_channels}")
    print(f"radiance z-score stats (mean, std) per band {rad_bands}:\n{train_ds.rad_stats}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=True, persistent_workers=args.num_workers > 0)
        if val_ds is not None else None
    )

    module = StudentWindsModule(
        in_channels=train_ds.in_channels, hidden=args.hidden, n_layers=args.n_layers,
        context=not args.no_context, learning_rate=args.lr,
        w_u=args.w_u, w_v=args.w_v, w_h=args.w_h, nll_mode=args.nll_mode,
        use_teacher_sigma=args.use_teacher_sigma,
        flow_bands=flow_bands, rad_bands=rad_bands,
        rad_stats=train_ds.rad_stats.tolist(),
    )

    # Always keep a CSV logger for programmatic access; add wandb for the
    # live dashboard + image panels unless disabled.
    loggers = [CSVLogger(save_dir=args.output_dir, name=args.run_name)]
    if not args.no_wandb:
        loggers.append(WandbLogger(project=args.wandb_project, name=args.run_name))

    ckpt_cb = ModelCheckpoint(
        save_top_k=3, monitor="val/rmsvd" if val_loader else "train/rmsvd",
        mode="min", every_n_train_steps=args.ckpt_every_steps,
        filename="step{step:05d}",
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, max_steps=args.max_steps,
        accelerator=args.accelerator, devices=1, precision="32-true",
        gradient_clip_val=1.0, log_every_n_steps=10,
        callbacks=[ckpt_cb], logger=loggers, default_root_dir=args.output_dir,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
