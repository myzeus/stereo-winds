"""Train the single-satellite wind student with the zeus tooling.

xbatcher dataset (``StudentXBatchDataset``) + zeus ``BaseLightningModule``
(``StudentWindsModel``).  The radiance ``StandardScalar`` is fit from the train
loader via ``prepare_data_transformation`` before ``trainer.fit``, matching the
earthnetv2 training recipe.
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "zeus"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, ConcatDataset

from stereo_winds.student_xbatcher import StudentXBatchDataset
from stereo_winds.student_zeus_model import StudentWindsModel
from stereo_winds.student_dataset import DEFAULT_FLOW_BANDS, DEFAULT_RAD_BANDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-zarr", required=True, nargs="+",
                   help="One or more feature zarr store(s); a glob of chunk_*.zarr "
                        "is concatenated along time.")
    p.add_argument("--label-zarr", nargs="*", default=[],
                   help="Teacher label store(s); leave empty when the feature "
                        "store(s) already contain the label vars (combined store).")
    p.add_argument("--valid-mask", default=None)
    p.add_argument("--flow-bands", default=",".join(DEFAULT_FLOW_BANDS))
    p.add_argument("--rad-bands", default=",".join(DEFAULT_RAD_BANDS))
    p.add_argument("--rad-time-frames", type=int, choices=[1, 3], default=1,
                   help="1 = single-frame rad at t0 (legacy). 3 = three-frame "
                        "stack t-Δ, t0, t+Δ (per band) — requires chunks "
                        "generated with the temporal-frame schema.")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--no-preload", action="store_true",
                   help="Stream from zarr (dask) instead of loading the crop to RAM")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--trunk", choices=["pixelwise", "unet"], default="pixelwise",
                   help="Network trunk. 'pixelwise' is the legacy 1x1-conv "
                        "MLP with optional 3x3 context. 'unet' is a small "
                        "encoder-decoder for genuine spatial context.")
    p.add_argument("--unet-base-channels", type=int, default=32,
                   help="(--trunk unet) base channels in the first encoder "
                        "level. Levels double width: 32, 64, 128, 256.")
    p.add_argument("--unet-n-levels", type=int, default=3,
                   help="(--trunk unet) number of encoder downsampling steps. "
                        "3 → bottleneck at 1/8 spatial.")
    p.add_argument("--nll-mode", choices=["gaussian", "huber_learned"], default="gaussian",
                   help="Per-component NLL mode for the gaussian wind-loss path "
                        "and for the height loss in all modes.")
    p.add_argument("--wind-loss", choices=["gaussian", "vector"], default="vector",
                   help="'gaussian': per-component NLL on z-scored u, v "
                        "(legacy).  'vector': bivariate Gaussian NLL on (u,v) "
                        "in physical m/s — data term is RMSVD-aligned and the "
                        "model emits a single joint wind logvar per pixel.")
    p.add_argument("--logvar-init-offset", type=float, default=5.0,
                   help="(vector mode) offset added to the model's joint "
                        "logvar at training time so the initial wind-error "
                        "variance estimate sits at e^offset ≈ 150 (m/s)².")
    p.add_argument("--w-u", type=float, default=1.0)
    p.add_argument("--w-v", type=float, default=1.0)
    p.add_argument("--w-h", type=float, default=3.0,
                   help="Height-loss weight.  Default bumped to 3.0 from 1.0 "
                        "because h has been the laggard target.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-ckpt", default=None,
                   help="Optional StudentWindsModel ckpt to warm-start from. "
                        "Weights/buffers are loaded with strict=False; the "
                        "optimizer/scheduler start fresh (not resumed).")
    p.add_argument("--resume-from-ckpt", default=None,
                   help="Full Lightning training-state resume (weights, "
                        "optimizer, scheduler, EarlyStopping/ckpt callback "
                        "state, global_step). Use this to pick up where a "
                        "crashed run left off — distinct from --init-ckpt "
                        "which only restores weights.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--ckpt-every-steps", type=int, default=5000)
    p.add_argument("--ckpt-save-top-k", type=int, default=-1,
                   help="-1 keeps every ckpt at --ckpt-every-steps cadence")
    p.add_argument("--transform-batches", type=int, default=50,
                   help="#train batches to fit the radiance StandardScalar")
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--output-dir", default=str(BASE / "output" / "student_zeus"))
    p.add_argument("--wandb-project", default="stereo-winds-student")
    p.add_argument("--run-name", default="student-zeus")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--val-months", default=None,
                   help="Comma-separated YYYY-MM months held out as val "
                        "(e.g. '2025-02' or '2024-10,2025-02').  Overrides the "
                        "val_mod=5 intra-month decimation, which lets the model "
                        "overfit to the training months.  Required for any run "
                        "claiming generalization.")
    p.add_argument("--random-crop", dest="random_crop",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Random spatial crop per train sample within each "
                        "chunk's overlap window (round-robin over times). "
                        "Train-only — val keeps deterministic xbatcher tiling "
                        "so eval/rmsvd is a stable signal. Flips/rotations are "
                        "intentionally NOT applied (world-frame u,v).")
    p.add_argument("--predict-chi2", dest="predict_chi2",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Distill the teacher's chi² as an extra head channel "
                        "(log-space L1 loss). The student then emits its own "
                        "chi² at inference, enabling downstream QA filtering "
                        "without --qa-from teacher.")
    p.add_argument("--w-chi2", type=float, default=0.1,
                   help="Weight on the chi²-distillation L1 loss term "
                        "(only used when --predict-chi2). Default 0.1 keeps "
                        "it modest vs the wind/h NLL budget.")
    p.add_argument("--w-chi2-dist", type=float, default=0.0,
                   help="(--predict-chi2 only) Weight on the symmetric "
                        "Gaussian KL between batch-level (μ, σ²) of "
                        "predicted vs teacher log(chi²) on masked pixels. "
                        "Counters the distribution-compression failure where "
                        "the chi² head over-narrows its predictions and the "
                        "QA gate ends up excluding all high-wind pixels.")
    p.add_argument("--w-speed", type=float, default=0.0,
                   help="Weight on the speed-magnitude MSE penalty "
                        "((|V_pred| - |V_teacher|)²). 0 disables. Use to "
                        "counteract the systematic ~-3 m/s under-prediction "
                        "of wind speed observed with vector NLL alone "
                        "(componentwise smoothing biases magnitude → 0).")
    p.add_argument("--chi2-separate-head", dest="chi2_separate_head",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="(--predict-chi2 only) Use a dedicated 1×1 conv for "
                        "the chi² output instead of sharing the wind head's "
                        "channels. Prerequisite for --chi2-stop-grad.")
    p.add_argument("--chi2-stop-grad", dest="chi2_stop_grad",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="(--chi2-separate-head only) Detach the trunk "
                        "features before the chi² head so chi² gradients "
                        "never update the U-Net trunk. The trunk is shaped "
                        "solely by wind+h losses; the chi² head learns to "
                        "predict log(teacher_chi²) from those features. "
                        "Targets the chi²-vs-wind capacity competition "
                        "we measured (~2.5 m/s apples-to-apples cost).")
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="Stop training if eval/rmsvd doesn't improve for N "
                        "consecutive evals (only active when --val-months is "
                        "set, since val_mod=5 val_loss is meaningless for "
                        "generalization).")
    args = p.parse_args()

    flow_bands = [b for b in args.flow_bands.split(",") if b]
    rad_bands = [b for b in args.rad_bands.split(",") if b]
    feats = args.feature_zarr
    labs = args.label_zarr if args.label_zarr else [None] * len(feats)
    if len(labs) != len(feats):
        raise ValueError(f"feature/label zarr count mismatch: {len(feats)} vs {len(labs)}")
    val_months_list = (
        [m.strip() for m in args.val_months.split(",") if m.strip()]
        if args.val_months else None
    )
    if val_months_list:
        print(f"Held-out month-CV: val_months={val_months_list} "
              "(val_mod=5 decimation disabled)")
    common = dict(
        valid_mask_path=args.valid_mask, flow_bands=flow_bands, rad_bands=rad_bands,
        patch_size=args.patch_size, overlap=args.overlap,
        preload=not args.no_preload, seed=42,
        rad_time_frames=args.rad_time_frames,
        val_months=val_months_list,
        random_crop=args.random_crop,
    )

    def _build(train):
        parts = []
        for f, l in zip(feats, labs):
            try:
                parts.append(StudentXBatchDataset(
                    feature_zarr=f, label_zarr=l, train=train, **common))
            except ValueError as e:
                print(f"  skip {f} ({'train' if train else 'val'}): {e}")
        return ConcatDataset(parts) if parts else None

    train_ds = _build(True)
    val_ds = _build(False)
    if train_ds is None:
        raise RuntimeError("No training data — all chunks failed")
    if val_ds is None:
        print("No validation data; training without validation.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)
                  if val_ds is not None else None)

    model = StudentWindsModel(
        n_flow_bands=len(flow_bands), n_rad_bands=len(rad_bands),
        hidden=args.hidden, n_layers=args.n_layers, context=not args.no_context,
        w_u=args.w_u, w_v=args.w_v, w_h=args.w_h, nll_mode=args.nll_mode,
        wind_loss=args.wind_loss, logvar_init_offset=args.logvar_init_offset,
        trunk=args.trunk, unet_base_channels=args.unet_base_channels,
        unet_n_levels=args.unet_n_levels,
        rad_time_frames=args.rad_time_frames,
        predict_chi2=args.predict_chi2,
        w_chi2=args.w_chi2,
        w_chi2_dist=args.w_chi2_dist,
        w_speed=args.w_speed,
        chi2_separate_head=args.chi2_separate_head,
        chi2_stop_grad=args.chi2_stop_grad,
        learning_rate=args.lr,
    )

    if args.init_ckpt:
        import torch
        print(f"warm-start: loading weights from {args.init_ckpt}")
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  missing keys (left at init): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"  unexpected keys (skipped): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    # Fit the radiance StandardScalar from the data (zeus convention).
    if rad_bands:
        print(f"Fitting radiance StandardScalar on {args.transform_batches} batches...")
        model.prepare_data_transformation(train_loader, n_batches=args.transform_batches)
        print("  rad mean:", model.transform.mu["rad"].detach().cpu().numpy())
        print("  rad std :", model.transform.sd["rad"].detach().cpu().numpy())
    # Fit per-target mu/sd so the NLL is dimensionless and balanced.
    print(f"Fitting target stats on {args.transform_batches} batches...")
    model.fit_target_stats(train_loader, n_batches=args.transform_batches)

    loggers = [CSVLogger(save_dir=args.output_dir, name=args.run_name)]
    if not args.no_wandb:
        loggers.append(WandbLogger(project=args.wandb_project, name=args.run_name))
    # ckpt every --ckpt-every-steps; save_top_k=-1 keeps every one (intended
    # for long runs where we want a checkpoint trail).  Always also save_last.
    ckpt = ModelCheckpoint(
        save_last=True,
        save_top_k=args.ckpt_save_top_k,
        monitor="train/rmsvd" if args.ckpt_save_top_k > 0 else None,
        mode="min",
        every_n_train_steps=args.ckpt_every_steps,
        filename="step-{step:06d}",
    )
    callbacks = [ckpt]
    # When --val-months is set, eval/rmsvd reflects true generalization, so
    # also track best-val checkpoints and arm early stopping.  When the user
    # leaves val_mod=5 in place, eval/rmsvd is just decimated training data —
    # using it for selection would replicate the original overfit.
    if val_months_list and val_ds is not None:
        callbacks.append(ModelCheckpoint(
            save_last=False, save_top_k=3,
            monitor="eval/rmsvd", mode="min",
            filename="best-val-{step:06d}",
        ))
        callbacks.append(EarlyStopping(
            monitor="eval/rmsvd", mode="min",
            patience=args.early_stop_patience, verbose=True,
        ))
        print(f"  callbacks: best-val ckpt + EarlyStopping (patience="
              f"{args.early_stop_patience}) on eval/rmsvd")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, max_steps=args.max_steps,
        accelerator=args.accelerator, devices=1, precision="32-true",
        gradient_clip_val=1.0, log_every_n_steps=10,
        callbacks=callbacks, logger=loggers, default_root_dir=args.output_dir,
    )
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=args.resume_from_ckpt)


if __name__ == "__main__":
    main()
