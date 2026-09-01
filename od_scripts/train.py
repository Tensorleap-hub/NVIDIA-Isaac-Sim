"""RF-DETR fine-tune with in-process ReduceLROnPlateau + early stopping (the repo's
standing training recipe, formerly train_warehouse_reduce_lr.py).

Two monkeypatches on the installed rfdetr package (applied at import, site-packages untouched):
  1. build_trainer -> num_sanity_val_steps=0. The 2-batch sanity eval is otherwise logged as a
     real epoch-0 "best" and poisons best-checkpoint tracking.
  2. RFDETRModelModule.configure_optimizers -> keep rf-detr's AdamW + param groups but replace
     its LambdaLR with torch ReduceLROnPlateau (mode=max on val/ema_mAP_50_95, factor 0.1,
     patience 3, abs threshold 5e-4, per-group min_lr = base_lr * lr_min_factor).

Defaults reproduce the 3-stage ladder lr 1e-4 -> 1e-5 -> 1e-6. EarlyStopping (patience >
scheduler patience) ends the run once the LR has bottomed out and plateaued again.

Deployable checkpoint: <output-dir>/checkpoint_best_ema.pth.
NOTE: rf-detr silently resumes from <output-dir>/last.ckpt if the dir is non-empty.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import rfdetr.training as _training  # noqa: E402
from rfdetr.training.module_model import RFDETRModelModule  # noqa: E402
from rfdetr import RFDETRBase, RFDETRLarge  # noqa: E402
from common import load_class_names, arm_output_dir  # noqa: E402

MONITOR = "val/ema_mAP_50_95"

# --- patch 1: no sanity-val ---
_orig_build_trainer = _training.build_trainer


def _no_sanity_build_trainer(train_config, model_config, **kwargs):
    kwargs.setdefault("num_sanity_val_steps", 0)
    return _orig_build_trainer(train_config, model_config, **kwargs)


_training.build_trainer = _no_sanity_build_trainer

# --- patch 2: ReduceLROnPlateau ---
_orig_configure = RFDETRModelModule.configure_optimizers
_SCHED = {"factor": 0.1, "patience": 3, "min_delta": 5e-4, "lr_min_factor": 1e-2}


def _plateau_configure_optimizers(self):
    out = _orig_configure(self)
    optimizer = out["optimizer"] if isinstance(out, dict) else out[0][0]
    min_lrs = [g["lr"] * _SCHED["lr_min_factor"] for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=_SCHED["factor"], patience=_SCHED["patience"],
        threshold=_SCHED["min_delta"], threshold_mode="abs", min_lr=min_lrs,
    )
    return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": MONITOR}}


RFDETRModelModule.configure_optimizers = _plateau_configure_optimizers


def main():
    p = argparse.ArgumentParser(description="RF-DETR + in-process ReduceLROnPlateau")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", default=None, help="default: <dataset-dir>/output/rfdetr_reducelr")
    p.add_argument("--pretrain-weights", default=None, help="omit -> RF-DETR COCO pretrain (the normal choice)")
    p.add_argument("--resume", default=None, help="Lightning .ckpt (e.g. <output-dir>/last.ckpt) to continue "
                   "an interrupted/capped run from — resumes optimizer, scheduler, epoch, EMA, best-ckpt state. "
                   "Raise --epochs past the previous cap or the run will see it's already 'done' and stop immediately.")
    p.add_argument("--epochs", type=int, default=60, help="hard cap; early stopping usually ends sooner")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--warmup-epochs", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--large", action="store_true")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--sched-factor", type=float, default=0.1)
    p.add_argument("--sched-patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=5e-4)
    p.add_argument("--lr-min-factor", type=float, default=1e-2, help="LR floor = lr_min_factor * base_lr")
    p.add_argument("--es-patience", type=int, default=8, help="keep > sched-patience so the LR can drop first")
    args = p.parse_args()

    _SCHED.update(factor=args.sched_factor, patience=args.sched_patience,
                  min_delta=args.min_delta, lr_min_factor=args.lr_min_factor)

    ds = Path(args.dataset_dir)
    output_dir = args.output_dir or str(arm_output_dir(ds.name) if ds.parent.name == "datasets_coco"
                                        else ds / "output" / "rfdetr_reducelr")
    class_names = load_class_names(ds)
    assert len(class_names) == 3, f"expected 3 classes, got {class_names}"
    print(f"class_names (ordered by category_id): {class_names}")
    print(f"[reduce_lr] {_SCHED} lr_floor={args.lr * args.lr_min_factor:g} es_patience={args.es_patience}")
    print(f"[output] {output_dir}")

    model_kwargs = {"num_classes": 3}
    if args.pretrain_weights:
        model_kwargs["pretrain_weights"] = args.pretrain_weights
    model = (RFDETRLarge if args.large else RFDETRBase)(**model_kwargs)
    model.train(
        dataset_dir=str(ds), output_dir=output_dir, epochs=args.epochs,
        lr=args.lr, lr_encoder=args.lr_encoder, warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, num_workers=args.num_workers,
        tensorboard=not args.no_tensorboard, class_names=class_names,
        early_stopping=True, early_stopping_patience=args.es_patience,
        early_stopping_min_delta=args.min_delta, early_stopping_use_ema=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
