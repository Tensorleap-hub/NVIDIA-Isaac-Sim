"""Eval-only: score an rf-detr checkpoint on a dataset's valid/ split via trainer.validate,
reusing rf-detr's own COCOEvalCallback so numbers match training-time logs.

With checkpoint_best_ema.pth loaded, the live weights ARE the EMA weights, so val/mAP_50_95
is the deployable-checkpoint number. Optionally writes all metrics to --json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rfdetr.training as _training  # noqa: E402
from rfdetr import RFDETRBase, RFDETRLarge  # noqa: E402
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer  # noqa: E402
from common import load_class_names  # noqa: E402

_orig_build = _training.build_trainer


def _no_sanity(train_config, model_config, **kw):
    kw.setdefault("num_sanity_val_steps", 0)
    return _orig_build(train_config, model_config, **kw)


_training.build_trainer = _no_sanity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True, help="dataset whose valid/ split to eval on")
    p.add_argument("--pretrain-weights", required=True, help="checkpoint (.pth) to score")
    p.add_argument("--output-dir", default="/tmp/claude-1000/eval_ckpt_out")
    p.add_argument("--json", default=None, help="write metrics dict here")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--large", action="store_true")
    args = p.parse_args()

    class_names = load_class_names(args.dataset_dir)
    model = (RFDETRLarge if args.large else RFDETRBase)(num_classes=len(class_names),
                                                        pretrain_weights=args.pretrain_weights)
    tc = model.get_train_config(dataset_dir=args.dataset_dir, output_dir=args.output_dir, epochs=1,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                class_names=class_names, tensorboard=False)
    module = RFDETRModelModule(model.model_config, tc)
    dm = RFDETRDataModule(model.model_config, tc)
    trainer = build_trainer(tc, model.model_config)
    results = trainer.validate(module, dm)

    metrics = {k: float(v) for d in results for k, v in d.items()}
    print("\n===== VALIDATE RESULTS =====")
    print(f"checkpoint: {args.pretrain_weights}\nvalid: {args.dataset_dir}/valid")
    for k in sorted(metrics):
        if "mAP" in k or "AP" in k or "mAR" in k:
            print(f"  {k} = {metrics[k]:.4f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"checkpoint": args.pretrain_weights, "dataset": args.dataset_dir,
                       "class_names": class_names, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
