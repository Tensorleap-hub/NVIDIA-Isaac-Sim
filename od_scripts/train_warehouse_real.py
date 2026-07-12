import argparse
import json
import os

from rfdetr import RFDETRBase, RFDETRLarge


def load_class_names(dataset_dir: str) -> list[str]:
    """Return class names ordered to match RF-DETR's label indices.

    RF-DETR's roboflow COCO loader builds ``cat2label`` as
    ``{cat_id: i for i, cat_id in enumerate(sorted(cat_ids))}`` — i.e. label
    index = position of the category_id in ascending order. ``class_names`` must
    therefore be ordered by ascending category_id, or every per-class metric and
    the deployed model's label map will be silently permuted. Deriving the names
    from the dataset itself makes that impossible to get wrong.
    """
    ann = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    with open(ann) as f:
        cats = json.load(f)["categories"]
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on 3-class warehouse dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to prepared COCO dataset (train/ and valid/ subdirs)")
    parser.add_argument("--output-dir", default=None, help="Where to save checkpoints (default: <dataset-dir>/output/rfdetr_base)")
    parser.add_argument("--pretrain-weights", default=None, help="Path to checkpoint to fine-tune from (default: RF-DETR COCO pretrained)")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze backbone, train decoder/head only")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-drop", type=int, default=30, help="Epoch at which LR drops 10x (step schedule)")
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--large", action="store_true", help="Use RFDETRLarge instead of RFDETRBase")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--resume", default=None, help="Resume from a lightning checkpoint")
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.dataset_dir}/output/rfdetr_{'large' if args.large else 'base'}"

    model_cls = RFDETRLarge if args.large else RFDETRBase
    model_kwargs = {"num_classes": 3, "freeze_encoder": args.freeze_encoder}
    if args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = args.pretrain_weights
    class_names = load_class_names(args.dataset_dir)
    print(f"class_names (ordered by category_id): {class_names}")
    assert len(class_names) == 3, f"expected 3 classes, got {class_names}"

    model = model_cls(**model_kwargs)
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_drop=args.lr_drop,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        tensorboard=not args.no_tensorboard,
        class_names=class_names,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
