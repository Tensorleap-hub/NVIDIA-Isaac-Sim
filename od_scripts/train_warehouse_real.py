import argparse
from rfdetr import RFDETRBase, RFDETRLarge


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
        class_names=["pallet_truck", "forklift", "pallet"],
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
