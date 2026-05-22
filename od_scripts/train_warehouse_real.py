import argparse
from rfdetr import RFDETRBase, RFDETRLarge


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR on 3-class warehouse dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to prepared COCO dataset (train/ and valid/ subdirs)")
    parser.add_argument("--output-dir", default=None, help="Where to save checkpoints (default: <dataset-dir>/output/rfdetr_base)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--large", action="store_true", help="Use RFDETRLarge instead of RFDETRBase")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.dataset_dir}/output/rfdetr_{'large' if args.large else 'base'}"

    model_cls = RFDETRLarge if args.large else RFDETRBase
    model = model_cls(num_classes=3)
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        tensorboard=not args.no_tensorboard,
        class_names=["pallet_truck", "forklift", "pallet"],
    )


if __name__ == "__main__":
    main()
