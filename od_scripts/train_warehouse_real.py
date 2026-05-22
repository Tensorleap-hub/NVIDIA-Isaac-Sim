from rfdetr import RFDETRBase

if __name__ == "__main__":
    model = RFDETRBase(num_classes=3)
    model.train(
        dataset_dir="/Users/orram/Tensorleap/data/warehouse/warehouse3cls_real",
        output_dir="/Users/orram/Tensorleap/data/warehouse/warehouse3cls_real/output/rfdetr_base",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        num_workers=4,
        tensorboard=True,
        class_names=["pallet_truck", "forklift", "pallet"],
    )
