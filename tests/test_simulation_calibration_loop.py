from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from simulation_calibration_loop.config import (
    DINOv2Config,
    RFDETREmbedderConfig,
    WorkflowConfig,
    load_workflow_config,
)
from simulation_calibration_loop.parameter_schema import (
    flatten_config,
    infer_parameter_schema,
    materialize_config,
)


class ParameterSchemaTest(unittest.TestCase):
    def test_flatten_and_materialize_round_trip(self) -> None:
        config_a = {
            "environment": {"name": "warehouse"},
            "camera": {
                "camera_height_mean": 1.0,
                "camera_height_std": 0.2,
                "dataset_noise": {"mode": "gaussian_jpeg", "sigma_mean": 6.0, "sigma_std": 1.5, "jpeg_quality_mean": 28, "jpeg_quality_std": 5, "shot_scale_mean": 0, "shot_scale_std": 0, "seed": 0},
            },
            "materials": {
                "textures": ["a.jpg", "b.jpg", "c.jpg"],
            },
            "lighting": {"visibility_choices": [True, False, False]},
        }
        config_b = {
            "environment": {"name": "warehouse_with_forklifts"},
            "camera": {
                "camera_height_mean": 2.0,
                "camera_height_std": 0.5,
                "dataset_noise": {"mode": "shot_jpeg", "sigma_mean": 0, "sigma_std": 0, "jpeg_quality_mean": 18, "jpeg_quality_std": 3, "shot_scale_mean": 22.0, "shot_scale_std": 3.0, "seed": 0},
            },
            "materials": {
                "textures": ["d.jpg", "e.jpg", "f.jpg"],
            },
            "lighting": {"visibility_choices": [True, True, False]},
        }

        specs = infer_parameter_schema([config_a, config_b])
        flattened = flatten_config(config_b, specs)
        rebuilt = materialize_config(config_a, flattened, specs)

        self.assertEqual(rebuilt, config_b)

    def test_schema_indexes_fixed_length_lists(self) -> None:
        config = {
            "palletjacks": {"position_std": [3.2, 4.8, 0.0]},
        }
        specs = infer_parameter_schema([config, config])
        flattened = flatten_config(config, specs)

        self.assertEqual(flattened["palletjacks.position_std[0]"], 3.2)
        self.assertEqual(flattened["palletjacks.position_std[1]"], 4.8)
        self.assertEqual(flattened["palletjacks.position_std[2]"], 0.0)


class EmbedderConfigTest(unittest.TestCase):
    def test_dinov2_config_defaults(self) -> None:
        cfg = DINOv2Config()
        self.assertEqual(cfg.model_name, "dinov2_vitb14_reg")
        self.assertEqual(cfg.repo, "facebookresearch/dinov2")
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.image_size, 224)
        self.assertEqual(cfg.resize_size, 256)

    def test_rfdetr_embedder_config_defaults(self) -> None:
        cfg = RFDETREmbedderConfig()
        self.assertEqual(cfg.checkpoint_path, "")
        self.assertEqual(cfg.num_classes, 3)
        self.assertEqual(cfg.layer_index, 3)
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.image_size, 224)
        self.assertEqual(cfg.resize_size, 256)

    def test_workflow_config_embedder_backend_defaults_to_dinov2(self) -> None:
        # WorkflowConfig without embedder_backend should default to "dinov2"
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "seeds"
            seed_dir.mkdir()
            seed_yaml = {"environment": {"name": "warehouse"}}
            (seed_dir / "seed.yaml").write_text(yaml.safe_dump(seed_yaml))

            config_dict = {
                "project_name": "test_project",
                "workspace_dir": tmp,
                "seed_config_dir": str(seed_dir),
                "real_dataset_root": tmp,
                "real_annotations_file": str(Path(tmp) / "ann.json"),
                "max_iterations": 1,
                "iteration_batch_size": 2,
                "search_space": {"include": ["environment.name"]},
                "isaac": {"script_path": "/tmp/script.py"},
            }
            config_path = Path(tmp) / "project_config.yaml"
            config_path.write_text(yaml.safe_dump(config_dict))

            cfg = load_workflow_config(config_path)
            self.assertEqual(cfg.embedder_backend, "dinov2")
            self.assertIsInstance(cfg.rfdetr_embedder, RFDETREmbedderConfig)
            self.assertEqual(cfg.rfdetr_embedder.checkpoint_path, "")
            self.assertEqual(cfg.rfdetr_embedder.layer_index, 3)

    def test_load_workflow_config_with_rfdetr_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "seeds"
            seed_dir.mkdir()
            (seed_dir / "seed.yaml").write_text(yaml.safe_dump({"environment": {"name": "w"}}))
            ckpt_path = Path(tmp) / "checkpoint.pth"
            ckpt_path.touch()

            config_dict = {
                "project_name": "test_rfdetr",
                "workspace_dir": tmp,
                "seed_config_dir": str(seed_dir),
                "real_dataset_root": tmp,
                "real_annotations_file": str(Path(tmp) / "ann.json"),
                "max_iterations": 1,
                "iteration_batch_size": 2,
                "search_space": {"include": ["environment.name"]},
                "isaac": {"script_path": "/tmp/script.py"},
                "embedder_backend": "rfdetr",
                "rfdetr_embedder": {
                    "checkpoint_path": str(ckpt_path),
                    "num_classes": 4,
                    "layer_index": 2,
                    "batch_size": 8,
                },
            }
            config_path = Path(tmp) / "project_config.yaml"
            config_path.write_text(yaml.safe_dump(config_dict))

            cfg = load_workflow_config(config_path)
            self.assertEqual(cfg.embedder_backend, "rfdetr")
            self.assertEqual(cfg.rfdetr_embedder.num_classes, 4)
            self.assertEqual(cfg.rfdetr_embedder.layer_index, 2)
            self.assertEqual(cfg.rfdetr_embedder.batch_size, 8)
            self.assertTrue(Path(cfg.rfdetr_embedder.checkpoint_path).is_absolute())


if __name__ == "__main__":
    unittest.main()
