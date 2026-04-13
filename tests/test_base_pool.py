from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from simulation_calibration_loop.base_pool import BasePoolManager, PoolEntry
from simulation_calibration_loop.controller import SimulationCalibrationController
from simulation_calibration_loop.parameter_schema import infer_parameter_schema
from simulation_calibration_loop.run_theme_rounds import _write_round_config


def _pool_entry(
    *,
    entry_id: str,
    score: float | None,
    created_at: str,
    centroid: list[float],
) -> PoolEntry:
    return PoolEntry(
        entry_id=entry_id,
        config={"run": {"data_dir": f"/tmp/{entry_id}"}},
        flattened_params={"camera.camera_height_mean": float(len(centroid))},
        score=score,
        created_at=created_at,
        iteration_index=0,
        theme_label="camera",
        stage_lineage="theme=camera theme_step=1/1 theme_round=1/1",
        artifact_path=f"/tmp/{entry_id}",
        yaml_path=f"/tmp/{entry_id}.yaml",
        embedding_path=f"/tmp/{entry_id}.npy",
        diversity_metadata={
            "backend": "embedding_centroid",
            "embedding_centroid": centroid,
        },
    )


class _FakePool:
    def __init__(self, entries: list[PoolEntry]):
        self.enabled = True
        self._entries = entries

    def has_scored_entries(self) -> bool:
        return True

    def sample_entries(self, count: int) -> list[PoolEntry]:
        return self._entries[:count]


class _FakeUI:
    def __init__(self):
        self.logs: list[str] = []

    def append_log(self, line: str) -> None:
        self.logs.append(line)


class BasePoolManagerTest(unittest.TestCase):
    def test_prune_drops_weaker_near_duplicate_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool_path = Path(temp_dir) / "base_pool.json"
            manager = BasePoolManager(
                state_path=pool_path,
                enabled=True,
                max_size=3,
                elite_size=1,
                recent_size=1,
                score_weight=0.6,
                diversity_weight=0.3,
                recency_weight=0.1,
                near_duplicate_threshold=0.02,
                random_seed=7,
            )
            manager.admit_entries(
                [
                    _pool_entry(
                        entry_id="best_duplicate",
                        score=0.10,
                        created_at="2026-04-10T00:00:00+00:00",
                        centroid=[1.0, 0.0],
                    ),
                    _pool_entry(
                        entry_id="weak_duplicate",
                        score=0.90,
                        created_at="2026-04-11T00:00:00+00:00",
                        centroid=[0.99995, 0.01],
                    ),
                    _pool_entry(
                        entry_id="diverse_a",
                        score=0.20,
                        created_at="2026-04-09T00:00:00+00:00",
                        centroid=[0.0, 1.0],
                    ),
                    _pool_entry(
                        entry_id="diverse_b",
                        score=0.30,
                        created_at="2026-04-12T00:00:00+00:00",
                        centroid=[-1.0, 0.0],
                    ),
                ]
            )

            kept_ids = {entry.entry_id for entry in manager.entries}
            self.assertEqual(len(kept_ids), 3)
            self.assertIn("best_duplicate", kept_ids)
            self.assertNotIn("weak_duplicate", kept_ids)
            self.assertIn("diverse_a", kept_ids)
            self.assertIn("diverse_b", kept_ids)

            reloaded = BasePoolManager(
                state_path=pool_path,
                enabled=True,
                max_size=3,
                elite_size=1,
                recent_size=1,
                score_weight=0.6,
                diversity_weight=0.3,
                recency_weight=0.1,
                near_duplicate_threshold=0.02,
                random_seed=7,
            )
            self.assertEqual({entry.entry_id for entry in reloaded.entries}, kept_ids)

    def test_sampling_uses_more_than_one_pool_member(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BasePoolManager(
                state_path=Path(temp_dir) / "base_pool.json",
                enabled=True,
                max_size=5,
                elite_size=2,
                recent_size=1,
                score_weight=0.6,
                diversity_weight=0.3,
                recency_weight=0.1,
                near_duplicate_threshold=0.01,
                random_seed=11,
            )
            manager.admit_entries(
                [
                    _pool_entry(
                        entry_id="elite",
                        score=0.05,
                        created_at="2026-04-08T00:00:00+00:00",
                        centroid=[1.0, 0.0],
                    ),
                    _pool_entry(
                        entry_id="diverse",
                        score=0.20,
                        created_at="2026-04-09T00:00:00+00:00",
                        centroid=[0.0, 1.0],
                    ),
                    _pool_entry(
                        entry_id="recent",
                        score=0.40,
                        created_at="2026-04-12T00:00:00+00:00",
                        centroid=[-1.0, 0.0],
                    ),
                ]
            )

            sampled = manager.sample_entries(3)
            sampled_ids = {entry.entry_id for entry in sampled}
            self.assertEqual(len(sampled), 3)
            self.assertEqual(len(sampled_ids), 3)


class ThemeRoundConfigTest(unittest.TestCase):
    def test_round_config_injects_shared_pool_state_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_config = temp_root / "project_config_camera.yaml"
            destination_dir = temp_root / "generated"
            destination_dir.mkdir(parents=True, exist_ok=True)
            source_config.write_text(
                yaml.safe_dump(
                    {
                        "project_name": "demo",
                        "workspace_dir": "./workspace_camera",
                        "promoted_baseline_dir": "./promoted_baseline_theme_rounds",
                        "seed_config_dir": "../palletjack_sdg/experiments/ec2-loop/base_v2",
                        "real_dataset_root": "../loco_dataset",
                        "real_annotations_file": "../loco_dataset/labels/loco-sub3-v1-train.json",
                        "max_iterations": 1,
                        "iteration_batch_size": 1,
                        "base_pool": {"enabled": True},
                    },
                    sort_keys=False,
                )
            )

            derived_config = _write_round_config(
                source_config_path=source_config,
                destination_dir=destination_dir,
                round_index=0,
                workspace_root=None,
                synthetic_rgb_base_dir=None,
            )
            payload = yaml.safe_load(derived_config.read_text())

            self.assertEqual(
                payload["base_pool"]["state_path"],
                str((temp_root / "promoted_baseline_theme_rounds" / "base_pool.json").resolve()),
            )


class ControllerPoolBootstrapTest(unittest.TestCase):
    def test_iteration_zero_uses_pool_params_instead_of_seed_params(self) -> None:
        entry_a_config = {
            "camera": {"camera_height_mean": 1.5},
            "run": {"data_dir": "/tmp/a"},
        }
        entry_b_config = {
            "camera": {"camera_height_mean": 2.5},
            "run": {"data_dir": "/tmp/b"},
        }
        controller = SimulationCalibrationController.__new__(SimulationCalibrationController)
        controller.schema = infer_parameter_schema([entry_a_config, entry_b_config])
        controller.seed_rows = [
            {
                "suggestion_id": "seed_0",
                "optuna_trial_number": None,
                "params": {"camera.camera_height_mean": 9.0, "run.data_dir": "/seed/0"},
            },
            {
                "suggestion_id": "seed_1",
                "optuna_trial_number": None,
                "params": {"camera.camera_height_mean": 8.0, "run.data_dir": "/seed/1"},
            },
        ]
        controller.seed_base_records = []
        controller.base_template = entry_a_config
        controller.base_pool = _FakePool(
            [
                _pool_entry(
                    entry_id="entry_a",
                    score=0.1,
                    created_at="2026-04-10T00:00:00+00:00",
                    centroid=[1.0, 0.0],
                ),
                _pool_entry(
                    entry_id="entry_b",
                    score=0.2,
                    created_at="2026-04-11T00:00:00+00:00",
                    centroid=[0.0, 1.0],
                ),
            ]
        )
        controller.base_pool._entries[0].config = entry_a_config
        controller.base_pool._entries[1].config = entry_b_config

        rows = controller._load_iteration_rows({"iterations": []}, start_iteration=0)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["params"]["camera.camera_height_mean"], 1.5)
        self.assertEqual(rows[1]["params"]["camera.camera_height_mean"], 2.5)
        self.assertEqual(rows[0]["base_config"], entry_a_config)
        self.assertEqual(rows[1]["base_config"], entry_b_config)
        self.assertEqual(rows[0]["base_pool_entry_id"], "entry_a")
        self.assertEqual(rows[1]["base_pool_entry_id"], "entry_b")
        self.assertTrue(rows[0]["direct_pool_replay"])
        self.assertTrue(rows[1]["direct_pool_replay"])
        self.assertNotEqual(rows[0]["params"]["camera.camera_height_mean"], 9.0)
        self.assertNotEqual(rows[1]["params"]["camera.camera_height_mean"], 8.0)

    def test_direct_pool_replay_copies_existing_artifacts(self) -> None:
        controller = SimulationCalibrationController.__new__(SimulationCalibrationController)
        controller.ui = _FakeUI()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_output_dir = temp_root / "source_output"
            source_rgb_dir = source_output_dir / "Camera" / "rgb"
            source_rgb_dir.mkdir(parents=True, exist_ok=True)
            source_image_path = source_rgb_dir / "frame_000.png"
            source_image_path.write_bytes(b"rgb")

            source_embedding_path = temp_root / "source_embedding.npy"
            np.save(source_embedding_path, np.asarray([[1.0, 2.0]], dtype=float))
            source_manifest_path = source_embedding_path.with_suffix(".manifest.json")
            source_manifest_path.write_text("{}")

            output_dir = temp_root / "target_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            embedding_path = temp_root / "target_embedding.npy"
            row_record = {
                "direct_pool_replay": True,
                "pool_artifact_path": str(source_output_dir),
                "pool_embedding_path": str(source_embedding_path),
                "base_pool_entry_id": "entry_a",
            }

            image_paths, reused = controller._copy_synthetic_artifacts_from_pool_replay(
                output_dir=output_dir,
                embedding_path=embedding_path,
                row_record=row_record,
            )

            self.assertTrue(reused)
            self.assertEqual(len(image_paths), 1)
            self.assertTrue((output_dir / "Camera" / "rgb" / "frame_000.png").exists())
            self.assertTrue(embedding_path.exists())
            self.assertTrue(embedding_path.with_suffix(".manifest.json").exists())
            self.assertIn("[reuse] copied direct pool replay artifacts", controller.ui.logs[0])


if __name__ == "__main__":
    unittest.main()
