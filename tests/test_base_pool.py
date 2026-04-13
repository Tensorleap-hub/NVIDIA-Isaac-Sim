from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from simulation_calibration_loop.base_pool import BasePoolManager, PoolEntry
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
            )
            payload = yaml.safe_load(derived_config.read_text())

            self.assertEqual(
                payload["base_pool"]["state_path"],
                "promoted_baseline_theme_rounds/base_pool.json",
            )


if __name__ == "__main__":
    unittest.main()
