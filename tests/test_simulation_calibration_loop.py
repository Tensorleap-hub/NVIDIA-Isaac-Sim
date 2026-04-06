from __future__ import annotations

import unittest

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
                "dataset_noise": {"enabled": True, "mode": "gaussian_jpeg"},
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
                "dataset_noise": {"enabled": True, "mode": "shot_jpeg"},
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


if __name__ == "__main__":
    unittest.main()
