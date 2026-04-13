from __future__ import annotations

import argparse

from simulation_calibration_loop import SimulationCalibrationController, load_workflow_config


def main() -> None:
    parser = argparse.ArgumentParser("Run the DINOv2 + Optuna simulation calibration workflow")
    parser.add_argument(
        "--config",
        type=str,
        default="simulation_calibration_loop/project_config.yaml",
        help="Path to the workflow config YAML",
    )
    args = parser.parse_args()

    config = load_workflow_config(args.config)
    controller = SimulationCalibrationController(config)
    controller.run()


if __name__ == "__main__":
    main()
