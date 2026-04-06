from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibration_optuna import DEFAULT_CONFIG
from calibration_optuna.data_utils import infer_bounds_and_types_from_metadata
from calibration_optuna.experiment_runner import ExperimentRunner

from .config import WorkflowConfig
from .data import (
    DINOv2Embedder,
    RunArtifact,
    StateStore,
    discover_generated_images,
    make_cache_key,
    prepare_output_dir,
    run_isaac_generation,
    select_real_image_paths,
)
from .parameter_schema import (
    filter_parameter_specs,
    flatten_config,
    infer_parameter_schema,
    load_yaml_configs,
    materialize_config,
    save_yaml_config,
)
from .ui import WorkflowUI


class SimulationCalibrationController:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.workspace_dir = Path(config.workspace_dir)
        self.seed_config_dir = Path(config.seed_config_dir)
        self.state_store = StateStore(self.workspace_dir / "state.json")
        self.ui = WorkflowUI()

        seed_items = load_yaml_configs(self.seed_config_dir)
        if not seed_items:
            raise ValueError(f"No YAML files found in {self.seed_config_dir}")

        self.seed_configs = seed_items
        self.base_template = deepcopy(seed_items[0][1])
        inferred_schema = infer_parameter_schema([item[1] for item in seed_items])
        self.schema = filter_parameter_specs(
            inferred_schema,
            include=config.search_space.include,
            exclude=config.search_space.exclude,
        )
        if not self.schema:
            raise ValueError("Search-space filtering removed all Isaac parameters; update search_space.include/exclude")
        self.seed_rows = [flatten_config(item[1], self.schema) for item in seed_items]
        self.group_name = "simulation_1"
        self.seed_metadata = self._build_distribution_metadata(self.seed_rows)
        self.param_bounds, self.param_type = infer_bounds_and_types_from_metadata(
            self.seed_metadata,
            [self.group_name],
        )

        self.optimizer_config = deepcopy(DEFAULT_CONFIG)
        self.optimizer_config["experiment_name"] = config.project_name
        self.optimizer_config["experiments_base_dir"] = str(self.workspace_dir / "optuna")
        self.optimizer_config["iteration_batch_size"] = config.iteration_batch_size
        self.optimizer_config["top_n_best_trials"] = config.top_n_best_trials
        self.optimizer_config["mmd_max_samples"] = config.mmd_max_samples
        self.optimizer_config["random_seed"] = config.random_seed

        self.runner = ExperimentRunner(
            config=self.optimizer_config,
            param_bounds=self.param_bounds,
            param_type=self.param_type,
        )
        self.embedder = DINOv2Embedder(
            repo=config.dino.repo,
            model_name=config.dino.model_name,
            device=config.dino.device,
            image_size=config.dino.image_size,
            resize_size=config.dino.resize_size,
        )

    def run(self) -> None:
        self.ui.start()
        self.ui.set_status(max_iterations=self.config.max_iterations)
        real_embeddings = self._prepare_real_embeddings()
        self.runner.set_real_embeddings(real_embeddings)

        state = self.state_store.load()
        start_iteration = len(state["iterations"])
        self._replay_completed_iterations(state)
        current_rows = self._load_iteration_rows(state, start_iteration)

        for iteration_index in range(start_iteration, self.config.max_iterations):
            self.ui.set_status(
                phase="generate",
                iteration_index=iteration_index + 1,
                total_runs=len(current_rows),
                completed_runs=0,
                note=f"materializing {len(current_rows)} YAMLs",
            )
            artifacts = self._materialize_and_execute_iteration(iteration_index, current_rows)
            self.ui.set_status(phase="optimize", note="computing embeddings and Optuna suggestions")
            suggestions = self._run_optimizer_iteration(artifacts)
            best_trials = self.runner.get_best_trials(top_n=self.config.top_n_best_trials)
            next_rows = [suggestion["params"] for suggestion in suggestions]

            state["iterations"].append(
                {
                    "iteration_index": iteration_index,
                    "input_rows": current_rows,
                    "artifacts": [self._serialize_artifact(item) for item in artifacts],
                    "suggestions": suggestions,
                    "best_trials": [
                        {"trial_id": trial_id, "params": params}
                        for trial_id, params in best_trials
                    ],
                }
            )
            self.state_store.save(state)
            current_rows = next_rows

        self.ui.set_status(phase="complete", note="workflow finished")
        self.ui.stop()

    def _prepare_real_embeddings(self) -> np.ndarray:
        self.ui.set_status(phase="real-cache", note="loading subset-3 reference embeddings")
        real_image_paths = select_real_image_paths(
            self.config.real_dataset_root,
            self.config.real_annotations_file,
        )
        if not real_image_paths:
            raise ValueError("No real subset images were resolved from the dataset root and annotation file")
        cache_dir = self.workspace_dir / "cache" / "real"
        cache_key = make_cache_key(
            [
                self.config.dino.model_name,
                *(str(path) for path in real_image_paths),
            ]
        )
        cache_path = cache_dir / f"{cache_key}.npy"
        manifest = {
            "model_name": self.config.dino.model_name,
            "repo": self.config.dino.repo,
            "image_paths": [str(path) for path in real_image_paths],
        }
        status = "hit" if cache_path.exists() else "miss"
        self.ui.set_status(real_cache_status=status)
        return self.embedder.embed_paths(
            real_image_paths,
            batch_size=self.config.dino.batch_size,
            cache_path=cache_path,
            manifest=manifest,
        )

    def _load_iteration_rows(self, state: dict[str, Any], start_iteration: int) -> list[dict[str, Any]]:
        if start_iteration == 0:
            return self.seed_rows
        return [item["params"] for item in state["iterations"][-1]["suggestions"]]

    def _replay_completed_iterations(self, state: dict[str, Any]) -> None:
        if not state["iterations"]:
            return

        self.ui.set_status(phase="resume", note=f"replaying {len(state['iterations'])} completed iterations")
        for iteration in state["iterations"]:
            artifacts = [
                RunArtifact(
                    run_id=item["run_id"],
                    yaml_path=Path(item["yaml_path"]),
                    output_dir=Path(item["output_dir"]),
                    log_path=Path(item["log_path"]),
                    embedding_path=Path(item["embedding_path"]),
                    image_count=int(item["image_count"]),
                    flattened_params=item["flattened_params"],
                )
                for item in iteration["artifacts"]
            ]
            self._run_optimizer_iteration(artifacts)

    def _materialize_and_execute_iteration(
        self,
        iteration_index: int,
        rows: list[dict[str, Any]],
    ) -> list[RunArtifact]:
        iteration_dir = self.workspace_dir / f"iteration_{iteration_index:03d}"
        yaml_dir = iteration_dir / "yamls"
        outputs_dir = iteration_dir / "outputs"
        cache_dir = iteration_dir / "cache"
        prepare_output_dir(yaml_dir, clean=False)
        prepare_output_dir(outputs_dir, clean=False)
        prepare_output_dir(cache_dir, clean=False)

        artifacts: list[RunArtifact] = []
        for run_index, row in enumerate(rows):
            run_id = f"iter{iteration_index:03d}_run{run_index:03d}"
            yaml_path = yaml_dir / f"{run_id}.yaml"
            output_dir = outputs_dir / run_id
            log_path = output_dir / "isaac.log"
            embedding_path = cache_dir / f"{run_id}_{self.config.dino.model_name}.npy"
            config_dict = materialize_config(self.base_template, row, self.schema)
            save_yaml_config(yaml_path, config_dict)

            self.ui.set_status(current_run=run_id, completed_runs=run_index, total_runs=len(rows))
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            if not log_path.exists():
                run_isaac_generation(
                    isaac_sim_path=Path(self.config.isaac.isaac_sim_path),
                    script_path=Path(self.config.isaac.script_path),
                    yaml_path=yaml_path,
                    output_dir=output_dir,
                    log_path=log_path,
                    headless=self.config.isaac.headless,
                    num_frames_override=self.config.isaac.num_frames_override,
                    log_callback=self.ui.append_log,
                )

            image_paths = discover_generated_images(output_dir)
            if not image_paths:
                raise ValueError(f"No generated images discovered under {output_dir}")
            manifest = {
                "model_name": self.config.dino.model_name,
                "repo": self.config.dino.repo,
                "image_paths": [str(path) for path in image_paths],
                "yaml_path": str(yaml_path),
            }
            self.embedder.embed_paths(
                image_paths,
                batch_size=self.config.dino.batch_size,
                cache_path=embedding_path,
                manifest=manifest,
            )
            artifacts.append(
                RunArtifact(
                    run_id=run_id,
                    yaml_path=yaml_path,
                    output_dir=output_dir,
                    log_path=log_path,
                    embedding_path=embedding_path,
                    image_count=len(image_paths),
                    flattened_params=row,
                )
            )
            self.ui.set_status(completed_runs=run_index + 1)
        return artifacts

    def _run_optimizer_iteration(self, artifacts: list[RunArtifact]) -> list[dict[str, Any]]:
        embeddings = []
        current_distributions = []
        embeddings_indices_by_dist = {}
        start_index = 0

        for dist_index, artifact in enumerate(artifacts):
            embedding_array = np.load(artifact.embedding_path)
            embeddings.append(embedding_array)
            end_index = start_index + len(embedding_array)
            params = {
                f"shape_logit_{self.group_name}": 0.0,
            }
            for key, value in artifact.flattened_params.items():
                params[f"{self.group_name}__{key}"] = value
            current_distributions.append((artifact.run_id, params))
            embeddings_indices_by_dist[dist_index] = [(0, np.arange(start_index, end_index))]
            start_index = end_index

        embeddings_by_shape = [np.concatenate(embeddings, axis=0)]
        raw_suggestions = self.runner.run_iteration(
            current_distributions=current_distributions,
            embeddings_by_shape=embeddings_by_shape,
            embeddings_indices_by_dist=embeddings_indices_by_dist,
        )

        suggestions = []
        for suggestion_id, params in raw_suggestions:
            flattened = {}
            for key, value in params.items():
                if key.startswith(f"{self.group_name}__"):
                    flattened[key[len(f"{self.group_name}__"):]] = value
            suggestions.append({"suggestion_id": suggestion_id, "params": flattened})
        return suggestions

    def _build_distribution_metadata(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        metadata_rows = []
        for distribution_id, row in enumerate(rows):
            metadata_row = {"distribution_id": distribution_id, f"shape_logit_{self.group_name}": 0.0}
            for key, value in row.items():
                metadata_row[f"{self.group_name}__{key}"] = value
            metadata_rows.append(metadata_row)
        return pd.DataFrame(metadata_rows)

    def _serialize_artifact(self, artifact: RunArtifact) -> dict[str, Any]:
        return {
            "run_id": artifact.run_id,
            "yaml_path": str(artifact.yaml_path),
            "output_dir": str(artifact.output_dir),
            "log_path": str(artifact.log_path),
            "embedding_path": str(artifact.embedding_path),
            "image_count": artifact.image_count,
            "flattened_params": artifact.flattened_params,
        }
