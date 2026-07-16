"""Main orchestration for the Isaac -> DINOv2 -> Optuna calibration loop."""

from __future__ import annotations

from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import optuna
import numpy as np
import yaml

from calibration_optuna import DEFAULT_CONFIG
from calibration_optuna.experiment_runner import ExperimentRunner

from . import diversity
from .base_pool import BasePoolManager, PoolEntry
from .config import WorkflowConfig
from .data import (
    DINOv2Embedder,
    RFDETREmbedder,
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
    validate_configs_against_schema,
)
from .ui import WorkflowUI


class SimulationCalibrationController:
    """Own the iterative calibration workflow and its durable state."""

    def __init__(self, config: WorkflowConfig):
        """Build the controller, optimizer search space, and runtime helpers."""
        self.config = config
        self.workspace_dir = Path(config.workspace_dir)
        self.s3_best_runs_prefix = config.s3_best_runs_prefix.rstrip("/") if config.s3_best_runs_prefix else None
        self.promoted_baseline_dir = Path(config.promoted_baseline_dir) if config.promoted_baseline_dir else None
        self.baseline_state_path = Path(config.baseline_state_path) if config.baseline_state_path else None
        self.synthetic_rgb_base_dir = Path(config.synthetic_rgb_base_dir) if config.synthetic_rgb_base_dir else None
        self.seed_config_dir = Path(config.seed_config_dir)
        self.base_pool_state_path = (
            Path(config.base_pool.state_path)
            if config.base_pool.state_path is not None
            else self.workspace_dir / "base_pool.json"
        )
        self.state_store = StateStore(self.workspace_dir / "state.json")
        self.ui = WorkflowUI(log_path=self.workspace_dir / "main_loop_screen.log")
        self.meta_label = os.environ.get("SIM_CAL_LOOP_META_LABEL", "").strip()

        seed_items = load_yaml_configs(self.seed_config_dir)
        if not seed_items:
            raise ValueError(f"No YAML files found in {self.seed_config_dir}")

        self.seed_configs = seed_items
        inferred_schema = infer_parameter_schema([item[1] for item in seed_items])
        self.full_schema = inferred_schema
        self.schema = filter_parameter_specs(
            inferred_schema,
            include=config.search_space.include,
            exclude=config.search_space.exclude,
        )
        if not self.schema:
            raise ValueError("Search-space filtering removed all Isaac parameters; update search_space.include/exclude")
        validate_configs_against_schema([item[1] for item in seed_items], self.schema)
        self.base_template = self._load_base_template(seed_items)
        # Seed rows are external observations, not Optuna-issued trials. They are
        # imported into the study with `add_trial(...)` on the first iteration.
        self.seed_rows = [
            {
                "suggestion_id": f"seed_{index}",
                "optuna_trial_number": None,
                "params": flatten_config(item[1], self.schema),
            }
            for index, item in enumerate(seed_items)
        ]
        self.seed_base_records = [
            {
                "pool_entry_id": self._make_seed_pool_entry_id(path),
                "config": deepcopy(seed_config),
                "yaml_path": str(path),
            }
            for path, seed_config in seed_items
        ]
        # calibration_optuna expects grouped parameter names even though this
        # workflow currently optimizes a single synthetic family.
        self.group_name = "simulation_1"
        self.param_bounds, self.param_type = self._build_param_bounds_from_config()

        self.optimizer_config = deepcopy(DEFAULT_CONFIG)
        self.optimizer_config["experiment_name"] = config.project_name
        self.optimizer_config["experiments_base_dir"] = str(self.workspace_dir / "optuna")
        self.optimizer_config["iteration_batch_size"] = config.iteration_batch_size
        self.optimizer_config["top_n_best_trials"] = config.top_n_best_trials
        self.optimizer_config["mmd_max_samples"] = config.mmd_max_samples
        self.optimizer_config["random_seed"] = config.random_seed
        self.optimizer_config["optimizer"]["multivariate"] = config.tpe_sampler.multivariate
        self.optimizer_config["optimizer"]["group"] = config.tpe_sampler.group
        self.optimizer_config["optimizer"]["constant_liar"] = config.tpe_sampler.constant_liar
        if config.tpe_sampler.n_startup_trials is not None:
            self.optimizer_config["optimizer"]["n_startup_trials"] = int(
                config.tpe_sampler.n_startup_trials
            )

        self.runner = ExperimentRunner(
            config=self.optimizer_config,
            param_bounds=self.param_bounds,
            param_type=self.param_type,
        )
        if config.embedder_backend == "rfdetr":
            ckpt_stem = (
                Path(config.rfdetr_embedder.checkpoint_path).stem
                if config.rfdetr_embedder.checkpoint_path
                else "pretrained"
            )
            self._embedder_id = f"rfdetr_{ckpt_stem}"
            self._embedder_repo = ""
            self._embedder_batch_size = config.rfdetr_embedder.batch_size
            self.embedder: DINOv2Embedder | RFDETREmbedder = RFDETREmbedder(
                checkpoint_path=config.rfdetr_embedder.checkpoint_path,
                num_classes=config.rfdetr_embedder.num_classes,
                layer_index=config.rfdetr_embedder.layer_index,
                device=config.rfdetr_embedder.device,
                image_size=config.rfdetr_embedder.image_size,
                resize_size=config.rfdetr_embedder.resize_size,
            )
        else:
            self._embedder_id = config.dino.model_name
            self._embedder_repo = config.dino.repo
            self._embedder_batch_size = config.dino.batch_size
            self.embedder = DINOv2Embedder(
                repo=config.dino.repo,
                model_name=config.dino.model_name,
                device=config.dino.device,
                image_size=config.dino.image_size,
                resize_size=config.dino.resize_size,
            )
        self.base_pool = BasePoolManager(
            state_path=self.base_pool_state_path,
            enabled=config.base_pool.enabled,
            max_size=config.base_pool.max_size,
            elite_size=config.base_pool.elite_size,
            recent_size=config.base_pool.recent_size,
            score_weight=config.base_pool.score_weight,
            diversity_weight=config.base_pool.diversity_weight,
            recency_weight=config.base_pool.recency_weight,
            near_duplicate_threshold=config.base_pool.near_duplicate_threshold,
            random_seed=config.random_seed,
            pin_seeds=config.base_pool.pin_seeds,
        )
        self._bootstrap_base_pool(seed_items)

    def run(self) -> None:
        """Run the full calibration loop from real-cache setup through completion."""
        self.ui.start()
        self.ui.set_status(max_iterations=self.config.max_iterations, note=self._compose_note(""))
        real_embeddings = self._prepare_real_embeddings()
        self.runner.set_real_embeddings(real_embeddings)

        state = self.state_store.load()
        start_iteration = len(state["iterations"])
        self._replay_completed_iterations(state)
        self._sync_base_pool_from_state(state)
        self._promote_global_baseline(state)
        self._export_top_trials(state)
        self._export_best_runs_to_s3(state)
        initial_distance = "-"
        if state["iterations"]:
            initial_distance = state["iterations"][0]["iteration_summary"]["iteration_best"]
        self.ui.set_status(initial_distance=initial_distance)
        current_rows = self._load_iteration_rows(state, start_iteration)

        for iteration_index in range(start_iteration, self.config.max_iterations):
            self.ui.set_status(
                phase="generate",
                iteration_index=iteration_index + 1,
                total_runs=len(current_rows),
                completed_runs=0,
                note=self._compose_note(f"materializing {len(current_rows)} YAMLs"),
            )
            artifacts = self._materialize_and_execute_iteration(iteration_index, current_rows)
            self.ui.set_status(
                phase="optimize",
                note=self._compose_note("computing embeddings and Optuna suggestions"),
            )
            suggestions, iteration_summary, objective_values = self._run_optimizer_iteration(artifacts)
            for artifact, objective_value in zip(artifacts, objective_values, strict=True):
                artifact.objective_value = objective_value
            self._admit_artifacts_to_base_pool(artifacts, iteration_index)
            best_trials = self.runner.get_best_trials(top_n=self.config.top_n_best_trials)
            # Suggestions returned here are Optuna-issued trials. Their trial
            # numbers are persisted and completed with `tell(...)` next round.
            next_rows = self._attach_base_configs_to_rows(
                suggestions,
                iteration_index=iteration_index + 1,
                use_seed_defaults=False,
            )
            best_trial_id = best_trials[0][0] if best_trials else "-"
            best_objective = self._get_best_objective_string()
            if initial_distance == "-":
                initial_distance = iteration_summary["iteration_best"]
            self.ui.set_status(
                best_trial_id=best_trial_id,
                best_objective=best_objective,
                initial_distance=initial_distance,
                iteration_best=iteration_summary["iteration_best"],
                iteration_mean=iteration_summary["iteration_mean"],
                iteration_median=iteration_summary["iteration_median"],
                note=self._compose_note("iteration complete"),
            )

            state["iterations"].append(
                {
                    "iteration_index": iteration_index,
                    "input_rows": current_rows,
                    "artifacts": [self._serialize_artifact(item) for item in artifacts],
                    "suggestions": next_rows,
                    "iteration_summary": iteration_summary,
                    "best_trials": [
                        {"trial_id": trial_id, "params": params}
                        for trial_id, params in best_trials
                    ],
                }
            )
            param_importances = self._compute_param_importances()
            if param_importances is not None:
                state["param_importances"] = param_importances
            self.state_store.save(state)
            self._promote_global_baseline(state)
            self._export_top_trials(state)
            self._export_best_runs_to_s3(state)
            current_rows = next_rows

        self.ui.set_status(phase="complete", note=self._compose_note("workflow finished"))
        self.ui.stop()

    def _prepare_real_embeddings(self) -> np.ndarray:
        """Load or compute the fixed reference embeddings for the real dataset."""
        self.ui.set_status(
            phase="real-cache",
            note=self._compose_note("loading subset-3 reference embeddings"),
        )
        real_image_paths = select_real_image_paths(
            self.config.real_dataset_root,
            self.config.real_annotations_file,
        )
        if not real_image_paths:
            raise ValueError("No real subset images were resolved from the dataset root and annotation file")
        cache_dir = self.workspace_dir / "cache" / "real"
        cache_key = make_cache_key(
            [
                self._embedder_id,
                *(str(path) for path in real_image_paths),
            ]
        )
        cache_path = cache_dir / f"{cache_key}.npy"
        manifest = {
            "model_name": self._embedder_id,
            "repo": self._embedder_repo,
            "image_paths": [str(path) for path in real_image_paths],
        }
        status = "hit" if cache_path.exists() else "miss"
        self.ui.set_status(real_cache_status=status)
        return self.embedder.embed_paths(
            real_image_paths,
            batch_size=self._embedder_batch_size,
            cache_path=cache_path,
            manifest=manifest,
        )

    def _compose_note(self, note: str) -> str:
        """Prefix workflow notes with the optional theme-round label."""
        if not self.meta_label:
            return note
        if not note:
            return self.meta_label
        return f"{self.meta_label} | {note}"

    def _bootstrap_base_pool(self, seed_items: list[tuple[Path, dict[str, Any]]]) -> None:
        """Seed a new pool with the configured base YAML family."""
        if not self.base_pool.enabled:
            return
        seed_entries = [
            PoolEntry(
                entry_id=self._make_seed_pool_entry_id(path),
                config=deepcopy(seed_config),
                flattened_params=self._flatten_pool_config(seed_config),
                score=None,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                iteration_index=None,
                theme_label="seed",
                stage_lineage=f"seed_config={path.name}",
                artifact_path=None,
                yaml_path=str(path),
                embedding_path=None,
                diversity_metadata={
                    "backend": "parameter_space",
                    "is_seed": True,
                },
            )
            for path, seed_config in seed_items
        ]
        self.base_pool.ensure_bootstrap_entries(seed_entries)

    def _sync_base_pool_from_state(self, state: dict[str, Any]) -> None:
        """Rebuild scored pool entries from checkpointed iterations when needed."""
        if not self.base_pool.enabled or not state["iterations"] or self.base_pool.has_scored_entries():
            return
        artifacts = [
            artifact
            for artifact in self._collect_completed_artifacts(state)
            if artifact.objective_value is not None
        ]
        self._admit_artifacts_to_base_pool(artifacts, iteration_index=None)

    def _attach_base_configs_to_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        iteration_index: int,
        use_seed_defaults: bool,
    ) -> list[dict[str, Any]]:
        """Attach row-specific base configs from the pool without changing Optuna params."""
        attached_rows = [deepcopy(row) for row in rows]
        if not self.base_pool.enabled:
            return attached_rows

        if use_seed_defaults and not self.base_pool.has_scored_entries():
            for row, seed_record in zip(attached_rows, self.seed_base_records, strict=True):
                row["base_config"] = deepcopy(seed_record["config"])
                row["base_pool_entry_id"] = seed_record["pool_entry_id"]
                row["base_pool_lineage"] = f"bootstrap_seed:{Path(seed_record['yaml_path']).name}"
            return attached_rows

        rows_needing_base = [row for row in attached_rows if "base_config" not in row]
        sampled_entries = self.base_pool.sample_entries(len(rows_needing_base))
        for row, entry in zip(rows_needing_base, sampled_entries, strict=False):
            row["base_config"] = deepcopy(entry.config)
            row["base_pool_entry_id"] = entry.entry_id
            row["base_pool_lineage"] = entry.stage_lineage
        for row in attached_rows:
            row.setdefault("base_config", deepcopy(self.base_template))
            row.setdefault("base_pool_entry_id", None)
            row.setdefault("base_pool_lineage", "fallback:best_yaml")
        return attached_rows

    def _admit_artifacts_to_base_pool(
        self,
        artifacts: list[RunArtifact],
        iteration_index: int | None,
    ) -> None:
        """Admit successful artifacts into the persistent base pool."""
        if not self.base_pool.enabled:
            return
        new_entries = []
        for artifact in artifacts:
            if artifact.objective_value is None:
                continue
            new_entries.append(self._build_pool_entry_from_artifact(artifact, iteration_index))
        if not new_entries:
            return
        self.base_pool.admit_entries(new_entries)

    def _build_pool_entry_from_artifact(
        self,
        artifact: RunArtifact,
        iteration_index: int | None,
    ) -> PoolEntry:
        """Convert one completed artifact into a durable base-pool entry."""
        config_dict = yaml.safe_load(artifact.yaml_path.read_text())
        flattened_params = self._flatten_pool_config(config_dict)
        embedding_array = np.load(artifact.embedding_path)
        centroid = embedding_array.mean(axis=0).astype(float).tolist()
        entry_id = make_cache_key(
            [
                str(artifact.yaml_path.resolve()),
                artifact.run_id,
                artifact.run_fingerprint,
            ]
        )
        return PoolEntry(
            entry_id=entry_id,
            config=config_dict,
            flattened_params=flattened_params,
            score=artifact.objective_value,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            iteration_index=iteration_index,
            theme_label=self._current_theme_label(),
            stage_lineage=self.meta_label or f"iteration={iteration_index}",
            artifact_path=str(artifact.output_dir),
            yaml_path=str(artifact.yaml_path),
            embedding_path=str(artifact.embedding_path),
            source_pool_entry_id=artifact.base_pool_entry_id,
            diversity_metadata={
                "backend": "embedding_centroid",
                "embedding_centroid": centroid,
                "fallback_backend": "parameter_space",
                "source_base_pool_entry_id": artifact.base_pool_entry_id,
            },
        )

    def _flatten_pool_config(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Flatten one pool config using only paths that actually exist in that config."""
        config_schema = infer_parameter_schema([config_dict])
        return flatten_config(config_dict, config_schema)

    def _make_seed_pool_entry_id(self, path: Path) -> str:
        """Generate a stable pool id for a seed YAML."""
        return f"seed::{make_cache_key([str(path.resolve())])}"

    def _current_theme_label(self) -> str:
        """Extract the current theme name from the metadata label when available."""
        if not self.meta_label:
            return "default"
        for token in self.meta_label.split():
            if token.startswith("theme="):
                return token.split("=", 1)[1]
        return self.meta_label

    def _load_iteration_rows(self, state: dict[str, Any], start_iteration: int) -> list[dict[str, Any]]:
        """Choose the current batch source: seeds on iteration 0, suggestions otherwise."""
        if start_iteration == 0:
            if (
                self.base_pool.enabled
                and self.base_pool.has_scored_entries()
                and len(self.base_pool.entries) > len(self.seed_rows)
            ):
                pool_rows = self._prime_optuna_from_pool_embeddings()
                if pool_rows:
                    return pool_rows
            return self._attach_base_configs_to_rows(
                self.seed_rows,
                iteration_index=0,
                use_seed_defaults=True,
            )
        return self._attach_base_configs_to_rows(
            state["iterations"][-1]["suggestions"],
            iteration_index=start_iteration,
            use_seed_defaults=False,
        )

    def _prime_optuna_from_pool_embeddings(self) -> list[dict[str, Any]]:
        """Initialize Optuna with pool entries using their cached DINOv2 embeddings.

        Loads each pool entry's existing embeddings from disk, feeds them all into
        evaluate_iteration as external (add_trial) observations, then returns the
        resulting ask() suggestions as iter-0 rows — no Isaac simulation needed.
        """
        scored = [
            e for e in self.base_pool.entries
            if e.score is not None and e.embedding_path and Path(e.embedding_path).exists()
        ]
        if not scored:
            return []

        current_distributions = []
        all_embeddings = []
        embeddings_indices_by_dist: dict[int, list] = {}
        start_index = 0

        for dist_index, entry in enumerate(scored):
            embedding_array = np.load(entry.embedding_path)
            all_embeddings.append(embedding_array)
            end_index = start_index + len(embedding_array)
            params = {f"shape_logit_{self.group_name}": 0.0}
            theme_params = flatten_config(entry.config, self.schema)
            for key, value in theme_params.items():
                params[f"{self.group_name}__{key}"] = value
            current_distributions.append((entry.entry_id, params))
            embeddings_indices_by_dist[dist_index] = [(0, np.arange(start_index, end_index))]
            start_index = end_index

        embeddings_by_shape = [np.concatenate(all_embeddings, axis=0)]
        trial_numbers = [None] * len(scored)

        print(f"Priming Optuna with {len(scored)} pool entries (cached embeddings)...")
        raw_suggestions, _ = self.runner.evaluate_iteration(
            current_distributions=current_distributions,
            embeddings_by_shape=embeddings_by_shape,
            embeddings_indices_by_dist=embeddings_indices_by_dist,
            trial_numbers=trial_numbers,
        )

        suggestions = []
        for suggestion_id, params in raw_suggestions:
            flattened = {
                key[len(f"{self.group_name}__"):]: value
                for key, value in params.items()
                if key.startswith(f"{self.group_name}__")
            }
            trial_number = None
            if suggestion_id.startswith("trial_"):
                trial_number = int(suggestion_id.split("_", 1)[1])
            suggestions.append({
                "suggestion_id": suggestion_id,
                "optuna_trial_number": trial_number,
                "params": flattened,
            })

        return self._attach_base_configs_to_rows(
            suggestions,
            iteration_index=0,
            use_seed_defaults=False,
        )

    # NOT USED! Was used to re-sample pool entries and re-run them as fresh simulations
    # to initialize iter-0 — i.e. replay the pool as a new batch instead of using
    # cached embeddings. Superseded by _prime_optuna_from_pool_embeddings.
    def _build_iteration_zero_rows_from_pool(self, target_count: int) -> list[dict[str, Any]]:
        """Use sampled scored pool entries as the direct iteration-0 candidates."""
        sampled_entries = self.base_pool.sample_entries(target_count)
        rows = []
        for index, entry in enumerate(sampled_entries):
            rows.append(
                {
                    "suggestion_id": f"pool_bootstrap_{index}",
                    "optuna_trial_number": None,
                    "params": flatten_config(entry.config, self.schema),
                    "base_config": deepcopy(entry.config),
                    "base_pool_entry_id": entry.entry_id,
                    "base_pool_lineage": entry.stage_lineage,
                    "direct_pool_replay": True,
                    "pool_artifact_path": entry.artifact_path,
                    "pool_embedding_path": entry.embedding_path,
                }
            )
        return rows

    def _replay_completed_iterations(self, state: dict[str, Any]) -> None:
        """Rebuild the in-memory Optuna study by replaying saved completed iterations."""
        if not state["iterations"]:
            return

        self.ui.set_status(phase="resume", note=f"replaying {len(state['iterations'])} completed iterations")
        for iteration in state["iterations"]:
            artifacts = [
                RunArtifact(
                    run_id=item["run_id"],
                    run_fingerprint=item.get("run_fingerprint", "legacy"),
                    yaml_path=Path(item["yaml_path"]),
                    output_dir=Path(item["output_dir"]),
                    log_path=Path(item["log_path"]),
                    embedding_path=Path(item["embedding_path"]),
                    image_count=int(item["image_count"]),
                    flattened_params=item["flattened_params"],
                    optuna_trial_number=item.get("optuna_trial_number"),
                    objective_value=item.get("objective_value"),
                    base_pool_entry_id=item.get("base_pool_entry_id"),
                    base_pool_lineage=item.get("base_pool_lineage"),
                )
                for item in iteration["artifacts"]
            ]
            _, iteration_summary, _ = self._run_optimizer_iteration(artifacts)
            best_trials = self.runner.get_best_trials(top_n=self.config.top_n_best_trials)
            best_trial_id = best_trials[0][0] if best_trials else "-"
            self.ui.set_status(
                best_trial_id=best_trial_id,
                best_objective=self._get_best_objective_string(),
                initial_distance=state["iterations"][0]["iteration_summary"]["iteration_best"],
                iteration_best=iteration_summary["iteration_best"],
                iteration_mean=iteration_summary["iteration_mean"],
                iteration_median=iteration_summary["iteration_median"],
            )

    def _materialize_and_execute_iteration(
        self,
        iteration_index: int,
        rows: list[dict[str, Any]],
    ) -> list[RunArtifact]:
        """Materialize one batch of YAMLs, run Isaac, and cache synthetic embeddings."""
        iteration_started_at = time.perf_counter()
        iteration_dir = self.workspace_dir / f"iteration_{iteration_index:03d}"
        yaml_dir = iteration_dir / "yamls"
        outputs_dir = iteration_dir / "outputs"
        cache_dir = iteration_dir / "cache"
        prepare_output_dir(yaml_dir, clean=False)
        prepare_output_dir(outputs_dir, clean=False)
        prepare_output_dir(cache_dir, clean=False)

        artifacts: list[RunArtifact] = []
        reused_seed_runs = 0
        generated_runs = 0
        for run_index, row_record in enumerate(rows):
            run_id = f"iter{iteration_index:03d}_run{run_index:03d}"
            yaml_path = yaml_dir / f"{run_id}.yaml"
            params_row = row_record["params"]
            base_config = row_record.get("base_config", self.base_template)
            config_dict = materialize_config(base_config, params_row, self.schema)
            run_fingerprint = self._make_run_fingerprint(config_dict)
            output_dir = outputs_dir / f"{run_id}__{run_fingerprint[:12]}"
            log_path = output_dir / "isaac.log"
            embedding_path = cache_dir / f"{run_id}__{run_fingerprint[:12]}_{self._embedder_id}.npy"
            # Keep the generated YAML self-contained so Isaac writes into the
            # iteration-specific output directory even if the base template had a
            # different `run.data_dir`.
            run_section = config_dict.setdefault("run", {})
            run_section["data_dir"] = str(output_dir)
            save_yaml_config(yaml_path, config_dict)

            self.ui.set_status(current_run=run_id, completed_runs=run_index, total_runs=len(rows))
            self._prepare_run_output_dir(output_dir, run_id, run_fingerprint)
            embedding_reused = False
            eval_seeds = list(self.config.isaac.eval_seeds) or [0]

            # Single-seed reuse fast-paths (pool replay / prior synthetic base)
            # only apply to the legacy one-run layout (output_dir/Camera/rgb) and
            # are inactive for trajectory configs (no synthetic_rgb_base_dir, no
            # direct pool replay). Keep them only when a single seed is requested.
            image_paths: list[Path] = []
            if len(eval_seeds) == 1:
                single_seed_dir = output_dir / f"seed_{eval_seeds[0]}"
                image_paths, embedding_reused = self._copy_synthetic_artifacts_from_pool_replay(
                    output_dir=single_seed_dir,
                    embedding_path=embedding_path,
                    row_record=row_record,
                )
                if not image_paths:
                    image_paths, embedding_reused = self._copy_synthetic_artifacts_from_base(
                        output_dir=single_seed_dir,
                        embedding_path=embedding_path,
                        run_id=run_id,
                        run_fingerprint=run_fingerprint,
                        yaml_path=yaml_path,
                    )
                if image_paths:
                    reused_seed_runs += 1

            if not image_paths:
                # Multi-seed evaluation: generate the SAME candidate YAML once per
                # seed (each into its own subdir), then POOL every seed's RGB
                # frames before embedding. Re-rolling the seed re-rolls the scene
                # layout + trajectory, so the pooled embedding samples the
                # config's distribution instead of one realization. Only RGB
                # images feed the embedder — discovery is scoped to each seed's
                # Camera/rgb tree so depth/semantic renders never leak in.
                for seed in eval_seeds:
                    seed_dir = output_dir / f"seed_{seed}"
                    seed_rgb_dir = seed_dir / "Camera" / "rgb"
                    seed_images = discover_generated_images(seed_rgb_dir)
                    if not seed_images:
                        # Seed-retry (mirrors run_base_v4_train.sh): a run can fail
                        # because THIS layout draw leaves no navigable freespace
                        # (UniformPoseSampler "high <= 0" / no occupancy path).
                        # That is a property of the layout, not the config, and it
                        # is DETERMINISTIC for (config, seed) — without resampling
                        # the outer retry wrapper loops on it forever. Re-roll with
                        # seed + k*1000; the seed_dir keeps the original label.
                        last_exc: Exception | None = None
                        for attempt in range(3):
                            try_seed = seed + attempt * 1000
                            if attempt:
                                self.ui.append_log(
                                    f"[isaac-seed-retry] {run_id} seed {seed}: "
                                    f"attempt {attempt + 1} with seed {try_seed}"
                                )
                            try:
                                run_isaac_generation(
                                    isaac_sim_path=Path(self.config.isaac.isaac_sim_path),
                                    script_path=Path(self.config.isaac.script_path),
                                    yaml_path=yaml_path,
                                    output_dir=seed_dir,
                                    log_path=seed_dir / "isaac.log",
                                    headless=self.config.isaac.headless,
                                    num_frames_override=self.config.isaac.num_frames_override,
                                    seed=try_seed,
                                    log_callback=self.ui.append_log,
                                )
                                last_exc = None
                                break
                            except RuntimeError as exc:
                                last_exc = exc
                        if last_exc is not None:
                            raise last_exc
                        seed_images = discover_generated_images(seed_rgb_dir)
                        if seed_images:
                            generated_runs += 1
                    if not seed_images:
                        raise ValueError(
                            f"No generated images discovered under {seed_dir} (seed={seed})"
                        )
                    image_paths.extend(seed_images)
                image_paths = sorted(image_paths)
            if not image_paths:
                raise ValueError(f"No generated images discovered under {output_dir}")
            self._write_run_manifest(output_dir, run_id, run_fingerprint, yaml_path)
            manifest = {
                "model_name": self._embedder_id,
                "repo": self._embedder_repo,
                "image_paths": [str(path) for path in image_paths],
                "yaml_path": str(yaml_path),
                "run_fingerprint": run_fingerprint,
            }
            if embedding_reused:
                embedding_manifest_path = embedding_path.with_suffix(".manifest.json")
                if not embedding_path.exists() or not embedding_manifest_path.exists():
                    raise ValueError(f"Reused embedding artifacts are incomplete for {run_id}")
            else:
                self.embedder.embed_paths(
                    image_paths,
                    batch_size=self._embedder_batch_size,
                    cache_path=embedding_path,
                    manifest=manifest,
                )
            artifacts.append(
                RunArtifact(
                    run_id=run_id,
                    run_fingerprint=run_fingerprint,
                    yaml_path=yaml_path,
                    output_dir=output_dir,
                    log_path=log_path,
                    embedding_path=embedding_path,
                    image_count=len(image_paths),
                    flattened_params=params_row,
                    optuna_trial_number=row_record.get("optuna_trial_number"),
                    base_pool_entry_id=row_record.get("base_pool_entry_id"),
                    base_pool_lineage=row_record.get("base_pool_lineage"),
                )
            )
            self.ui.set_status(completed_runs=run_index + 1)
        if iteration_index == 0:
            elapsed_seconds = time.perf_counter() - iteration_started_at
            self.ui.append_log(
                "[seed-timing] "
                f"iteration_000 image prep took {elapsed_seconds:.1f}s "
                f"(reused={reused_seed_runs}, generated={generated_runs}, total_runs={len(rows)})"
            )
        return artifacts

    def _run_optimizer_iteration(self, artifacts: list[RunArtifact]) -> tuple[list[dict[str, Any]], dict[str, str], list[float]]:
        """Evaluate one completed synthetic batch and request the next suggestions."""
        embeddings = []
        current_distributions = []
        embeddings_indices_by_dist = {}
        trial_numbers = []
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
            # `None` means "external trial" and is imported with `add_trial`.
            # A real trial number means this row originated from `ask()` and must
            # be completed with `tell(...)`.
            trial_numbers.append(artifact.optuna_trial_number)
            start_index = end_index

        embeddings_by_shape = [np.concatenate(embeddings, axis=0)]
        raw_suggestions, metrics_list = self.runner.evaluate_iteration(
            current_distributions=current_distributions,
            embeddings_by_shape=embeddings_by_shape,
            embeddings_indices_by_dist=embeddings_indices_by_dist,
            trial_numbers=trial_numbers,
        )

        suggestions = []
        for suggestion_id, params in raw_suggestions:
            flattened = {}
            for key, value in params.items():
                if key.startswith(f"{self.group_name}__"):
                    flattened[key[len(f"{self.group_name}__"):]] = value
            trial_number = None
            if suggestion_id.startswith("trial_"):
                trial_number = int(suggestion_id.split("_", 1)[1])
            suggestions.append(
                {
                    "suggestion_id": suggestion_id,
                    "optuna_trial_number": trial_number,
                    "params": flattened,
                }
            )
        objective_name = self.optimizer_config["optimization_metrics"][0]
        objective_values = [metrics[objective_name] for metrics in metrics_list]
        iteration_summary = {
            "objective_name": objective_name,
            "iteration_best": f"{min(objective_values):.6f}",
            "iteration_mean": f"{float(np.mean(objective_values)):.6f}",
            "iteration_median": f"{float(np.median(objective_values)):.6f}",
        }
        return suggestions, iteration_summary, objective_values

    def _build_param_bounds_from_config(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
        """Map yaml-declared search-space bounds onto the filtered Isaac schema.

        Bounds come from `search_space.bounds` in the project YAML. Types come
        from the inferred parameter schema's `value_kind`. Raises with a clear
        list of missing paths so the user knows exactly what to add.
        """
        declared_bounds = self.config.search_space.bounds
        flat_keys: list[tuple[str, str]] = []
        for spec in self.schema:
            if spec.kind == "indexed_list":
                length = spec.length or 0
                for index in range(length):
                    flat_keys.append((f"{spec.path}[{index}]", spec.value_kind))
            else:
                flat_keys.append((spec.path, spec.value_kind))

        missing = [key for key, _ in flat_keys if key not in declared_bounds]
        if missing:
            joined = "\n  - ".join(missing)
            raise ValueError(
                "Missing search_space.bounds entries for the following parameter paths:\n  - "
                f"{joined}\nAdd them to the project YAML under `search_space.bounds`."
            )

        bounds: dict[str, Any] = {}
        types: dict[str, str] = {}
        for key, value_kind in flat_keys:
            declared = declared_bounds[key]
            if value_kind in ("int", "float"):
                if not isinstance(declared, list) or len(declared) != 2:
                    raise ValueError(
                        f"Numeric bound for '{key}' must be [min, max]; got {declared!r}"
                    )
                bounds[key] = list(declared)
                types[key] = value_kind
            else:
                if not isinstance(declared, list) or len(declared) == 0:
                    raise ValueError(
                        f"Categorical bound for '{key}' must be a non-empty list; got {declared!r}"
                    )
                bounds[key] = list(declared)
                types[key] = "categorical"

        return {self.group_name: bounds}, {self.group_name: types}

    def _serialize_artifact(self, artifact: RunArtifact) -> dict[str, Any]:
        """Convert a runtime artifact into a JSON-serializable checkpoint record."""
        return {
            "run_id": artifact.run_id,
            "run_fingerprint": artifact.run_fingerprint,
            "yaml_path": str(artifact.yaml_path),
            "output_dir": str(artifact.output_dir),
            "log_path": str(artifact.log_path),
            "embedding_path": str(artifact.embedding_path),
            "image_count": artifact.image_count,
            "flattened_params": artifact.flattened_params,
            "optuna_trial_number": artifact.optuna_trial_number,
            "objective_value": artifact.objective_value,
            "base_pool_entry_id": artifact.base_pool_entry_id,
            "base_pool_lineage": artifact.base_pool_lineage,
        }

    def _copy_synthetic_artifacts_from_pool_replay(
        self,
        *,
        output_dir: Path,
        embedding_path: Path,
        row_record: dict[str, Any],
    ) -> tuple[list[Path], bool]:
        """Reuse artifacts directly from a replayed pool entry when available."""
        if not row_record.get("direct_pool_replay"):
            return [], False

        artifact_path_value = row_record.get("pool_artifact_path")
        embedding_path_value = row_record.get("pool_embedding_path")
        if not artifact_path_value or not embedding_path_value:
            return [], False

        source_output_dir = Path(artifact_path_value)
        source_embedding_path = Path(embedding_path_value)
        source_manifest_path = source_embedding_path.with_suffix(".manifest.json")
        source_rgb_dir = source_output_dir / "Camera" / "rgb"
        if not source_output_dir.exists() or not source_rgb_dir.exists():
            return [], False
        if not source_embedding_path.exists() or not source_manifest_path.exists():
            return [], False

        shutil.copytree(source_output_dir, output_dir, dirs_exist_ok=True)
        shutil.copy2(source_embedding_path, embedding_path)
        shutil.copy2(source_manifest_path, embedding_path.with_suffix(".manifest.json"))
        image_paths = discover_generated_images(output_dir / "Camera" / "rgb")
        if not image_paths:
            return [], False
        self.ui.append_log(
            f"[reuse] copied direct pool replay artifacts for {row_record.get('base_pool_entry_id', 'unknown')}"
        )
        return image_paths, True

    def _get_best_objective_string(self) -> str:
        """Return the best completed objective value currently known to Optuna."""
        completed_trials = [trial for trial in self.runner.optimizer.study.trials if trial.values is not None]
        if not completed_trials:
            return "-"
        best_value = min(trial.values[0] for trial in completed_trials)
        return f"{best_value:.6f}"

    def _compute_param_importances(self) -> dict[str, float] | None:
        """Compute fANOVA parameter importances from the current Optuna study.

        Returns None when fewer than 5 completed trials exist (fANOVA is unreliable
        below that threshold). The group prefix is stripped from each param name so
        the result maps directly to Isaac config paths.
        """
        study = self.runner.optimizer.study
        completed = [t for t in study.trials if t.values is not None]
        if len(completed) < 5:
            return None
        raw = optuna.importance.get_param_importances(study)
        group_prefix = f"{self.group_name}__"
        return {
            (k[len(group_prefix):] if k.startswith(group_prefix) else k): float(v)
            for k, v in raw.items()
        }

    def _copy_synthetic_artifacts_from_base(
        self,
        *,
        output_dir: Path,
        embedding_path: Path,
        run_id: str,
        run_fingerprint: str,
        yaml_path: Path,
    ) -> tuple[list[Path], bool]:
        """Reuse a prior synthetic artifact bundle when it matches this exact run."""
        if self.synthetic_rgb_base_dir is None:
            return [], False

        source_output_dir = self._find_reusable_source_output_dir(run_id, run_fingerprint)
        if source_output_dir is None:
            return [], False

        source_rgb_dir = source_output_dir / "Camera" / "rgb"
        if not source_rgb_dir.exists():
            return [], False

        target_rgb_dir = output_dir / "Camera" / "rgb"
        if not target_rgb_dir.exists():
            target_rgb_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_rgb_dir, target_rgb_dir)

        image_paths = discover_generated_images(target_rgb_dir)
        if not image_paths:
            return [], False

        expected_manifest = {
            "model_name": self._embedder_id,
            "repo": self._embedder_repo,
            "image_paths": [str(path) for path in image_paths],
            "yaml_path": str(yaml_path),
            "run_fingerprint": run_fingerprint,
        }
        source_embedding_path = self._find_reusable_source_embedding_path(source_output_dir)
        if source_embedding_path is None:
            return image_paths, False

        source_manifest_path = source_embedding_path.with_suffix(".manifest.json")
        if not source_manifest_path.exists():
            return image_paths, False

        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest.get("model_name") != self._embedder_id:
            return image_paths, False
        if source_manifest.get("repo") != self._embedder_repo:
            return image_paths, False
        if source_manifest.get("run_fingerprint") != run_fingerprint:
            return image_paths, False
        if len(source_manifest.get("image_paths", [])) != len(image_paths):
            return image_paths, False

        shutil.copy2(source_embedding_path, embedding_path)
        embedding_path.with_suffix(".manifest.json").write_text(json.dumps(expected_manifest, indent=2, sort_keys=True))
        self.ui.append_log(f"[reuse] copied matching embedding cache for {run_id}")
        return image_paths, True

    def _find_reusable_source_output_dir(self, run_id: str, run_fingerprint: str) -> Path | None:
        """Resolve a reusable output directory from the configured synthetic base."""
        assert self.synthetic_rgb_base_dir is not None
        fingerprint_prefix = run_fingerprint[:12]
        candidate_paths = [
            self.synthetic_rgb_base_dir / f"{run_id}__{fingerprint_prefix}",
            self.synthetic_rgb_base_dir / run_id,
        ]
        candidate_paths.extend(sorted(self.synthetic_rgb_base_dir.glob(f"{run_id}__*")))
        for candidate in candidate_paths:
            manifest_path = candidate / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("run_id") != run_id:
                continue
            if manifest.get("run_fingerprint") != run_fingerprint:
                continue
            return candidate
        return None

    def _find_reusable_source_embedding_path(self, source_output_dir: Path) -> Path | None:
        """Resolve a reusable embedding file from a prior iteration cache."""
        cache_dir = source_output_dir.parent.parent / "cache"
        if not cache_dir.exists():
            return None

        preferred_name = f"{source_output_dir.name}_{self._embedder_id}.npy"
        preferred_path = cache_dir / preferred_name
        if preferred_path.exists():
            return preferred_path

        run_prefix = f"{source_output_dir.name}_"
        matching = sorted(cache_dir.glob(f"{run_prefix}*.npy"))
        if matching:
            return matching[0]
        return None

    def _make_run_fingerprint(self, config_dict: dict[str, Any]) -> str:
        """Hash the effective Isaac config and execution knobs for one run."""
        fingerprint_payload = deepcopy(config_dict)
        run_section = fingerprint_payload.get("run")
        if isinstance(run_section, dict):
            run_section.pop("data_dir", None)
        serialized_config = yaml.safe_dump(fingerprint_payload, sort_keys=True)
        return make_cache_key(
            [
                serialized_config,
                str(self.config.isaac.script_path),
                str(self.config.isaac.headless),
                str(self.config.isaac.num_frames_override),
                str(list(self.config.isaac.eval_seeds)),
            ]
        )

    def _prepare_run_output_dir(self, output_dir: Path, run_id: str, run_fingerprint: str) -> None:
        """Ensure existing outputs are only reused when they match this exact run."""
        manifest_path = output_dir / "run_manifest.json"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            return

        if not manifest_path.exists():
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.ui.append_log(f"[reuse] cleared legacy output without manifest for {run_id}")
            return

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_fingerprint") != run_fingerprint or manifest.get("run_id") != run_id:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.ui.append_log(f"[reuse] cleared stale output for {run_id}")

    def _write_run_manifest(self, output_dir: Path, run_id: str, run_fingerprint: str, yaml_path: Path) -> None:
        """Record the trial identity that produced a reusable output directory."""
        manifest = {
            "run_id": run_id,
            "run_fingerprint": run_fingerprint,
            "yaml_path": str(yaml_path),
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _load_base_template(self, seed_items: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
        """Pick the nested YAML template used for materializing future suggestions."""
        promoted_yaml_path = self._get_promoted_baseline_yaml_path()
        if promoted_yaml_path is not None:
            self.ui.append_log(f"[baseline] using promoted baseline from {promoted_yaml_path}")
            seed_template = deepcopy(seed_items[0][1])
            baseline_template = yaml.safe_load(promoted_yaml_path.read_text())
            return self._merge_dicts(seed_template, baseline_template)

        if self.baseline_state_path is None:
            return deepcopy(seed_items[0][1])

        if not self.baseline_state_path.exists():
            raise ValueError(f"Baseline state file does not exist: {self.baseline_state_path}")

        baseline_state = json.loads(self.baseline_state_path.read_text())
        if not baseline_state.get("iterations"):
            raise ValueError(f"Baseline state file has no iterations: {self.baseline_state_path}")

        best_trials = baseline_state["iterations"][-1].get("best_trials", [])
        if not best_trials:
            raise ValueError(f"Baseline state file has no best trials: {self.baseline_state_path}")

        best_trial_id = best_trials[0]["trial_id"]
        best_trial_number = self._trial_number_from_trial_id(best_trial_id)
        artifact = self._find_artifact_for_trial_number(baseline_state, best_trial_number)
        if artifact is None:
            raise ValueError(
                f"Could not find artifact for baseline trial {best_trial_id} in {self.baseline_state_path}"
            )

        yaml_path = Path(artifact["yaml_path"])
        if not yaml_path.exists():
            raise ValueError(f"Baseline yaml does not exist: {yaml_path}")
        self.ui.append_log(f"[baseline] using best base trial {best_trial_id} from {yaml_path}")
        seed_template = deepcopy(seed_items[0][1])
        baseline_template = yaml.safe_load(yaml_path.read_text())
        return self._merge_dicts(seed_template, baseline_template)

    def _get_promoted_baseline_yaml_path(self) -> Path | None:
        """Return the promoted baseline YAML when configured and already created."""
        if self.promoted_baseline_dir is None:
            return None
        candidate = self.promoted_baseline_dir / "best.yaml"
        if candidate.exists():
            return candidate
        return None

    def _merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively overlay one nested config onto another."""
        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def _trial_number_from_trial_id(self, trial_id: str) -> int:
        """Parse the workflow's `trial_<n>` identifier format."""
        if not trial_id.startswith("trial_"):
            raise ValueError(f"Unsupported trial id format: {trial_id}")
        return int(trial_id.split("_", 1)[1])

    def _find_artifact_for_trial_number(
        self,
        state: dict[str, Any],
        trial_number: int,
    ) -> dict[str, Any] | None:
        """Find the saved artifact that corresponds to one Optuna trial number."""
        for iteration in state.get("iterations", []):
            for artifact in iteration.get("artifacts", []):
                if artifact.get("optuna_trial_number") == trial_number:
                    return artifact
        return None

    def _promote_global_baseline(self, state: dict[str, Any]) -> None:
        """Persist the current best completed YAML as a shared promoted baseline."""
        if self.promoted_baseline_dir is None or not state["iterations"]:
            return

        artifacts = [
            artifact for artifact in self._collect_completed_artifacts(state)
            if artifact.objective_value is not None
        ]
        if not artifacts:
            return

        best_artifact = min(artifacts, key=lambda item: item.objective_value)
        assert best_artifact.objective_value is not None

        self.promoted_baseline_dir.mkdir(parents=True, exist_ok=True)
        promoted_yaml_path = self.promoted_baseline_dir / "best.yaml"
        promoted_metadata_path = self.promoted_baseline_dir / "best.json"
        existing_objective = None
        if promoted_metadata_path.exists():
            metadata = json.loads(promoted_metadata_path.read_text())
            if metadata.get("objective_value") is not None:
                existing_objective = float(metadata["objective_value"])
        if existing_objective is not None and existing_objective <= best_artifact.objective_value:
            return

        shutil.copy2(best_artifact.yaml_path, promoted_yaml_path)
        promoted_metadata = {
            "project_name": self.config.project_name,
            "run_id": best_artifact.run_id,
            "objective_value": best_artifact.objective_value,
            "yaml_path": str(best_artifact.yaml_path),
            "workspace_dir": str(self.workspace_dir),
            "updated_from_state": str(self.state_store.state_path),
        }
        promoted_metadata_path.write_text(json.dumps(promoted_metadata, indent=2, sort_keys=True))
        self.ui.append_log(
            "[baseline] promoted "
            f"{best_artifact.run_id} ({best_artifact.objective_value:.6f}) "
            f"to {promoted_yaml_path}"
        )

    def _export_top_trials(self, state: dict[str, Any]) -> None:
        """Write the top-k and top-k-diverse trial exports next to the promoted best.

        Three YAML files are rewritten from scratch after every iteration:
        - best_top{k}.yaml: the k best unique trials by objective value.
        - best_top{k}_diverse.yaml: k trials drawn from the best
          `diverse_candidate_pool` candidates via greedy max-min selection on
          normalized parameter distance — the best trial is always included,
          then each pick maximizes the minimum distance to the already-selected
          set, so the exported configs share as few parameter values as possible.
        - best_top{k}_diverse_latent.yaml: same greedy max-min selection, but
          the distance is the pairwise MMD (RBF, shared pool-level gamma)
          between the runs' cached embedding sets — configs whose *rendered
          images* look most different from each other, regardless of how close
          their parameters are.
        """
        if not state["iterations"]:
            return
        export_dir = (
            self.promoted_baseline_dir
            if self.promoted_baseline_dir is not None
            else self.workspace_dir / "top_trials"
        )

        artifacts = [
            artifact for artifact in self._collect_completed_artifacts(state)
            if artifact.objective_value is not None
        ]
        if not artifacts:
            return

        # Direct pool replays re-evaluate an identical config across iterations;
        # keep only the best-scoring run per fingerprint so duplicates don't
        # occupy multiple export slots.
        best_by_fingerprint: dict[str, RunArtifact] = {}
        for artifact in artifacts:
            existing = best_by_fingerprint.get(artifact.run_fingerprint)
            if existing is None or artifact.objective_value < existing.objective_value:
                best_by_fingerprint[artifact.run_fingerprint] = artifact
        ranked = sorted(best_by_fingerprint.values(), key=lambda item: item.objective_value)

        k = self.config.top_k_export
        top_artifacts = ranked[:k]
        threshold = self.config.diverse_objective_threshold
        if threshold is not None:
            pool = [
                artifact for artifact in ranked
                if artifact.objective_value <= threshold
            ]
            if len(pool) < k:
                self.ui.append_log(
                    f"[baseline] diverse export: only {len(pool)} trials pass "
                    f"objective <= {threshold}; lists will be short"
                )
        else:
            pool = ranked[: max(self.config.diverse_candidate_pool, k)]
        diverse_artifacts, min_distances = self._select_diverse_trials(
            pool, k, self._param_distance_fn(pool)
        )
        latent_pool = [
            artifact for artifact in pool if artifact.embedding_path.exists()
        ]
        if len(latent_pool) < len(pool):
            self.ui.append_log(
                f"[baseline] latent-diverse export: {len(pool) - len(latent_pool)} of "
                f"{len(pool)} pool candidates dropped (embedding cache missing)"
            )
        latent_artifacts, latent_distances = self._select_diverse_trials(
            latent_pool, k, self._latent_distance_fn(latent_pool)
        )

        export_dir.mkdir(parents=True, exist_ok=True)
        save_yaml_config(export_dir / f"best_top{k}.yaml", {
            "project_name": self.config.project_name,
            "selection": "objective",
            "trials": [
                self._top_trial_entry(rank, artifact)
                for rank, artifact in enumerate(top_artifacts, start=1)
            ],
        })
        save_yaml_config(export_dir / f"best_top{k}_diverse.yaml", {
            "project_name": self.config.project_name,
            "selection": "objective+diversity",
            "diversity_metric": "normalized_param_distance_full_config",
            "objective_threshold": threshold,
            "candidate_pool_size": len(pool),
            "trials": [
                {
                    **self._top_trial_entry(rank, artifact),
                    "min_param_distance_to_selected": distance,
                }
                for rank, (artifact, distance) in enumerate(
                    zip(diverse_artifacts, min_distances, strict=True), start=1
                )
            ],
        })
        save_yaml_config(export_dir / f"best_top{k}_diverse_latent.yaml", {
            "project_name": self.config.project_name,
            "selection": "objective+diversity",
            "diversity_metric": "embedding_mmd_rbf",
            "objective_threshold": threshold,
            "candidate_pool_size": len(latent_pool),
            "trials": [
                {
                    **self._top_trial_entry(rank, artifact),
                    "min_latent_mmd_to_selected": distance,
                }
                for rank, (artifact, distance) in enumerate(
                    zip(latent_artifacts, latent_distances, strict=True), start=1
                )
            ],
        })
        self.ui.append_log(
            f"[baseline] exported top{k} ({len(top_artifacts)} trials), "
            f"top{k}-diverse ({len(diverse_artifacts)} from pool of {len(pool)}), "
            f"top{k}-diverse-latent ({len(latent_artifacts)} from pool of {len(latent_pool)}) "
            f"to {export_dir}"
        )

    def _top_trial_entry(self, rank: int, artifact: RunArtifact) -> dict[str, Any]:
        """Build one self-contained export record for a top trial."""
        entry: dict[str, Any] = {
            "rank": rank,
            "trial_id": (
                f"trial_{artifact.optuna_trial_number}"
                if artifact.optuna_trial_number is not None
                else artifact.run_id
            ),
            "run_id": artifact.run_id,
            "objective_value": artifact.objective_value,
            "yaml_path": str(artifact.yaml_path),
            "embedding_path": str(artifact.embedding_path),
            "params": artifact.flattened_params,
        }
        # Inline the full materialized config so the export stays usable even
        # if the workspace run directories are cleaned up later.
        if artifact.yaml_path.exists():
            entry["config"] = yaml.safe_load(artifact.yaml_path.read_text())
        return entry

    def _param_distance_fn(self, pool: list[RunArtifact]):
        """Build a Gower-style distance over the full materialized configs.

        See diversity.build_gower_distance: numeric values normalize by the
        pool range, other values contribute 0/1 mismatch, constant keys are
        dropped, and comparing full configs (not just the searched params)
        keeps cross-study comparisons fair.
        """
        params_by_id = {
            artifact.run_id: diversity.full_config_flat(
                artifact.yaml_path, artifact.flattened_params, log=self.ui.append_log
            )
            for artifact in pool
        }
        id_distance = diversity.build_gower_distance(params_by_id, log=self.ui.append_log)

        def distance(a: RunArtifact, b: RunArtifact) -> float:
            return id_distance(a.run_id, b.run_id)

        return distance

    def _latent_distance_fn(self, pool: list[RunArtifact]):
        """Build a latent-space distance: pairwise MMD over cached run embeddings.

        Each run's cached (n_images, D) embedding array is loaded once; pairwise
        distances use the same RBF-kernel MMD as the optimization objective. The
        RBF gamma is computed ONCE with the median heuristic over a pooled
        subsample of all candidates — a shared bandwidth keeps the pairwise
        values mutually comparable, which per-pair gamma would not.
        """
        from calibration_optuna.metrics import DistributionMetrics

        embeddings = {
            artifact.run_id: np.load(artifact.embedding_path)
            for artifact in pool
        }
        gamma = None
        if len(pool) >= 2:
            rng = np.random.default_rng(self.config.random_seed)
            stacked = np.vstack(list(embeddings.values()))
            if stacked.shape[0] > 2000:
                stacked = stacked[rng.choice(stacked.shape[0], 2000, replace=False)]
            half = stacked.shape[0] // 2
            gamma = DistributionMetrics._compute_gamma_median_heuristic(
                stacked[:half], stacked[half:]
            )
        cache: dict[tuple[str, str], float] = {}

        def distance(a: RunArtifact, b: RunArtifact) -> float:
            key = (a.run_id, b.run_id) if a.run_id <= b.run_id else (b.run_id, a.run_id)
            if key not in cache:
                cache[key] = DistributionMetrics.mmd(
                    embeddings[a.run_id], embeddings[b.run_id], kernel="rbf", gamma=gamma
                )
            return cache[key]

        return distance

    def _select_diverse_trials(
        self,
        pool: list[RunArtifact],
        k: int,
        distance,
    ) -> tuple[list[RunArtifact], list[float | None]]:
        """Greedy max-min (farthest-point) selection under a pluggable distance.

        Starts from the best trial by objective (pool is objective-sorted), then
        repeatedly adds the pool candidate whose minimum distance to the
        already-selected set is largest (ties broken by better objective).
        Returns the selected artifacts (best first) and, per artifact, its
        minimum distance to the previously selected set (None for the first).
        """
        if not pool:
            return [], []

        selected = [pool[0]]
        min_distances: list[float | None] = [None]
        remaining = list(pool[1:])
        while remaining and len(selected) < k:
            scored = [
                (min(distance(candidate, chosen) for chosen in selected), candidate)
                for candidate in remaining
            ]
            # Farthest first; among equally-far candidates prefer the better
            # objective (pool order is objective-sorted, so max() with a stable
            # tie-break on the earliest index achieves this).
            best_score = max(score for score, _ in scored)
            picked = next(candidate for score, candidate in scored if score == best_score)
            selected.append(picked)
            min_distances.append(best_score)
            remaining.remove(picked)
        return selected, min_distances

    def _export_best_runs_to_s3(self, state: dict[str, Any]) -> None:
        """Stage and upload the current top trials to a timestamped S3 snapshot."""
        if self.s3_best_runs_prefix is None or not state["iterations"]:
            return

        if shutil.which("aws") is None:
            raise RuntimeError("AWS CLI is required for S3 export, but 'aws' was not found on PATH")

        artifacts = self._collect_completed_artifacts(state)
        selected_artifacts = [
            artifact
            for artifact in sorted(
                artifacts,
                key=lambda item: float("inf") if item.objective_value is None else item.objective_value,
            )[: self.config.top_n_best_trials]
        ]
        if not selected_artifacts:
            return

        snapshot_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_prefix = f"{self.s3_best_runs_prefix.rstrip('/')}/{snapshot_timestamp}/"

        with tempfile.TemporaryDirectory(prefix=f"{self.config.project_name}_s3_") as temp_dir:
            stage_root = Path(temp_dir)
            manifest_runs = []
            for artifact in selected_artifacts:
                trial_id = f"trial_{artifact.optuna_trial_number}" if artifact.optuna_trial_number is not None else artifact.run_id
                trial_stage_dir = stage_root / trial_id
                output_stage_dir = trial_stage_dir / "outputs" / artifact.run_id
                cache_stage_dir = trial_stage_dir / "cache"
                yaml_stage_dir = trial_stage_dir / "yamls"
                output_stage_dir.parent.mkdir(parents=True, exist_ok=True)
                cache_stage_dir.mkdir(parents=True, exist_ok=True)
                yaml_stage_dir.mkdir(parents=True, exist_ok=True)

                shutil.copytree(artifact.output_dir, output_stage_dir)
                shutil.copy2(artifact.embedding_path, cache_stage_dir / artifact.embedding_path.name)
                manifest_path = artifact.embedding_path.with_suffix(".manifest.json")
                if manifest_path.exists():
                    shutil.copy2(manifest_path, cache_stage_dir / manifest_path.name)
                shutil.copy2(artifact.yaml_path, yaml_stage_dir / artifact.yaml_path.name)

                manifest_runs.append(
                    {
                        "trial_id": trial_id,
                        "run_id": artifact.run_id,
                        "iteration_index": int(artifact.run_id[4:7]),
                        "objective_value": artifact.objective_value,
                        "source_output_dir": str(artifact.output_dir),
                        "source_yaml_path": str(artifact.yaml_path),
                        "source_embedding_path": str(artifact.embedding_path),
                    }
                )

            manifest = {
                "project_name": self.config.project_name,
                "top_n_best_trials": self.config.top_n_best_trials,
                "s3_prefix": snapshot_prefix,
                "snapshot_timestamp": snapshot_timestamp,
                "best_trials": manifest_runs,
            }
            (stage_root / "best_runs_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

            self.ui.append_log(
                f"[s3] uploading top {len(selected_artifacts)} runs to {snapshot_prefix}"
            )
            self._sync_directory_to_s3(stage_root, snapshot_prefix)
            self.ui.append_log(f"[s3] upload complete: {snapshot_prefix}")

    def _collect_completed_artifacts(self, state: dict[str, Any]) -> list[RunArtifact]:
        """Rehydrate all completed run artifacts from the persisted state ledger."""
        artifacts: list[RunArtifact] = []
        for iteration in state["iterations"]:
            for item in iteration["artifacts"]:
                artifacts.append(
                    RunArtifact(
                        run_id=item["run_id"],
                        run_fingerprint=item.get("run_fingerprint", "legacy"),
                        yaml_path=Path(item["yaml_path"]),
                        output_dir=Path(item["output_dir"]),
                        log_path=Path(item["log_path"]),
                        embedding_path=Path(item["embedding_path"]),
                        image_count=int(item["image_count"]),
                        flattened_params=item["flattened_params"],
                        optuna_trial_number=(
                            int(item["optuna_trial_number"])
                            if item.get("optuna_trial_number") is not None
                            else None
                        ),
                        objective_value=(
                            float(item["objective_value"])
                            if item.get("objective_value") is not None
                            else None
                        ),
                        base_pool_entry_id=item.get("base_pool_entry_id"),
                        base_pool_lineage=item.get("base_pool_lineage"),
                    )
                )
        return artifacts

    def _sync_directory_to_s3(self, source_dir: Path, s3_prefix: str) -> None:
        """Upload a staged directory to S3 with the AWS CLI."""
        command = [
            "aws",
            "s3",
            "sync",
            str(source_dir),
            s3_prefix,
            "--only-show-errors",
        ]
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.stdout:
            for line in process.stdout.splitlines():
                self.ui.append_log(f"[s3] {line}")
        if process.stderr:
            for line in process.stderr.splitlines():
                self.ui.append_log(f"[s3] {line}")
        if process.returncode != 0:
            raise RuntimeError(
                f"S3 sync failed with exit code {process.returncode} for prefix {s3_prefix}"
            )
