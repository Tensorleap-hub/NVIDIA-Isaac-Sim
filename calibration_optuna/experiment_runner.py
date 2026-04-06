"""
Production Experiment Runner for orchestrating the optimization loop.

Modified for production use:
- Accepts config dict instead of YAML file path
- No YAML dependency
- Simplified initialization

Agnostic to data source - receives embeddings and distributions from external source.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from .metrics import compute_per_param_set_metrics
from .optimizer import OptunaOptimizer


class ExperimentRunner:
    """
    Orchestrates the optimization loop.

    Receives data from external source, computes metrics, updates optimizer.
    """

    def __init__(
        self,
        config: Dict,
        param_bounds: Dict[str, Dict],
        param_type: Dict[str, Dict[str, str]]
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration dict (from config.DEFAULT_CONFIG)
            param_bounds: Parameter bounds dict {simulation_name: {param: [min, max]}}
                         Simulation names are inferred from param_bounds keys
            param_type: Parameter type dict {simulation_name: {param: "int"|"float"|"categorical"}}
        """
        self.config = config
        self._setup_experiment_dir()

        # Optuna optimizer (creates its own directory and SQLite DB)
        self.optimizer = OptunaOptimizer(
            experiment_dir=Path(self.config['experiment_dir']),
            config=self.config,
            param_bounds=param_bounds,
            param_type=param_type
        )

        self.param_type = param_type

        # Real embeddings reference (set via set_real_embeddings)
        self.real_embeddings_400d = None

        print(f"Initialized ExperimentRunner")
        print(f"Experiment directory: {self.config['experiment_dir']}")

    def _setup_experiment_dir(self):
        """
        Setup experiment directory based on experiment name.

        Creates directory as {experiments_base_dir}/{experiment_name}.
        If directory exists, reuses it to continue optimization from existing state.
        Updates self.config['experiment_dir'] with the resolved path.
        """
        base_dir = Path(self.config.get('experiments_base_dir', 'data/experiments'))
        exp_name = self.config['experiment_name']

        exp_dir = base_dir / exp_name
        self.config['experiment_dir'] = str(exp_dir)

        if exp_dir.exists():
            print(f"Reusing existing experiment directory: {exp_dir}")
        else:
            print(f"Creating new experiment directory: {exp_dir}")

    def set_real_embeddings(self, embeddings_400d: np.ndarray):
        """
        Set reference distribution (one-time setup).

        Args:
            embeddings_400d: Real embeddings (400D) to use as reference
        """
        self.real_embeddings_400d = embeddings_400d
        print(f"Set real embeddings: {embeddings_400d.shape}")

    def run_iteration(
        self,
        current_distributions: List[Tuple[str, Dict]],
        embeddings_by_shape: List[np.ndarray],
        embeddings_indices_by_dist: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> List[Tuple[str, Dict]]:
        """
        Process external data and get next suggestions.

        Every iteration is the same:
        1. Receive external data (distributions + embeddings)
        2. Compute metrics
        3. Tell optimizer and get next suggestions

        Args:
            current_distributions: What was evaluated [(dist_id, params_dict), ...]
                                   params_dict contains shape_logit_* and {shape}__{param} keys
            embeddings_by_shape: Original embeddings arrays from each simulation source
            embeddings_indices_by_dist: Dict mapping distribution_id to list of (source_idx, indices) tuples

        Returns:
            next_suggestions: What to try next [(dist_id, params_dict), ...]
        """
        if self.real_embeddings_400d is None:
            raise ValueError("Real embeddings not set. Call set_real_embeddings() first.")

        n_distributions = len(current_distributions)

        # Compute per-distribution metrics with MMD subsampling
        mmd_max_samples = self.config.get('mmd_max_samples', 2000)
        metrics_list = compute_per_param_set_metrics(
            embeddings_by_shape,
            embeddings_indices_by_dist,
            self.real_embeddings_400d,
            n_param_sets=n_distributions,
            mmd_max_samples=mmd_max_samples
        )

        # Tell optimizer and get next suggestions
        next_suggestions = self.optimizer.suggest_next_distributions(
            current_distributions=current_distributions,
            metrics_list=metrics_list,
            config=self.config
        )

        return next_suggestions

    def get_best_trials(self, top_n: int = None) -> List[Tuple[str, Dict]]:
        """
        Get the best trials seen so far.

        Args:
            top_n: Number of best trials to return. If None, returns all.

        Returns:
            List of (trial_id, params_dict) tuples with probabilities
        """
        return self.optimizer.get_best_trials_as_distributions(top_n=top_n)
