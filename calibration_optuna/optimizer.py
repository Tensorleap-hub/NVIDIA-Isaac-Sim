"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.
"""

import math
import numbers
import optuna
from pathlib import Path
from typing import Dict, List, Tuple


class OptunaOptimizer:
    """
    Optuna-based optimizer using TPE (Tree-structured Parzen Estimator) sampler.

    Jointly optimizes shape probabilities and all shape-specific parameters.
    Each trial suggests:
    1. Shape logits (converted to probabilities via softmax downstream)
    2. All parameters for all shapes simultaneously

    Features:
    - Multi-objective optimization (configurable metrics)
    - Joint shape probability + parameter optimization
    - Proper ask/tell pattern with pending trials tracking
    - Pareto front tracking for trade-off analysis
    - SQLite persistence for study state

    Min/max handling:
    - If a parameter pair uses *_min and *_max bounds in param_bounds, the optimizer
      transforms it internally to *_min and *_delta where delta = max - min.
    - Optuna operates on min + delta to guarantee max >= min; outgoing suggestions
      are converted back to min/max and clamped to the original max bounds.
    - current_distributions provided to suggest_next_distributions should use
      min/max keys; they are normalized to min/delta before add_trial().
    - Ensure min/max bounds are consistent (max upper bound > min lower bound),
      or the derived delta range may be too small.
    """

    def __init__(
        self,
        experiment_dir: Path,
        config: Dict,
        param_bounds: Dict[str, Dict],
        param_type: Dict[str, Dict[str, str]],
        logit_bounds: Tuple[float, float] = (-5.0, 5.0)
    ):
        """
        Initialize Optuna optimizer.

        Args:
            experiment_dir: Path to experiment directory for SQLite storage
            config: Experiment configuration dict with optimization_metrics, etc.
            param_bounds: Dict mapping simulation names to their parameter bounds
                          e.g., {'simulation_1': {'void_count_mean': [1.0, 10.0], ...}}
                          Simulation names (group_names) are inferred from the keys
            param_type: Dict mapping simulation names to parameter type strings
            logit_bounds: Min/max bounds for shape logits (default: -5.0 to 5.0)
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.study_path = self.experiment_dir / "optuna_study.db"
        self.logit_bounds = logit_bounds
        self.param_type = param_type

        # Validate inputs
        if not param_bounds:
            raise ValueError("param_bounds is required and cannot be empty")
        if not param_type:
            raise ValueError("param_type is required and cannot be empty")
        if set(param_type.keys()) != set(param_bounds.keys()):
            raise ValueError("param_type must have the same group keys as param_bounds")

        for group_name, group_bounds in param_bounds.items():
            if group_name not in param_type:
                raise ValueError(f"Missing param_type for group '{group_name}'")
            bounds_keys = set(group_bounds.keys())
            type_keys = set(param_type[group_name].keys())
            if bounds_keys != type_keys:
                raise ValueError(
                    f"param_type keys for '{group_name}' must match param_bounds keys"
                )

        # Find and swich min/max params to min + delta
        # 2. All params for all shapes
        self.param_bounds, self.param_type = self._arrange_param_bounds(param_bounds, param_type)

        # Infer group names from param_bounds keys (sorted for deterministic order)
        self.group_names = sorted(param_bounds.keys())

        # Get optimization metrics from config
        self.optimization_metrics = config.get('optimization_metrics', ['mmd_rbf', 'mean_nn_distance'])
        n_objectives = len(self.optimization_metrics)

        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create or load Optuna study with SQLite persistence
        #storage = f"sqlite:///{self.study_path}"
        study_name = config.get('experiment_name', 'optuna_study')

        # Get optimizer config
        optimizer_config = config.get('optimizer', {})
        multivariate = optimizer_config.get('multivariate', True)

        # Set n_startup_trials: higher for joint optimization due to larger search space
        # Default: 50 trials (more than per-group mode due to ~18 params)
        if 'n_startup_trials' in optimizer_config:
            n_startup_trials = optimizer_config['n_startup_trials']
        else:
            # Count total params: logits + all shape params
            total_params = len(self.group_names)  # logits
            for group_bounds in self.param_bounds.values():
                total_params += len(group_bounds)
            n_startup_trials = max(50, 3 * total_params)

        # Multi-objective optimization with configurable metrics
        self.study = optuna.create_study(
            study_name=study_name,
            storage=None,
            load_if_exists=True,
            directions=['minimize'] * n_objectives,  # minimize all metrics
            sampler=optuna.samplers.TPESampler(
                seed=config.get('random_seed', 42),
                n_startup_trials=n_startup_trials,
                multivariate=multivariate,
                warn_independent_sampling=False
            )
        )

        # Track pending trials from last ask() for potential future completion
        self.pending_trials = []

        print(f"Initialized OptunaOptimizer (joint mode)")
        print(f"  Study storage: {self.study_path}")
        print(f"  Study name: {study_name}")
        print(f"  Objectives: {n_objectives} ({', '.join(self.optimization_metrics)})")
        print(f"  Logit bounds: {logit_bounds}")
        print(f"  TPE startup trials: {n_startup_trials}")
        print(f"  TPE multivariate: {multivariate}")
        print(f"  Groups: {', '.join(self.group_names)}")
        for group_name in self.group_names:
            params = list(self.param_bounds[group_name].keys())
            print(f"    {group_name}: {len(params)} params")

    def _define_joint_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define joint search space: shape logits + all shape params.

        Each trial suggests:
        1. Shape logits for all groups (converted to probabilities downstream)
        2. All parameters for all shapes

        Returns:
            Dict with shape_logit_* keys and {shape}__{param} keys
        """
        params = {}

        # 1. Shape logits (converted to probs downstream via softmax)
        for group_name in self.group_names:
            logit = trial.suggest_float(
                f'shape_logit_{group_name}',
                self.logit_bounds[0],
                self.logit_bounds[1]
            )
            params[f'shape_logit_{group_name}'] = logit

        # 2. All params for all shapes
        for group_name in self.group_names:
            group_bounds = self.param_bounds[group_name]
            for param_name, bounds in group_bounds.items():
                optuna_key = f'{group_name}__{param_name}'
                params[optuna_key] = self._suggest_single_param(
                    trial,
                    optuna_key,
                    bounds,
                    self.param_type[group_name][param_name]
                )

        return params

    def _suggest_single_param(
        self,
        trial: optuna.Trial,
        param_name: str,
        bounds,
        param_type: str
    ):
        """
        Suggest a single parameter value based on its bounds.

        Args:
            trial: Optuna trial object
            param_name: Name of the parameter
            bounds: Either [min, max] for numerical or list of categories for categorical
            param_type: "int", "float", or "categorical"

        Returns:
            Suggested parameter value
        """
        if param_type == "categorical":
            if not isinstance(bounds, list) or len(bounds) == 0:
                raise ValueError(f"Invalid categorical bounds for '{param_name}': {bounds}")
            return trial.suggest_categorical(param_name, bounds)

        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Invalid numeric bounds for '{param_name}': {bounds}")

        min_val, max_val = bounds[0], bounds[1]
        if param_type == "int":
            return trial.suggest_int(param_name, int(min_val), int(max_val))
        if param_type == "float":
            return trial.suggest_float(param_name, float(min_val), float(max_val))

        raise ValueError(f"Invalid param_type for '{param_name}': {param_type}")

    def _bounds_to_distribution(
        self,
        bounds,
        param_type: str
    ) -> optuna.distributions.BaseDistribution:
        """
        Convert bounds to an Optuna distribution object.

        Args:
            bounds: Either [min, max] for numerical or list of categories for categorical
            param_type: "int", "float", or "categorical"

        Returns:
            Optuna distribution object
        """
        if param_type == "categorical":
            if not isinstance(bounds, list) or len(bounds) == 0:
                raise ValueError(f"Invalid categorical bounds: {bounds}")
            return optuna.distributions.CategoricalDistribution(bounds)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Invalid numeric bounds: {bounds}")

        min_val, max_val = bounds[0], bounds[1]
        if param_type == "int":
            return optuna.distributions.IntDistribution(int(min_val), int(max_val))
        if param_type == "float":
            return optuna.distributions.FloatDistribution(float(min_val), float(max_val))

        raise ValueError(f"Invalid param_type: {param_type}")

    def _expand_logit_bounds_from_data(self, current_distributions: List[Tuple[str, Dict]]):
        min_logit = float('inf')
        max_logit = float('-inf')

        for _, params in current_distributions:
            for param_name, value in params.items():
                if param_name.startswith('shape_logit_'):
                    min_logit = min(min_logit, value)
                    max_logit = max(max_logit, value)

        if min_logit < float('inf') and max_logit > float('-inf'):
            margin = max(abs(max_logit - min_logit) * 0.2, 1.0)
            new_lower = min(self.logit_bounds[0], min_logit - margin)
            new_upper = max(self.logit_bounds[1], max_logit + margin)

            if new_lower < self.logit_bounds[0] or new_upper > self.logit_bounds[1]:
                print(f"  Expanding logit bounds from {self.logit_bounds} to ({new_lower:.2f}, {new_upper:.2f})")
                self.logit_bounds = (new_lower, new_upper)

    def _build_full_distributions(self) -> Dict:
        """
        Build Optuna distributions for all params (logits + all shape params).

        Required for add_trial() to tell Optuna the type and range of each parameter.

        Returns:
            Dict mapping param names to optuna.distributions objects
        """
        distributions = {}

        # Logit distributions
        for group_name in self.group_names:
            distributions[f'shape_logit_{group_name}'] = optuna.distributions.FloatDistribution(
                self.logit_bounds[0], self.logit_bounds[1]
            )

        # All shape param distributions
        for group_name in self.group_names:
            group_bounds = self.param_bounds[group_name]
            for param_name, bounds in group_bounds.items():
                optuna_key = f'{group_name}__{param_name}'
                distributions[optuna_key] = self._bounds_to_distribution(
                    bounds,
                    self.param_type[group_name][param_name]
                )

        return distributions

    def get_pareto_front(self) -> List[optuna.trial.FrozenTrial]:
        """
        Get non-dominated trials from Pareto front.

        Returns:
            List of trials on the Pareto front (non-dominated solutions)
        """
        return self.study.best_trials

    def get_best_trials_as_distributions(
        self,
        top_n: int = None
    ) -> List[Tuple[str, Dict]]:
        """
        Get the best trials seen so far as distribution specifications.

        For single-objective: returns trials sorted by metric (best first)
        For multi-objective: returns Pareto front trials

        Args:
            top_n: Number of best trials to return. If None, returns all best trials.
                   For single-objective, this limits the sorted list.
                   For multi-objective, this limits the Pareto front.

        Returns:
            List of (dist_id, params_dict) tuples with probabilities (not logits)
        """
        # Get best trials
        if len(self.optimization_metrics) == 1:
            # Single objective: sort all completed trials by metric
            completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(completed, key=lambda t: t.values[0])
            best_trials = sorted_trials[:top_n] if top_n else sorted_trials
        else:
            # Multi-objective: use Pareto front
            pareto_trials = self.get_pareto_front()
            best_trials = pareto_trials[:top_n] if top_n else pareto_trials

        # Convert to distribution format with probabilities
        distributions = []
        for trial in best_trials:
            dist_id = f"trial_{trial.number}"

            # Convert logits to probabilities for output
            params_with_probs = self.convert_logits_to_probs_in_params(trial.params)

            distributions.append((dist_id, params_with_probs))

        return distributions

    @staticmethod
    def sample_counts_to_logits(sample_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Convert sample counts to logits (inverse softmax).

        Used to infer initial shape probabilities from data where sample counts
        across shape CSVs determine the distribution.

        Args:
            sample_counts: Dict mapping group names to sample counts
                          e.g., {'circle': 100, 'ellipse': 80, 'irregular': 70}

        Returns:
            Dict with shape_logit_* keys
            e.g., {'shape_logit_circle': -0.22, 'shape_logit_ellipse': -0.44, ...}
        """
        total = sum(sample_counts.values())
        if total == 0:
            raise ValueError("Total sample count cannot be zero")

        logits = {}
        for shape, count in sample_counts.items():
            # Compute probability, clamp to avoid log(0)
            prob = max(count / total, 1e-6)
            # Inverse softmax: logit = log(prob)
            # (constant offset cancels out in softmax)
            logits[f'shape_logit_{shape}'] = math.log(prob)

        return logits

    @staticmethod
    def logits_to_probabilities(params: Dict) -> Dict[str, float]:
        """
        Convert shape logits in params dict to probabilities via softmax.

        Args:
            params: Dict containing shape_logit_* keys

        Returns:
            Dict mapping shape names to probabilities (sum to 1.0)
        """
        # Extract logits
        logit_prefix = 'shape_logit_'
        logits = {}
        for key, value in params.items():
            if key.startswith(logit_prefix):
                shape = key[len(logit_prefix):]
                logits[shape] = value

        if not logits:
            return {}

        # Softmax: exp(z_i) / sum(exp(z_j))
        # Subtract max for numerical stability
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())

        return {k: v / total for k, v in exp_logits.items()}

    @staticmethod
    def convert_logits_to_probs_in_params(params: Dict) -> Dict:
        """
        Convert shape_logit_* keys to shape_prob_* keys with softmax probabilities.

        This is for output formatting only - internally the optimizer still uses logits.

        Args:
            params: Dict with shape_logit_* and other parameter keys

        Returns:
            New dict with shape_prob_* instead of shape_logit_*, plus all other params
        """
        # Extract logits and compute probabilities
        logit_prefix = 'shape_logit_'
        logits = {}
        other_params = {}

        for key, value in params.items():
            if key.startswith(logit_prefix):
                shape = key[len(logit_prefix):]
                logits[shape] = value
            else:
                other_params[key] = value

        # Compute softmax probabilities
        if logits:
            max_logit = max(logits.values())
            exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
            total = sum(exp_logits.values())
            probs = {f'shape_prob_{k}': v / total for k, v in exp_logits.items()}
        else:
            probs = {}

        # Return combined dict with probabilities + other params
        return {**probs, **other_params}

    def _arrange_param_bounds(self, param_bounds, param_type):
        self.max_bound_storage = {}
        self.max_type_storage = {}
        rel_eps = 1e-4
        for group_name in sorted(param_bounds.keys()):
            group_bounds = param_bounds[group_name]
            group_types = param_type[group_name]
            if group_name not in self.max_bound_storage:
                self.max_bound_storage[group_name] = {}
            if group_name not in self.max_type_storage:
                self.max_type_storage[group_name] = {}
            for param_name, bounds in list(group_bounds.items()):
                if param_name.endswith('_min'):
                    base = param_name[:-4]
                    max_name = f'{base}_max'
                    delta_name = f'{base}_delta'
                    if max_name in group_bounds:
                        max_bounds = group_bounds.pop(max_name)
                        max_type = group_types.pop(max_name)
                        self.max_bound_storage[group_name][max_name] = max_bounds
                        self.max_type_storage[group_name][max_name] = max_type
                        width_scale = max_bounds[1] - group_bounds[param_name][0]
                        epsilon = max(rel_eps * width_scale, 1e-12)

                        max_diff = max_bounds[1] - group_bounds[param_name][0]
                        max_diff = max(max_diff, epsilon)

                        diff_min_max = max_bounds[0] - group_bounds[param_name][1]
                        if max_bounds[0] <= group_bounds[param_name][1]:
                            min_diff = epsilon
                        else:
                            min_diff =  diff_min_max

                        min_type = group_types[param_name]
                        if min_type == "float" or max_type == "float":
                            group_types[delta_name] = "float"
                        else:
                            group_types[delta_name] = "int"
                            min_diff = int(round(max(min_diff, 1)))
                            max_diff = int(round(max(max_diff, 1)))

                        group_bounds[delta_name] = [min_diff, max_diff]
        return param_bounds, param_type

    @staticmethod
    def _split_group_prefix(param_name: str) -> Tuple[str, str, str]:
        if '__' in param_name:
            group_name, raw_name = param_name.split('__', 1)
            name_prefix = f"{group_name}__"
            return group_name, raw_name, name_prefix
        return None, param_name, ""

    @staticmethod
    def _suffix_name(base: str, suffix: str, name_prefix: str) -> str:
        return f"{name_prefix}{base}_{suffix}"

    @staticmethod
    def _edit_current_distribution(current_distributions: List[Tuple[str,Dict[str, float]]]) -> List[Tuple[str,Dict[str, float]]]:
        for name, group in current_distributions:
            for param_name, val in list(group.items()):
                group_name, raw_name, name_prefix = OptunaOptimizer._split_group_prefix(param_name)

                if raw_name.endswith('_min'):
                    base = raw_name[:-4]
                    max_name = OptunaOptimizer._suffix_name(base, 'max', name_prefix)
                    if max_name in group and isinstance(group[max_name], numbers.Real):
                        max_val = group.pop(max_name)
                        delta = max_val - val
                        delta_name = OptunaOptimizer._suffix_name(base, 'delta', name_prefix)
                        group[delta_name] = delta
        return current_distributions

    def _delta_to_max(self, suggestions: List[Tuple[str,Dict[str, float]]]) -> List[Tuple[str,Dict[str, float]]]:
        for name, group in suggestions:
            for param_name, val in list(group.items()):
                group_name, raw_name, name_prefix = self._split_group_prefix(param_name)

                if raw_name.endswith('_min'):
                    base = raw_name[:-4]
                    delta_name = self._suffix_name(base, 'delta', name_prefix)
                    if delta_name in group and isinstance(group[delta_name], numbers.Real):
                        delta_val = group.pop(delta_name)
                        max_value = val + delta_val
                        max_name = self._suffix_name(base, 'max', name_prefix)
                        min_param_name = f"{base}_min"
                        max_param_name = f"{base}_max"

                        min_is_int = (group_name and group_name in self.param_type and
                                     min_param_name in self.param_type[group_name] and
                                     self.param_type[group_name][min_param_name] == "int")
                        max_is_int = (group_name and group_name in self.max_type_storage and
                                     max_param_name in self.max_type_storage[group_name] and
                                     self.max_type_storage[group_name][max_param_name] == "int")

                        if min_is_int:
                            val = int(round(val))
                            group[param_name] = val

                        if max_is_int:
                            max_value = int(round(max_value))

                        group[max_name] = max_value
                        if group_name and group_name in self.max_bound_storage and max_param_name in self.max_bound_storage[group_name]:
                            max_bounds = self.max_bound_storage[group_name][max_param_name]

                            if max_value > max_bounds[-1]:
                                group[max_name] = int(round(max_bounds[-1])) if max_is_int else max_bounds[-1]
                            elif max_value < max_bounds[0]:
                                group[max_name] = int(round(max_bounds[0])) if max_is_int else max_bounds[0]
        return suggestions

    def suggest_next_distributions(
        self,
        current_distributions: List[Tuple[str, Dict]],
        metrics_list: List[Dict[str, float]],
        config: Dict
    ) -> List[Tuple[str, Dict]]:
        """
        Register current results and suggest next distribution specifications.

        This is the main optimization interface. Each call:
        1. Registers current distributions and their metrics with Optuna via add_trial()
        2. Asks Optuna for the next batch of suggestions

        The optimizer is agnostic to data source - it only sees distributions and metrics.
        Works uniformly for all iterations (initial external data or optimizer-suggested).

        Args:
            current_distributions: List of (dist_id, params_dict) where params_dict
                                   contains shape_logit_* and {shape}__{param} keys
            metrics_list: List of metric dicts corresponding to current_distributions
            config: Experiment configuration dict

        Returns:
            suggestions: List of (dist_id, params_dict) tuples for next iteration
        """
        if len(current_distributions) != len(metrics_list):
            raise ValueError(
                f"Mismatch: {len(current_distributions)} distributions but "
                f"{len(metrics_list)} metric dicts"
            )

        current_distributions = self._edit_current_distribution(current_distributions)

        # Expand logit bounds to accommodate actual data range with safety margin
        self._expand_logit_bounds_from_data(current_distributions)

        # Build full distributions once (same for all trials in joint mode)
        distributions = self._build_full_distributions()

        # Tell: Register current results with Optuna using add_trial()
        print(f"  Registering {len(current_distributions)} results with Optuna...")

        for idx, ((dist_id, params), metrics) in enumerate(zip(current_distributions, metrics_list)):
            # Get metric values
            trial_values = [metrics[metric_name] for metric_name in self.optimization_metrics]
            # Clamp params to distribution bounds (real data may have values slightly outside
            # the optimizer's inferred bounds, e.g. delta=0.0 when low=epsilon)
            clamped_params = {}
            for key, val in params.items():
                dist = distributions.get(key)
                if dist is not None and hasattr(dist, 'low') and hasattr(dist, 'high'):
                    val = max(dist.low, min(dist.high, val))
                clamped_params[key] = val
            # Create and add the completed trial
            trial = optuna.trial.create_trial(
                params=clamped_params,
                distributions=distributions,
                values=trial_values,
                state=optuna.trial.TrialState.COMPLETE
            )
            self.study.add_trial(trial)

            # Log probabilities for readability
            probs = self.logits_to_probabilities(params)
            probs_str = ', '.join([f"{k}={v:.2%}" for k, v in sorted(probs.items())])
            metrics_str = ', '.join([f"{name}={metrics[name]:.4f}"
                                    for name in self.optimization_metrics])
            print(f"    [{idx + 1}/{len(current_distributions)}] {dist_id}: {metrics_str} | {probs_str}")

        # Ask: Get next batch of suggestions
        n_distributions = config.get('iteration_batch_size', 8)
        suggestions = []
        self.pending_trials = []

        # Get current trial count for generating dist_ids
        completed_count = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        print(f"\n  Suggesting {n_distributions} distributions for next iteration...")

        for i in range(n_distributions):
            trial = self.study.ask()
            params_with_logits = self._define_joint_search_space(trial)
            dist_id = f"dist_{completed_count + i}"

            # Convert logits to probabilities for output only
            params_with_probs = self.convert_logits_to_probs_in_params(params_with_logits)
            suggestions.append((dist_id, params_with_probs))
            self.pending_trials.append(trial)
            # Log suggestion with probabilities
            probs = self.logits_to_probabilities(params_with_logits)
            probs_str = ', '.join([f"{k}={v:.2%}" for k, v in sorted(probs.items())])

        pareto_size = len(self.get_pareto_front())
        print(f"  Total completed trials: {completed_count}")
        print(f"  Current Pareto front size: {pareto_size}")

        suggestions = self._delta_to_max(suggestions)
        return suggestions
