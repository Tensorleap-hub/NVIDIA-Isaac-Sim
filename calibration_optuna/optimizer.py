"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.
"""

import math
import optuna
from scipy.stats.qmc import Sobol
from typing import Dict, List, Optional, Tuple


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

    """

    def __init__(
        self,
        config: Dict,
        param_bounds: Dict[str, Dict],
        param_type: Dict[str, Dict[str, str]],
        logit_bounds: Tuple[float, float] = (-2.0, 2.0)
    ):
        """
        Initialize Optuna optimizer.

        Args:
            config: Experiment configuration dict with optimization_metrics, etc.
            param_bounds: Dict mapping simulation names to their parameter bounds
                          e.g., {'simulation_1': {'void_count_mean': [1.0, 10.0], ...}}
                          Simulation names (group_names) are inferred from the keys
            param_type: Dict mapping simulation names to parameter type strings
            logit_bounds: Min/max bounds for shape logits (default: -2.0 to 2.0).
                Mix-weight prior: tighter bounds bias the uniform-on-the-cube
                Sobol startup and TPE proposals toward equal softmax weights.
                (-2, 2) allows max softmax ratio of e^4 ~ 55x — enough to
                express any realistic per-sim skew (e.g., 98%/1%/1%) while
                keeping the bulk of the search volume near uniform mix.
        """
        self.config = config
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

        self.param_bounds = param_bounds
        self.param_type = param_type

        # Infer group names from param_bounds keys (sorted for deterministic order)
        self.group_names = sorted(param_bounds.keys())

        # Get optimization metrics from config
        self.optimization_metrics = config.get('optimization_metrics', ['mmd_rbf', 'mean_nn_distance'])
        n_objectives = len(self.optimization_metrics)

        study_name = config.get('experiment_name', 'optuna_study')

        # Get optimizer config
        optimizer_config = config.get('optimizer', {})
        multivariate = optimizer_config.get('multivariate', True)
        group = optimizer_config.get('group', True)
        constant_liar = optimizer_config.get('constant_liar', True)

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
                group=group,
                constant_liar=constant_liar,
                warn_independent_sampling=False
            )
        )

        self.n_startup_trials = n_startup_trials
        self.random_seed = config.get('random_seed', 42)

        # Track distributions asked-but-not-yet-told. Keyed by dist_id.
        # Value is (Optional[Trial], params_with_logits):
        #   Trial present  → TPE-asked, complete via study.tell()
        #   Trial is None  → Sobol-asked, complete via add_trial(create_trial(...))
        self._asked: Dict[str, Tuple[Optional[optuna.Trial], Dict]] = {}

        print(f"Initialized OptunaOptimizer (joint mode)")
        print(f"  Study name: {study_name}")
        print(f"  Objectives: {n_objectives} ({', '.join(self.optimization_metrics)})")
        print(f"  Logit bounds: {logit_bounds}")
        print(f"  TPE startup trials: {n_startup_trials}")
        print(f"  TPE multivariate: {multivariate}")
        print(f"  TPE group: {group}")
        print(f"  TPE constant_liar: {constant_liar}")
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
            Dict with simulation_logit_* keys and {shape}__{param} keys
        """
        params = {}

        # 1. Shape logits (converted to probs downstream via softmax)
        for group_name in self.group_names:
            logit = trial.suggest_float(
                f'simulation_logit_{group_name}',
                self.logit_bounds[0],
                self.logit_bounds[1]
            )
            params[f'simulation_logit_{group_name}'] = logit

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
                if param_name.startswith('simulation_logit_'):
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
            distributions[f'simulation_logit_{group_name}'] = optuna.distributions.FloatDistribution(
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
            Dict with simulation_logit_* keys
            e.g., {'simulation_logit_circle': -0.22, 'simulation_logit_ellipse': -0.44, ...}
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
            logits[f'simulation_logit_{shape}'] = math.log(prob)

        return logits

    @staticmethod
    def logits_to_probabilities(params: Dict) -> Dict[str, float]:
        """
        Convert shape logits in params dict to probabilities via softmax.

        Args:
            params: Dict containing simulation_logit_* keys

        Returns:
            Dict mapping shape names to probabilities (sum to 1.0)
        """
        # Extract logits
        logit_prefix = 'simulation_logit_'
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
        Convert simulation_logit_* keys to simulation_prob_* keys with softmax probabilities.

        This is for output formatting only - internally the optimizer still uses logits.

        Args:
            params: Dict with simulation_logit_* and other parameter keys

        Returns:
            New dict with simulation_prob_* instead of simulation_logit_*, plus all other params
        """
        # Extract logits and compute probabilities
        logit_prefix = 'simulation_logit_'
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
            probs = {f'simulation_prob_{k}': v / total for k, v in exp_logits.items()}
        else:
            probs = {}

        # Return combined dict with probabilities + other params
        return {**probs, **other_params}

    def _sobol_params(self, offset: int, n_points: int) -> List[Dict]:
        param_specs = []
        for group_name in self.group_names:
            param_specs.append((f'simulation_logit_{group_name}', self.logit_bounds, 'float'))
        for group_name in self.group_names:
            for param_name, bounds in self.param_bounds[group_name].items():
                param_specs.append((f'{group_name}__{param_name}', bounds, self.param_type[group_name][param_name]))

        sampler = Sobol(d=len(param_specs), scramble=True, seed=self.random_seed)
        if offset > 0:
            sampler.fast_forward(offset)
        unit_samples = sampler.random(n_points)

        results = []
        for sample in unit_samples:
            params = {}
            for j, (name, bounds, ptype) in enumerate(param_specs):
                u = float(sample[j])
                if ptype == 'categorical':
                    idx = min(int(u * len(bounds)), len(bounds) - 1)
                    params[name] = bounds[idx]
                elif ptype == 'int':
                    low, high = int(bounds[0]), int(bounds[1])
                    params[name] = min(int(low + u * (high - low + 1)), high)
                else:
                    low, high = float(bounds[0]), float(bounds[1])
                    params[name] = low + u * (high - low)
            results.append(params)
        return results

    @staticmethod
    def _clamp(params: Dict, distributions: Dict) -> Dict:
        clamped = {}
        for key, val in params.items():
            dist = distributions.get(key)
            if dist is not None and hasattr(dist, 'low') and hasattr(dist, 'high'):
                val = max(dist.low, min(dist.high, val))
            clamped[key] = val
        return clamped

    def replay(
        self,
        distributions: List[Tuple[str, Dict]],
        metrics_list: List[Dict[str, float]],
    ) -> None:
        if len(distributions) != len(metrics_list):
            raise ValueError(
                f"Mismatch: {len(distributions)} distributions but "
                f"{len(metrics_list)} metric dicts"
            )
        self._expand_logit_bounds_from_data(distributions)
        full_distributions = self._build_full_distributions()
        for (_dist_id, params), metrics in zip(distributions, metrics_list):
            values = [metrics[name] for name in self.optimization_metrics]
            clamped = self._clamp(params, full_distributions)
            self.study.add_trial(optuna.trial.create_trial(
                params=clamped,
                distributions=full_distributions,
                values=values,
                state=optuna.trial.TrialState.COMPLETE,
            ))

    def ask(self, n: int) -> List[Tuple[str, Dict]]:
        # Advance both the Sobol offset and the dist_id off of *asked* trials
        # (COMPLETE + FAIL + any pending _asked entry), not off of COMPLETE
        # trials alone. If we counted only completions, a mark_failed() call
        # would leave the counter frozen — the next ask() would re-draw the
        # same Sobol positions and re-use the same dist_ids as the previous
        # batch, causing wasted re-evaluations and duplicate COMPLETE trials
        # in the study once the re-drawn point gets tell()'d successfully.
        asked_count = len(self.study.trials) + len(self._asked)
        in_startup = asked_count < self.n_startup_trials
        suggestions: List[Tuple[str, Dict]] = []
        if in_startup:
            sobol_points = self._sobol_params(offset=asked_count, n_points=n)
            for i, params_with_logits in enumerate(sobol_points):
                dist_id = f"dist_{asked_count + i}"
                self._asked[dist_id] = (None, params_with_logits)
                suggestions.append(
                    (dist_id, self.convert_logits_to_probs_in_params(params_with_logits))
                )
        else:
            for i in range(n):
                trial = self.study.ask()
                params_with_logits = self._define_joint_search_space(trial)
                dist_id = f"dist_{asked_count + i}"
                self._asked[dist_id] = (trial, params_with_logits)
                suggestions.append(
                    (dist_id, self.convert_logits_to_probs_in_params(params_with_logits))
                )
        return suggestions

    def tell(self, dist_id: str, score: float) -> None:
        if dist_id not in self._asked:
            raise KeyError(f"No pending ask for dist_id={dist_id!r}")
        trial_or_none, params_with_logits = self._asked.pop(dist_id)
        if trial_or_none is not None:
            self.study.tell(trial_or_none, score)
        else:
            full_distributions = self._build_full_distributions()
            clamped = self._clamp(params_with_logits, full_distributions)
            self.study.add_trial(optuna.trial.create_trial(
                params=clamped,
                distributions=full_distributions,
                values=[score],
                state=optuna.trial.TrialState.COMPLETE,
            ))

    def mark_failed(self, dist_id: str) -> None:
        # Drop a trial from TPE's posterior without polluting it with a
        # noise-spike MMD. Optuna's TPE skips FAIL-state trials when
        # fitting its kernel density estimate; the ask/tell bookkeeping
        # still completes so batched proposals stay aligned.
        if dist_id not in self._asked:
            raise KeyError(f"No pending ask for dist_id={dist_id!r}")
        trial_or_none, params_with_logits = self._asked.pop(dist_id)
        if trial_or_none is not None:
            self.study.tell(trial_or_none, state=optuna.trial.TrialState.FAIL)
        else:
            full_distributions = self._build_full_distributions()
            clamped = self._clamp(params_with_logits, full_distributions)
            self.study.add_trial(optuna.trial.create_trial(
                params=clamped,
                distributions=full_distributions,
                values=None,
                state=optuna.trial.TrialState.FAIL,
            ))

    def suggest_next_distributions(
        self,
        current_distributions: List[Tuple[str, Dict]],
        metrics_list: List[Dict[str, float]],
        config: Dict,
    ) -> List[Tuple[str, Dict]]:
        n = config.get('iteration_batch_size', 8)
        self.replay(current_distributions, metrics_list)
        return self.ask(n)
