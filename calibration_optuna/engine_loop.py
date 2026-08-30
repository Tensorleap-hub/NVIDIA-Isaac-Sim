# Vendored from engine src_tensorleap/trainer/ds_curation/calibration_optuna/loop.py.
# Deviations from the engine source, kept to the minimum needed to run standalone:
# - contract dataclasses come from .contracts instead of src_tensorleap.contract
# - LatentSpaceType / OPTUNA_CALIBRATION_LS_TYPE dropped (only the engine's
#   dispatcher used it as a dict key; local dispatchers return arrays directly)
# - leaplogger replaced by a stdlib-logging adapter with the same call shape
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_CONFIG
from .contracts import SimulationInstance, SingleSimulationData
from .convergence import (
    ConvergenceTracker,
    StopReason,
)
from .optimizer import OptunaOptimizer
from .sample_dispatcher import SampleDispatcher


_logger = logging.getLogger(__name__)


class _ExtraLogger:
    @staticmethod
    def info(msg, extra=None):
        _logger.info("%s %s", msg, extra or {})

    @staticmethod
    def warning(msg, extra=None):
        _logger.warning("%s %s", msg, extra or {})


leaplogger = _ExtraLogger()

_CALIBRATION_SEED: int = 42

_OUTLIER_MAD_K: float = 5.0
_OUTLIER_HISTORY: int = 30
_OUTLIER_MIN_HISTORY: int = 12

# Dummy-ping auto-sizing of n_samples_total.
# See project_optuna_calibration_backlog memory for the theory: noise floor
# on MMD scales as sqrt(K_var / n), so n = K_var * R^2 / M_ping^2 picks the
# sample count needed for the noise floor to sit a factor R below the ping
# trial's MMD (which anchors the problem's natural MMD scale).
_AUTO_N_PING: int = 128
_AUTO_HEADROOM_R: float = 5.0
_AUTO_K_VAR_PAIRS: int = 10
_AUTO_K_VAR_SUBSAMPLE: int = 50
_AUTO_N_SAMPLES_MIN: int = 32
_AUTO_N_SAMPLES_MAX: int = 1024
_AUTO_N_SAMPLES_FALLBACK: int = 64

# Multiplier on the K_var-derived noise floor when deciding "further gains
# are indistinguishable from measurement noise." 1.0 = one std of a single
# MMD measurement — an improvement smaller than that is inside the noise
# band and doesn't count as progress.
_NOISE_FLOOR_STOP_C: float = 1.0
# Absolute fallback used when K_var is 0 (degenerate real set) or n_samples
# is 0, so the loop still terminates.
_MIN_ABS_IMPROVEMENT_FALLBACK: float = 1e-4
_MMD_MAX_SAMPLES: int = 2000

_STALE_WINDOW_TRIALS: int = 20
_STARTUP_PER_DIM: int = 11
_BATCH_MIN: int = 2
_BATCH_MAX: int = 8
_N_STARTUP_MIN: int = 10
_MAX_ITERS_MIN_TAIL: int = 15
_CATEGORICAL_SMALL_THRESHOLD: int = 4


@dataclass
class TrialRecord:
    dist_id: str
    sim_data: List[SingleSimulationData]
    sample_ids: List[str]
    ls_vectors: np.ndarray
    mmd: float
    iteration_idx: int = 0
    trial_idx: int = 0


@dataclass
class CalibrationResult:
    all_sample_ids: List[str] = field(default_factory=list)
    best_trial_records: List[TrialRecord] = field(default_factory=list)
    all_trial_records: List[TrialRecord] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0
    stop_reason: Optional[str] = None
    noise_floor: Optional[float] = None
    n_real: Optional[int] = None
    n_samples_total: Optional[int] = None


_TYPE_TO_OPTUNA = {
    'float': 'float',
    'int': 'int',
    'string': 'categorical',
}


def _stable_seed(seed_base: int, dist_id: str, sim_name: str) -> int:
    digest = hashlib.md5(f"{seed_base}|{dist_id}|{sim_name}".encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') & 0x7FFFFFFF


def _is_mmd_outlier(score: float, history: List[float]) -> bool:
    # Tukey-style threshold on the median absolute deviation. Robust to the
    # outliers themselves; only fires once we have enough trials to estimate
    # a stable median.
    if len(history) < _OUTLIER_MIN_HISTORY:
        return False
    arr = np.asarray(history, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad <= 0.0:
        return False
    return score > median + _OUTLIER_MAD_K * mad


def _estimate_k_var(
    mmd_calc: "MMDCalculator",
    real: np.ndarray,
    n_pairs: int = _AUTO_K_VAR_PAIRS,
    subsample: int = _AUTO_K_VAR_SUBSAMPLE,
    seed: int = _CALIBRATION_SEED,
) -> float:
    # K_var: the proportionality constant relating MMD's noise floor to
    # sample count via  noise_floor_in_MMD  ~=  sqrt(K_var / n).
    #
    # Derivation: under H0 (identical distributions), Var(MMD^2) scales as
    # K_var^2 / n^2 (Gretton 2012 Thm 7). Equivalently, std(MMD^2) ~= K_var / n
    # so K_var ~= std(MMD^2_pairs) * n0 when MMD^2 is measured between
    # subsamples of size n0 drawn from the same distribution.
    #
    # We must use the *signed* unbiased MMD^2 here — DistributionMetrics.mmd
    # clips negative unbiased MMD^2 to 0 (correct for a distance report;
    # wrong for a noise-distribution measurement, since it collapses the
    # negative half of the H0 noise to a spike at 0 and biases std downward).
    #
    # Downstream use: n_samples_total = K_var * R^2 / M_ping^2.
    if real.size == 0 or len(real) < 2 * subsample:
        return 0.0
    if mmd_calc.gammas is None:
        return 0.0
    from .metrics import DistributionMetrics
    gammas = list(mmd_calc.gammas)
    rng = np.random.RandomState(seed)
    mmd2_values: List[float] = []
    for _ in range(n_pairs):
        idx = rng.choice(len(real), size=2 * subsample, replace=False)
        a = real[idx[:subsample]]
        b = real[idx[subsample:]]
        mmd2_values.append(
            DistributionMetrics.mmd_squared_signed(a, b, kernel='rbf', gamma=gammas)
        )
    if not mmd2_values:
        return 0.0
    return float(np.std(mmd2_values, ddof=0) * subsample)


def _effective_dimensionality(simulations: List[SimulationInstance]) -> float:
    d = 0.0
    for sim in simulations:
        if not sim.sim_config:
            continue
        for spec in sim.sim_config.values():
            raw_type = spec['type']
            if raw_type in ('float', 'int'):
                d += 1.0
            elif raw_type == 'string':
                n_values = len(spec['bounds'].get('values', []))
                if n_values <= 1:
                    continue
                if n_values <= _CATEGORICAL_SMALL_THRESHOLD:
                    d += 1.0
                else:
                    d += math.log2(n_values)
    return max(1.0, d)


def calibration_budget(simulations: List[SimulationInstance]) -> Dict[str, int]:
    d = _effective_dimensionality(simulations)
    n_startup_trials = max(_N_STARTUP_MIN, int(round(_STARTUP_PER_DIM * d)))
    batch_size = min(_BATCH_MAX, max(_BATCH_MIN, int(round(d))))
    warmup_iters = math.ceil(n_startup_trials / batch_size)
    stale_window = max(2, math.ceil(_STALE_WINDOW_TRIALS / batch_size))
    min_iters_before_stale = warmup_iters + stale_window
    max_iters = min_iters_before_stale + max(_MAX_ITERS_MIN_TAIL, 2 * warmup_iters)
    return {
        "n_startup_trials": int(n_startup_trials),
        "batch_size": int(batch_size),
        "warmup_iters": int(warmup_iters),
        "stale_window": int(stale_window),
        "max_iters": int(max_iters),
        "min_iters_before_stale": int(min_iters_before_stale),
    }


def simulations_to_param_specs(
    simulations: List[SimulationInstance],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    param_bounds: Dict[str, Dict[str, Any]] = {}
    param_type: Dict[str, Dict[str, str]] = {}
    for sim in simulations:
        if not sim.sim_config:
            raise ValueError(
                f"Simulation '{sim.name}' has empty sim_config; "
                f"engine cannot construct a search space without per-param type+bounds."
            )
        bounds_for_sim: Dict[str, Any] = {}
        types_for_sim: Dict[str, str] = {}
        for param_name, spec in sim.sim_config.items():
            raw_type = spec['type']
            if raw_type not in _TYPE_TO_OPTUNA:
                raise ValueError(
                    f"Simulation '{sim.name}' param '{param_name}' has unsupported "
                    f"type '{raw_type}' (expected one of {list(_TYPE_TO_OPTUNA)})"
                )
            types_for_sim[param_name] = _TYPE_TO_OPTUNA[raw_type]
            bounds = spec['bounds']
            if raw_type == 'string':
                bounds_for_sim[param_name] = list(bounds['values'])
            else:
                bounds_for_sim[param_name] = [bounds['min'], bounds['max']]
        param_bounds[sim.name] = bounds_for_sim
        param_type[sim.name] = types_for_sim
    return param_bounds, param_type


class CalibrationLoop:
    def __init__(
        self,
        simulations: List[SimulationInstance],
        sample_dispatcher: SampleDispatcher,
        real_embeddings: np.ndarray,
        n_samples_total: Optional[int] = None,
        batch_size: Optional[int] = None,
        top_k_to_persist: int = 5,
        n_startup_trials: Optional[int] = None,
        max_iters: Optional[int] = None,
        stale_window: Optional[int] = None,
        min_iters_before_stale: Optional[int] = None,
    ):
        if not simulations:
            raise ValueError("simulations list is empty")
        from .metrics import MMDCalculator
        self.simulations = simulations
        self.sample_dispatcher = sample_dispatcher
        self.mmd_calc = MMDCalculator(
            real_embeddings, max_samples=_MMD_MAX_SAMPLES, seed=_CALIBRATION_SEED,
        )
        self._n_samples_total_override = n_samples_total
        # Set to the override now so any callers that read self.n_samples_total
        # before run() (e.g. tests) get a sane value. Auto-sizing happens in
        # run() once we can actually dispatch a ping trial.
        self.n_samples_total = (
            n_samples_total if n_samples_total is not None else _AUTO_N_SAMPLES_FALLBACK
        )

        budget = calibration_budget(simulations)
        self.batch_size = batch_size if batch_size is not None else budget["batch_size"]
        n_startup_trials_eff = (
            n_startup_trials if n_startup_trials is not None else budget["n_startup_trials"]
        )
        max_iters_eff = max_iters if max_iters is not None else budget["max_iters"]
        stale_window_eff = stale_window if stale_window is not None else budget["stale_window"]
        min_iters_before_stale_eff = (
            min_iters_before_stale
            if min_iters_before_stale is not None
            else budget["min_iters_before_stale"]
        )
        self.top_k_to_persist = top_k_to_persist
        self.budget = budget

        param_bounds, param_type = simulations_to_param_specs(simulations)
        config = dict(DEFAULT_CONFIG)
        config['random_seed'] = _CALIBRATION_SEED
        config['experiment_name'] = f"engine_calibration"
        optimizer_cfg = dict(config.get('optimizer', {}))
        optimizer_cfg['n_startup_trials'] = n_startup_trials_eff
        config['optimizer'] = optimizer_cfg
        self.optimizer = OptunaOptimizer(
            config=config,
            param_bounds=param_bounds,
            param_type=param_type,
        )
        # min_abs_improvement is set from the K_var / n_samples_total noise
        # floor at the start of run() — both inputs may not be known here
        # yet (auto-sizing may kick in during run()). Seed with fallback so
        # the tracker is well-formed even if callers construct one and
        # never call run().
        self._tracker_max_iters = max_iters_eff
        self._tracker_stale_window = stale_window_eff
        self._tracker_min_iters_before_stale = min_iters_before_stale_eff
        self.tracker = ConvergenceTracker(
            max_iters=max_iters_eff,
            stale_window=stale_window_eff,
            min_abs_improvement=_MIN_ABS_IMPROVEMENT_FALLBACK,
            min_iters_before_stale=min_iters_before_stale_eff,
        )
        self._k_var: float = 0.0

    def _distribution_to_sim_data(
        self, dist_id: str, params: Dict[str, Any],
    ) -> List[SingleSimulationData]:
        min_samples = 2
        sim_data: List[SingleSimulationData] = []
        for sim in self.simulations:
            prob = params.get(f'simulation_prob_{sim.name}', 0.0)
            if prob <= 0.0:
                continue
            n_samples = max(min_samples, int(round(prob * self.n_samples_total)))
            sim_params = {
                key[len(f'{sim.name}__'):]: val
                for key, val in params.items()
                if key.startswith(f'{sim.name}__')
            }
            per_trial_seed = _stable_seed(_CALIBRATION_SEED, dist_id, sim.name)
            sim_data.append(SingleSimulationData(
                sim_name=sim.name,
                params=sim_params,
                n_samples=n_samples,
                seed=per_trial_seed,
            ))
        return sim_data

    def _mmd_rbf(self, synth_ls: np.ndarray) -> float:
        return self.mmd_calc.mmd_rbf(synth_ls)

    def _center_of_bounds_params(self, sim: SimulationInstance) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if not sim.sim_config:
            return params
        for pname, spec in sim.sim_config.items():
            raw_type = spec.get('type')
            bounds = spec.get('bounds', {}) or {}
            if raw_type == 'string':
                values = list(bounds.get('values', []))
                params[pname] = values[0] if values else ""
            elif raw_type == 'int':
                lo = int(bounds.get('min', 0))
                hi = int(bounds.get('max', lo))
                params[pname] = (lo + hi) // 2
            else:
                lo = float(bounds.get('min', 0.0))
                hi = float(bounds.get('max', lo))
                params[pname] = 0.5 * (lo + hi)
        return params

    def _auto_size_n_samples_total(self, leaplogger) -> int:
        try:
            real = self.mmd_calc.real
            if real is None or real.size == 0:
                leaplogger.info(
                    "Synthetic job auto n_samples: no real embeddings, using fallback",
                    extra={"n_samples_total": _AUTO_N_SAMPLES_FALLBACK})
                return _AUTO_N_SAMPLES_FALLBACK

            # K_var was estimated once at the top of run(); reuse the value
            # so the noise-floor stopping threshold and the auto-sizer see
            # the same measurement.
            k_var = self._k_var

            n_per_sim = max(2, _AUTO_N_PING // len(self.simulations))
            ping_sim_data: List[SingleSimulationData] = []
            for sim in self.simulations:
                params = self._center_of_bounds_params(sim)
                ping_sim_data.append(SingleSimulationData(
                    sim_name=sim.name,
                    params=params,
                    n_samples=n_per_sim,
                    seed=_stable_seed(_CALIBRATION_SEED, "ping", sim.name),
                ))
            ping_dist_id = "ping_0"
            sample_ids_by_dist = self.sample_dispatcher.generate({ping_dist_id: ping_sim_data})
            ls_by_dist = self.sample_dispatcher.collect_ls(sample_ids_by_dist)
            ping_ls = ls_by_dist.get(ping_dist_id, np.empty((0, 0)))
            if ping_ls.size == 0:
                leaplogger.info(
                    "Synthetic job auto n_samples: ping returned no embeddings, using fallback",
                    extra={"n_samples_total": _AUTO_N_SAMPLES_FALLBACK})
                return _AUTO_N_SAMPLES_FALLBACK

            m_ping = float(self._mmd_rbf(ping_ls))
            if not (m_ping > 0.0 and k_var > 0.0):
                leaplogger.info(
                    "Synthetic job auto n_samples: degenerate ping/K_var, using fallback",
                    extra={"k_var": k_var, "m_ping": m_ping,
                           "n_samples_total": _AUTO_N_SAMPLES_FALLBACK})
                return _AUTO_N_SAMPLES_FALLBACK

            target_floor = m_ping / _AUTO_HEADROOM_R
            raw_n = math.ceil(k_var / (target_floor * target_floor))
            n_samples = max(_AUTO_N_SAMPLES_MIN, min(_AUTO_N_SAMPLES_MAX, raw_n))
            leaplogger.info(
                "Synthetic job auto n_samples derived",
                extra={"k_var": k_var, "m_ping": m_ping,
                       "target_floor": target_floor, "raw_n": int(raw_n),
                       "n_samples_total": n_samples})
            return int(n_samples)
        except Exception as e:
            leaplogger.warning(
                "Synthetic job auto n_samples failed, using fallback",
                extra={"error": str(e), "n_samples_total": _AUTO_N_SAMPLES_FALLBACK})
            return _AUTO_N_SAMPLES_FALLBACK

    def run(self) -> CalibrationResult:
        # K_var is used both by the auto-sizer (if it runs) and by the
        # stopping criterion (always). Estimate it once here, up front.
        self._k_var = _estimate_k_var(self.mmd_calc, self.mmd_calc.real)

        if self._n_samples_total_override is None:
            self.n_samples_total = self._auto_size_n_samples_total(leaplogger)

        # Wire the stopping threshold to the objective's measurement noise:
        # noise_floor_on_MMD ~= sqrt(K_var / n_samples_total). We stop when
        # improvement over `stale_window` iterations is below `c` noise
        # stds of a single MMD measurement — i.e. indistinguishable from
        # sampling noise. Falls back to a small absolute for degenerate
        # K_var / n_samples so the loop still terminates.
        if self._k_var > 0.0 and self.n_samples_total > 0:
            noise_floor = math.sqrt(self._k_var / self.n_samples_total)
            min_abs_improvement = _NOISE_FLOOR_STOP_C * noise_floor
        else:
            noise_floor = 0.0
            min_abs_improvement = _MIN_ABS_IMPROVEMENT_FALLBACK
        self.tracker = ConvergenceTracker(
            max_iters=self._tracker_max_iters,
            stale_window=self._tracker_stale_window,
            min_abs_improvement=min_abs_improvement,
            min_iters_before_stale=self._tracker_min_iters_before_stale,
        )

        leaplogger.info(
            "Synthetic job calibration budget",
            extra={"d_eff": _effective_dimensionality(self.simulations),
                   "batch_size": self.batch_size,
                   "n_samples_total": self.n_samples_total,
                   "k_var": self._k_var,
                   "noise_floor_mmd": noise_floor,
                   "min_abs_improvement": min_abs_improvement,
                   "n_startup_trials": self.optimizer.n_startup_trials,
                   "max_iters": self.tracker.max_iters,
                   "stale_window": self.tracker.stale_window,
                   "min_iters_before_stale": self.tracker.min_iters_before_stale})

        all_trial_records: List[TrialRecord] = []
        recent_scores: List[float] = []
        outlier_count = 0
        iter_count = 0
        trial_counter = 0
        stop_reason: Optional[StopReason] = None
        while stop_reason is None:
            suggestions = self.optimizer.ask(self.batch_size)
            sim_data_by_dist: Dict[str, List[SingleSimulationData]] = {
                dist_id: self._distribution_to_sim_data(dist_id, params)
                for dist_id, params in suggestions
            }
            sample_ids_by_dist = self.sample_dispatcher.generate(sim_data_by_dist)
            ls_by_dist = self.sample_dispatcher.collect_ls(sample_ids_by_dist)

            iter_records: List[TrialRecord] = []
            for dist_id, params in suggestions:
                if not sample_ids_by_dist.get(dist_id):
                    self.optimizer.mark_failed(dist_id)
                    iter_records.append(TrialRecord(
                        dist_id=dist_id,
                        sim_data=sim_data_by_dist[dist_id],
                        sample_ids=[],
                        ls_vectors=np.empty((0, 0)),
                        mmd=float('inf'),
                        iteration_idx=iter_count,
                        trial_idx=trial_counter,
                    ))
                    trial_counter += 1
                    continue
                ls = ls_by_dist.get(dist_id, np.empty((0, 0)))
                score = self._mmd_rbf(ls)
                if _is_mmd_outlier(score, recent_scores):
                    self.optimizer.mark_failed(dist_id)
                    outlier_count += 1
                    leaplogger.info(
                        "Synthetic job calibration outlier suppressed",
                        extra={"dist_id": dist_id, "score": score,
                               "n_recent": len(recent_scores)})
                else:
                    self.optimizer.tell(dist_id, score)
                    recent_scores.append(score)
                    if len(recent_scores) > _OUTLIER_HISTORY:
                        recent_scores.pop(0)
                iter_records.append(TrialRecord(
                    dist_id=dist_id,
                    sim_data=sim_data_by_dist[dist_id],
                    sample_ids=sample_ids_by_dist.get(dist_id, []),
                    ls_vectors=ls,
                    mmd=score,
                    iteration_idx=iter_count,
                    trial_idx=trial_counter,
                ))
                trial_counter += 1

            iter_count += 1
            all_trial_records.extend(iter_records)
            # Bound both memory sites (worker inference cache + generic-pod
            # _synthetic_lookup) to the running top-K trials' samples; a trial's
            # MMD is fixed, so one that drops out of top-K never re-enters. The
            # top-K stay resident so eval can resolve + persist their LS.
            prune = getattr(self.sample_dispatcher, 'prune_caches', None)
            if prune is not None:
                top_k = sorted(all_trial_records, key=lambda r: r.mmd)[:self.top_k_to_persist]
                prune([sid for r in top_k for sid in r.sample_ids])
            best_mmd_this_iter = min((r.mmd for r in iter_records), default=float('inf'))
            best_mmd_so_far = min((r.mmd for r in all_trial_records), default=float('inf'))
            stop_reason = self.tracker.update(best_mmd_so_far)
            leaplogger.info(
                "Synthetic job calibration iteration",
                extra={"iter": iter_count,
                       "n_dists": len(iter_records),
                       "n_samples_this_iter": sum(len(r.sample_ids) for r in iter_records),
                       "best_mmd_this_iter": best_mmd_this_iter,
                       "best_mmd_so_far": best_mmd_so_far,
                       "stale_count": self.tracker._stale_count,
                       "outlier_count": outlier_count,
                       "stop_reason": stop_reason.value if stop_reason else None})

        return CalibrationResult(
            all_sample_ids=[sid for r in all_trial_records for sid in r.sample_ids],
            best_trial_records=sorted(all_trial_records, key=lambda r: r.mmd)[:self.top_k_to_persist],
            all_trial_records=all_trial_records,
            converged=self.tracker.converged,
            n_iterations=iter_count,
            stop_reason=stop_reason.value if stop_reason else None,
            noise_floor=(math.sqrt(self._k_var / self.n_samples_total)
                         if self._k_var and self.n_samples_total else None),
            n_real=(int(self.mmd_calc.real.shape[0])
                    if getattr(self.mmd_calc, "real", None) is not None else None),
            n_samples_total=int(self.n_samples_total) if self.n_samples_total else None,
        )
