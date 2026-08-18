"""Local ask/tell adapter around the engine's OptunaOptimizer + MMDCalculator.

The engine's calibration optimizer (Sobol startup + TPE) is driven with the
synchronous ``ask() -> generate -> tell()/mark_failed()`` flow. Each
distribution's synthetic embeddings are scored against a fixed real reference
via the shared :class:`MMDCalculator`, so the objective computed here is
byte-for-byte the same one the engine's in-process CalibrationLoop uses.

This module is local orchestration glue — it has no engine counterpart. The
optimizer/metrics it wraps are vendored verbatim from the engine.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import MMDCalculator
from .optimizer import OptunaOptimizer


class ExperimentRunner:
    """Own the Optuna study + the real-reference MMD objective.

    Exposes the engine optimizer's ``ask``/``tell``/``mark_failed``/``replay``
    surface plus a ``score`` helper so the controller can run one iteration as
    ``ask -> generate -> score -> tell/mark_failed`` and inject seed/pool
    observations via ``replay``.
    """

    def __init__(
        self,
        config: Dict,
        param_bounds: Dict[str, Dict],
        param_type: Dict[str, Dict[str, str]],
    ):
        self.config = config
        self.optimizer = OptunaOptimizer(
            config=config,
            param_bounds=param_bounds,
            param_type=param_type,
        )
        self.param_type = param_type
        self.mmd_calc: Optional[MMDCalculator] = None

    def set_real_embeddings(self, real_embeddings: np.ndarray) -> None:
        """Set the fixed real reference and precompute its multi-bandwidth gammas."""
        self.mmd_calc = MMDCalculator(
            real_embeddings,
            max_samples=self.config.get("mmd_max_samples", 2000),
            seed=self.config.get("random_seed", 42),
        )
        print(f"Set real embeddings: {real_embeddings.shape}")

    def score(self, synth_embeddings: np.ndarray) -> float:
        """Multi-bandwidth RBF MMD between synthetic embeddings and the real reference."""
        if self.mmd_calc is None:
            raise ValueError("Real embeddings not set. Call set_real_embeddings() first.")
        return float(self.mmd_calc.mmd_rbf(synth_embeddings))

    def ask(self, n: int) -> List[Tuple[str, Dict]]:
        """Draw the next batch of distributions (Sobol during startup, then TPE)."""
        return self.optimizer.ask(n)

    def has_pending(self, dist_id: str) -> bool:
        """True if ``dist_id`` was issued by ``ask()`` this session and not yet completed."""
        return dist_id in self.optimizer._asked

    def tell(self, dist_id: str, score: float) -> None:
        """Complete an asked distribution with its measured objective."""
        self.optimizer.tell(dist_id, score)

    def mark_failed(self, dist_id: str) -> None:
        """Drop an asked distribution from TPE's posterior (e.g. a noise-spike outlier)."""
        self.optimizer.mark_failed(dist_id)

    def replay(
        self,
        distributions: List[Tuple[str, Dict]],
        metrics_list: List[Dict[str, float]],
    ) -> None:
        """Import external observations (seeds, pool priming, resume) as completed trials."""
        self.optimizer.replay(distributions, metrics_list)

    def get_best_trials(self, top_n: Optional[int] = None) -> List[Tuple[str, Dict]]:
        """Best completed trials as (trial_id, params-with-probs), best objective first."""
        return self.optimizer.get_best_trials_as_distributions(top_n=top_n)
