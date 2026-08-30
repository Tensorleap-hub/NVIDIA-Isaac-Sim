from enum import Enum
from typing import Optional


class StopReason(Enum):
    MAX_ITERS = "max_iters"
    STALE = "stale"


class ConvergenceTracker:
    """
    Objective-space convergence detector.

    Stops the loop when successive best-MMD improvements fall below
    `min_abs_improvement` — expected to be one standard deviation of a
    single MMD measurement, derived from the pre-flight K_var estimate
    as `sqrt(K_var / n_samples_total)`. That ties "further gains are
    indistinguishable from sampling noise" directly to the objective's
    measurement precision rather than to a fixed relative threshold.
    """

    def __init__(
        self,
        max_iters: int,
        stale_window: int,
        min_abs_improvement: float,
        min_iters_before_stale: int = 0,
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if stale_window < 1:
            raise ValueError("stale_window must be >= 1")
        if min_abs_improvement < 0:
            raise ValueError("min_abs_improvement must be >= 0")
        if min_iters_before_stale < 0:
            raise ValueError("min_iters_before_stale must be >= 0")
        self.max_iters = max_iters
        self.stale_window = stale_window
        self.min_abs_improvement = min_abs_improvement
        self.min_iters_before_stale = min_iters_before_stale
        self._iter_count = 0
        self._stale_count = 0
        self._best: Optional[float] = None
        self.stop_reason: Optional[StopReason] = None

    @property
    def converged(self) -> bool:
        return self.stop_reason == StopReason.STALE

    def update(self, best_mmd_so_far: float) -> Optional[StopReason]:
        if self.stop_reason is not None:
            return self.stop_reason
        self._iter_count += 1

        if self._best is None:
            self._best = best_mmd_so_far
        else:
            abs_imp = self._best - best_mmd_so_far
            if abs_imp > self.min_abs_improvement:
                self._best = best_mmd_so_far
                self._stale_count = 0
            else:
                self._stale_count += 1

        if self._iter_count >= self.min_iters_before_stale:
            if self._stale_count >= self.stale_window:
                self.stop_reason = StopReason.STALE
                return StopReason.STALE
        if self._iter_count >= self.max_iters:
            self.stop_reason = StopReason.MAX_ITERS
            return StopReason.MAX_ITERS
        return None
