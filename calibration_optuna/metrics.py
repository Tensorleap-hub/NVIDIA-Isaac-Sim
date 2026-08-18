"""
Metrics for evaluating distribution similarity and sample distances.

Includes MMD, Wasserstein distance, and per-sample distance metrics.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional, Sequence, Tuple, Union


# Multi-bandwidth MMD: rather than a single RBF bandwidth (which has to be
# guessed correctly per task), average MMD^2 over a set of bandwidths that
# are guaranteed to sit where the data actually lives — the sigmas are
# quantiles of the pair-distance distribution in the combined sample. This
# is scale-invariant and multi-modal-aware: it automatically adapts to LS
# geometry (tight clusters vs continuous manifolds vs cluster-of-clusters)
# without any user knobs, and guarantees no bandwidth is dead weight (each
# one sits at a distance-value that actually occurs in the data).
_QUANTILE_BANDWIDTH_LEVELS: Tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)


GammaSpec = Union[float, Sequence[float]]


class DistributionMetrics:
    """Calculate distribution similarity metrics"""

    @staticmethod
    def mmd_squared_signed(
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = 'rbf',
        gamma: Optional[GammaSpec] = None,
    ) -> float:
        """
        Unbiased MMD^2 estimator, un-clipped.

        Returns the raw MMD^2 including its natural negative values under
        H0 (identical distributions), where sampling noise routinely
        pushes the estimator below zero. Callers that report a distance
        clip this to zero and take the square root (see mmd()); callers
        that measure the *distribution* of MMD^2 under H0 (K_var noise
        estimation) MUST use this signed variant, otherwise the negative
        tail of the noise distribution collapses to a spike at 0 and
        std(MMD^2) is biased downward.
        """
        if kernel == 'rbf':
            if gamma is None:
                gammas = DistributionMetrics._compute_gammas_multi_bandwidth(X, Y)
            else:
                gammas = DistributionMetrics._as_gamma_list(gamma)
            XX = DistributionMetrics._rbf_kernel_multi(X, X, gammas)
            YY = DistributionMetrics._rbf_kernel_multi(Y, Y, gammas)
            XY = DistributionMetrics._rbf_kernel_multi(X, Y, gammas)
        elif kernel == 'linear':
            XX = X @ X.T
            YY = Y @ Y.T
            XY = X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Unbiased MMD^2 estimator: average over off-diagonal pairs only.
        # Including the diagonal (the biased estimator) adds a 1/n term to
        # XX.mean() / YY.mean(), which produces a small offset on the reported
        # MMD value. Excluding the diagonal removes that offset; the result
        # can be slightly negative for near-identical distributions.
        n_x, n_y = X.shape[0], Y.shape[0]
        if n_x > 1 and n_y > 1:
            xx_mean = (XX.sum() - np.trace(XX)) / (n_x * (n_x - 1))
            yy_mean = (YY.sum() - np.trace(YY)) / (n_y * (n_y - 1))
        else:
            xx_mean = XX.mean()
            yy_mean = YY.mean()
        return float(xx_mean + yy_mean - 2 * XY.mean())

    @staticmethod
    def mmd(
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = 'rbf',
        gamma: Optional[GammaSpec] = None,
    ) -> float:
        """
        Calculate Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            X: (N, D) array of samples from distribution 1
            Y: (M, D) array of samples from distribution 2
            kernel: Kernel type ('rbf' or 'linear')
            gamma: RBF kernel parameter. Accepts either a single float (single
                bandwidth) or a sequence of floats (multi-bandwidth — the
                kernel is the mean of single-bandwidth RBF kernels and the
                resulting MMD^2 is the mean of single-bandwidth MMD^2 values).
                If None, multi-bandwidth defaults are derived from the
                median-heuristic gamma.

        Returns:
            mmd_value: MMD distance (lower is better). Non-negative;
            slightly-negative unbiased MMD^2 values are clipped to 0.
        """
        return float(np.sqrt(max(DistributionMetrics.mmd_squared_signed(
            X, Y, kernel=kernel, gamma=gamma,
        ), 0.0)))

    @staticmethod
    def _as_gamma_list(gamma: GammaSpec) -> List[float]:
        if isinstance(gamma, (int, float)):
            return [float(gamma)]
        return [float(g) for g in gamma]

    @staticmethod
    def _compute_gamma_median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute gamma for RBF kernel using median heuristic.

        gamma = 1 / (2 * sigma^2), where sigma is the median pairwise distance.
        """
        from sklearn.metrics import pairwise_distances

        # Combine both distributions for computing median
        Z = np.vstack([X, Y])

        # Compute pairwise distances
        D = pairwise_distances(Z)

        # Get median of non-zero distances
        sigma = np.median(D[D > 0])

        # gamma = 1 / (2 * sigma^2)
        gamma = 1.0 / (2 * sigma ** 2)

        return gamma

    @staticmethod
    def _compute_gammas_multi_bandwidth(
        X: np.ndarray,
        Y: np.ndarray,
        quantile_levels: Sequence[float] = _QUANTILE_BANDWIDTH_LEVELS,
    ) -> List[float]:
        """
        Multi-bandwidth gammas at quantiles of the pair-distance distribution.

        For an RBF kernel, only bandwidths sitting near actual pair-distance
        values contribute discriminating signal: if sigma is much smaller
        than every pair distance the kernel matrix is ~identity (no signal);
        if sigma is much larger the kernel matrix is ~constant (also no
        signal). So we set sigma = quantile-of-pair-distances, which
        guarantees each bandwidth lands where some pairs actually live.

        This also adapts naturally to multi-modal distance distributions
        (e.g. classifier LS with well-separated clusters where pair
        distances cluster around within-cluster and between-cluster scales
        — quantiles land in both regions).

        Each gamma = 1 / (2 * sigma^2).
        """
        from sklearn.metrics import pairwise_distances
        Z = np.vstack([X, Y])
        D = pairwise_distances(Z)
        nz = D[D > 0]
        if nz.size == 0:
            # Degenerate: all points identical. Fall back to a single tiny
            # sigma so mmd() still returns a well-defined 0.
            return [1.0]
        sigmas = np.quantile(nz, list(quantile_levels))
        # Guard against sigma == 0 in case a quantile happens to hit a zero
        # entry (shouldn't after the D > 0 filter, but be defensive).
        sigmas = np.maximum(sigmas, 1e-12)
        return [float(1.0 / (2.0 * s * s)) for s in sigmas]

    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel between X and Y for a single bandwidth."""
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        distances_sq = X_norm + Y_norm - 2 * X @ Y.T
        return np.exp(-gamma * distances_sq)

    @staticmethod
    def _rbf_kernel_multi(
        X: np.ndarray, Y: np.ndarray, gammas: Sequence[float],
    ) -> np.ndarray:
        """
        Multi-bandwidth RBF kernel: mean of single-bandwidth RBF kernels.

        Distances are computed once; only the exponential is repeated per
        bandwidth.
        """
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        d2 = X_norm + Y_norm - 2 * X @ Y.T
        d2 = np.maximum(d2, 0.0)
        K = np.zeros_like(d2)
        for g in gammas:
            K += np.exp(-g * d2)
        return K / float(len(gammas))

    @staticmethod
    def wasserstein_1d(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate 1D Wasserstein distance (Earth Mover's Distance).

        For high-dimensional data, compute average over all dimensions.

        Args:
            X: (N, D) array of samples from distribution 1
            Y: (M, D) array of samples from distribution 2

        Returns:
            wasserstein: Average 1D Wasserstein distance across dimensions
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        distances = []
        for dim in range(X.shape[1]):
            dist = wasserstein_distance(X[:, dim], Y[:, dim])
            distances.append(dist)

        return float(np.mean(distances))


class MMDCalculator:
    def __init__(self, real_embeddings: np.ndarray, max_samples: int = 2000, seed: int = 42) -> None:
        if real_embeddings.size > 0 and len(real_embeddings) > max_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(real_embeddings), size=max_samples, replace=False)
            self.real = real_embeddings[idx]
        else:
            self.real = real_embeddings
        if self.real.size > 0:
            self.gammas: Optional[List[float]] = DistributionMetrics._compute_gammas_multi_bandwidth(
                self.real, self.real,
            )
        else:
            self.gammas = None

    @property
    def gamma(self) -> Optional[List[float]]:
        # Back-compat alias for callers that read `mmd_calc.gamma`. The list
        # is forwarded to DistributionMetrics.mmd() as the multi-bandwidth
        # spec.
        return self.gammas

    def mmd_rbf(self, synth_embeddings: np.ndarray) -> float:
        if synth_embeddings.size == 0 or self.real.size == 0 or self.gammas is None:
            return float('inf')
        return float(DistributionMetrics.mmd(
            synth_embeddings, self.real, kernel='rbf', gamma=self.gammas,
        ))


class SampleMetrics:
    """Calculate per-sample distance metrics"""

    @staticmethod
    def nearest_neighbor_distances(
        synthetic: np.ndarray,
        real: np.ndarray,
        metric: str = 'euclidean',
        bidirectional: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Calculate nearest neighbor distances between synthetic and real samples.

        Args:
            synthetic: (N, D) array of synthetic embeddings
            real: (M, D) array of real embeddings
            metric: Distance metric ('euclidean', 'cosine', etc.)
            bidirectional: If True, compute distances in both directions

        Returns:
            syn_to_real_distances: (N,) array of distances from synthetic to nearest real
            syn_to_real_indices: (N,) array of nearest real neighbor indices
            bidirectional_info: Dictionary with bidirectional metrics (if bidirectional=True)
        """
        # Compute pairwise distances
        dist_matrix = cdist(synthetic, real, metric=metric)

        # Synthetic -> Real (precision: are synthetic close to real?)
        syn_to_real_distances = dist_matrix.min(axis=1)
        syn_to_real_indices = dist_matrix.argmin(axis=1)

        bidirectional_info = {}

        if bidirectional:
            # Real -> Synthetic (recall: are real covered by synthetic?)
            real_to_syn_distances = dist_matrix.min(axis=0)
            real_to_syn_indices = dist_matrix.argmin(axis=0)

            # Bidirectional metrics
            bidirectional_info = {
                'syn_to_real_mean': float(syn_to_real_distances.mean()),
                'real_to_syn_mean': float(real_to_syn_distances.mean()),
                'max_nn_distance': float(max(syn_to_real_distances.mean(), real_to_syn_distances.mean())),
                'mean_nn_distance': float((syn_to_real_distances.mean() + real_to_syn_distances.mean()) / 2),
                'real_to_syn_max': float(real_to_syn_distances.max()),  # worst-case uncovered real
                'unique_real_neighbors': int(len(np.unique(syn_to_real_indices)))  # diversity check
            }

        return syn_to_real_distances, syn_to_real_indices, bidirectional_info

    @staticmethod
    def coverage(
        synthetic: np.ndarray,
        real: np.ndarray,
        threshold: float = None,
        metric: str = 'euclidean'
    ) -> Dict[str, float]:
        """
        Calculate coverage: fraction of synthetic samples within threshold of real.

        Args:
            synthetic: (N, D) array of synthetic embeddings
            real: (M, D) array of real embeddings
            threshold: Distance threshold (if None, uses median real-real distance)
            metric: Distance metric

        Returns:
            Dictionary with coverage metrics
        """
        # Calculate nearest neighbor distances
        nn_distances, _, _ = SampleMetrics.nearest_neighbor_distances(synthetic, real, metric, bidirectional=False)

        # If no threshold provided, use median of real-real distances
        if threshold is None:
            real_real_distances = cdist(real, real, metric=metric)
            # Exclude diagonal (self-distances)
            np.fill_diagonal(real_real_distances, np.inf)
            threshold = np.median(real_real_distances[np.isfinite(real_real_distances)])

        # Calculate coverage
        within_threshold = (nn_distances <= threshold).sum()
        coverage_ratio = within_threshold / len(synthetic)

        return {
            'coverage': float(coverage_ratio),
            'threshold': float(threshold),
            'mean_distance': float(nn_distances.mean()),
            'median_distance': float(np.median(nn_distances)),
            'within_threshold_count': int(within_threshold)
        }


def compute_all_metrics(
    synthetic_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    rbf_gamma: float = None
) -> Dict[str, float]:
    """
    Compute all metrics for synthetic vs real comparison.

    Args:
        synthetic_embeddings: (N, D) array
        real_embeddings: (M, D) array
        rbf_gamma: Pre-computed RBF gamma parameter. If None, uses median heuristic.

    Returns:
        Dictionary with all metric values
    """
    metrics = {}

    # Distribution-level metrics
    # MMD with pre-computed or auto-computed gamma
    metrics['mmd_rbf'] = DistributionMetrics.mmd(synthetic_embeddings, real_embeddings, kernel='rbf', gamma=rbf_gamma)
    metrics['mmd_linear'] = DistributionMetrics.mmd(synthetic_embeddings, real_embeddings, kernel='linear')
    metrics['wasserstein'] = DistributionMetrics.wasserstein_1d(synthetic_embeddings, real_embeddings)

    # Sample-level bidirectional metrics
    nn_distances, _, bidirectional_info = SampleMetrics.nearest_neighbor_distances(
        synthetic_embeddings, real_embeddings, bidirectional=True
    )

    # Add bidirectional NN metrics
    metrics.update(bidirectional_info)

    # Keep legacy metrics for backwards compatibility
    metrics['median_nn_distance'] = float(np.median(nn_distances))

    # Coverage
    coverage_info = SampleMetrics.coverage(synthetic_embeddings, real_embeddings)
    metrics.update(coverage_info)

    return metrics


def stratified_subsample_for_mmd(
    embeddings_by_source: List[np.ndarray],
    max_samples: int = 2000,
    seed: int = None
) -> np.ndarray:
    """
    Subsample embeddings while preserving source proportions.

    Args:
        embeddings_by_source: List of embedding arrays, one per source/simulation
        max_samples: Maximum total samples to keep
        seed: Random seed for reproducibility

    Returns:
        Subsampled embeddings with source proportions preserved
    """
    total_samples = sum(len(arr) for arr in embeddings_by_source)

    if total_samples <= max_samples:
        return np.concatenate(embeddings_by_source, axis=0)

    rng = np.random.RandomState(seed)
    subsampled_parts = []

    for arr in embeddings_by_source:
        source_proportion = len(arr) / total_samples
        n_samples_from_source = max(1, int(round(source_proportion * max_samples)))

        if len(arr) <= n_samples_from_source:
            subsampled_parts.append(arr)
        else:
            indices = rng.choice(len(arr), size=n_samples_from_source, replace=False)
            subsampled_parts.append(arr[indices])

    return np.concatenate(subsampled_parts, axis=0)


def compute_per_param_set_metrics(
    embeddings_by_simulation: List[np.ndarray],
    embeddings_indices_by_dist: Dict[int, List[Tuple[int, np.ndarray]]],
    real_embeddings: np.ndarray,
    n_param_sets: int,
    mmd_max_samples: int = 2000
) -> List[Dict[str, float]]:
    """
    Compute metrics for each distribution separately using on-demand embedding fetching.
    Uses stratified subsampling for MMD computation to optimize performance.

    Args:
        embeddings_by_simulation: Original embeddings arrays from each simulation source
        embeddings_indices_by_dist: Dict mapping distribution_id to list of (source_idx, indices) tuples
        real_embeddings: (M, D) embeddings of real samples
        n_param_sets: Number of distributions
        mmd_max_samples: Maximum samples for MMD computation (default: 2000)

    Returns:
        List of metric dicts, one per distribution
    """
    # Verify we have the expected number of distributions
    if len(embeddings_indices_by_dist) != n_param_sets:
        raise ValueError(
            f"Expected {n_param_sets} distributions, but found {len(embeddings_indices_by_dist)} "
            f"in indices mapping. Distribution IDs found: {sorted(embeddings_indices_by_dist.keys())}"
        )

    # Subsample real ONCE + precompute gamma — shared with the engine path
    # via MMDCalculator so both flows compute the calibration objective identically.
    mmd_calc = MMDCalculator(real_embeddings, max_samples=mmd_max_samples, seed=42)
    real_subsampled = mmd_calc.real
    rbf_gamma = mmd_calc.gamma

    # Compute metrics for each distribution
    metrics_list = []

    for distribution_id in sorted(embeddings_indices_by_dist.keys()):
        # Fetch embeddings per source (preserving source structure)
        embeddings_by_source = []
        for source_idx, indices in embeddings_indices_by_dist[distribution_id]:
            embeddings_by_source.append(embeddings_by_simulation[source_idx][indices])

        # Stratified subsample (preserves source proportions)
        embeddings_subsampled = stratified_subsample_for_mmd(
            embeddings_by_source,
            max_samples=mmd_max_samples,
            seed=distribution_id
        )

        # Compute metrics on subsampled data with pre-computed gamma
        metrics = compute_all_metrics(embeddings_subsampled, real_subsampled, rbf_gamma=rbf_gamma)

        # Add distribution_id for tracking
        metrics['distribution_id'] = distribution_id

        metrics_list.append(metrics)

    return metrics_list
