"""
Metrics for evaluating distribution similarity and sample distances.

Includes MMD, Wasserstein distance, and per-sample distance metrics.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple


class DistributionMetrics:
    """Calculate distribution similarity metrics"""

    @staticmethod
    def mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', gamma: float = None) -> float:
        """
        Calculate Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            X: (N, D) array of samples from distribution 1
            Y: (M, D) array of samples from distribution 2
            kernel: Kernel type ('rbf' or 'linear')
            gamma: RBF kernel parameter. If None, uses median heuristic.

        Returns:
            mmd_value: MMD distance (lower is better)
        """
        if kernel == 'rbf':
            # Auto-compute gamma using median heuristic if not provided
            if gamma is None:
                gamma = DistributionMetrics._compute_gamma_median_heuristic(X, Y)

            # RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
            XX = DistributionMetrics._rbf_kernel(X, X, gamma)
            YY = DistributionMetrics._rbf_kernel(Y, Y, gamma)
            XY = DistributionMetrics._rbf_kernel(X, Y, gamma)
        elif kernel == 'linear':
            # Linear kernel: k(x,y) = x^T y
            XX = X @ X.T
            YY = Y @ Y.T
            XY = X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return float(np.sqrt(max(mmd, 0)))  # Ensure non-negative due to numerical errors

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
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel between X and Y"""
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        distances_sq = X_norm + Y_norm - 2 * X @ Y.T
        return np.exp(-gamma * distances_sq)

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
    embeddings_by_shape: List[np.ndarray],
    embeddings_indices_by_dist: Dict[int, List[Tuple[int, np.ndarray]]],
    real_embeddings: np.ndarray,
    n_param_sets: int,
    mmd_max_samples: int = 2000
) -> List[Dict[str, float]]:
    """
    Compute metrics for each distribution separately using on-demand embedding fetching.
    Uses stratified subsampling for MMD computation to optimize performance.

    Args:
        embeddings_by_shape: Original embeddings arrays from each simulation source
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

    # Subsample real embeddings ONCE (reused for all distributions)
    if len(real_embeddings) > mmd_max_samples:
        rng = np.random.RandomState(42)
        real_indices = rng.choice(len(real_embeddings), size=mmd_max_samples, replace=False)
        real_subsampled = real_embeddings[real_indices]
    else:
        real_subsampled = real_embeddings

    # Compute gamma ONCE from real embeddings (reused for all distributions)
    rbf_gamma = DistributionMetrics._compute_gamma_median_heuristic(real_subsampled, real_subsampled)

    # Compute metrics for each distribution
    metrics_list = []

    for distribution_id in sorted(embeddings_indices_by_dist.keys()):
        # Fetch embeddings per source (preserving source structure)
        embeddings_by_source = []
        for source_idx, indices in embeddings_indices_by_dist[distribution_id]:
            embeddings_by_source.append(embeddings_by_shape[source_idx][indices])

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
