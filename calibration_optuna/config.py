"""
Production configuration module.

All configuration values are hardcoded (no YAML dependency).
Based on configs/experiment_config.yaml.
"""

from typing import Dict

# Default experiment configuration
DEFAULT_CONFIG: Dict = {
    'experiment_name': 'synth_data_optimization',
    'experiments_base_dir': 'data/experiments',
    'random_seed': 42,
    'iteration_batch_size': 30,
    'replications_per_iteration': 10,
    'max_iterations': 10,
    'param_precision': {
        'base_size': 1,
        'rotation': 1,
        'center_x': 2,
        'center_y': 2,
        'position_spread': 2,
    },
    'optimization_metrics': ['mmd_rbf'],
    'convergence_threshold': 0.05,
    'early_stop_patience': 3,
    'optimizer': {
        'n_startup_trials': 60,
        'multivariate': True,
    },
    'max_top_n_suggestions': 100,
    'top_n_best_trials': 1,
    'n_samples_to_generate': 5000,
    'samples_per_distribution': 128,
    'min_samples_to_return': 2,
    'mmd_max_samples': 1000,
}
