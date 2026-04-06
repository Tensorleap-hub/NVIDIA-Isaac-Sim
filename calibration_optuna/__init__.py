"""
Production-ready standalone optimizer module.

This module provides a minimal, dependency-light implementation
of the synthetic data optimizer for production use.

Dependencies:
- optuna: Bayesian optimization
- numpy: Numerical arrays
- scipy: Distance metrics
- pandas: Data handling
- scikit-learn: Only for RBF kernel gamma heuristic (minimal use)

No dependencies on:
- torch, torchvision (embedding models)
- opencv, pillow (image processing)
- Data generators or visualization tools
"""

from .config import DEFAULT_CONFIG

__all__ = ['DEFAULT_CONFIG']
