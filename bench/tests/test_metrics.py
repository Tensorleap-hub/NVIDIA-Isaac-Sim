import numpy as np
import pytest
from convergence.metrics import MetricsLogger, param_gap, spread, normalize_theta

_STAR = {
    "blur_sigma": 1.5, "noise_std": 0.12, "brightness_shift": 0.08,
    "color_shift_r": 0.06, "color_shift_g": -0.04, "color_shift_b": 0.02,
    "clutter_count": 7, "background_id": 1,
}


def test_normalize_theta_in_unit_range():
    vec = normalize_theta(_STAR)
    assert vec.shape == (8,)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


def test_param_gap_zero_for_star():
    assert param_gap(_STAR, _STAR) < 1e-10


def test_param_gap_positive_for_different():
    other = {**_STAR, "blur_sigma": 5.0}
    assert param_gap(other, _STAR) > 0


def test_spread_zero_for_identical():
    assert spread([_STAR] * 4) < 1e-10


def test_spread_positive_for_varied():
    thetas = [
        {**_STAR, "blur_sigma": 0.0},
        {**_STAR, "blur_sigma": 5.0},
    ]
    assert spread(thetas) > 0


def test_logger_creates_csv_on_first_log(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.csv", _STAR)
    trial_results = [({**_STAR, "blur_sigma": 0.5}, 0.4), ({**_STAR}, 0.35)]
    record = logger.log(0, trial_results)
    assert record.iteration == 0
    assert abs(record.best_objective - 0.35) < 1e-6


def test_logger_tracks_global_best_across_iterations(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.csv", _STAR)
    logger.log(0, [(_STAR, 0.5)])
    record = logger.log(1, [({**_STAR, "blur_sigma": 4.0}, 0.8)])
    assert abs(record.best_objective - 0.5) < 1e-6


def test_logger_load_roundtrip(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.csv", _STAR)
    logger.log(0, [(_STAR, 0.5)])
    logger.log(1, [({**_STAR, "blur_sigma": 1.0}, 0.3)])
    records = logger.load()
    assert len(records) == 2
    assert records[0].iteration == 0
    assert records[1].iteration == 1
    assert abs(records[1].best_objective - 0.3) < 1e-6
