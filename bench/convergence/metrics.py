from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
from .config import THETA_KEYS, THETA_BOUNDS


def normalize_theta(theta: dict) -> np.ndarray:
    vec = []
    for k in THETA_KEYS:
        lo, hi = THETA_BOUNDS[k]
        vec.append((float(theta[k]) - lo) / (hi - lo))
    return np.array(vec, dtype=np.float32)


def param_gap(theta: dict, theta_star: dict) -> float:
    return float(np.linalg.norm(normalize_theta(theta) - normalize_theta(theta_star)))


def spread(thetas: list[dict]) -> float:
    mat = np.stack([normalize_theta(t) for t in thetas])
    return float(mat.std(axis=0).mean())


@dataclass
class IterationRecord:
    iteration: int
    best_objective: float
    best_theta_json: str
    param_gap: float
    spread: float
    median_objective: float
    mean_objective: float
    all_samples_objective: float


_FIELDNAMES = [f.name for f in fields(IterationRecord)]


class MetricsLogger:
    def __init__(self, csv_path: Path, theta_star: dict):
        self._path = Path(csv_path)
        self._theta_star = theta_star
        self._global_best = float("inf")
        self._global_best_theta: dict | None = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

    def log(self, iteration: int, trial_results: list[tuple[dict, float]], all_samples_objective: float = float("nan")) -> IterationRecord:
        objectives = [r[1] for r in trial_results]
        thetas = [r[0] for r in trial_results]
        iter_best = min(objectives)
        iter_best_theta = thetas[objectives.index(iter_best)]
        if iter_best < self._global_best:
            self._global_best = iter_best
            self._global_best_theta = iter_best_theta
        record = IterationRecord(
            iteration=iteration,
            best_objective=self._global_best,
            best_theta_json=json.dumps(self._global_best_theta),
            param_gap=param_gap(self._global_best_theta, self._theta_star),
            spread=spread(thetas),
            median_objective=float(np.median(objectives)),
            mean_objective=float(np.mean(objectives)),
            all_samples_objective=all_samples_objective,
        )
        with self._path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow(asdict(record))
        return record

    def load(self) -> list[IterationRecord]:
        records = []
        with self._path.open() as f:
            for row in csv.DictReader(f):
                records.append(IterationRecord(
                    iteration=int(row["iteration"]),
                    best_objective=float(row["best_objective"]),
                    best_theta_json=row["best_theta_json"],
                    param_gap=float(row["param_gap"]),
                    spread=float(row["spread"]),
                    median_objective=float(row["median_objective"]),
                    mean_objective=float(row["mean_objective"]),
                    all_samples_objective=float(row.get("all_samples_objective", float("nan"))),
                ))
        return records
