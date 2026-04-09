from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock, Thread
import sys
import time


@dataclass
class UISnapshot:
    phase: str = "idle"
    iteration_index: int = 0
    max_iterations: int = 0
    current_run: str = "-"
    total_runs: int = 0
    completed_runs: int = 0
    real_cache_status: str = "-"
    note: str = ""
    best_trial_id: str = "-"
    best_objective: str = "-"
    initial_distance: str = "-"
    iteration_best: str = "-"
    iteration_mean: str = "-"
    iteration_median: str = "-"
    recent_logs: deque[str] = field(default_factory=lambda: deque(maxlen=20))


class WorkflowUI:
    def __init__(self, log_path: str | Path | None = None) -> None:
        self.snapshot = UISnapshot()
        self._lock = RLock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._interactive = sys.stdout.isatty()
        self._last_emitted_signature: tuple | None = None
        self._log_path = Path(log_path) if log_path is not None else None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("")

    def start(self) -> None:
        if not self._interactive:
            return
        self._thread = Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_status(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self.snapshot, key, value)
            snapshot = UISnapshot(
                phase=self.snapshot.phase,
                iteration_index=self.snapshot.iteration_index,
                max_iterations=self.snapshot.max_iterations,
                current_run=self.snapshot.current_run,
                total_runs=self.snapshot.total_runs,
                completed_runs=self.snapshot.completed_runs,
                real_cache_status=self.snapshot.real_cache_status,
                note=self.snapshot.note,
                best_trial_id=self.snapshot.best_trial_id,
                best_objective=self.snapshot.best_objective,
                initial_distance=self.snapshot.initial_distance,
                iteration_best=self.snapshot.iteration_best,
                iteration_mean=self.snapshot.iteration_mean,
                iteration_median=self.snapshot.iteration_median,
                recent_logs=deque(self.snapshot.recent_logs, maxlen=20),
            )
        self._emit_status_snapshot(snapshot)

    def append_log(self, line: str) -> None:
        with self._lock:
            self.snapshot.recent_logs.append(line)
        self._write_line(f"[isaac] {line}")

    def _render_loop(self) -> None:
        while not self._stop.is_set():
            self._render_once()
            time.sleep(0.2)
        self._render_once()

    def _render_once(self) -> None:
        with self._lock:
            snapshot = UISnapshot(
                phase=self.snapshot.phase,
                iteration_index=self.snapshot.iteration_index,
                max_iterations=self.snapshot.max_iterations,
                current_run=self.snapshot.current_run,
                total_runs=self.snapshot.total_runs,
                completed_runs=self.snapshot.completed_runs,
                real_cache_status=self.snapshot.real_cache_status,
                note=self.snapshot.note,
                best_trial_id=self.snapshot.best_trial_id,
                best_objective=self.snapshot.best_objective,
                initial_distance=self.snapshot.initial_distance,
                iteration_best=self.snapshot.iteration_best,
                iteration_mean=self.snapshot.iteration_mean,
                iteration_median=self.snapshot.iteration_median,
                recent_logs=deque(self.snapshot.recent_logs, maxlen=20),
            )

        lines = [
            "Simulation Calibration Loop",
            "",
            f"Phase: {snapshot.phase}",
            f"Iteration: {snapshot.iteration_index}/{snapshot.max_iterations}",
            f"Runs: {snapshot.completed_runs}/{snapshot.total_runs}",
            f"Current run: {snapshot.current_run}",
            f"Real cache: {snapshot.real_cache_status}",
            f"Best trial: {snapshot.best_trial_id}",
            f"Best objective: {snapshot.best_objective}",
            f"Initial distance: {snapshot.initial_distance}",
            f"Iteration best: {snapshot.iteration_best}",
            f"Iteration mean: {snapshot.iteration_mean}",
            f"Iteration median: {snapshot.iteration_median}",
            f"Note: {snapshot.note}",
            "",
            "Isaac logs:",
            *list(snapshot.recent_logs),
        ]
        screen = "\033[2J\033[H" + "\n".join(lines)
        sys.stdout.write(screen)
        sys.stdout.flush()

    def _emit_status_snapshot(self, snapshot: UISnapshot) -> None:
        signature = (
            snapshot.phase,
            snapshot.iteration_index,
            snapshot.max_iterations,
            snapshot.completed_runs,
            snapshot.total_runs,
            snapshot.current_run,
            snapshot.best_trial_id,
            snapshot.best_objective,
            snapshot.initial_distance,
            snapshot.iteration_best,
            snapshot.iteration_mean,
            snapshot.iteration_median,
            snapshot.note,
        )
        if signature == self._last_emitted_signature:
            return
        self._last_emitted_signature = signature
        summary = (
            f"[workflow] phase={snapshot.phase} "
            f"iter={snapshot.iteration_index}/{snapshot.max_iterations} "
            f"runs={snapshot.completed_runs}/{snapshot.total_runs} "
            f"current={snapshot.current_run} "
            f"best_trial={snapshot.best_trial_id} "
            f"best_obj={snapshot.best_objective} "
            f"initial={snapshot.initial_distance} "
            f"iter_best={snapshot.iteration_best} "
            f"iter_mean={snapshot.iteration_mean} "
            f"iter_med={snapshot.iteration_median} "
            f"note={snapshot.note}"
        )
        self._write_line(summary)
        if not self._interactive:
            print(summary, flush=True)

    def _write_line(self, line: str) -> None:
        if self._log_path is None:
            return
        with self._lock:
            with self._log_path.open("a") as log_file:
                log_file.write(line + "\n")
