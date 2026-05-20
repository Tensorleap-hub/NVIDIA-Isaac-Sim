"""Persistent base-pool management for staged theme optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass
class PoolEntry:
    """A reusable candidate base configuration for future theme stages."""

    entry_id: str
    config: dict[str, Any]
    flattened_params: dict[str, Any]
    score: float | None
    created_at: str
    iteration_index: int | None
    theme_label: str
    stage_lineage: str
    artifact_path: str | None
    yaml_path: str | None
    embedding_path: str | None
    source_pool_entry_id: str | None = None
    diversity_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize one pool entry into JSON."""
        return {
            "entry_id": self.entry_id,
            "config": self.config,
            "flattened_params": self.flattened_params,
            "score": self.score,
            "created_at": self.created_at,
            "iteration_index": self.iteration_index,
            "theme_label": self.theme_label,
            "stage_lineage": self.stage_lineage,
            "artifact_path": self.artifact_path,
            "yaml_path": self.yaml_path,
            "embedding_path": self.embedding_path,
            "source_pool_entry_id": self.source_pool_entry_id,
            "diversity_metadata": self.diversity_metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PoolEntry:
        """Deserialize one pool entry from JSON."""
        score = payload.get("score")
        return cls(
            entry_id=str(payload["entry_id"]),
            config=payload["config"],
            flattened_params=payload["flattened_params"],
            score=float(score) if score is not None else None,
            created_at=str(payload["created_at"]),
            iteration_index=(
                int(payload["iteration_index"])
                if payload.get("iteration_index") is not None
                else None
            ),
            theme_label=str(payload.get("theme_label", "unknown")),
            stage_lineage=str(payload.get("stage_lineage", "")),
            artifact_path=payload.get("artifact_path"),
            yaml_path=payload.get("yaml_path"),
            embedding_path=payload.get("embedding_path"),
            source_pool_entry_id=payload.get("source_pool_entry_id"),
            diversity_metadata=payload.get("diversity_metadata") or {},
        )


class BasePoolManager:
    """Maintain a bounded pool of strong and diverse base candidates."""

    def __init__(
        self,
        *,
        state_path: Path,
        enabled: bool,
        max_size: int,
        elite_size: int,
        recent_size: int,
        score_weight: float,
        diversity_weight: float,
        recency_weight: float,
        near_duplicate_threshold: float,
        random_seed: int,
        pin_seeds: bool = True,
    ):
        self.state_path = state_path
        self.enabled = enabled
        self.max_size = max(1, int(max_size))
        self.elite_size = max(0, int(elite_size))
        self.recent_size = max(0, int(recent_size))
        self.score_weight = float(score_weight)
        self.diversity_weight = float(diversity_weight)
        self.recency_weight = float(recency_weight)
        self.near_duplicate_threshold = max(0.0, float(near_duplicate_threshold))
        self.pin_seeds = bool(pin_seeds)
        self.random = np.random.default_rng(random_seed)
        self.entries: list[PoolEntry] = []
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def has_entries(self) -> bool:
        """Return whether the pool already has any candidates."""
        return bool(self.entries)

    def has_scored_entries(self) -> bool:
        """Return whether the pool contains at least one evaluated candidate."""
        return any(entry.score is not None for entry in self.entries)

    def ensure_bootstrap_entries(self, entries: list[PoolEntry]) -> None:
        """Populate the initial pool from seed YAMLs when no pool exists yet."""
        if not self.enabled or self.entries or not entries:
            return
        self.entries = [deepcopy(entry) for entry in entries]
        self._save()

    def get_entry(self, entry_id: str) -> PoolEntry | None:
        """Resolve one pool entry by id."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return deepcopy(entry)
        return None

    def admit_entries(self, entries: list[PoolEntry]) -> None:
        """Add scored candidates and prune the pool back to the configured bound."""
        if not self.enabled or not entries:
            return

        merged: dict[str, PoolEntry] = {entry.entry_id: deepcopy(entry) for entry in self.entries}
        for entry in entries:
            merged[entry.entry_id] = deepcopy(entry)

        updated_entries = list(merged.values())
        if len(updated_entries) > self.max_size:
            updated_entries = self._prune_entries(updated_entries)
        self.entries = updated_entries
        self._save()

    def sample_entries(self, count: int) -> list[PoolEntry]:
        """Sample future starting bases from the current bounded pool."""
        if not self.enabled or count <= 0 or not self.entries:
            return []

        available = [deepcopy(entry) for entry in self.entries]
        selected: list[PoolEntry] = []
        ranges = self._compute_numeric_ranges(available)
        elite_ids = {
            entry.entry_id
            for entry in self._scored_entries(available)[: self.elite_size]
        }
        recency_map = self._recency_scores(available)
        quality_map = self._quality_scores(available)

        for _ in range(min(count, len(available))):
            weights = np.array(
                [
                    self._sampling_weight(
                        entry=entry,
                        selected=selected,
                        candidates=available,
                        elite_ids=elite_ids,
                        ranges=ranges,
                        recency_map=recency_map,
                        quality_map=quality_map,
                    )
                    for entry in available
                ],
                dtype=float,
            )
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                weights = np.ones(len(available), dtype=float)
            probabilities = weights / weights.sum()
            index = int(self.random.choice(len(available), p=probabilities))
            selected.append(available.pop(index))

        return selected

    def _load(self) -> None:
        """Load pool state from disk when present."""
        if not self.enabled or not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text())
        self.entries = [PoolEntry.from_dict(item) for item in payload.get("entries", [])]

    def _save(self) -> None:
        """Persist the pool state to disk."""
        if not self.enabled:
            return
        payload = {
            "version": 1,
            "updated_at": _utcnow(),
            "max_size": self.max_size,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _prune_entries(self, entries: list[PoolEntry]) -> list[PoolEntry]:
        """Retain elites, recent challengers, and diverse survivors.

        When pin_seeds is True, entries whose entry_id starts with 'seed::' are
        always kept and serve as deduplication anchors for the remaining candidates.
        """
        if self.pin_seeds:
            pinned = [e for e in entries if e.entry_id.startswith("seed::")]
            candidates = [e for e in entries if not e.entry_id.startswith("seed::")]
        else:
            pinned = []
            candidates = entries

        ranked = sorted(candidates, key=self._rank_key)
        ranges = self._compute_numeric_ranges(entries)

        # Deduplicate candidates; seeds act as anchors so near-duplicate scored
        # runs don't waste slots that the seeds already cover.
        deduped: list[PoolEntry] = []
        reference: list[PoolEntry] = list(pinned)
        for entry in ranked:
            if any(
                self._entry_distance(entry, kept, ranges) < self.near_duplicate_threshold
                for kept in reference
            ):
                continue
            deduped.append(entry)
            reference.append(entry)

        non_seed_budget = max(0, self.max_size - len(pinned))

        if len(deduped) <= non_seed_budget:
            return pinned + deduped

        selected: list[PoolEntry] = []
        selected_ids: set[str] = set()

        for entry in self._scored_entries(deduped)[: self.elite_size]:
            selected.append(entry)
            selected_ids.add(entry.entry_id)
            if len(selected) >= non_seed_budget:
                return pinned + selected

        for entry in self._recent_entries(deduped):
            if entry.entry_id in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(entry.entry_id)
            if len(selected_ids) >= min(non_seed_budget, self.elite_size + self.recent_size):
                break

        remaining = [entry for entry in deduped if entry.entry_id not in selected_ids]
        while remaining and len(selected) < non_seed_budget:
            index = max(
                range(len(remaining)),
                key=lambda current_index: self._diverse_selection_score(
                    remaining[current_index],
                    selected=selected + pinned,
                    pool=deduped + pinned,
                    ranges=ranges,
                ),
            )
            entry = remaining.pop(index)
            selected.append(entry)
            selected_ids.add(entry.entry_id)

        return pinned + selected

    def _rank_key(self, entry: PoolEntry) -> tuple[float, float]:
        """Sort better-scoring and newer entries first."""
        score_key = float(entry.score) if entry.score is not None else float("inf")
        created_at = self._timestamp_seconds(entry.created_at)
        return (score_key, -created_at)

    def _scored_entries(self, entries: list[PoolEntry]) -> list[PoolEntry]:
        """Return scored entries sorted by objective value."""
        return sorted(
            [entry for entry in entries if entry.score is not None],
            key=lambda entry: float(entry.score),
        )

    def _recent_entries(self, entries: list[PoolEntry]) -> list[PoolEntry]:
        """Return entries from newest to oldest."""
        return sorted(entries, key=lambda entry: self._timestamp_seconds(entry.created_at), reverse=True)

    def _diverse_selection_score(
        self,
        entry: PoolEntry,
        *,
        selected: list[PoolEntry],
        pool: list[PoolEntry],
        ranges: dict[str, tuple[float, float]],
    ) -> float:
        """Bias diverse slots toward novel but still competitive candidates."""
        if selected:
            diversity = min(self._entry_distance(entry, other, ranges) for other in selected)
        else:
            peers = [other for other in pool if other.entry_id != entry.entry_id]
            if not peers:
                diversity = 1.0
            else:
                diversity = float(np.mean([self._entry_distance(entry, other, ranges) for other in peers]))
        return self._clip_unit(diversity) + 0.15 * self._quality_scores(pool).get(entry.entry_id, 0.0)

    def _sampling_weight(
        self,
        *,
        entry: PoolEntry,
        selected: list[PoolEntry],
        candidates: list[PoolEntry],
        elite_ids: set[str],
        ranges: dict[str, tuple[float, float]],
        recency_map: dict[str, float],
        quality_map: dict[str, float],
    ) -> float:
        """Compute a weighted sample score for one entry."""
        if selected:
            diversity = min(self._entry_distance(entry, other, ranges) for other in selected)
        else:
            peers = [other for other in candidates if other.entry_id != entry.entry_id]
            if peers:
                diversity = float(np.mean([self._entry_distance(entry, other, ranges) for other in peers]))
            else:
                diversity = 1.0

        weight = 0.05
        weight += self.score_weight * quality_map.get(entry.entry_id, 0.35)
        weight += self.diversity_weight * self._clip_unit(diversity)
        weight += self.recency_weight * recency_map.get(entry.entry_id, 0.0)
        if entry.entry_id in elite_ids:
            weight += 0.25
        return max(weight, 0.01)

    def _quality_scores(self, entries: list[PoolEntry]) -> dict[str, float]:
        """Normalize objective quality into the [0, 1] range."""
        scored = [entry for entry in entries if entry.score is not None]
        if not scored:
            return {}
        best = min(float(entry.score) for entry in scored)
        worst = max(float(entry.score) for entry in scored)
        denominator = max(worst - best, 1e-9)
        return {
            entry.entry_id: 1.0 - ((float(entry.score) - best) / denominator)
            for entry in scored
            if entry.score is not None
        }

    def _recency_scores(self, entries: list[PoolEntry]) -> dict[str, float]:
        """Normalize recency into the [0, 1] range."""
        ordered = self._recent_entries(entries)
        if len(ordered) == 1:
            return {ordered[0].entry_id: 1.0}
        denominator = max(len(ordered) - 1, 1)
        return {
            entry.entry_id: 1.0 - (index / denominator)
            for index, entry in enumerate(ordered)
        }

    def _entry_distance(
        self,
        left: PoolEntry,
        right: PoolEntry,
        ranges: dict[str, tuple[float, float]],
    ) -> float:
        """Compute embedding-based diversity with parameter fallback."""
        left_centroid = left.diversity_metadata.get("embedding_centroid")
        right_centroid = right.diversity_metadata.get("embedding_centroid")
        if left_centroid and right_centroid:
            left_vector = np.asarray(left_centroid, dtype=float)
            right_vector = np.asarray(right_centroid, dtype=float)
            if left_vector.shape == right_vector.shape and left_vector.size > 0:
                left_norm = np.linalg.norm(left_vector)
                right_norm = np.linalg.norm(right_vector)
                if left_norm > 0 and right_norm > 0:
                    cosine_similarity = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
                    return self._clip_unit(1.0 - cosine_similarity)
        return self._parameter_distance(left.flattened_params, right.flattened_params, ranges)

    def _parameter_distance(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        ranges: dict[str, tuple[float, float]],
    ) -> float:
        """Compute normalized distance in flattened parameter space."""
        keys = sorted(set(left) | set(right))
        if not keys:
            return 0.0

        total = 0.0
        for key in keys:
            left_value = left.get(key)
            right_value = right.get(key)
            left_numeric = self._to_numeric(left_value)
            right_numeric = self._to_numeric(right_value)
            if left_numeric is not None and right_numeric is not None:
                lower, upper = ranges.get(key, (min(left_numeric, right_numeric), max(left_numeric, right_numeric)))
                denominator = max(upper - lower, 1e-9)
                total += abs(left_numeric - right_numeric) / denominator
                continue
            total += 0.0 if left_value == right_value else 1.0
        return total / len(keys)

    def _compute_numeric_ranges(self, entries: list[PoolEntry]) -> dict[str, tuple[float, float]]:
        """Compute observed ranges for numeric flattened params."""
        observed: dict[str, list[float]] = {}
        for entry in entries:
            for key, value in entry.flattened_params.items():
                numeric_value = self._to_numeric(value)
                if numeric_value is None:
                    continue
                observed.setdefault(key, []).append(numeric_value)
        return {
            key: (min(values), max(values))
            for key, values in observed.items()
        }

    def _to_numeric(self, value: Any) -> float | None:
        """Convert flat parameter values into a normalized numeric domain when possible."""
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
            return None
        if isinstance(value, str):
            lower_value = value.strip().lower()
            if lower_value == "true":
                return 1.0
            if lower_value == "false":
                return 0.0
            try:
                numeric = float(value)
            except ValueError:
                return None
            if math.isfinite(numeric):
                return numeric
        return None

    def _timestamp_seconds(self, created_at: str) -> float:
        """Parse ISO timestamps into numeric sort keys."""
        try:
            return datetime.datetime.fromisoformat(created_at).timestamp()
        except ValueError:
            return 0.0

    def _clip_unit(self, value: float) -> float:
        """Clamp a numeric score into the [0, 1] interval."""
        return max(0.0, min(1.0, float(value)))
