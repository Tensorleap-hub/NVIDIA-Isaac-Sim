"""Schema utilities for mapping Isaac YAMLs to Optuna parameter rows.

The controller works with flat parameter dictionaries because Optuna expects a
tabular search space, while Isaac expects nested YAMLs. This module owns that
translation in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json

import yaml


@dataclass(frozen=True)
class ParameterSpec:
    """Describes one optimizable flattened parameter path."""

    path: str
    kind: str
    value_kind: str
    length: int | None = None


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "none"
    return "json"


def _collect_specs(value: Any, prefix: str, observed: dict[str, list[Any]]) -> None:
    observed.setdefault(prefix, []).append(value)

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_specs(child, child_prefix, observed)


def _all_primitive(values: list[Any]) -> bool:
    return all(_is_scalar(item) for item in values)


def infer_parameter_schema(configs: list[dict[str, Any]]) -> list[ParameterSpec]:
    """Infer the flattening schema from a family of resolved seed YAMLs.

    Scalars stay as single Optuna parameters, homogeneous fixed-length lists are
    expanded into indexed parameters, and everything else is serialized as JSON.
    """
    observed: dict[str, list[Any]] = {}
    for config in configs:
        for key, value in config.items():
            _collect_specs(value, str(key), observed)

    specs: list[ParameterSpec] = []
    for path in sorted(observed):
        values = observed[path]
        sample = values[0]

        if isinstance(sample, dict):
            continue

        if _is_scalar(sample):
            specs.append(ParameterSpec(path=path, kind="scalar", value_kind=_value_kind(sample)))
            continue

        if isinstance(sample, list) and all(isinstance(v, list) for v in values):
            same_length = len({len(v) for v in values}) == 1
            if same_length and _all_primitive([item for seq in values for item in seq]):
                element_kinds = {_value_kind(item) for seq in values for item in seq}
                if element_kinds <= {"int"}:
                    value_kind = "int"
                elif element_kinds <= {"int", "float"}:
                    value_kind = "float"
                elif len(element_kinds) == 1:
                    value_kind = next(iter(element_kinds))
                else:
                    value_kind = "json"
                specs.append(
                    ParameterSpec(
                        path=path,
                        kind="indexed_list",
                        value_kind=value_kind,
                        length=len(sample),
                    )
                )
                continue

        specs.append(ParameterSpec(path=path, kind="serialized", value_kind="json"))

    return specs


def _encode_serialized(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_scalar(value: Any, value_kind: str) -> Any:
    if value_kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"
    if value_kind == "int":
        return int(value)
    if value_kind == "float":
        return float(value)
    if value_kind == "none":
        return None
    return value


def _encode_scalar(value: Any, value_kind: str) -> Any:
    if value_kind == "bool":
        return "true" if value else "false"
    if value_kind == "none":
        return "null"
    return value


def flatten_config(config: dict[str, Any], specs: list[ParameterSpec]) -> dict[str, Any]:
    """Flatten a nested Isaac config into an Optuna-compatible parameter row."""
    row: dict[str, Any] = {}
    for spec in specs:
        value = _get_by_path(config, spec.path)
        if spec.kind == "scalar":
            row[spec.path] = _encode_scalar(value, spec.value_kind)
        elif spec.kind == "indexed_list":
            for index, item in enumerate(value):
                row[f"{spec.path}[{index}]"] = _encode_scalar(item, spec.value_kind)
        else:
            row[spec.path] = _encode_serialized(value)
    return row


def materialize_config(base_config: dict[str, Any], params: dict[str, Any], specs: list[ParameterSpec]) -> dict[str, Any]:
    """Write a flat parameter row back into a nested Isaac config template."""
    config = copy.deepcopy(base_config)
    for spec in specs:
        if spec.kind == "scalar":
            _set_by_path(config, spec.path, _decode_scalar(params[spec.path], spec.value_kind))
            continue

        if spec.kind == "indexed_list":
            items = []
            for index in range(spec.length or 0):
                key = f"{spec.path}[{index}]"
                items.append(_decode_scalar(params[key], spec.value_kind))
            _set_by_path(config, spec.path, items)
            continue

        _set_by_path(config, spec.path, json.loads(params[spec.path]))

    return config


def filter_parameter_specs(
    specs: list[ParameterSpec],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[ParameterSpec]:
    """Filter the inferred schema down to the configured optimization subset."""
    include = include or []
    exclude = exclude or []

    filtered = []
    for spec in specs:
        if include and spec.path not in include:
            continue
        if spec.path in exclude:
            continue
        filtered.append(spec)
    return filtered


def load_yaml_configs(config_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load and resolve all YAML configs in a seed directory."""
    config_dir = Path(config_dir)
    items = []
    for path in sorted(config_dir.glob("*.yaml")):
        items.append((path, _load_yaml_with_extends(path)))
    return items


def save_yaml_config(path: str | Path, config: dict[str, Any]) -> None:
    """Persist a materialized Isaac config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _get_by_path(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        current = current[part]
    return current


def _set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    """Resolve the repo's lightweight YAML inheritance via `extends`."""
    raw = yaml.safe_load(path.read_text())
    if "extends" not in raw:
        return raw

    base_path = (path.parent / raw["extends"]).resolve()
    override = dict(raw)
    override.pop("extends")
    return _deep_merge(_load_yaml_with_extends(base_path), override)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base YAML dictionary."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
