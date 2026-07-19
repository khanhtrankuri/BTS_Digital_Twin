"""Small recursive YAML configuration loader used by BTS-GeoGS-v4."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml_with_base(path, _stack=None) -> dict:
    """Load YAML with an optional relative ``BASE`` string or list."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for BTS configuration files") from error
    path = Path(path).resolve()
    stack = [] if _stack is None else list(_stack)
    if path in stack:
        chain = " -> ".join(str(value) for value in stack + [path])
        raise ValueError(f"Cyclic BASE configuration: {chain}")
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = data.pop("BASE", [])
    if isinstance(bases, str):
        bases = [bases]
    result = {}
    for base in bases:
        result = deep_merge(result, load_yaml_with_base(path.parent / base, stack + [path]))
    return deep_merge(result, data)
