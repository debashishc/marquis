"""Shared Hydra config glue for the retrieval branch.

The single source of truth for paths and knobs is the YAML tree under
``configs/retrieval/``; override any value on the command line with Hydra
syntax, e.g. ``runtime.rrf_k=10`` or ``data.base_dir=/tmp/run``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def _config_dir() -> str:
    """Locate ``configs/retrieval`` (override with MARQUIS_RETRIEVAL_CONFIG_DIR)."""
    env_dir = os.environ.get("MARQUIS_RETRIEVAL_CONFIG_DIR")
    if env_dir:
        return env_dir
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "configs" / "retrieval")


def build_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the Hydra config, applying any ``key=value`` overrides."""
    overrides = list(overrides or [])
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def resolve_query_ids(raw: Any, available: Iterable[str]) -> list[str]:
    """Resolve a ``query_ids`` value (None / scalar / str / list) to an id list."""
    if raw is None:
        return sorted(available, key=lambda x: int(x))
    if isinstance(raw, str):
        return [q.strip() for q in raw.split(",") if q.strip()]
    if isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__"):
        return [str(q) for q in raw]
    return [str(raw)]
