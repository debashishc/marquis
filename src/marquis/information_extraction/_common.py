"""Shared Hydra config glue for the information-extraction branch.

The single source of truth for paths, models and knobs is the YAML tree under
``configs/information_extraction/``; override any value on the command line with
Hydra syntax, e.g. ``data.query_ids=3`` or ``model=unli_lora``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def _config_dir() -> str:
    """Locate ``configs/information_extraction`` (override with MARQUIS_IE_CONFIG_DIR)."""
    env_dir = os.environ.get("MARQUIS_IE_CONFIG_DIR")
    if env_dir:
        return env_dir
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "configs" / "information_extraction")


def build_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the Hydra config, applying any ``key=value`` overrides."""
    overrides = list(overrides or [])
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def query_id_filter(raw: Any) -> set | None:
    """Turn ``data.query_ids`` (None / scalar / str / list) into a set, or None."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        ids = {q.strip() for q in raw.split(",") if q.strip()}
    elif isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__"):
        ids = {str(q) for q in raw}
    else:
        ids = {str(raw)}
    return ids or None
