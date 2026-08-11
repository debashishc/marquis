"""Shared Hydra config glue for the RLM controller."""

from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def _config_dir() -> str:
    env_dir = os.environ.get("MARQUIS_RLM_CONFIG_DIR")
    if env_dir:
        return env_dir
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "configs" / "rlm_controller")


def build_config(overrides: list[str] | None = None) -> DictConfig:
    overrides = list(overrides or [])
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        return compose(config_name="config", overrides=overrides)
