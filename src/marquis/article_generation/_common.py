"""Shared infrastructure for the article-generation strategies.

Holds the Hydra config glue plus the claim loading, model loading, generation
and JSON-parsing helpers reused across the ``baseline`` / ``ginger`` / ``qa``
strategies. The single source of truth for paths, model and runtime knobs is the
YAML tree under ``configs/article_generation/``; override any value on the
command line with Hydra syntax, e.g. ``data.query_ids=3``.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Hydra config


def _config_dir() -> str:
    """Locate ``configs/article_generation`` (override with MARQUIS_CONFIG_DIR)."""
    env_dir = os.environ.get("MARQUIS_CONFIG_DIR")
    if env_dir:
        return env_dir
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "configs" / "article_generation")


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


# Claim loading


def load_claims(claims_path: str) -> dict[str, list[dict]]:
    """Load claims from a JSONL file and group them by ``query_id``."""
    claims_by_query: dict[str, list[dict]] = defaultdict(list)
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            claims_by_query[claim["query_id"]].append(claim)

    print(f"Loaded claims for {len(claims_by_query)} queries")
    for qid, claims in sorted(claims_by_query.items(), key=lambda x: int(x[0])):
        topics = set(c["topic"] for c in claims)
        print(f"  Query {qid}: {len(claims)} claims, topic(s): {topics}")
    return claims_by_query


# Model loading and generation


def load_model(model_name: str, cache_dir: str | None):
    """Load a causal LM and its tokenizer for report writing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = cache_dir or None
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    print(f"Loading model from {model_name} (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded successfully!")
    return model, tokenizer


def llm_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    system_msg: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generic single-turn chat generation helper."""
    import torch

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )

    generated_ids = output[0][inputs["input_ids"].shape[1] :]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clear GPU cache after each generation
    del inputs, output, generated_ids
    torch.cuda.empty_cache()
    return result.strip()


def parse_json_from_response(response: str) -> dict | None:
    """Extract a JSON object from an LLM response, handling markdown fences."""
    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to the first { ... last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            pass

    print(f"WARNING: Failed to parse JSON from response:\n{response[:500]}")
    return None
