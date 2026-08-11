"""Query expansion: decompose each MAGMaR query into searchable sub-queries.

Runs a generative LLM over the query set and writes ``expanded_queries.json``.
Model, paths and decoding knobs come from ``configs/retrieval/``::

    python -m retrieval.cli expand model.model=Qwen/Qwen3.5-2B runtime.expansion.temperature=0.7
"""

from __future__ import annotations

import json
import os
import sys

from omegaconf import DictConfig

from marquis.retrieval._common import build_config
from marquis.retrieval.prompts import QUERY_EXPANSION_PROMPT


def load_queries(queries_file: str) -> list[dict]:
    queries = []
    with open(queries_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries")
    return queries


def load_model(model_name: str, cache_dir: str | None):
    """Load the query-expansion causal LM and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = cache_dir or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def expand_one(model, tokenizer, q: dict, *, max_new_tokens, temperature, top_p) -> list[str]:
    """Decompose a single query into sub-queries, with a stricter retry + fallback."""
    import torch

    prompt = QUERY_EXPANSION_PROMPT.format(
        title=q["title"],
        query=q["query"],
        language=q.get("language", "english"),
        persona_title=q.get("persona_title", ""),
        background=q.get("background", ""),
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"  Response: {response[:300]}")

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            sub_queries = json.loads(response[start:end])
            if not (
                isinstance(sub_queries, list)
                and all(isinstance(s, str) for s in sub_queries)
                and len(sub_queries) >= 2
            ):
                raise ValueError("Invalid format")
        else:
            raise ValueError("No JSON array found")
        return sub_queries
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Failed to parse ({e}), retrying with stricter prompt...")

    retry_prompt = (
        f"List 4 short video search queries about: {q['title']} - {q['query'][:200]}\n"
        "Return ONLY a JSON array: "
    )
    messages2 = [{"role": "user", "content": retry_prompt}]
    text2 = tokenizer.apply_chat_template(
        messages2, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output2 = model.generate(**inputs2, max_new_tokens=200, temperature=0.5, do_sample=True)
    response2 = tokenizer.decode(
        output2[0][inputs2["input_ids"].shape[1] :], skip_special_tokens=True
    )
    try:
        start2 = response2.find("[")
        end2 = response2.rfind("]") + 1
        return json.loads(response2[start2:end2])
    except Exception:
        print("  Retry also failed, generating manual sub-queries")
        return [
            f"{q['title']} overview",
            f"{q['title']} impact effects",
            f"{q['title']} latest updates details",
            f"{q['title']} analysis response",
        ]


def run(cfg: DictConfig) -> None:
    """Expand all queries into sub-queries and write expanded_queries.json."""
    queries = load_queries(cfg.data.queries_file)
    model, tokenizer = load_model(cfg.model.model, cfg.model.cache_dir or None)
    ex = cfg.runtime.expansion

    results = {}
    for q in queries:
        sub_queries = expand_one(
            model,
            tokenizer,
            q,
            max_new_tokens=ex.max_new_tokens,
            temperature=ex.temperature,
            top_p=ex.top_p,
        )
        print(f"  Sub-queries ({len(sub_queries)}):")
        for i, sq in enumerate(sub_queries):
            print(f"    {i + 1}. {sq}")
        results[q["query_id"]] = {
            "original_query": q["query"],
            "title": q["title"],
            "sub_queries": sub_queries,
        }

    output_path = cfg.data.expanded_queries
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved expanded queries to {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
