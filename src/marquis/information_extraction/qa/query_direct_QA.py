"""Query decomposition: expand each official query into searchable sub-questions.

Despite the historical filename, this is NOT a QA step. It loads a text LLM and
rewrites every query into 10-25 atomic, retrieval-friendly English sub-questions.
Output feeds the QA runners (qa-answer / qa) via ``data.qa_queries``.

Config comes from ``configs/information_extraction/``::

    python -m information_extraction.cli qa-decompose data.subqueries_output=out.jsonl
"""

from __future__ import annotations

import json
import sys

from omegaconf import DictConfig

from marquis.information_extraction._common import build_config
from marquis.information_extraction.prompts import QA_DECOMPOSE_PROMPT


def load_model(model_name: str, cache_dir: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = cache_dir or None
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return tokenizer, model


def decompose(q, tokenizer, model, max_new_tokens=1024):
    import torch

    prompt = QA_DECOMPOSE_PROMPT.format(
        title=q.get("title", ""),
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
            **inputs, max_new_tokens=max_new_tokens, temperature=0.4, top_p=0.9, do_sample=True
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON found")
        sub_queries = json.loads(response[start:end])
        if not isinstance(sub_queries, list):
            raise ValueError("Not a list")
        if not all(isinstance(s, str) for s in sub_queries):
            raise ValueError("Items must be strings")
        if len(sub_queries) < 5:
            raise ValueError("Too few queries")
    except Exception as e:
        print("Parsing failed:", e)
        title = q.get("title", q.get("query", "the event"))
        sub_queries = [
            f"What were the main results of the {title} event?",
            f"What statistical data is available about the {title}?",
            f"What official sources publish data about the {title}?",
            f"What were the major outcomes associated with the {title}?",
        ]
    return sub_queries


def run(cfg: DictConfig) -> None:
    """Decompose each query into sub-questions and write the augmented JSONL."""
    queries_path = cfg.data.queries_jsonl
    output_path = cfg.data.subqueries_output
    model_name = cfg.runtime.qa.qa_model
    cache_dir = cfg.model.download_dir or None
    max_new_tokens = cfg.runtime.qa.max_new_tokens

    with open(queries_path, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(queries)} queries from {queries_path}")

    tokenizer, model = load_model(model_name, cache_dir)
    for q in queries:
        sub_queries = decompose(q, tokenizer, model, max_new_tokens=max_new_tokens)
        print(f"Generated {len(sub_queries)} questions for: {q.get('title', q['query'])[:60]}")
        q["sub_queries"] = sub_queries

    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"[ok] wrote {len(queries)} rows -> {output_path}")


def main(argv: list | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    run(build_config(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
