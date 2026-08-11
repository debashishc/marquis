"""Baseline article generation: synthesize per-query claims into a cited report.

Configuration is driven entirely by the Hydra config tree under
``configs/article_generation/``. The single source of truth for paths, model and
runtime knobs is that YAML tree; override any value on the command line with
Hydra syntax, e.g.::

    python -m article_generation.cli baseline data.query_ids=3 runtime.max_claims_per_query=10
"""

from __future__ import annotations

import json
import os
import sys

from omegaconf import DictConfig

from marquis.article_generation._common import (
    build_config,
    llm_generate,
    load_claims,
    load_model,
    resolve_query_ids,
)
from marquis.article_generation.citations import format_timestamp
from marquis.article_generation.prompts import REPORT_PROMPT


def format_claims_for_prompt(claims: list[dict]) -> str:
    """Format a list of claims into text for the prompt."""
    lines = []
    for i, c in enumerate(claims, 1):
        timestamp = format_timestamp(c.get("timestamp"))
        lines.append(
            f"Claim {i}:\n"
            f"  - Content: {c.get('claim', '')}\n"
            f"  - Video ID: {c.get('video_id', '?')}\n"
            f"  - Timestamp: {timestamp}\n"
            f"  - Source type: {c.get('source') or 'unknown'}\n"
            f"  - Evidence: {c.get('evidence') or ''}\n"
            f"  - Confidence: {c.get('confidence')}"
        )
    return "\n\n".join(lines)


def generate_report(model, tokenizer, topic, claims, *, max_claims, temperature, top_p):
    """Generate a report for a single query/topic."""
    # Truncate claims if too many to avoid OOM
    if len(claims) > max_claims:
        print(
            f"  WARNING: {len(claims)} claims exceeds limit of {max_claims}, "
            f"using top {max_claims} by confidence"
        )
        claims = sorted(claims, key=lambda x: float(x.get("confidence") or 0.0), reverse=True)[
            :max_claims
        ]

    claims_text = format_claims_for_prompt(claims)
    prompt = REPORT_PROMPT.format(topic=topic, claims_text=claims_text)

    return llm_generate(
        model,
        tokenizer,
        prompt,
        system_msg=(
            "You are a helpful report writing assistant that synthesizes information "
            "from multiple video sources into coherent reports with proper citations."
        ),
        max_tokens=1024,
        temperature=temperature,
        top_p=top_p,
    )


def run(cfg: DictConfig) -> None:
    """Generate baseline reports for the queries selected by ``cfg``."""
    import torch

    claims_path = cfg.data.claims_path
    output_dir = cfg.output.out_dir
    model_name = cfg.model.model
    cache_dir = cfg.model.cache_dir or None
    max_claims = cfg.runtime.max_claims_per_query
    temperature = cfg.runtime.temperature
    top_p = cfg.runtime.top_p

    os.makedirs(output_dir, exist_ok=True)

    # Load claims
    claims_by_query = load_claims(claims_path)

    # Load model
    model, tokenizer = load_model(model_name, cache_dir)

    # Determine which queries to process
    query_ids = resolve_query_ids(cfg.data.query_ids, claims_by_query.keys())

    # Load existing reports if any (to resume from crash)
    output_path = os.path.join(output_dir, "reports_baseline.json")
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            all_reports = json.load(f)
        print(f"Loaded {len(all_reports)} existing reports from {output_path}")
    else:
        all_reports = {}

    # Generate reports
    for qid in query_ids:
        # Skip if already in JSON
        if qid in all_reports:
            print(f"\nSkipping Query {qid} (already generated)")
            continue

        # Also check if txt file exists (from previous run)
        existing_txts = [
            f
            for f in os.listdir(output_dir)
            if f.startswith(f"report_q{qid}_")
            and not f.startswith("report_ginger")
            and f.endswith(".txt")
        ]
        if existing_txts:
            print(f"\nSkipping Query {qid} (txt file exists: {existing_txts[0]})")
            txt_path = os.path.join(output_dir, existing_txts[0])
            with open(txt_path, encoding="utf-8") as f:
                content = f.read()
            parts = content.split("=" * 60)
            report_text = parts[-1].strip() if len(parts) > 1 else content
            claims = claims_by_query.get(qid, [])
            topic = claims[0]["topic"] if claims else "unknown"
            all_reports[qid] = {
                "query_id": qid,
                "topic": topic,
                "num_claims": len(claims),
                "report": report_text,
            }
            continue

        if qid not in claims_by_query:
            print(f"WARNING: No claims found for query {qid}, skipping")
            continue

        claims = claims_by_query[qid]
        topic = claims[0]["topic"]
        print(f"\n{'=' * 60}")
        print(f"Generating report for Query {qid} | Topic: {topic}")
        print(f"Number of claims: {len(claims)}")
        print(f"{'=' * 60}")

        try:
            report = generate_report(
                model,
                tokenizer,
                topic,
                claims,
                max_claims=max_claims,
                temperature=temperature,
                top_p=top_p,
            )

            all_reports[qid] = {
                "query_id": qid,
                "topic": topic,
                "num_claims": len(claims),
                "report": report,
            }

            print("\n--- Report Preview (first 500 chars) ---")
            print(report[:500])
            print("--- End Preview ---\n")

            # Save individual report immediately
            txt_path = os.path.join(output_dir, f"report_q{qid}_{topic}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Query ID: {qid}\n")
                f.write(f"Topic: {topic}\n")
                f.write(f"Number of claims used: {len(claims)}\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(report)
            print(f"  Saved: {txt_path}")

            # Save all reports after each query (incremental save)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_reports, f, indent=2, ensure_ascii=False)

        except torch.cuda.OutOfMemoryError:
            print(f"  ERROR: OOM on Query {qid} with {len(claims)} claims. Skipping.")
            torch.cuda.empty_cache()
            continue

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nAll reports saved to: {output_path}")
    print(f"Total reports generated: {len(all_reports)}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
