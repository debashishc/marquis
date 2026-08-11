"""Ginger article generation: cluster → rank → summarize → fluency over claims.

A multi-stage refinement of the baseline: claims are grouped into facet clusters,
the clusters are ranked by relevance, the top-N are summarized one sentence each,
and the summaries are polished into a fluent report.

Configuration is driven entirely by the Hydra config tree under
``configs/article_generation/`` (see ``runtime.ginger.*`` and
``runtime.top_n_clusters``). Override on the command line with Hydra syntax::

    python -m article_generation.cli ginger data.query_ids=3 runtime.top_n_clusters=3
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
    parse_json_from_response,
    resolve_query_ids,
)
from marquis.article_generation.citations import format_timestamp
from marquis.article_generation.prompts import (
    GINGER_CLUSTER_PROMPT as CLUSTER_PROMPT,
)
from marquis.article_generation.prompts import (
    GINGER_FLUENCY_PROMPT as FLUENCY_PROMPT,
)
from marquis.article_generation.prompts import (
    GINGER_RANK_PROMPT as RANK_PROMPT,
)
from marquis.article_generation.prompts import (
    GINGER_SUMMARIZE_PROMPT as SUMMARIZE_PROMPT,
)


def format_claims_for_prompt(claims: list[dict]) -> str:
    """Format claims with IDs for clustering."""
    lines = []
    for c in claims:
        timestamp = format_timestamp(c.get("timestamp"))
        lines.append(
            f'[{c.get("claim_id", "?")}] "{c.get("claim", "")}" '
            f"(video: {c.get('video_id', '?')}, {timestamp}, "
            f"source: {c.get('source') or 'unknown'}, confidence: {c.get('confidence')})"
        )
    return "\n".join(lines)


def format_clusters_for_ranking(clusters: list[dict], claims_lookup: dict) -> str:
    """Format clusters for the ranking prompt."""
    lines = []
    for cluster in clusters:
        label = cluster["label"]
        claim_texts = []
        for cid in cluster["claim_ids"]:
            if cid in claims_lookup:
                claim_texts.append(f"  - {claims_lookup[cid]['claim']}")
        lines.append(f'Cluster: "{label}"\n' + "\n".join(claim_texts))
    return "\n\n".join(lines)


def format_cluster_claims_for_summary(claim_ids: list[str], claims_lookup: dict) -> str:
    """Format claims in a cluster for summarization."""
    lines = []
    for cid in claim_ids:
        if cid in claims_lookup:
            c = claims_lookup[cid]
            timestamp = format_timestamp(c.get("timestamp"))
            lines.append(
                f'- "{c.get("claim", "")}" [video: {c.get("video_id", "?")}, {timestamp}]'
            )
    return "\n".join(lines)


# Pipeline Stages


def stage2_cluster_claims(model, tokenizer, topic, claims, *, max_tokens, temperature):
    """Stage 2: Cluster claims by facet/sub-topic."""
    print(f"  [Stage 2] Clustering {len(claims)} claims...")

    # If very few claims, skip clustering — each claim is its own cluster
    if len(claims) <= 4:
        print(f"  [Stage 2] Only {len(claims)} claims, skipping clustering")
        return [{"label": c["claim"][:50], "claim_ids": [c["claim_id"]]} for c in claims]

    claims_text = format_claims_for_prompt(claims)
    prompt = CLUSTER_PROMPT.format(topic=topic, claims_text=claims_text)

    response = llm_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, temperature=temperature
    )
    result = parse_json_from_response(response)

    if result and "clusters" in result:
        clusters = result["clusters"]
        print(f"  [Stage 2] Found {len(clusters)} clusters")
        for cl in clusters:
            print(f'    - "{cl["label"]}": {len(cl["claim_ids"])} claims')
        return clusters

    # Fallback: put all claims in one cluster
    print("  [Stage 2] Clustering failed, using single cluster fallback")
    return [{"label": topic, "claim_ids": [c["claim_id"] for c in claims]}]


def stage3_rank_clusters(
    model, tokenizer, topic, clusters, claims_lookup, *, max_tokens, temperature
):
    """Stage 3: Rank clusters by relevance to query."""
    print(f"  [Stage 3] Ranking {len(clusters)} clusters...")

    if len(clusters) <= 1:
        print(f"  [Stage 3] Only {len(clusters)} cluster(s), skipping ranking")
        return clusters

    clusters_text = format_clusters_for_ranking(clusters, claims_lookup)
    prompt = RANK_PROMPT.format(topic=topic, clusters_text=clusters_text)

    response = llm_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, temperature=temperature
    )
    result = parse_json_from_response(response)

    if result and "ranked_labels" in result:
        ranked_labels = result["ranked_labels"]
        # Reorder clusters according to ranking
        label_to_cluster = {cl["label"]: cl for cl in clusters}
        ranked_clusters = [
            label_to_cluster[label] for label in ranked_labels if label in label_to_cluster
        ]

        # Add any clusters that weren't in the ranking (fallback)
        ranked_set = set(cl["label"] for cl in ranked_clusters)
        for cl in clusters:
            if cl["label"] not in ranked_set:
                ranked_clusters.append(cl)

        print("  [Stage 3] Ranking result:")
        for i, cl in enumerate(ranked_clusters, 1):
            print(f'    {i}. "{cl["label"]}"')
        return ranked_clusters

    print("  [Stage 3] Ranking failed, keeping original order")
    return clusters


def stage4_summarize_clusters(
    model, tokenizer, topic, ranked_clusters, claims_lookup, top_n, *, max_tokens, temperature
):
    """Stage 4: Summarize top-n clusters, one sentence each."""
    clusters_to_summarize = ranked_clusters[:top_n]
    print(f"  [Stage 4] Summarizing top {len(clusters_to_summarize)} clusters...")

    summaries = []
    for cl in clusters_to_summarize:
        cluster_claims_text = format_cluster_claims_for_summary(cl["claim_ids"], claims_lookup)
        prompt = SUMMARIZE_PROMPT.format(
            cluster_label=cl["label"],
            cluster_claims_text=cluster_claims_text,
        )

        summary = llm_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, temperature=temperature
        )
        summaries.append({"label": cl["label"], "summary": summary})
        print(f'    - "{cl["label"]}": {summary[:100]}...')

    return summaries


def stage5_improve_fluency(model, tokenizer, topic, summaries, *, max_tokens, temperature):
    """Stage 5: Combine summaries into a fluent report."""
    print("  [Stage 5] Improving fluency...")

    draft = "\n\n".join(s["summary"] for s in summaries)
    prompt = FLUENCY_PROMPT.format(topic=topic, draft_report=draft)

    return llm_generate(model, tokenizer, prompt, max_tokens=max_tokens, temperature=temperature)


def run_ginger_pipeline(model, tokenizer, topic, claims, top_n, g):
    """Run the full Ginger-style pipeline for one query.

    ``g`` is the ``runtime.ginger`` config sub-tree carrying per-stage knobs.
    """
    claims_lookup = {c["claim_id"]: c for c in claims}

    clusters = stage2_cluster_claims(
        model,
        tokenizer,
        topic,
        claims,
        max_tokens=g.cluster_max_tokens,
        temperature=g.cluster_temperature,
    )
    ranked_clusters = stage3_rank_clusters(
        model,
        tokenizer,
        topic,
        clusters,
        claims_lookup,
        max_tokens=g.rank_max_tokens,
        temperature=g.rank_temperature,
    )
    summaries = stage4_summarize_clusters(
        model,
        tokenizer,
        topic,
        ranked_clusters,
        claims_lookup,
        top_n,
        max_tokens=g.summarize_max_tokens,
        temperature=g.summarize_temperature,
    )
    final_report = stage5_improve_fluency(
        model,
        tokenizer,
        topic,
        summaries,
        max_tokens=g.fluency_max_tokens,
        temperature=g.fluency_temperature,
    )

    return {
        "clusters": clusters,
        "ranked_clusters": [cl["label"] for cl in ranked_clusters],
        "summaries": summaries,
        "final_report": final_report,
    }


def run(cfg: DictConfig) -> None:
    """Run the Ginger pipeline for the queries selected by ``cfg``."""
    claims_path = cfg.data.claims_path
    output_dir = cfg.output.out_dir
    model_name = cfg.model.model
    cache_dir = cfg.model.cache_dir or None
    top_n = cfg.runtime.top_n_clusters
    g = cfg.runtime.ginger

    os.makedirs(output_dir, exist_ok=True)

    # Load claims
    claims_by_query = load_claims(claims_path)

    # Load model
    model, tokenizer = load_model(model_name, cache_dir)

    # Determine which queries to process
    query_ids = resolve_query_ids(cfg.data.query_ids, claims_by_query.keys())

    output_path = os.path.join(output_dir, "reports_ginger.json")

    # Run pipeline for each query, writing each result to disk as it finishes so
    # progress is never lost if a later query fails.
    all_results = {}
    for qid in query_ids:
        if qid not in claims_by_query:
            print(f"WARNING: No claims found for query {qid}, skipping")
            continue

        claims = claims_by_query[qid]
        topic = claims[0]["topic"]
        print(f"\n{'=' * 60}")
        print(f"GINGER Pipeline — Query {qid} | Topic: {topic}")
        print(f"Number of claims: {len(claims)}")
        print(f"{'=' * 60}")

        result = run_ginger_pipeline(model, tokenizer, topic, claims, top_n, g)

        data = {
            "query_id": qid,
            "topic": topic,
            "num_claims": len(claims),
            "num_clusters": len(result["clusters"]),
            "cluster_ranking": result["ranked_clusters"],
            "summaries": result["summaries"],
            "report": result["final_report"],
        }
        all_results[qid] = data

        print("\n--- Final Report Preview (first 500 chars) ---")
        print(result["final_report"][:500])
        print("--- End Preview ---\n")

        # Persist this query's individual report immediately.
        txt_path = os.path.join(output_dir, f"report_ginger_q{qid}_{topic}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Query ID: {qid}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Claims used: {data['num_claims']}\n")
            f.write(f"Clusters found: {data['num_clusters']}\n")
            f.write(f"Cluster ranking: {data['cluster_ranking']}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write("## Cluster Summaries:\n\n")
            for s in data["summaries"]:
                f.write(f"[{s['label']}]: {s['summary']}\n\n")
            f.write(f"{'=' * 60}\n\n")
            f.write("## Final Report:\n\n")
            f.write(data["report"])
        print(f"  Saved: {txt_path}")

        # Update the combined JSON with everything completed so far.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll reports saved to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
