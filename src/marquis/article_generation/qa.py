"""QA article generation: synthesize a QA system's outputs into a cited report.

Consumes the question-answer pairs produced by the retrieval QA stage (``iter_q``
or ``ss`` answer dumps), groups them back under their seed query, then writes a
report via the ``baseline`` and/or ``ginger`` method.

- ``iter_q`` uses SEQUENTIAL grouping: each seed subquery plus its dynamically
  generated follow-ups belong to the same query_id.
- ``ss`` uses text-matching: each subquery matches one original seed.

Configuration is driven entirely by the Hydra config tree under
``configs/article_generation/`` (see ``data.qa_*`` and ``runtime.qa_*``).
Override on the command line with Hydra syntax::

    python -m article_generation.cli qa data.qa_file=ss runtime.qa_method=baseline
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from omegaconf import DictConfig

from marquis.article_generation._common import (
    build_config,
    llm_generate,
    load_model,
    parse_json_from_response,
    resolve_query_ids,
)
from marquis.article_generation.prompts import (
    QA_BASELINE_PROMPT as BASELINE_PROMPT,
)
from marquis.article_generation.prompts import (
    QA_CLUSTER_PROMPT as CLUSTER_PROMPT,
)
from marquis.article_generation.prompts import (
    QA_FLUENCY_PROMPT as FLUENCY_PROMPT,
)
from marquis.article_generation.prompts import (
    QA_RANK_PROMPT as RANK_PROMPT,
)
from marquis.article_generation.prompts import (
    QA_SUMMARIZE_PROMPT as SUMMARIZE_PROMPT,
)

# QA loading / grouping


def load_queries_info(queries_file: str):
    """Load query metadata and the seed subqueries per query."""
    queries_info = {}
    seed_subqueries = {}
    with open(queries_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            qid = d["query_id"]
            queries_info[qid] = {
                "query_id": qid,
                "title": d.get("title", ""),
                "query": d.get("query", ""),
            }
            seed_subqueries[qid] = [sq.strip() for sq in d.get("sub_queries", [])]
    return queries_info, seed_subqueries


def extract_video_id(video_path: str) -> str:
    if not video_path:
        return ""
    name = os.path.basename(str(video_path))
    name = name.rsplit(".", 1)[0]
    if name.endswith("_ds"):
        name = name[:-3]
    return name


def normalize_qa_item(d: dict, qa_type: str) -> dict:
    if qa_type == "iter_q":
        videos = d.get("sources", [])
        video_ids = [extract_video_id(v) for v in videos] if isinstance(videos, list) else []
        answer = d.get("final_answer", "")
    else:  # ss
        answer_field = d.get("answer", ["", []])
        if isinstance(answer_field, list) and len(answer_field) >= 2:
            answer = answer_field[0]
            videos = answer_field[1] if isinstance(answer_field[1], list) else []
            video_ids = [extract_video_id(v) for v in videos]
        else:
            answer = str(answer_field)
            video_ids = []
    return {
        "subquery": d.get("subquery", "").strip(),
        "video_ids": video_ids,
        "answer": answer,
    }


def load_qa_items(qa_path: str, qa_type: str) -> list[dict]:
    items = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(normalize_qa_item(json.loads(line), qa_type))
    print(f"Loaded {len(items)} QA items from {qa_path}")
    return items


def is_useful_answer(answer: str) -> bool:
    if not answer:
        return False
    a = answer.strip().lower()
    if a in ("i don't know", "i don't know.", "i do not know", "i do not know."):
        return False
    if len(answer.strip()) < 10:
        return False
    return True


def group_qa_by_query_iter_q(qa_items: list[dict], seed_subqueries: dict) -> dict:
    """iter_q: a known seed subquery starts a new query block; follow-ups inherit it."""
    seed_to_qid = {}
    for qid, seeds in seed_subqueries.items():
        for sq in seeds:
            seed_to_qid[sq] = qid

    by_query = defaultdict(list)
    current_qid = None
    unmatched_before_seed = 0
    for item in qa_items:
        sq = item["subquery"]
        if sq in seed_to_qid:
            current_qid = seed_to_qid[sq]
        if current_qid is None:
            unmatched_before_seed += 1
            continue
        if not is_useful_answer(item["answer"]):
            continue
        by_query[current_qid].append(item)

    if unmatched_before_seed > 0:
        print(f"  Note: {unmatched_before_seed} items before any seed subquery were skipped")
    return by_query


def group_qa_by_query_ss(qa_items: list[dict], seed_subqueries: dict) -> dict:
    seed_to_qid = {}
    for qid, seeds in seed_subqueries.items():
        for sq in seeds:
            seed_to_qid[sq] = qid

    by_query = defaultdict(list)
    unmatched = 0
    for item in qa_items:
        qid = seed_to_qid.get(item["subquery"])
        if qid is None:
            unmatched += 1
            continue
        if not is_useful_answer(item["answer"]):
            continue
        by_query[qid].append(item)

    if unmatched > 0:
        print(f"  Note: {unmatched} subqueries did not match any seed")
    return by_query


def group_qa(qa_items: list[dict], seed_subqueries: dict, qa_type: str) -> dict:
    if qa_type == "iter_q":
        by_query = group_qa_by_query_iter_q(qa_items, seed_subqueries)
    else:
        by_query = group_qa_by_query_ss(qa_items, seed_subqueries)

    print(f"Grouped into {len(by_query)} queries")
    for qid in sorted(by_query.keys(), key=lambda x: int(x)):
        print(f"  Query {qid}: {len(by_query[qid])} QA pairs")
    return by_query


# QA formatting


def format_qa_for_baseline(qa_items: list[dict]) -> str:
    lines = []
    for i, qa in enumerate(qa_items, 1):
        vids = ", ".join(qa["video_ids"]) if qa["video_ids"] else "no source"
        lines.append(f"Q{i}: {qa['subquery']}\nA{i}: {qa['answer']}\nSources: [{vids}]")
    return "\n\n".join(lines)


def format_qa_for_clustering(qa_items: list[dict]) -> str:
    lines = []
    for i, qa in enumerate(qa_items):
        vids = ", ".join(qa["video_ids"]) if qa["video_ids"] else "no source"
        ans = qa["answer"][:300] + ("..." if len(qa["answer"]) > 300 else "")
        lines.append(f"[{i}] Q: {qa['subquery']}\n    A: {ans}\n    Sources: [{vids}]")
    return "\n\n".join(lines)


def format_qa_for_summary(qa_ids: list[int], qa_items: list[dict]) -> str:
    lines = []
    for qid in qa_ids:
        if qid >= len(qa_items):
            continue
        qa = qa_items[qid]
        vids = ", ".join(qa["video_ids"]) if qa["video_ids"] else "no source"
        ans = qa["answer"][:400] + ("..." if len(qa["answer"]) > 400 else "")
        lines.append(f"- Q: {qa['subquery']}\n  A: {ans}\n  [{vids}]")
    return "\n".join(lines)


# Report methods


def run_baseline(model, tokenizer, query, qa_items, *, max_qa):
    if len(qa_items) > max_qa:
        print(f"  WARNING: {len(qa_items)} QA pairs exceeds limit, truncating to {max_qa}")
        qa_items = qa_items[:max_qa]
    qa_text = format_qa_for_baseline(qa_items)
    prompt = BASELINE_PROMPT.format(query=query, qa_text=qa_text)
    return llm_generate(model, tokenizer, prompt)


def run_ginger(model, tokenizer, query, qa_items, top_n, *, max_qa):
    if len(qa_items) > max_qa:
        qa_items = qa_items[:max_qa]

    print(f"  [Stage 2] Clustering {len(qa_items)} QA pairs...")
    if len(qa_items) <= 4:
        clusters = [{"label": qa["subquery"][:60], "qa_ids": [i]} for i, qa in enumerate(qa_items)]
    else:
        qa_text = format_qa_for_clustering(qa_items)
        prompt = CLUSTER_PROMPT.format(query=query, qa_text=qa_text)
        response = llm_generate(model, tokenizer, prompt, max_tokens=2048, temperature=0.3)
        result = parse_json_from_response(response)
        if result and "clusters" in result:
            clusters = result["clusters"]
        else:
            clusters = [{"label": "General", "qa_ids": list(range(len(qa_items)))}]
    print(f"  [Stage 2] Found {len(clusters)} clusters")

    print("  [Stage 3] Ranking clusters...")
    if len(clusters) > 1:
        clusters_text = ""
        for cl in clusters:
            sqs = [
                f"  - {qa_items[qid]['subquery']}" for qid in cl["qa_ids"] if qid < len(qa_items)
            ]
            clusters_text += f'Cluster: "{cl["label"]}"\n' + "\n".join(sqs) + "\n\n"
        prompt = RANK_PROMPT.format(query=query, clusters_text=clusters_text)
        response = llm_generate(model, tokenizer, prompt, max_tokens=512, temperature=0.3)
        result = parse_json_from_response(response)
        if result and "ranked_labels" in result:
            label_to_cluster = {cl["label"]: cl for cl in clusters}
            ranked = [
                label_to_cluster[label]
                for label in result["ranked_labels"]
                if label in label_to_cluster
            ]
            ranked_set = set(cl["label"] for cl in ranked)
            for cl in clusters:
                if cl["label"] not in ranked_set:
                    ranked.append(cl)
            clusters = ranked

    top_clusters = clusters[:top_n]
    print(f"  [Stage 4] Summarizing top {len(top_clusters)} clusters...")
    summaries = []
    for cl in top_clusters:
        cluster_qa_text = format_qa_for_summary(cl["qa_ids"], qa_items)
        prompt = SUMMARIZE_PROMPT.format(cluster_label=cl["label"], cluster_qa_text=cluster_qa_text)
        summary = llm_generate(model, tokenizer, prompt, max_tokens=256, temperature=0.5)
        summaries.append({"label": cl["label"], "summary": summary})

    print("  [Stage 5] Improving fluency...")
    draft = "\n\n".join(s["summary"] for s in summaries)
    prompt = FLUENCY_PROMPT.format(query=query, draft_report=draft)
    final_report = llm_generate(model, tokenizer, prompt)

    return {
        "clusters": [cl["label"] for cl in clusters],
        "summaries": summaries,
        "report": final_report,
    }


def run(cfg: DictConfig) -> None:
    """Run the QA-based report writer for the queries selected by ``cfg``."""
    import torch

    qa_dir = cfg.data.qa_dir
    qa_file = cfg.data.qa_file
    queries_with_subqueries = cfg.data.queries_with_subqueries
    qa_path = os.path.join(qa_dir, "qa_outputs", f"answers_{qa_file}.json")

    output_dir = cfg.output.out_dir
    model_name = cfg.model.model
    cache_dir = cfg.model.cache_dir or None
    top_n = cfg.runtime.top_n_clusters
    max_qa = cfg.runtime.max_qa_per_query
    method = cfg.runtime.qa_method

    run_baseline_flag = method in ("baseline", "both")
    run_ginger_flag = method in ("ginger", "both")

    os.makedirs(output_dir, exist_ok=True)

    queries_info, seed_subqueries = load_queries_info(queries_with_subqueries)
    qa_items = load_qa_items(qa_path, qa_file)
    qa_by_query = group_qa(qa_items, seed_subqueries, qa_file)

    model, tokenizer = load_model(model_name, cache_dir)

    query_ids = resolve_query_ids(cfg.data.query_ids, qa_by_query.keys())

    baseline_results = {}
    ginger_results = {}

    out_baseline = os.path.join(output_dir, f"reports_qa_{qa_file}_baseline.json")
    out_ginger = os.path.join(output_dir, f"reports_qa_{qa_file}_ginger.json")

    for qid in query_ids:
        if qid not in qa_by_query:
            print(f"No QA for query {qid}, skipping")
            continue

        qa_list = qa_by_query[qid]
        query_info = queries_info.get(qid, {})
        query_text = query_info.get("query", "")
        title = query_info.get("title", qid)

        print(f"\n{'=' * 60}")
        print(f"Query {qid} | Title: {title} | QA pairs: {len(qa_list)}")
        print(f"{'=' * 60}")

        if run_baseline_flag:
            print("\n--- Running Baseline ---")
            try:
                baseline_report = run_baseline(model, tokenizer, query_text, qa_list, max_qa=max_qa)
                baseline_results[qid] = {
                    "query_id": qid,
                    "title": title,
                    "query": query_text,
                    "num_qa": len(qa_list),
                    "report": baseline_report,
                }
                print(f"Preview: {baseline_report[:300]}...")
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on baseline for query {qid}")
                torch.cuda.empty_cache()

        if run_ginger_flag:
            print("\n--- Running Ginger ---")
            try:
                ginger_result = run_ginger(
                    model, tokenizer, query_text, qa_list, top_n, max_qa=max_qa
                )
                ginger_results[qid] = {
                    "query_id": qid,
                    "title": title,
                    "query": query_text,
                    "num_qa": len(qa_list),
                    "clusters": ginger_result["clusters"],
                    "summaries": ginger_result["summaries"],
                    "report": ginger_result["report"],
                }
                print(f"Preview: {ginger_result['report'][:300]}...")
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on ginger for query {qid}")
                torch.cuda.empty_cache()

        if run_baseline_flag:
            with open(out_baseline, "w", encoding="utf-8") as f:
                json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        if run_ginger_flag:
            with open(out_ginger, "w", encoding="utf-8") as f:
                json.dump(ginger_results, f, indent=2, ensure_ascii=False)

    print("\nDone!")
    if run_baseline_flag:
        print(f"  Baseline: {out_baseline}")
    if run_ginger_flag:
        print(f"  Ginger:   {out_ginger}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
