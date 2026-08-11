"""IR run evaluation adapter."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Any


def default_metrics() -> list[Any]:
    """Return the standard retrieval metrics."""
    from ir_measures import AP, RR, P, Recall, nDCG

    return [
        nDCG @ 10,
        nDCG @ 20,
        nDCG @ 100,
        Recall @ 10,
        Recall @ 20,
        Recall @ 100,
        P @ 5,
        P @ 10,
        AP,
        RR,
    ]


def evaluate_single_run(
    qrels_path: str, run_path: str, metrics: list[Any]
) -> tuple[dict[Any, float], dict[str, dict[str, float]]]:
    """Evaluate one TREC run file."""
    import ir_measures

    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(run_path)
    aggregate = ir_measures.calc_aggregate(metrics, qrels, run)

    per_query: dict[str, dict[str, float]] = defaultdict(dict)
    for measure in ir_measures.iter_calc(metrics, qrels, run):
        per_query[measure.query_id][str(measure.measure)] = measure.value
    return aggregate, dict(per_query)


def collect_run_files(run_dir: str | None, run_file: str | None) -> list[str]:
    """Collect run files from either one file or a directory."""
    if run_file:
        return [run_file]
    if run_dir:
        return sorted(glob.glob(os.path.join(run_dir, "*.trec")))
    raise ValueError("must specify --run-dir or --run-file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate IR runs with ir-measures")
    parser.add_argument("--qrels", required=True, help="Path to qrels file in TREC format")
    parser.add_argument(
        "--run-dir", "--run_dir", default=None, help="Directory containing .trec run files"
    )
    parser.add_argument("--run-file", "--run_file", default=None, help="Single .trec run file")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args(argv)

    try:
        run_files = collect_run_files(args.run_dir, args.run_file)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    if not run_files:
        print(f"No .trec files found in {args.run_dir}")
        return 1

    metrics = default_metrics()
    all_results: dict[str, dict[Any, float]] = {}
    for run_path in run_files:
        run_name = os.path.basename(run_path).replace(".trec", "")
        try:
            aggregate, _ = evaluate_single_run(args.qrels, run_path, metrics)
            all_results[run_name] = aggregate
            print(f"Evaluating: {run_name}")
            for metric, score in aggregate.items():
                print(f"  {metric}: {score:.4f}")
        except Exception as exc:
            print(f"  ERROR evaluating {run_name}: {exc}", file=sys.stderr)

    if args.output and all_results:
        metric_names = [str(metric) for metric in metrics]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run"] + metric_names)
            for run_name, aggregate in all_results.items():
                writer.writerow([run_name] + [f"{aggregate[metric]:.4f}" for metric in metrics])
        print(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
