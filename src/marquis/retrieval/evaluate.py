"""Evaluate fusion run files with ir-measures against a qrels file.

Computes nDCG / Recall / P / AP / RR for one run or all ``*.trec`` runs in a
directory and prints aggregate + per-query tables (optional CSV). Paths/knobs
come from ``configs/retrieval/`` (``runtime.eval.*`` defaults to output.out_dir)::

    python -m retrieval.cli evaluate data.qrels=<qrels> runtime.eval.run_dir=<dir>
"""

from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

from omegaconf import DictConfig

from marquis.retrieval._common import build_config


def _metrics():
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


def evaluate_single_run(qrels_path: str, run_path: str, metrics):
    """Evaluate one run file; return (aggregate, per_query)."""
    import ir_measures

    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(run_path)
    agg = ir_measures.calc_aggregate(metrics, qrels, run)
    per_query = defaultdict(dict)
    for m in ir_measures.iter_calc(metrics, qrels, run):
        per_query[m.query_id][str(m.measure)] = m.value
    return agg, per_query


def run(cfg: DictConfig) -> None:
    """Evaluate one or all run files and print metric tables."""
    qrels_path = cfg.data.qrels
    run_dir = cfg.runtime.eval.run_dir or cfg.output.out_dir
    run_file = cfg.runtime.eval.run_file
    output_csv = cfg.runtime.eval.output_csv

    metrics = _metrics()

    if run_file:
        run_files = [run_file]
    elif run_dir:
        run_files = sorted(glob.glob(os.path.join(run_dir, "*.trec")))
    else:
        raise SystemExit("must set runtime.eval.run_file or runtime.eval.run_dir")
    if not run_files:
        raise SystemExit(f"No .trec files found in {run_dir}")

    print(f"Qrels: {qrels_path}")
    print(f"Found {len(run_files)} run file(s) to evaluate\n")

    metric_names = [str(m) for m in metrics]
    all_results = {}
    for run_path in run_files:
        run_name = os.path.basename(run_path).replace(".trec", "")
        print(f"Evaluating: {run_name}")
        try:
            agg, per_query = evaluate_single_run(qrels_path, run_path, metrics)
            all_results[run_name] = agg
            for metric, score in agg.items():
                print(f"  {metric}: {score:.4f}")
            print()

            print(f"  Per-query breakdown for {run_name}:")
            header = f"  {'Query':<10}" + "".join(f"{m:<15}" for m in metric_names)
            print(header)
            print("  " + "-" * len(header))
            for qid in sorted(per_query.keys()):
                row = f"  {qid:<10}" + "".join(
                    f"{per_query[qid].get(m, 0.0):<15.4f}" for m in metric_names
                )
                print(row)
            print()
        except Exception as e:  # noqa: BLE001 - report and continue to next run
            print(f"  ERROR: {e}\n")

    if len(all_results) > 1:
        print("=" * 80)
        print("SUMMARY TABLE (all runs)")
        print("=" * 80)
        header = f"{'Run':<35}" + "".join(f"{m:<15}" for m in metric_names)
        print(header)
        print("-" * len(header))
        for run_name, agg in all_results.items():
            row = f"{run_name:<35}" + "".join(f"{agg[m]:<15.4f}" for m in metrics)
            print(row)
        print()

    if output_csv and all_results:
        import csv

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run"] + metric_names)
            for run_name, agg in all_results.items():
                writer.writerow([run_name] + [f"{agg[m]:.4f}" for m in metrics])
        print(f"Results saved to: {output_csv}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
