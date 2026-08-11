"""Command-line entrypoint for retrieval utilities."""

from __future__ import annotations

import argparse
import sys

# expand / prepare-subqueries / retrieve / rrf / fusion / qrels / evaluate. Each
# route reads its config from the Hydra tree under configs/retrieval/; remaining
# argv are passed through as `key=value` Hydra overrides.
COMMANDS = (
    "expand",
    "prepare-subqueries",
    "retrieve",
    "rrf",
    "fusion",
    "rerank",
    "qrels",
    "evaluate",
)

HELP_TEXT = {
    "expand": (
        "usage: marquis-retrieve expand [Hydra key=value overrides]\n\n"
        "Example: marquis-retrieve expand data.queries_file=... output.out_dir=..."
    ),
    "prepare-subqueries": (
        "usage: marquis-retrieve prepare-subqueries [Hydra key=value overrides]\n\n"
        "Common overrides: data.expanded_queries=... data.subqueries_jsonl=... "
        "data.subquery_mapping=..."
    ),
    "retrieve": (
        "usage: marquis-retrieve retrieve [Hydra key=value overrides]\n\n"
        "Common overrides: data.video_links=... data.video_dir=... "
        "data.retrieved_videos=..."
    ),
    "rrf": (
        "usage: marquis-retrieve rrf [Hydra key=value overrides]\n\n"
        "Common overrides: data.subquery_results=... data.subquery_mapping=... "
        "output.out_dir=..."
    ),
    "fusion": (
        "usage: marquis-retrieve fusion [Hydra key=value overrides]\n\n"
        "Common overrides: data.subquery_results=... data.subquery_mapping=... "
        "output.out_dir=..."
    ),
    "rerank": (
        "usage: marquis-retrieve rerank --run RUN --rankvideo-scores SCORES "
        "--output OUT [--depth 100]"
    ),
    "qrels": (
        "usage: marquis-retrieve qrels [Hydra key=value overrides]\n\n"
        "Common overrides: data.reference=... data.queries_file=... "
        "data.qrels=..."
    ),
    "evaluate": (
        "usage: marquis-retrieve evaluate [Hydra key=value overrides]\n\n"
        "Common overrides: data.qrels=... runtime.eval.run_dir=... "
        "runtime.eval.output_csv=..."
    ),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        command, rest = argv[0], argv[1:]
        if rest and rest[0] in {"--help", "-h"}:
            print(HELP_TEXT[command])
            return 0
        if command == "expand":
            from marquis.retrieval.query_expansion import main as command_main
        elif command == "prepare-subqueries":
            from marquis.retrieval.prepare_subqueries import main as command_main
        elif command == "retrieve":
            from marquis.retrieval.video_retrieval import main as command_main
        elif command == "rrf":
            from marquis.retrieval.rrf import main as command_main
        elif command == "fusion":
            from marquis.retrieval.fusion import main as command_main
        elif command == "rerank":
            from marquis.retrieval.reranking import main as command_main
        elif command == "qrels":
            from marquis.retrieval.qrels import main as command_main
        else:
            from marquis.retrieval.evaluate import main as command_main
        return int(command_main(rest) or 0)

    parser = argparse.ArgumentParser(
        prog="marquis-retrieve",
        description="MARQUIS retrieval utilities",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("expand", help="Decompose queries into sub-queries (LLM)")
    subparsers.add_parser("prepare-subqueries", help="Flatten expanded queries for retrieval")
    subparsers.add_parser("retrieve", help="OmniEmbed dense search → sub-query rank list (GPU)")
    subparsers.add_parser("rrf", help="Single-method fusion → submission JSON")
    subparsers.add_parser("fusion", help="Generate the standard fusion run suite (.trec)")
    subparsers.add_parser("rerank", help="Rerank a first-stage run with RankVideo scores")
    subparsers.add_parser("qrels", help="Build TREC qrels from topic mappings")
    subparsers.add_parser("evaluate", help="Evaluate run files with ir-measures")

    args = parser.parse_args(argv)
    if args.command:
        parser.error(f"unknown command: {args.command}")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
