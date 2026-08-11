"""Command-line entrypoint for MARQUIS evaluation."""

from __future__ import annotations

import argparse
import sys

COMMANDS = {"extraction", "retrieval"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        if argv[0] == "retrieval":
            from marquis.evaluation.retrieval import main as retrieval_main

            return retrieval_main(argv[1:])
        if argv[0] == "extraction":
            from marquis.evaluation.extraction import main as extraction_main

            result = extraction_main(argv[1:])
            return int(result or 0)

    parser = argparse.ArgumentParser(
        prog="marquis-evaluate",
        description="MARQUIS evaluation commands",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("retrieval", help="Run retrieval evaluation")
    for command in sorted(COMMANDS - {"retrieval"}):
        subparsers.add_parser(command, help=f"Run {command} evaluation")

    args = parser.parse_args(argv)
    if args.command:
        parser.error(f"unknown command: {args.command}")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
