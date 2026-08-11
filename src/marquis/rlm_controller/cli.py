"""Command-line entrypoint for MARQUIS RLM controller runs."""

from __future__ import annotations

import argparse
import sys

COMMANDS = {"magmar", "magmar-notes"}

HELP_TEXT = {
    "magmar": (
        "usage: marquis-rlm magmar [options] [Hydra key=value overrides]\n\n"
        "Self-extraction RLM over MAGMaR videos.\n"
        "Common options: --query_id --queries-jsonl --topic-mapping --video-root "
        "--local-magmar-dir --model --sub_model --vlm_backend --out-dir"
    ),
    "magmar-notes": (
        "usage: marquis-rlm magmar-notes [options] [Hydra key=value overrides]\n\n"
        "RLM over pre-extracted claims.\n"
        "Common options: --query_id --queries-jsonl --topic-mapping --claims-jsonl "
        "--video-root --local-magmar-dir --model --sub_model --out-dir"
    ),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        command, rest = argv[0], argv[1:]
        if rest and rest[0] in {"--help", "-h"}:
            print(HELP_TEXT[command])
            return 0
        if command == "magmar":
            from marquis.rlm_controller.magmar import main as command_main
        else:
            from marquis.rlm_controller.magmar_with_notes import main as command_main
        result = command_main(rest)
        return int(result or 0)

    parser = argparse.ArgumentParser(
        prog="marquis-rlm",
        description="MARQUIS RLM controller commands",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in sorted(COMMANDS):
        subparsers.add_parser(command, help=f"Run {command}")

    args = parser.parse_args(argv)
    if args.command:
        parser.error(f"unknown command: {args.command}")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
