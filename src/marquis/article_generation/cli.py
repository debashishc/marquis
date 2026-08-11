"""Command-line entrypoint for article generation strategies."""

from __future__ import annotations

import argparse
import sys

# Strategies wired into the dispatcher. ``baseline`` / ``ginger`` / ``qa`` are
# single-shot generators; ``bullet`` is a structurally distinct three-step
# (infer→report→annotate) pipeline that takes its own positional sub-step.
COMMANDS = ("baseline", "bullet", "ginger", "qa")

HELP_TEXT = {
    "baseline": (
        "usage: marquis-generate baseline [Hydra key=value overrides]\n\n"
        "Common overrides: data.claims_path=... output.out_dir=... data.query_ids=..."
    ),
    "bullet": (
        "usage: marquis-generate bullet {infer|report|annotate} "
        "[--api URL] [--api-model NAME] [Hydra key=value overrides]\n\n"
        "Common overrides: runtime.bullet.stream=... runtime.bullet.packets_dir=... "
        "output.out_dir=...\n"
        "--api URL routes inference to a remote OpenAI-compatible server "
        "(shorthand for model.api=URL) instead of loading the VLM locally; "
        "--api-model NAME picks the model ID on that server (model.api_model=NAME)."
    ),
    "ginger": (
        "usage: marquis-generate ginger [Hydra key=value overrides]\n\n"
        "Common overrides: data.claims_path=... output.out_dir=... data.query_ids=..."
    ),
    "qa": (
        "usage: marquis-generate qa [Hydra key=value overrides]\n\n"
        "Common overrides: data.qa_dir=... data.qa_file=... output.out_dir=..."
    ),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        command, rest = argv[0], argv[1:]
        if rest and rest[0] in {"--help", "-h"}:
            print(HELP_TEXT[command])
            return 0
        if command == "baseline":
            from marquis.article_generation.baseline import main as baseline_main

            return baseline_main(rest)
        if command == "bullet":
            from marquis.article_generation.bullet import main as bullet_main

            return bullet_main(rest)
        if command == "ginger":
            from marquis.article_generation.ginger import main as ginger_main

            return ginger_main(rest)
        from marquis.article_generation.qa import main as qa_main

        return qa_main(rest)

    parser = argparse.ArgumentParser(
        prog="marquis-generate",
        description="MARQUIS article generation commands",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        subparsers.add_parser(command, help=f"Run {command} generation")

    args = parser.parse_args(argv)
    if args.command:
        parser.error(f"unknown command: {args.command}")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
