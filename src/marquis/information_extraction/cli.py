"""Command-line entrypoint for information extraction stages."""

from __future__ import annotations

import argparse
import sys

# Each route reads its config from the Hydra tree under
# configs/information_extraction/; remaining argv are passed through as
# `key=value` Hydra overrides. calibrate stages map onto runtime.calibrate.stage.
COMMANDS = (
    "build-query-mapping",
    "prepare-videos",
    "prepare-audio",
    "general-notes",
    "query-claims",
    "predict-unli",
    "calibrate-unli",
    "calibrate",
    "packets",
    "qa-decompose",
    "prepare-transcripts",
    "qa-answer",
    "qa",
    "validate",
)

HELP_TEXT = {
    "build-query-mapping": (
        "usage: marquis-extract build-query-mapping [Hydra key=value overrides]\n\n"
        "Write query_video_mapping.json from data.reference + data.queries_jsonl\n"
        "(joined by topic_id). Common overrides: data.reference=... "
        "data.queries_jsonl=... data.query_video_mapping=..."
    ),
    "prepare-videos": (
        "usage: marquis-extract prepare-videos [Hydra key=value overrides]\n\n"
        "Materialize loose {video_id}.mp4 into data.video_root from the WebDataset\n"
        "tar shards under data.video_shards_root. Common overrides: "
        "data.video_shards_root=... data.video_root=... "
        "data.query_video_mapping=... data.query_ids=..."
    ),
    "prepare-audio": (
        "usage: marquis-extract prepare-audio [Hydra key=value overrides]\n\n"
        "Materialize loose {chunk_id}{audio_ext} into data.audio_root from the\n"
        "audio WebDataset shards under data.video_shards_root/audio (QA's Whisper\n"
        "transcribes these). Common overrides: data.audio_root=... data.audio_ext=... "
        "data.query_video_mapping=... data.query_ids=..."
    ),
    "general-notes": (
        "usage: marquis-extract general-notes [Hydra key=value overrides]\n\n"
        "Common overrides: data.query_video_mapping=... data.video_root=... output.out_dir=..."
    ),
    "query-claims": (
        "usage: marquis-extract query-claims [Hydra key=value overrides]\n\n"
        "Common overrides: data.queries_jsonl=... data.query_video_mapping=... data.video_root=..."
    ),
    "predict-unli": (
        "usage: marquis-extract predict-unli [Hydra key=value overrides]\n\n"
        "Common overrides: data.artifacts_jsonl=... data.unli_jsonl=..."
    ),
    "calibrate-unli": (
        "usage: marquis-extract calibrate-unli [Hydra key=value overrides]\n\n"
        "Common overrides: data.unli_jsonl=... output.out_dir=..."
    ),
    "calibrate": (
        "usage: marquis-extract calibrate [Hydra key=value overrides]\n\n"
        "Common overrides: runtime.calibrate.stage=predict|calibrate|all"
    ),
    "packets": (
        "usage: marquis-extract packets [Hydra key=value overrides]\n\n"
        "Common overrides: runtime.packets.stream=general-note|query-based "
        "data.notes=... data.claims=..."
    ),
    "qa-decompose": (
        "usage: marquis-extract qa-decompose [Hydra key=value overrides]\n\n"
        "Common overrides: data.queries_jsonl=... data.subqueries_output=..."
    ),
    "prepare-transcripts": (
        "usage: marquis-extract prepare-transcripts [Hydra key=value overrides]\n\n"
        "Run Whisper alone over every video in data.query_video_mapping and cache\n"
        "{video_id: transcript} to data.transcripts, then exit (releasing the GPU\n"
        "before the qa step loads the VLM). Avoids the Whisper/vLLM CUDA-context\n"
        "clash on EXCLUSIVE_PROCESS GPUs. Common overrides: data.query_video_mapping=... "
        "data.audio_root=... data.transcripts=... data.query_ids=..."
    ),
    "qa-answer": (
        "usage: marquis-extract qa-answer [Hydra key=value overrides]\n\n"
        "Common overrides: data.qa_queries=... data.qa_output=..."
    ),
    "qa": (
        "usage: marquis-extract qa [Hydra key=value overrides]\n\n"
        "QA-based iterative extraction over query subquestions."
    ),
    "validate": "usage: marquis-extract validate [Hydra key=value overrides]",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        command, rest = argv[0], argv[1:]
        if rest and rest[0] in {"--help", "-h"}:
            print(HELP_TEXT[command])
            return 0
        if command == "build-query-mapping":
            from marquis.information_extraction.extract import (
                build_query_mapping_main as command_main,
            )
        elif command == "prepare-videos":
            from marquis.information_extraction.extract import (
                prepare_videos_main as command_main,
            )
        elif command == "prepare-audio":
            from marquis.information_extraction.extract import (
                prepare_audio_main as command_main,
            )
        elif command == "general-notes":
            from marquis.information_extraction.extract import general_notes_main as command_main
        elif command == "query-claims":
            from marquis.information_extraction.extract import query_claims_main as command_main
        elif command == "predict-unli":
            from marquis.information_extraction.calibrate import main as _m

            return int(_m(["runtime.calibrate.stage=predict", *rest]) or 0)
        elif command == "calibrate-unli":
            from marquis.information_extraction.calibrate import main as _m

            return int(_m(["runtime.calibrate.stage=calibrate", *rest]) or 0)
        elif command == "calibrate":
            from marquis.information_extraction.calibrate import main as command_main
        elif command == "packets":
            from marquis.information_extraction.assemble_packets import main as command_main
        elif command == "qa-decompose":
            from marquis.information_extraction.qa.query_direct_QA import main as command_main
        elif command == "prepare-transcripts":
            from marquis.information_extraction.qa.prepare_transcripts import main as command_main
        elif command == "qa-answer":
            from marquis.information_extraction.qa.query_answering import main as command_main
        elif command == "qa":
            from marquis.information_extraction.qa.query_iterative_questions import (
                main as command_main,
            )
        else:  # validate
            from marquis.common.validate_contracts import main as command_main
        return int(command_main(rest) or 0)

    parser = argparse.ArgumentParser(
        prog="marquis-extract",
        description="MARQUIS information extraction commands",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        subparsers.add_parser(command, help=f"Run {command}")

    args = parser.parse_args(argv)
    if args.command:
        parser.error(f"unknown command: {args.command}")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
