from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from marquis.rlm_controller._common import build_config
from marquis.rlm_controller.rlm.rlm_repl import RLM_REPL
from marquis.rlm_controller.rlm.utils.magmar_data import (
    RLMDataPaths,
    add_data_path_args,
    build_context_vlm,
    build_query_string_vlm,
    data_paths_from_args,
    load_query,
    load_topic_video_mapping,
    local_query_video_dir,
    resolve_video_ids,
    resolve_video_paths,
    split_config_overrides,
)
from marquis.rlm_controller.rlm.utils.prompts import build_system_prompt_vlm
from marquis.rlm_controller.rlm.utils.vlm_backend import create_note_taking_fn
from marquis.rlm_controller.tool_api import create_toolkit


def main(argv=None) -> None:
    cli_args, config_overrides = split_config_overrides(argv)
    cfg = build_config(config_overrides)
    data_defaults = RLMDataPaths.from_config(cfg.data)

    parser = argparse.ArgumentParser(
        description="Run RLM VLM version (self-extraction) on a MAGMaR query"
    )
    parser.add_argument("--query_id", type=str, default=str(cfg.runtime.query_id))
    parser.add_argument("--model", type=str, default=str(cfg.model.root_model), help="root LM")
    parser.add_argument(
        "--sub_model", type=str, default=str(cfg.model.sub_model), help="sub-LM for llm_query()"
    )
    parser.add_argument("--max_iterations", type=int, default=int(cfg.runtime.max_steps))
    parser.add_argument("--enable_logging", action="store_true")
    parser.add_argument(
        "--vlm_backend",
        type=str,
        default=str(cfg.runtime.vlm_backend),
        choices=["local_qwen", "openai_vision"],
        help="VLM backend for note_taking()",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default=cfg.runtime.vlm_model,
        help="Model name for VLM backend (default depends on backend)",
    )
    parser.add_argument("--out-dir", default=str(cfg.output.out_dir))
    add_data_path_args(parser, data_defaults)
    args = parser.parse_args(cli_args)
    data_paths = data_paths_from_args(args)

    print(f"[VLM] Loading MAGMaR query_id={args.query_id} ...")
    query = load_query(args.query_id, data_paths)
    topic_map = load_topic_video_mapping(data_paths)
    topic, video_ids = resolve_video_ids(query, topic_map, data_paths)
    video_paths = resolve_video_paths(args.query_id, video_ids, data_paths)

    print(f"  topic:   {topic}")
    print(f"  videos:  {len(video_ids)}")
    print(f"  vlm:     {args.vlm_backend}")
    print(f"  queries: {data_paths.queries_jsonl}")
    print(f"  mapping: {data_paths.topic_mapping}")
    print(f"  videos root: {data_paths.video_root}")

    context = build_context_vlm(query, topic, video_ids, video_paths)
    context["artifacts"] = data_paths.as_dict()
    query_str = build_query_string_vlm(query)

    vlm_kwargs = {}
    if args.vlm_backend == "openai_vision":
        vlm_kwargs["openai_model"] = args.vlm_model or "gpt-5"
    elif args.vlm_backend == "local_qwen":
        vlm_kwargs["local_model"] = args.vlm_model or "Qwen/Qwen3.5-0.8B"

    note_taking_fn = create_note_taking_fn(backend=args.vlm_backend, **vlm_kwargs)
    tools = create_toolkit(context, note_taking_fn=note_taking_fn)
    tools["note_taking"] = note_taking_fn

    rlm = RLM_REPL(
        model=args.model,
        recursive_model=args.sub_model,
        max_iterations=args.max_iterations,
        enable_logging=args.enable_logging,
        extra_tools=tools,
        system_prompt_builder=build_system_prompt_vlm,
    )

    print(
        f"\n[VLM] Running RLM with self-extraction "
        f"(root={args.model}, sub={args.sub_model}, vlm={args.vlm_backend}, "
        f"max_iter={args.max_iterations}) ...\n"
    )
    wall_start = time.time()
    result = rlm.completion(context=context, query=query_str)
    wall_elapsed = time.time() - wall_start

    print("\n" + "=" * 80)
    print("FINAL REPORT (VLM — self-extraction)")
    print("=" * 80)
    print(result)

    out_dir = Path(args.out_dir) / f"query_{args.query_id}_vlm"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "report.txt").write_text(result, encoding="utf-8")
    (out_dir / "messages.json").write_text(
        json.dumps(rlm.messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_iterations = len(rlm.trajectory)
    total_executions = sum(len(t.get("executions", [])) for t in rlm.trajectory)
    total_root_time = sum(t.get("root_response_time", 0) for t in rlm.trajectory)
    total_exec_time = sum(
        e.get("execution_time", 0) for t in rlm.trajectory for e in t.get("executions", [])
    )
    has_final = any(t.get("final_answer") is not None for t in rlm.trajectory)

    global_exec_idx = 0
    for t in rlm.trajectory:
        for e in t.get("executions", []):
            e["global_execution_id"] = global_exec_idx
            global_exec_idx += 1

    trajectory_with_summary = list(rlm.trajectory) + [
        {
            "_summary": True,
            "version": "VLM_self_extraction",
            "vlm_backend": args.vlm_backend,
            "total_iterations": total_iterations,
            "total_code_executions": total_executions,
            "per_iteration_executions": [
                {"iteration": t["iteration"], "num_executions": len(t.get("executions", []))}
                for t in rlm.trajectory
            ],
            "total_root_response_time_sec": round(total_root_time, 2),
            "total_code_execution_time_sec": round(total_exec_time, 2),
            "total_wall_time_sec": round(wall_elapsed, 2),
            "termination": "FINAL" if has_final else "fallback",
            "max_iterations_budget": args.max_iterations,
        }
    ]

    (out_dir / "trajectory.json").write_text(
        json.dumps(trajectory_with_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[VLM] Wall time: {wall_elapsed:.1f}s")
    local_video_dir = local_query_video_dir(args.query_id, data_paths)
    if local_video_dir.exists():
        print(f"[VLM] Video preference: local -> {local_video_dir}")
    else:
        print(f"[VLM] Video preference: global -> {data_paths.video_root}")
    print(f"[VLM] Saved outputs to {out_dir}/")


if __name__ == "__main__":
    main()
