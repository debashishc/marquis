from __future__ import annotations

import argparse
import json
from pathlib import Path

from marquis.rlm_controller._common import build_config
from marquis.rlm_controller.rlm.rlm_repl import RLM_REPL
from marquis.rlm_controller.rlm.utils.magmar_data import (
    RLMDataPaths,
    add_data_path_args,
    build_context,
    build_query_string,
    data_paths_from_args,
    load_claims_for_query,
    load_query,
    load_topic_video_mapping,
    local_query_video_dir,
    resolve_video_ids,
    resolve_video_paths,
    split_config_overrides,
)
from marquis.rlm_controller.rlm.utils.prompts import build_system_prompt
from marquis.rlm_controller.tool_api import create_toolkit


def main(argv=None) -> None:
    cli_args, config_overrides = split_config_overrides(argv)
    cfg = build_config(config_overrides)
    data_defaults = RLMDataPaths.from_config(cfg.data)

    parser = argparse.ArgumentParser(description="Run the minimal RLM on one MAGMaR query")
    parser.add_argument(
        "--query_id",
        type=str,
        default=str(cfg.runtime.query_id),
        help="query_id from MAGMaR2026_queries.jsonl",
    )
    parser.add_argument(
        "--model", type=str, default=str(cfg.model.root_model), help="root LM model"
    )
    parser.add_argument(
        "--sub_model",
        type=str,
        default=str(cfg.model.sub_model),
        help="sub-LM model used by llm_query()",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=int(cfg.runtime.max_steps),
        help="maximum REPL iterations",
    )
    parser.add_argument("--enable_logging", action="store_true", help="enable colorful logging")
    parser.add_argument("--out-dir", default=str(cfg.output.out_dir))
    add_data_path_args(parser, data_defaults)
    args = parser.parse_args(cli_args)
    data_paths = data_paths_from_args(args)

    print(f"Loading MAGMaR query_id={args.query_id} ...")
    query = load_query(args.query_id, data_paths)
    topic_map = load_topic_video_mapping(data_paths)
    topic, video_ids = resolve_video_ids(query, topic_map, data_paths)
    video_paths = resolve_video_paths(args.query_id, video_ids, data_paths)
    claims = load_claims_for_query(args.query_id, data_paths)

    print(f"  topic:   {topic}")
    print(f"  videos:  {len(video_ids)}")
    print(f"  claims:  {len(claims)}")
    print(f"  queries: {data_paths.queries_jsonl}")
    print(f"  mapping: {data_paths.topic_mapping}")
    print(f"  claims:  {data_paths.claims_path}")
    local_video_dir = local_query_video_dir(args.query_id, data_paths)
    if local_video_dir.exists():
        print(f"  video source preference: local query dir -> {local_video_dir}")
    else:
        print(f"  video source preference: global root -> {data_paths.video_root}")

    context = build_context(query, topic, video_ids, video_paths, claims)
    context["artifacts"] = data_paths.as_dict()
    query_str = build_query_string(query)
    tools = create_toolkit(context)

    rlm = RLM_REPL(
        model=args.model,
        recursive_model=args.sub_model,
        max_iterations=args.max_iterations,
        enable_logging=args.enable_logging,
        extra_tools=tools,
        system_prompt_builder=build_system_prompt,
    )

    print(
        f"\nRunning minimal RLM on MAGMaR "
        f"(root={args.model}, sub={args.sub_model}, max_iter={args.max_iterations}) ...\n"
    )
    result = rlm.completion(context=context, query=query_str)

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(result)

    out_dir = Path(args.out_dir) / f"query_{args.query_id}"
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
    termination = "FINAL" if has_final else "fallback"
    per_iteration_execs = [
        {"iteration": t["iteration"], "num_executions": len(t.get("executions", []))}
        for t in rlm.trajectory
    ]

    global_exec_idx = 0
    for t in rlm.trajectory:
        for e in t.get("executions", []):
            e["global_execution_id"] = global_exec_idx
            global_exec_idx += 1

    trajectory_with_summary = list(rlm.trajectory) + [
        {
            "_summary": True,
            "version": "UNLI_pre_extracted",
            "total_iterations": total_iterations,
            "total_code_executions": total_executions,
            "per_iteration_executions": per_iteration_execs,
            "total_root_response_time_sec": round(total_root_time, 2),
            "total_code_execution_time_sec": round(total_exec_time, 2),
            "total_wall_time_sec": round(total_root_time + total_exec_time, 2),
            "termination": termination,
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
    print(f"\nSaved outputs to {out_dir}/")


if __name__ == "__main__":
    main()
