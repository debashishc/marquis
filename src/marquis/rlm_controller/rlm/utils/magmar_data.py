from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marquis.rlm_controller.tool_api import RLM_TOOL_SPECS


@dataclass(frozen=True)
class RLMDataPaths:
    """Path bundle for MAGMaR-backed RLM runs."""

    data_dir: Path
    queries_jsonl: Path
    topic_mapping: Path
    claims_path: Path
    local_magmar_dir: Path
    video_root: Path

    @classmethod
    def from_env(cls) -> RLMDataPaths:
        data_dir = Path(
            os.environ.get("MARQUIS_RLM_DATA_DIR", "")
        )
        return cls(
            data_dir=data_dir,
            queries_jsonl=Path(
                os.environ.get(
                    "MARQUIS_RLM_QUERIES_JSONL",
                    os.environ.get(
                        "MAGMAR_QUERIES_JSONL", str(data_dir / "MAGMaR2026_queries.jsonl")
                    ),
                )
            ),
            topic_mapping=Path(
                os.environ.get(
                    "MARQUIS_TOPIC_MAPPING",
                    os.environ.get(
                        "MAGMAR_TOPIC_MAPPING", str(data_dir / "topic_video_mapping.json")
                    ),
                )
            ),
            claims_path=Path(
                os.environ.get("MARQUIS_CLAIMS_PATH", str(data_dir / "features" / "claims"))
            ),
            local_magmar_dir=Path(os.environ.get("MARQUIS_LOCAL_MAGMAR_DIR", str(data_dir))),
            video_root=Path(os.environ.get("MAGMAR_VIDEO_ROOT", str(data_dir))),
        )

    @classmethod
    def from_config(cls, data_cfg: Any | None) -> RLMDataPaths:
        defaults = cls.from_env()
        if data_cfg is None:
            return defaults

        def value(name: str, default: Path) -> Path:
            if hasattr(data_cfg, "get"):
                raw = data_cfg.get(name, default)
            else:
                raw = getattr(data_cfg, name, default)
            return Path(str(raw))

        return cls(
            data_dir=value("data_dir", defaults.data_dir),
            queries_jsonl=value("queries_jsonl", defaults.queries_jsonl),
            topic_mapping=value("topic_mapping", defaults.topic_mapping),
            claims_path=value("claims_jsonl", defaults.claims_path),
            local_magmar_dir=value("local_magmar_dir", defaults.local_magmar_dir),
            video_root=value("video_root", defaults.video_root),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "queries_jsonl": str(self.queries_jsonl),
            "topic_mapping": str(self.topic_mapping),
            "claims_jsonl": str(self.claims_path),
            "local_magmar_dir": str(self.local_magmar_dir),
            "video_root": str(self.video_root),
        }


def _default_paths() -> RLMDataPaths:
    return RLMDataPaths.from_env()


_DEFAULT_PATHS = _default_paths()
DATA_DIR = _DEFAULT_PATHS.data_dir
QUERIES_PATH = _DEFAULT_PATHS.queries_jsonl
TOPIC_VIDEO_MAP_PATH = _DEFAULT_PATHS.topic_mapping
CLAIMS_PATH = _DEFAULT_PATHS.claims_path
LOCAL_MAGMAR_DIR = _DEFAULT_PATHS.local_magmar_dir
VIDEO_ROOT = _DEFAULT_PATHS.video_root


def add_data_path_args(parser: Any, defaults: RLMDataPaths) -> None:
    parser.add_argument("--data-dir", default=str(defaults.data_dir), help="RLM data directory")
    parser.add_argument(
        "--queries-jsonl", default=str(defaults.queries_jsonl), help="MAGMaR query JSONL"
    )
    parser.add_argument(
        "--topic-mapping",
        default=str(defaults.topic_mapping),
        help="Topic to video-id mapping JSON",
    )
    parser.add_argument(
        "--claims-jsonl",
        "--claims-path",
        dest="claims_jsonl",
        default=str(defaults.claims_path),
        help="Pre-extracted query-conditioned claims JSONL",
    )
    parser.add_argument(
        "--local-magmar-dir",
        default=str(defaults.local_magmar_dir),
        help="Optional local per-query MAGMaR directory with query_<id>/videos",
    )
    parser.add_argument(
        "--video-root", default=str(defaults.video_root), help="Fallback root for video files"
    )


def data_paths_from_args(args: Any) -> RLMDataPaths:
    return RLMDataPaths(
        data_dir=Path(args.data_dir),
        queries_jsonl=Path(args.queries_jsonl),
        topic_mapping=Path(args.topic_mapping),
        claims_path=Path(args.claims_jsonl),
        local_magmar_dir=Path(args.local_magmar_dir),
        video_root=Path(args.video_root),
    )


def split_config_overrides(argv: list[str] | None) -> tuple[list[str], list[str]]:
    cli_args: list[str] = []
    overrides: list[str] = []
    for token in list(argv or []):
        if "=" in token and not token.startswith("-"):
            overrides.append(token)
        else:
            cli_args.append(token)
    return cli_args, overrides


def load_query(query_id: str, paths: RLMDataPaths | None = None) -> dict:
    active_paths = paths or _default_paths()
    with open(active_paths.queries_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if str(item.get("query_id")) == str(query_id):
                return item
    raise ValueError(f"query_id={query_id} not found in {active_paths.queries_jsonl}")


def load_topic_video_mapping(paths: RLMDataPaths | None = None) -> dict:
    active_paths = paths or _default_paths()
    with open(active_paths.topic_mapping, encoding="utf-8") as f:
        return json.load(f)


def _normalize_key(text: str) -> str:
    return text.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_video_ids(
    query: dict,
    topic_map: dict,
    paths: RLMDataPaths | None = None,
) -> tuple[str, list[str]]:
    target = _normalize_key(str(query["title"]))
    for topic_key, vids in topic_map.items():
        if _normalize_key(topic_key) == target:
            return topic_key, list(vids)
    active_paths = paths or _default_paths()
    raise ValueError(f"No topic found for title={query['title']!r} in {active_paths.topic_mapping}")


def local_query_video_dir(query_id: str, paths: RLMDataPaths | None = None) -> Path:
    active_paths = paths or _default_paths()
    return active_paths.local_magmar_dir / f"query_{query_id}" / "videos"


def resolve_video_paths(
    query_id: str,
    video_ids: list[str],
    paths: RLMDataPaths | None = None,
) -> dict[str, str]:
    active_paths = paths or _default_paths()
    local_video_dir = local_query_video_dir(query_id, active_paths)
    resolved: dict[str, str] = {}
    for vid in video_ids:
        vid = str(vid)
        local_path = local_video_dir / f"{vid}.mp4"
        if local_path.exists():
            resolved[vid] = str(local_path)
        else:
            resolved[vid] = str(active_paths.video_root / f"{vid}.mp4")
    return resolved


def _topic_claims_path(topic: str, paths: RLMDataPaths) -> Path:
    if paths.claims_path.is_dir():
        candidates = [
            paths.claims_path / f"{topic}.json",
            paths.claims_path / f"{topic.replace(' ', '_')}.json",
            paths.claims_path / f"{topic.replace('-', '_')}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return paths.claims_path


def _load_claims_from_topic_json(
    *,
    query_id: str,
    topic: str,
    path: Path,
) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows: list[dict] = []
        for video_id, claims in data.items():
            if not isinstance(claims, list):
                continue
            for idx, claim in enumerate(claims):
                if isinstance(claim, dict):
                    text = claim.get("claim") or claim.get("text") or claim.get("fact")
                    evidence = claim.get("evidence")
                    confidence = claim.get("confidence")
                    timestamp = claim.get("timestamp")
                else:
                    text = str(claim)
                    evidence = None
                    confidence = None
                    timestamp = None
                rows.append(
                    {
                        "claim_id": f"qc-{query_id}-{video_id}-{idx:03d}",
                        "query_id": str(query_id),
                        "video_id": str(video_id),
                        "topic": topic,
                        "claim": text,
                        "confidence": confidence,
                        "evidence": evidence,
                        "source": "features_claims",
                        "timestamp": timestamp,
                        "source_path": str(path),
                        "is_post_grounded": False,
                    }
                )
        return rows

    if isinstance(data, list):
        rows = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                item.setdefault(
                    "claim_id", f"qc-{query_id}-{item.get('video_id', 'unknown')}-{idx:03d}"
                )
                item.setdefault("query_id", str(query_id))
                item.setdefault("topic", topic)
                item.setdefault("source_path", str(path))
                rows.append(item)
        return rows

    raise ValueError(f"Unsupported claims JSON shape in {path}")


def load_claims_for_query(query_id: str, paths: RLMDataPaths | None = None) -> list[dict]:
    active_paths = paths or _default_paths()
    if active_paths.claims_path.is_dir() or active_paths.claims_path.suffix.lower() == ".json":
        query = load_query(query_id, active_paths)
        topic_map = load_topic_video_mapping(active_paths)
        topic, _ = resolve_video_ids(query, topic_map, active_paths)
        claims_path = _topic_claims_path(topic, active_paths)
        if not claims_path.exists():
            raise FileNotFoundError(f"claims file not found for topic={topic}: {claims_path}")
        return _load_claims_from_topic_json(query_id=query_id, topic=topic, path=claims_path)

    claims: list[dict] = []
    with open(active_paths.claims_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if str(item.get("query_id")) == str(query_id):
                claims.append(item)
    return claims


def build_context(
    query: dict,
    topic: str,
    video_ids: list[str],
    video_paths: dict[str, str],
    claims: list[dict],
) -> dict:
    return {
        "task": {
            "query_id": query["query_id"],
            "title": query["title"],
            "persona_title": query["persona_title"],
            "background": query["background"],
            "query": query["query"],
            "language": query.get("language", ""),
            "query_type": query.get("query_type", ""),
        },
        "video_pool": {
            "topic": topic,
            "video_ids": video_ids,
            "video_paths": video_paths,
            "total_videos": len(video_ids),
        },
        "claims": [
            {
                "claim_id": c.get("claim_id"),
                "query_id": c.get("query_id"),
                "video_id": c.get("video_id"),
                "topic": c.get("topic"),
                "claim": c.get("claim"),
                "confidence": c.get("confidence"),
                "evidence": c.get("evidence"),
                "source": c.get("source"),
                "timestamp": c.get("timestamp"),
                "source_path": c.get("source_path"),
                "is_post_grounded": c.get("is_post_grounded"),
            }
            for c in claims
        ],
        "num_claims": len(claims),
        "tools": RLM_TOOL_SPECS,
    }


def build_query_string(query: dict) -> str:
    return (
        f"You are a {query['persona_title']}.\n"
        f"Background: {query['background']}\n\n"
        f"Task: {query['query']}\n\n"
        "The REPL variable `context` contains:\n"
        "- context['task']: query metadata\n"
        "- context['video_pool']: topic, video_ids, video_paths\n"
        "- context['claims']: pre-extracted video facts "
        "(claim_id, video_id, claim, confidence, evidence, source, timestamp)\n"
        "- context['num_claims']: total number of available claims\n\n"
        "Use the pre-loaded MARQUIS tools to inspect memory, collect query_claims by video, "
        "judge selected facts, and write a cited report. Only cite evidence that actually "
        "supports the sentence or claim you are making."
    )


def build_context_vlm(
    query: dict,
    topic: str,
    video_ids: list[str],
    video_paths: dict[str, str],
) -> dict:
    return {
        "task": {
            "query_id": query["query_id"],
            "title": query["title"],
            "persona_title": query["persona_title"],
            "background": query["background"],
            "query": query["query"],
            "language": query.get("language", ""),
            "query_type": query.get("query_type", ""),
        },
        "video_pool": {
            "topic": topic,
            "video_ids": video_ids,
            "video_paths": video_paths,
            "total_videos": len(video_ids),
        },
        "tools": RLM_TOOL_SPECS,
    }


def build_query_string_vlm(query: dict) -> str:
    return (
        f"You are a {query['persona_title']}.\n"
        f"Background: {query['background']}\n\n"
        f"Task: {query['query']}\n\n"
        "The REPL variable `context` contains:\n"
        "- context['task']: query metadata (persona, background, query text)\n"
        "- context['video_pool']: topic, video_ids, video_paths (video_id -> file path)\n\n"
        "You have the following tools available in the REPL:\n"
        "- llm_query(prompt): query a sub-LM for text-based semantic analysis\n"
        "- general_notes(vid), query_claims(vid), video_qa(vid, question): gather video evidence\n"
        "- llm_think(), llm_judge(), memory_summary(): update and inspect memory\n"
        "- write_report(facts): produce the final cited report\n"
        "- FINAL(answer) or FINAL_VAR(variable_name): submit your final report\n\n"
        "Strategy:\n"
        "1. Explore context['video_pool'] to see available videos\n"
        "2. Call MARQUIS evidence tools on selected videos to extract facts\n"
        "3. Analyze and organize the extracted facts\n"
        "4. Write a report with numbered citations (1), (2), (3)\n"
        "5. Include a References section mapping each number to video_id, evidence, timestamp\n"
        "6. Submit with FINAL() or FINAL_VAR()\n"
    )
