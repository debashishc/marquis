#!/usr/bin/env python3
"""
PR1: Frozen data contracts and dataset interfaces.

This module is the single source of truth for:
  - loading the official queries
  - loading reference.json (topic_id -> relevant chunk/video ids)
  - grouping queries by topic_id and deriving the query -> video mapping
  - artifact schemas for general notes, query-conditioned claims, note packets,
    claim packets, higher-level inferences, and claims (compat)
  - legacy schemas retained for observation_notes, grounded_notes, and query_packet
  - schema validation helpers

No later module may redefine query-topic matching or artifact shapes.
"""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# Paths / model identifiers (env-overridable defaults).
# These back the loader-function defaults below and the argparse / fallback
# defaults in the information_extraction pipeline. Runtime config is otherwise
# supplied by the Hydra tree under top-level `configs/`.
# ---------------------------------------------------------------------------


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DEFAULT_VIDEO_ROOT = _env("MAGMAR_VIDEO_ROOT", "")
# MicroVENT WebDataset root (<root>/videos/catalog.csv + shard_*.tar); source for
# `marquis-extract prepare-videos`, which materializes loose mp4s into VIDEO_ROOT.
DEFAULT_VIDEO_SHARDS_ROOT = _env("MARQUIS_VIDEO_SHARDS_ROOT", "")
DEFAULT_QUERIES_JSONL = _env(
    "MAGMAR_QUERIES_JSONL", os.path.join(DEFAULT_VIDEO_ROOT, "MAGMaR2026_queries.jsonl")
)
DEFAULT_REFERENCE = _env(
    "MARQUIS_REFERENCE", os.path.join(DEFAULT_VIDEO_ROOT, "reference.json")
)
DEFAULT_EXPANDED_QUERIES = _env(
    "MAGMAR_EXPANDED_QUERIES", os.path.join(DEFAULT_VIDEO_ROOT, "expanded_queries.json")
)
DEFAULT_CAPTIONS_JSONL = _env(
    "MAGMAR_CAPTIONS_JSONL",
    os.path.join(DEFAULT_VIDEO_ROOT, "captions/Qwen3-Omni-30B-A3B-Instruct/captions.jsonl"),
)

# Model identifiers (single source of truth for argparse / fallback defaults)
DEFAULT_VLM_MODEL = _env("MARQUIS_VLM_MODEL", "Qwen/Qwen3.5-2B")
DEFAULT_UNLI_MODEL = _env("MARQUIS_UNLI_MODEL", "AdoptedIrelia/UNLI")
DEFAULT_UNLI_LORA_PATH = _env("MARQUIS_UNLI_LORA_PATH", "AdoptedIrelia/UNLI/lora")

# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

QUERY_FIELDS = (
    "query_id",
    "query_type",
    "language",
    "persona_title",
    "background",
    "query",
)

def load_queries(path: str = DEFAULT_QUERIES_JSONL) -> list[dict]:
    """Load queries from JSONL. Each query carries a ``topic_id`` that joins it to
    its reference topic (see :func:`build_query_topic_map`)."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in QUERY_FIELDS:
                if field not in rec:
                    raise ValueError(f"Query record missing field '{field}': {rec}")
            if not rec.get("topic_id"):
                raise ValueError(f"Query record needs a 'topic_id' to join to a topic: {rec}")
            queries.append(rec)
    return queries


def load_expanded_queries(path: str = DEFAULT_EXPANDED_QUERIES) -> dict[str, dict]:
    """Load expanded sub-queries keyed by query_id."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reference (topic_id -> relevant chunk/video ids)
# ---------------------------------------------------------------------------


def load_reference(path: str = DEFAULT_REFERENCE) -> dict[str, list[str]]:
    """Load ``reference.json`` and return ``{topic_id: [chunk_id, ...]}``.

    ``reference.json`` is the curated source of truth for which videos are
    relevant to each topic: its ``topics`` list carries a ``topic_id`` and the
    ``chunks`` (video ids) that ground that topic's claims. Queries join to a
    topic by ``topic_id`` (see :func:`build_query_topic_map`).
    """
    with open(path) as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for topic in data.get("topics", []):
        tid = topic.get("topic_id")
        if tid is None:
            continue
        out[str(tid)] = [str(c) for c in (topic.get("chunks") or [])]
    return out


# ---------------------------------------------------------------------------
# Query -> topic grouping
# ---------------------------------------------------------------------------


def build_query_topic_map(queries: list[dict]) -> dict[str, list[dict]]:
    """Group queries by their ``topic_id``.

    Returns ``{topic_id: [query_dict, ...]}``. Every query must carry a
    ``topic_id`` (the join key into ``reference.json``); raises ValueError
    otherwise.
    """
    result: dict[str, list[dict]] = {}
    missing = [q for q in queries if not q.get("topic_id")]
    if missing:
        labels = [q.get("query_id") for q in missing]
        raise ValueError(f"{len(missing)} query/queries missing topic_id: {labels}")
    for q in queries:
        result.setdefault(str(q["topic_id"]), []).append(q)
    return result


def build_query_video_map(
    queries: list[dict],
    reference: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build ``{query_id: [video_id, ...]}`` by joining each query to its topic.

    ``reference`` is the ``{topic_id: [chunk_id, ...]}`` table from
    :func:`load_reference`. Each query's videos are its ``topic_id``'s chunks, so
    the biased/unbiased pair on one topic share that topic's video list. Used to
    *generate* the persisted ``query_video_mapping.json`` artifact that
    information extraction then loads with :func:`load_query_video_mapping`.
    """
    return {
        str(q["query_id"]): list(reference.get(str(q["topic_id"]), []))
        for q in queries
    }


def load_query_video_mapping(path: str) -> dict[str, list[str]]:
    """Load a ``query_id -> [video_id, ...]`` mapping (the query-keyed video table).

    This is the video-selection source for information extraction: each query
    reads its own video list directly, no topic matching at run time. Build the
    file from ``reference.json`` with :func:`build_query_video_map`.
    """
    with open(path) as f:
        data = json.load(f)
    return {
        str(qid): [str(v) for v in vids]
        for qid, vids in data.items()
        if isinstance(vids, list)
    }


# ---------------------------------------------------------------------------
# Caption loading
# ---------------------------------------------------------------------------


def load_captions_index(
    path: str = DEFAULT_CAPTIONS_JSONL,
) -> dict[str, dict]:
    """Build {normalized_video_id: caption_record} from captions JSONL."""
    idx: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            vid = meta.get("video_id") or rec.get("video_id") or rec.get("doc_id")
            vid_norm = normalize_video_id(str(vid or ""))
            if vid_norm and vid_norm not in idx:
                idx[vid_norm] = rec
    return idx


def normalize_video_id(video_id: str) -> str:
    """Strip known prefixes (youtube-, etc.) from video IDs."""
    s = str(video_id or "").strip()
    if s.startswith("youtube-"):
        return s[len("youtube-") :]
    return s


# ---------------------------------------------------------------------------
# Video file resolution
# ---------------------------------------------------------------------------


def resolve_video_path(video_root: str, video_id: str) -> str | None:
    """Find the .mp4 file for a video ID, handling common mismatches."""
    direct = os.path.join(video_root, f"{video_id}.mp4")
    if os.path.exists(direct):
        return direct
    # Try prefix match (handles S0C0crjCBvM3 -> S0C0crjCBvM.mp4)
    try:
        for fname in os.listdir(video_root):
            if fname.endswith(".mp4") and fname.startswith(video_id[:11]):
                candidate = os.path.join(video_root, fname)
                if os.path.exists(candidate):
                    return candidate
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Artifact schemas
# ---------------------------------------------------------------------------

OBSERVATION_NOTE_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
    "topic": str,
    "modality": str,  # visual | ocr | audio
    "text": str,
}

OBSERVATION_NOTE_OPTIONAL_FIELDS = {
    "timestamp": (list, type(None)),
    "extractor": str,
    "confidence": (float, int, type(None)),
    "source_path": (str, type(None)),
    "is_post_grounded": bool,
}

GROUNDED_NOTE_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
    "topic": str,
    "claim": str,
    "source_observation_ids": list,
}

GROUNDED_NOTE_OPTIONAL_FIELDS = {
    "timestamp_union": (list, type(None)),
    "calibration": (dict, type(None)),
    "extractor": (str, type(None)),
    "is_post_grounded": bool,
}

COMPAT_CLAIM_REQUIRED_FIELDS = {
    "claim": str,
}

COMPAT_CLAIM_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "evidence": (str, type(None)),
    "source": (str, type(None)),
    "timestamp": (list, type(None)),
    "calibration": (dict, type(None)),
}

FACT_REQUIRED_FIELDS = {
    "fact": str,
    "video_id": str,
}

FACT_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "evidence": (str, type(None)),
    "source": (str, type(None)),
    "timestamp": (list, type(None)),
    "video_path": (str, type(None)),
    "caption": (str, type(None)),
    "ocr": (str, type(None)),
}

GENERAL_NOTE_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
    "topic": str,
    "text": str,
    "modality": str,  # visual | ocr | audio
}
GENERAL_NOTE_OPTIONAL_FIELDS = {
    "timestamp": (list, type(None)),
    "extractor": (str, type(None)),
    "confidence": (float, int, type(None)),
    "source_path": (str, type(None)),
    "calibration": (dict, type(None)),
    "run_id": (str, type(None)),
    "is_post_grounded": bool,  # always false for Step 1a
}

QUERY_CONDITIONED_CLAIM_REQUIRED_FIELDS = {
    "claim_id": str,
    "query_id": str,
    "video_id": str,
    "topic": str,
    "claim": str,
}
QUERY_CONDITIONED_CLAIM_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "evidence": (str, type(None)),
    "source": (str, type(None)),
    "timestamp": (list, type(None)),
    "source_path": (str, type(None)),
    "calibration": (dict, type(None)),
    "run_id": (str, type(None)),
    "is_post_grounded": bool,  # always false for Step 1b
}

NOTE_PACKET_REQUIRED_FIELDS = {
    "query_id": str,
    "topic": str,
    "stream": str,  # "general_note"
}
NOTE_PACKET_OPTIONAL_FIELDS = {
    "note_ids": list,
    "scores": list,
    "provenance": dict,
}

CLAIM_PACKET_REQUIRED_FIELDS = {
    "query_id": str,
    "topic": str,
    "stream": str,  # "query_based"
}
CLAIM_PACKET_OPTIONAL_FIELDS = {
    "claim_ids": list,
    "scores": list,
    "provenance": dict,
}

HIGHER_LEVEL_INFERENCE_REQUIRED_FIELDS = {
    "inference_id": str,
    "query_id": str,
    "topic": str,
    "claim": str,
    "source_ids": list,  # note_ids or claim_ids from the packet
}
HIGHER_LEVEL_INFERENCE_OPTIONAL_FIELDS = {
    "confidence": (float, int, type(None)),
    "is_post_grounded": bool,  # always true for Step 2
    "calibration": (dict, type(None)),
    "stream": (str, type(None)),
    "run_id": (str, type(None)),
}

QUERY_PACKET_REQUIRED_FIELDS = {
    "query_id": str,
    "topic": str,
    "pipeline": str,  # "note_taking" | "single_query" | "expanded_query"
}

QUERY_PACKET_OPTIONAL_FIELDS = {
    "retrieved_note_ids": list,
    "retrieved_fact_ids": list,
    "scores": list,
    "provenance": dict,
}

REPORT_CITATION_REQUIRED_FIELDS = {
    "note_id": str,
    "video_id": str,
}

REPORT_CITATION_OPTIONAL_FIELDS = {
    "timestamp": (list, type(None)),
    "claim": (str, type(None)),
}

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

ENUM_FIELDS = {
    ("observation_note", "modality"): {"visual", "ocr", "audio"},
    ("general_note", "modality"): {"visual", "ocr", "audio"},
    ("compat_claim", "source"): {
        "caption",
        "video_visual",
        "video_text",
        "video_audio",
        "transcript",
        "visual",
        "ocr",
        "audio",
    },
    ("fact", "source"): {
        "caption",
        "video_visual",
        "video_text",
        "video_audio",
        "transcript",
        "visual",
        "ocr",
        "audio",
    },
    ("query_conditioned_claim", "source"): {
        "caption",
        "video_visual",
        "video_text",
        "video_audio",
        "transcript",
        "visual",
        "ocr",
        "audio",
    },
    ("note_packet", "stream"): {"general_note"},
    ("claim_packet", "stream"): {"query_based"},
    ("higher_level_inference", "stream"): {"general_note", "query_based"},
    ("query_packet", "pipeline"): {
        "note_taking",
        "single_query",
        "expanded_query",
        "general_note",
        "query_based",
    },
}

LIST_ELEMENT_TYPES = {
    ("observation_note", "timestamp"): (int, float, str),
    ("grounded_note", "source_observation_ids"): str,
    ("grounded_note", "timestamp_union"): (int, float, str),
    ("compat_claim", "timestamp"): (int, float, str),
    ("fact", "timestamp"): (int, float, str),
    ("general_note", "timestamp"): (int, float, str),
    ("query_conditioned_claim", "timestamp"): (int, float, str),
    ("note_packet", "note_ids"): str,
    ("note_packet", "scores"): (int, float),
    ("claim_packet", "claim_ids"): str,
    ("claim_packet", "scores"): (int, float),
    ("higher_level_inference", "source_ids"): str,
    ("query_packet", "retrieved_note_ids"): str,
    ("query_packet", "retrieved_fact_ids"): str,
    ("query_packet", "scores"): (int, float),
    ("report_citation", "timestamp"): (int, float, str),
}


def _type_name(expected_type) -> str:
    if isinstance(expected_type, tuple):
        return " | ".join(_type_name(t) for t in expected_type)
    return expected_type.__name__


def _matches_type(value, expected_type) -> bool:
    expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    if value is None and type(None) in expected_types:
        return True
    if bool in expected_types:
        return isinstance(value, bool)
    if isinstance(value, bool) and any(t in expected_types for t in (int, float)):
        return False
    return isinstance(value, expected_types)


def _check_list_elements(record: dict, field: str, label: str) -> list[str]:
    if field not in record or record[field] is None:
        return []
    expected = LIST_ELEMENT_TYPES.get((label, field))
    if expected is None:
        return []
    errors = []
    for idx, value in enumerate(record[field]):
        expected_types = expected if isinstance(expected, tuple) else (expected,)
        if isinstance(value, bool) and any(t in expected_types for t in (int, float)):
            ok = False
        else:
            ok = isinstance(value, expected)
        if not ok:
            errors.append(
                f"{label}: field '{field}[{idx}]' expected {_type_name(expected)}, "
                f"got {type(value).__name__}"
            )
    return errors


def _check_fields(
    record: dict,
    required: dict,
    optional: dict,
    label: str,
) -> list[str]:
    """Validate a single record against required/optional field specs. Returns errors."""
    errors = []
    allowed_fields = set(required) | set(optional)
    for field in record:
        if field not in allowed_fields:
            errors.append(f"{label}: unknown field '{field}'")
    for field, expected_type in required.items():
        if field not in record:
            errors.append(f"{label}: missing required field '{field}'")
        elif not _matches_type(record[field], expected_type):
            errors.append(
                f"{label}: field '{field}' expected {_type_name(expected_type)}, "
                f"got {type(record[field]).__name__}"
            )
        else:
            enum_values = ENUM_FIELDS.get((label, field))
            if enum_values is not None and record[field] not in enum_values:
                errors.append(
                    f"{label}: field '{field}' expected one of {sorted(enum_values)}, "
                    f"got {record[field]!r}"
                )
            errors.extend(_check_list_elements(record, field, label))
    for field, expected_types in optional.items():
        if field in record:
            if not _matches_type(record[field], expected_types):
                errors.append(
                    f"{label}: field '{field}' expected {_type_name(expected_types)}, "
                    f"got {type(record[field]).__name__}"
                )
            else:
                enum_values = ENUM_FIELDS.get((label, field))
                if (
                    enum_values is not None
                    and record[field] is not None
                    and record[field] not in enum_values
                ):
                    errors.append(
                        f"{label}: field '{field}' expected one of {sorted(enum_values)}, "
                        f"got {record[field]!r}"
                    )
                errors.extend(_check_list_elements(record, field, label))
    return errors


def validate_observation_note(note: dict) -> list[str]:
    return _check_fields(
        note,
        OBSERVATION_NOTE_REQUIRED_FIELDS,
        OBSERVATION_NOTE_OPTIONAL_FIELDS,
        "observation_note",
    )


def validate_grounded_note(note: dict) -> list[str]:
    return _check_fields(
        note,
        GROUNDED_NOTE_REQUIRED_FIELDS,
        GROUNDED_NOTE_OPTIONAL_FIELDS,
        "grounded_note",
    )


def validate_compat_claim(claim: dict) -> list[str]:
    return _check_fields(
        claim,
        COMPAT_CLAIM_REQUIRED_FIELDS,
        COMPAT_CLAIM_OPTIONAL_FIELDS,
        "compat_claim",
    )


def validate_query_packet(packet: dict) -> list[str]:
    return _check_fields(
        packet,
        QUERY_PACKET_REQUIRED_FIELDS,
        QUERY_PACKET_OPTIONAL_FIELDS,
        "query_packet",
    )


def validate_fact(fact: dict) -> list[str]:
    return _check_fields(
        fact,
        FACT_REQUIRED_FIELDS,
        FACT_OPTIONAL_FIELDS,
        "fact",
    )


def validate_report_citation(citation: dict) -> list[str]:
    return _check_fields(
        citation,
        REPORT_CITATION_REQUIRED_FIELDS,
        REPORT_CITATION_OPTIONAL_FIELDS,
        "report_citation",
    )


def validate_general_note(note: dict) -> list[str]:
    return _check_fields(
        note,
        GENERAL_NOTE_REQUIRED_FIELDS,
        GENERAL_NOTE_OPTIONAL_FIELDS,
        "general_note",
    )


def validate_query_conditioned_claim(claim: dict) -> list[str]:
    return _check_fields(
        claim,
        QUERY_CONDITIONED_CLAIM_REQUIRED_FIELDS,
        QUERY_CONDITIONED_CLAIM_OPTIONAL_FIELDS,
        "query_conditioned_claim",
    )


def validate_note_packet(packet: dict) -> list[str]:
    return _check_fields(
        packet,
        NOTE_PACKET_REQUIRED_FIELDS,
        NOTE_PACKET_OPTIONAL_FIELDS,
        "note_packet",
    )


def validate_claim_packet(packet: dict) -> list[str]:
    return _check_fields(
        packet,
        CLAIM_PACKET_REQUIRED_FIELDS,
        CLAIM_PACKET_OPTIONAL_FIELDS,
        "claim_packet",
    )


def validate_higher_level_inference(inference: dict) -> list[str]:
    return _check_fields(
        inference,
        HIGHER_LEVEL_INFERENCE_REQUIRED_FIELDS,
        HIGHER_LEVEL_INFERENCE_OPTIONAL_FIELDS,
        "higher_level_inference",
    )


# ---------------------------------------------------------------------------
# Convenience: load everything needed for a pipeline run
# ---------------------------------------------------------------------------


def load_all(
    *,
    queries_jsonl: str = DEFAULT_QUERIES_JSONL,
    reference: str = DEFAULT_REFERENCE,
    expanded_queries: str = DEFAULT_EXPANDED_QUERIES,
    captions_jsonl: str | None = None,
) -> dict:
    """
    Load all frozen data contracts in one call.

    Returns dict with keys: queries, reference, query_topic_map,
    query_video_map, expanded. Includes captions_idx only when captions_jsonl is
    provided for legacy flows.
    """
    queries = load_queries(queries_jsonl)
    ref = load_reference(reference)
    query_topic_map = build_query_topic_map(queries)
    query_video_map = build_query_video_map(queries, ref)
    expanded = load_expanded_queries(expanded_queries)
    bundle = {
        "queries": queries,
        "reference": ref,
        "query_topic_map": query_topic_map,
        "query_video_map": query_video_map,
        "expanded": expanded,
    }
    if captions_jsonl:
        bundle["captions_idx"] = load_captions_index(captions_jsonl)
    return bundle
