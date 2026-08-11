#!/usr/bin/env python3
"""
PR1 acceptance gate: validate all frozen contracts.

Checks:
  1. Every official query maps to exactly one topic (deterministic, reproducible).
  2. Fixtures validate against frozen schemas.
  3. claims.jsonl compatibility with calibrate_unli.py is testable.
  4. Expanded queries map to official query IDs.
  5. is_post_grounded semantics are correct across fixtures.

Usage:
    python marquis-validate-contracts
    python marquis-validate-contracts --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path

from marquis.common.contracts import (
    build_query_topic_map,
    load_expanded_queries,
    load_queries,
    validate_claim_packet,
    validate_compat_claim,
    validate_fact,
    validate_general_note,
    validate_grounded_note,
    validate_higher_level_inference,
    validate_note_packet,
    validate_observation_note,
    validate_query_conditioned_claim,
    validate_query_packet,
    validate_report_citation,
)

FIXTURES_DIR = Path(os.environ.get("MARQUIS_FIXTURES_DIR", "tests/fixtures"))


def _load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_query_topic_mapping(verbose: bool = False) -> list:
    """Every query maps to exactly one topic; mapping is deterministic."""
    errors = []
    queries = load_queries()

    try:
        qtm = build_query_topic_map(queries)
    except ValueError as e:
        return [f"query-topic mapping failed: {e}"]

    # Check every query appeared exactly once
    mapped_query_ids = set()
    for _topic, qs in qtm.items():
        for q in qs:
            qid = q["query_id"]
            if qid in mapped_query_ids:
                errors.append(f"query {qid} mapped to multiple topics")
            mapped_query_ids.add(qid)

    all_query_ids = {q["query_id"] for q in queries}
    unmapped = all_query_ids - mapped_query_ids
    if unmapped:
        errors.append(f"queries not mapped to any topic: {sorted(unmapped)}")

    # Check determinism: run twice
    qtm2 = build_query_topic_map(queries)
    for topic in qtm:
        ids1 = [q["query_id"] for q in qtm.get(topic, [])]
        ids2 = [q["query_id"] for q in qtm2.get(topic, [])]
        if ids1 != ids2:
            errors.append(f"mapping not deterministic for topic {topic}")

    if verbose and not errors:
        for topic in sorted(qtm.keys()):
            qs = qtm[topic]
            qids = [q["query_id"] for q in qs]
            print(f"  {topic}: queries {qids}")

    return errors


def check_fixtures(verbose: bool = False) -> list:
    """All fixtures validate against their schemas."""
    errors = []

    validators = [
        ("observation_note.json", validate_observation_note),
        ("grounded_note.json", validate_grounded_note),
        ("compat_claim.json", validate_compat_claim),
        ("query_packet.json", validate_query_packet),
        ("report_citation.json", validate_report_citation),
        ("fact.json", validate_fact),
        ("general_note.json", validate_general_note),
        ("query_conditioned_claim.json", validate_query_conditioned_claim),
        ("note_packet.json", validate_note_packet),
        ("claim_packet.json", validate_claim_packet),
        ("higher_level_inference.json", validate_higher_level_inference),
    ]

    for fname, validator in validators:
        path = FIXTURES_DIR / fname
        if not os.path.exists(path):
            errors.append(f"fixture missing: {fname}")
            continue
        fixture = _load_fixture(fname)
        errs = validator(fixture)
        if errs:
            errors.extend([f"fixture {fname}: {e}" for e in errs])
        elif verbose:
            print(f"  {fname}: OK")

    return errors


def check_claims_compat(verbose: bool = False) -> list:
    """Verify compat claim shape works with calibrate_unli.py expectations."""
    errors = []

    # calibrate_unli.py reads: video_rec.get("video_id"), c.get("claim")
    claim = _load_fixture("compat_claim.json")
    if "claim" not in claim:
        errors.append("compat_claim missing 'claim' field (calibrate_unli.py dependency)")

    # Also check that a claims.jsonl record shape would work
    test_record = {
        "topic": "test",
        "video_id": "test_vid",
        "video_path": None,
        "caption": None,
        "claims": [claim],
        "provenance": {"mode": "test", "method": "test"},
    }

    vid = test_record.get("video_id")
    claims = test_record.get("claims", [])
    if not vid or not isinstance(claims, list):
        errors.append("claims.jsonl record shape incompatible with calibrate_unli.py")
    else:
        for c in claims:
            if not isinstance(c, dict) or "claim" not in c:
                errors.append("claim item missing 'claim' key")

    if verbose and not errors:
        print("  claims.jsonl compat: OK")

    return errors


def check_expanded_queries(verbose: bool = False) -> list:
    """Expanded queries map to valid official query IDs."""
    errors = []
    queries = load_queries()
    official_ids = {q["query_id"] for q in queries}

    expanded = load_expanded_queries()
    for qid in expanded:
        if qid not in official_ids:
            errors.append(f"expanded query ID '{qid}' not in official queries")

    if verbose and not errors:
        print(f"  expanded queries: {len(expanded)} entries, all match official IDs")

    return errors


def check_is_post_grounded_semantics(verbose: bool = False) -> list:
    """Verify is_post_grounded semantics across fixtures."""
    errors = []

    # These should have is_post_grounded == false
    false_fixtures = [
        "observation_note.json",
        "general_note.json",
    ]
    for fname in false_fixtures:
        path = FIXTURES_DIR / fname
        if not os.path.exists(path):
            continue
        fixture = _load_fixture(fname)
        val = fixture.get("is_post_grounded")
        if val is not False:
            errors.append(f"{fname}: is_post_grounded should be false, got {val!r}")
        elif verbose:
            print(f"  {fname}: is_post_grounded=false OK")

    # These should have is_post_grounded == true
    true_fixtures = [
        "grounded_note.json",
        "higher_level_inference.json",
    ]
    for fname in true_fixtures:
        path = FIXTURES_DIR / fname
        if not os.path.exists(path):
            continue
        fixture = _load_fixture(fname)
        val = fixture.get("is_post_grounded")
        if val is not True:
            errors.append(f"{fname}: is_post_grounded should be true, got {val!r}")
        elif verbose:
            print(f"  {fname}: is_post_grounded=true OK")

    # query_conditioned_claim should be false
    qc_path = FIXTURES_DIR / "query_conditioned_claim.json"
    if os.path.exists(qc_path):
        fixture = _load_fixture("query_conditioned_claim.json")
        val = fixture.get("is_post_grounded")
        if val is not False:
            errors.append(
                f"query_conditioned_claim.json: is_post_grounded should be false, got {val!r}"
            )
        elif verbose:
            print("  query_conditioned_claim.json: is_post_grounded=false OK")

    return errors


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Validate PR1 contracts")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    all_errors = []

    checks = [
        ("Query-topic mapping", check_query_topic_mapping),
        ("Fixture schemas", check_fixtures),
        ("claims.jsonl compatibility", check_claims_compat),
        ("Expanded queries", check_expanded_queries),
        ("is_post_grounded semantics", check_is_post_grounded_semantics),
    ]

    for name, check_fn in checks:
        print(f"[check] {name}")
        errors = check_fn(verbose=args.verbose)
        if errors:
            for e in errors:
                print(f"  FAIL: {e}")
            all_errors.extend(errors)
        else:
            print("  PASS")

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
