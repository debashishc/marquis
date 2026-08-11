"""Small JSONL helpers used across MARQUIS stages."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-empty line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    """Write records to JSONL, creating the parent directory when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
