from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

_HEADING_ANCHOR = re.compile(
    r"\s*\{#([A-Za-z0-9][A-Za-z0-9_.:-]*)\}\s*$",
)


@dataclass(frozen=True)
class SourceUnit:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    evidence_anchor: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any]


def stable_chunk_id(
    document_id: str,
    section: str,
    chunk_index: int,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    safe_section = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in section
    ).strip("-") or "body"
    return f"{document_id}__{safe_section}__{chunk_index}__{digest}"


def parse_heading_anchor(title: str) -> tuple[str, str | None]:
    match = _HEADING_ANCHOR.search(title)
    if match is None:
        return title.strip(), None
    clean_title = title[:match.start()].strip()
    return clean_title, match.group(1)
