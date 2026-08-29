"""Structured citation mapping utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

_DOC_RE = re.compile(r"\[Doc\s+([^\]]+)\]", re.IGNORECASE)


class CitationTracker:
    def extract_raw_ids(self, text: str) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for match in _DOC_RE.finditer(text or ""):
            raw = match.group(1).strip()
            if raw and raw not in seen:
                seen.add(raw)
                ordered.append(raw)
        return ordered

    def build_index_lookup(self, documents: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        lookup: Dict[int, Dict[str, Any]] = {}
        for i, doc in enumerate(documents, start=1):
            lookup[i] = doc
        return lookup

    def map_citations(self, response_text: str, documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        known = {str(d.get("id")): d for d in documents}
        by_index = self.build_index_lookup(documents)
        mapped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for match in _DOC_RE.finditer(response_text or ""):
            raw_id = match.group(1).strip()
            if not raw_id or raw_id in seen:
                continue
            seen.add(raw_id)
            doc = known.get(raw_id)
            resolved = doc is not None
            chunk_id = raw_id
            if not resolved and raw_id.isdigit():
                indexed_doc = by_index.get(int(raw_id))
                if indexed_doc is not None:
                    doc = indexed_doc
                    chunk_id = str(indexed_doc.get("id"))
                    resolved = True
            metadata = (doc or {}).get("metadata") or {}
            mapped.append(
                {
                    "raw_id": raw_id,
                    "chunk_id": chunk_id,
                    "resolved": resolved,
                    "title": metadata.get("title") or (doc or {}).get("title"),
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "section_title": metadata.get("section_title"),
                    "source": metadata.get("source") or metadata.get("file_type"),
                    "evidence_anchor": metadata.get("evidence_anchor"),
                    "claim_text": self._claim_for_marker(
                        response_text,
                        match.start(),
                        match.end(),
                    ),
                    "text_preview": str((doc or {}).get("text", ""))[:240],
                }
            )
        return mapped

    @staticmethod
    def _claim_for_marker(text: str, start: int, end: int) -> str:
        boundaries = "。！？!?；;\n"
        left = max((text.rfind(char, 0, start) for char in boundaries), default=-1)
        right_candidates = [
            position
            for char in boundaries
            if (position := text.find(char, end)) >= 0
        ]
        right = min(right_candidates) + 1 if right_candidates else len(text)
        sentence = text[left + 1:right]
        clean = _DOC_RE.sub("", sentence).strip()
        clean = clean.rstrip("。！？!?；;").strip()
        if clean:
            return clean

        # Models often put a citation on the line immediately after the claim.
        # In that layout the marker's own sentence is empty, so associate it with
        # the nearest preceding non-empty sentence instead.
        prefix = text[:start].rstrip().rstrip(boundaries).rstrip()
        previous_left = max(
            (prefix.rfind(char) for char in boundaries),
            default=-1,
        )
        previous_sentence = prefix[previous_left + 1:]
        return _DOC_RE.sub("", previous_sentence).strip().rstrip("。！？!?；;").strip()
