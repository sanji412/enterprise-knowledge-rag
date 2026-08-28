"""Lightweight citation verification heuristics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.core.chinese_tokenizer import tokenize_search_text


class CitationVerifier:
    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(tokenize_search_text(text or ""))

    def score_citation(
        self,
        response_text: str,
        citation: Dict[str, Any],
        documents: Sequence[Dict[str, Any]],
    ) -> float:
        chunk_id = str(citation.get("chunk_id", ""))
        doc = next((d for d in documents if str(d.get("id")) == chunk_id), None)
        if not doc:
            return 0.0
        claim_text = str(citation.get("claim_text") or response_text)
        response_terms = self._tokenize(claim_text)
        doc_terms = self._tokenize(str(doc.get("text", "")))
        if not response_terms:
            return 0.0
        overlap = len(response_terms & doc_terms) / max(len(response_terms), 1)
        return max(0.0, min(1.0, 0.25 + overlap * 0.75))

    def verify(
        self,
        response_text: str,
        citations: Sequence[Dict[str, Any]],
        documents: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for citation in citations:
            score = self.score_citation(response_text, citation, documents)
            verdict = "supported" if score >= 0.55 else "weak_support"
            if not citation.get("resolved"):
                verdict = "unresolved"
            output.append({**citation, "verification_score": score, "verification": verdict})
        return output
