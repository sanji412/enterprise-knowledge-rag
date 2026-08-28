from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from src.core.retrieval_result import RetrievalResult


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass(frozen=True)
class GroundingDecision:
    status: AnswerStatus
    reason: str | None = None


class GroundingPolicy:
    def __init__(
        self,
        min_retrieval_confidence: float = 0.05,
        min_citation_score: float = 0.55,
    ) -> None:
        self.min_retrieval_confidence = min_retrieval_confidence
        self.min_citation_score = min_citation_score

    def before_generation(
        self,
        retrieved: Sequence[RetrievalResult],
    ) -> GroundingDecision:
        confidence = max(
            (row.confidence for row in retrieved),
            default=0.0,
        )
        if not retrieved or confidence < self.min_retrieval_confidence:
            return GroundingDecision(
                AnswerStatus.REFUSED,
                "no_relevant_evidence",
            )
        return GroundingDecision(AnswerStatus.ANSWERED)

    def after_generation(
        self,
        retrieved: Sequence[RetrievalResult],
        citations: Sequence[dict[str, Any]],
    ) -> GroundingDecision:
        initial = self.before_generation(retrieved)
        if initial.status is AnswerStatus.REFUSED:
            return initial
        all_supported = bool(citations) and all(
            citation.get("resolved")
            and citation.get("verification") == "supported"
            and float(citation.get("verification_score", 0.0))
            >= self.min_citation_score
            for citation in citations
        )
        if not all_supported:
            return GroundingDecision(
                AnswerStatus.REFUSED,
                "unverified_citations",
            )
        return GroundingDecision(AnswerStatus.ANSWERED)
