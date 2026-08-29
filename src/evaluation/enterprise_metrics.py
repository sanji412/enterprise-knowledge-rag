"""Deterministic metrics for the Chinese enterprise RAG benchmark."""

from __future__ import annotations


def hit_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Return 1 when any gold evidence anchor appears in the first *k* results."""
    return float(any(item in relevant for item in ranked[:k]))


def reciprocal_rank_at_k(
    ranked: list[str],
    relevant: set[str],
    k: int = 5,
) -> float:
    """Return the reciprocal rank of the first gold evidence anchor."""
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def fact_coverage(answer: str, required_facts: list[list[str]]) -> float:
    """Score required fact groups, accepting any alias inside each group."""
    if not required_facts:
        return 1.0
    normalized = answer.lower().replace(" ", "")
    hits = sum(any(alias.lower().replace(" ", "") in normalized for alias in group) for group in required_facts)
    return hits / len(required_facts)


def citation_accuracy(cited_anchors: list[str], gold: set[str]) -> float:
    """Return the share of cited evidence anchors that belong to the gold set."""
    if not cited_anchors:
        return 0.0
    return sum(anchor in gold for anchor in cited_anchors) / len(cited_anchors)


def refusal_correct(answerable: bool, status: str) -> float:
    """Score whether the pipeline answered or refused as the case requires."""
    return float((answerable and status == "answered") or (not answerable and status == "refused"))
