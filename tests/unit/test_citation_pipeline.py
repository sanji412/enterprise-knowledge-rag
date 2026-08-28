from src.core.citation_tracker import CitationTracker
from src.core.citation_verifier import CitationVerifier


def test_citation_tracker_resolves_chunk_ids_and_numeric_indices():
    docs = [
        {"id": "docA__chunk0", "text": "alpha beta", "metadata": {"title": "A"}},
        {"id": "docB__chunk1", "text": "gamma delta", "metadata": {"title": "B"}},
    ]
    tracker = CitationTracker()
    mapped = tracker.map_citations("Answer [Doc docA__chunk0] and [Doc 2]", docs)
    assert mapped[0]["resolved"] is True
    assert mapped[0]["chunk_id"] == "docA__chunk0"
    assert mapped[1]["chunk_id"] == "docB__chunk1"


def test_citation_verifier_scores_resolved_higher_than_unresolved():
    docs = [{"id": "docA", "text": "python retrieval citation mapping", "metadata": {}}]
    citations = [{"raw_id": "docA", "chunk_id": "docA", "resolved": True}]
    verifier = CitationVerifier()
    out = verifier.verify("python retrieval answer [Doc docA]", citations, docs)
    assert out[0]["verification"] in {"supported", "weak_support"}
    assert out[0]["verification_score"] > 0.0


def test_citations_include_structured_source_and_claim_text():
    docs = [
        {
            "id": "annual-leave",
            "text": "工作满 12 个月可享 5 天年假",
            "metadata": {
                "filename": "员工手册.pdf",
                "page_number": 2,
                "section_title": "年假",
                "file_type": ".pdf",
                "evidence_anchor": "handbook.leave.annual",
            },
        }
    ]
    response = "员工工作满 12 个月可享 5 天年假 [Doc annual-leave]。"

    mapped = CitationTracker().map_citations(response, docs)
    verified = CitationVerifier().verify(response, mapped, docs)

    assert mapped[0]["filename"] == "员工手册.pdf"
    assert mapped[0]["page_number"] == 2
    assert mapped[0]["section_title"] == "年假"
    assert mapped[0]["evidence_anchor"] == "handbook.leave.annual"
    assert mapped[0]["text_preview"] == "工作满 12 个月可享 5 天年假"
    assert mapped[0]["claim_text"] == "员工工作满 12 个月可享 5 天年假"
    assert verified[0]["verification"] == "supported"
