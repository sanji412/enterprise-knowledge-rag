from unittest.mock import MagicMock

from src.core.bm25_search import BM25Search
from src.core.hybrid_retriever import FusionConfig, HybridRetriever, reciprocal_rank_fusion
from src.core.rag_orchestrator import QueryRequest
from src.core.vector_search import VectorSearch


def test_reciprocal_rank_fusion_prefers_consensus_top():
    bm25 = ["a", "b", "c"]
    vec = ["b", "c", "a"]
    fused = reciprocal_rank_fusion([bm25, vec], k_rrf=60)
    assert fused[0][0] == "b"


def test_reciprocal_rank_fusion_stable_tiebreak():
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k_rrf=1)
    assert fused[0][1] == fused[1][1]
    assert fused[0][0] < fused[1][0]


def test_hybrid_retriever_rrf_not_concat_order():
    bm25 = MagicMock(spec=BM25Search)
    bm25.search.return_value = [
        {"id": "doc_a", "text": "ta", "metadata": {}, "score": 99.0},
        {"id": "doc_b", "text": "tb", "metadata": {}, "score": 1.0},
    ]
    vec = MagicMock(spec=VectorSearch)
    vec.search.return_value = [
        {"id": "doc_b", "text": "tb", "metadata": {}, "distance": 0.1},
        {"id": "doc_c", "text": "tc", "metadata": {}, "distance": 0.2},
    ]

    h = HybridRetriever(bm25, vec, enable_cache=False)
    out = h.retrieve("bm25 q", "vec q", k=3)
    ids = [r.id for r in out]
    assert ids[0] == "doc_b"
    assert set(ids) == {"doc_a", "doc_b", "doc_c"}


def test_hybrid_cache_returns_copy_and_skips_second_search():
    bm25 = MagicMock(spec=BM25Search)
    bm25.search.return_value = [{"id": "a", "text": "t", "metadata": {}, "score": 1.0}]
    vec = MagicMock(spec=VectorSearch)
    vec.search.return_value = [{"id": "a", "text": "t", "metadata": {}, "distance": 0.5}]

    h = HybridRetriever(
        bm25,
        vec,
        enable_cache=True,
        fusion_config=FusionConfig(cache_max_entries=4),
    )
    r1 = h.retrieve("q1", "q2", k=1)
    h.retrieve("q1", "q2", k=1)
    assert bm25.search.call_count == 1
    r1[0].text = "mutated"
    r3 = h.retrieve("q1", "q2", k=1)
    assert r3[0].text == "t"


def test_bm25_mode_does_not_call_vector():
    bm25 = MagicMock(spec=BM25Search)
    bm25.search.return_value = [
        {
            "id": "e03",
            "text": "长按复位键 8 秒",
            "metadata": {},
            "score": 2.0,
        }
    ]
    vector = MagicMock(spec=VectorSearch)
    retriever = HybridRetriever(bm25, vector, enable_cache=False)

    rows = retriever.retrieve("E03", "E03", k=5, mode="bm25")

    assert rows[0].sources == ["bm25"]
    assert rows[0].bm25_rank == 1
    vector.search.assert_not_called()


def test_vector_mode_does_not_call_bm25():
    bm25 = MagicMock(spec=BM25Search)
    vector = MagicMock(spec=VectorSearch)
    vector.search.return_value = [
        {
            "id": "e03",
            "text": "长按复位键 8 秒",
            "metadata": {},
            "distance": 0.1,
        }
    ]
    retriever = HybridRetriever(bm25, vector, enable_cache=False)

    rows = retriever.retrieve("如何复位", "如何复位", k=5, mode="vector")

    assert rows[0].sources == ["vector"]
    assert rows[0].vector_rank == 1
    bm25.search.assert_not_called()


def test_hybrid_mode_calls_both_legs():
    bm25 = MagicMock(spec=BM25Search)
    bm25.search.return_value = [
        {"id": "a", "text": "A", "metadata": {}, "score": 1.0},
    ]
    vector = MagicMock(spec=VectorSearch)
    vector.search.return_value = [
        {"id": "a", "text": "A", "metadata": {}, "distance": 0.1},
    ]
    retriever = HybridRetriever(bm25, vector, enable_cache=False)

    rows = retriever.retrieve("问题", "问题", k=5, mode="hybrid")

    assert rows[0].sources == ["bm25", "vector"]
    assert rows[0].fusion_score > 0.0
    bm25.search.assert_called_once()
    vector.search.assert_called_once()


def test_hybrid_rerank_experiment_maps_to_request_fields():
    request = QueryRequest(
        query_text="问题",
        retrieval_mode="hybrid",
        use_rerank=True,
    )

    assert request.retrieval_mode == "hybrid"
    assert request.use_rerank is True
