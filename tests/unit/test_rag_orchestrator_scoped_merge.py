from src.core.rag_orchestrator import RAGOrchestrator
from src.core.retrieval_result import RetrievalResult


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(id=chunk_id, text=chunk_id)


def test_both_scope_merge_keeps_session_candidate_when_global_is_full():
    global_results = [_result(f"global-{index}") for index in range(5)]
    session_results = [_result(f"session-{index}") for index in range(2)]

    merged = RAGOrchestrator._merge_scoped_results(
        global_results,
        session_results,
        top_k=5,
    )

    assert [item.id for item in merged] == [
        "global-0",
        "session-0",
        "global-1",
        "session-1",
        "global-2",
    ]


def test_both_scope_merge_deduplicates_without_reordering_each_scope():
    merged = RAGOrchestrator._merge_scoped_results(
        [_result("shared"), _result("global-1"), _result("global-2")],
        [_result("shared"), _result("session-1"), _result("session-2")],
        top_k=5,
    )

    assert [item.id for item in merged] == [
        "shared",
        "global-1",
        "session-1",
        "global-2",
        "session-2",
    ]
