from os import utime
from types import SimpleNamespace

from src.core.rag_orchestrator import QueryRequest, RAGOrchestrator
from src.utils.config import load_config


def _orchestrator() -> RAGOrchestrator:
    orchestrator = RAGOrchestrator.__new__(RAGOrchestrator)
    orchestrator.cfg = load_config("config.yaml")
    return orchestrator


def _selection():
    return SimpleNamespace(provider="deepseek", model="deepseek-chat")


def test_cache_key_changes_when_global_bm25_revision_changes(tmp_path, monkeypatch):
    index_path = tmp_path / "global.json"
    index_path.write_text("first", encoding="utf-8")
    monkeypatch.setattr("src.core.rag_orchestrator.BM25_INDEX_PATH", str(index_path))
    orchestrator = _orchestrator()
    request = QueryRequest(query_text="same query", knowledge_scope="global")

    first = orchestrator._make_cache_key(request, _selection())
    before = index_path.stat().st_mtime_ns
    index_path.write_text("other", encoding="utf-8")
    utime(index_path, ns=(before + 1_000_000, before + 1_000_000))
    second = orchestrator._make_cache_key(request, _selection())

    assert first != second


def test_cache_key_changes_when_active_session_bm25_revision_changes(tmp_path, monkeypatch):
    global_path = tmp_path / "global.json"
    session_path = tmp_path / "session.json"
    global_path.write_text("global", encoding="utf-8")
    session_path.write_text("first", encoding="utf-8")
    monkeypatch.setattr("src.core.rag_orchestrator.BM25_INDEX_PATH", str(global_path))
    orchestrator = _orchestrator()
    request = QueryRequest(
        query_text="same query",
        knowledge_scope="both",
        session_bm25_index_path=str(session_path),
        session_collection_name="sess_abc",
    )

    first = orchestrator._make_cache_key(request, _selection())
    before = session_path.stat().st_mtime_ns
    session_path.write_text("other", encoding="utf-8")
    utime(session_path, ns=(before + 1_000_000, before + 1_000_000))
    second = orchestrator._make_cache_key(request, _selection())

    assert first != second


def test_cache_key_ignores_session_bm25_without_active_collection(tmp_path, monkeypatch):
    global_path = tmp_path / "global.json"
    session_path = tmp_path / "session.json"
    global_path.write_text("global", encoding="utf-8")
    session_path.write_text("first", encoding="utf-8")
    monkeypatch.setattr("src.core.rag_orchestrator.BM25_INDEX_PATH", str(global_path))
    orchestrator = _orchestrator()
    request = QueryRequest(
        query_text="same query",
        knowledge_scope="session",
        session_bm25_index_path=str(session_path),
        session_collection_name=None,
    )

    first = orchestrator._make_cache_key(request, _selection())
    before = session_path.stat().st_mtime_ns
    session_path.write_text("other", encoding="utf-8")
    utime(session_path, ns=(before + 1_000_000, before + 1_000_000))

    assert orchestrator._make_cache_key(request, _selection()) == first


def test_cache_key_changes_with_resolved_embedding_profile():
    orchestrator = _orchestrator()
    default_request = QueryRequest(query_text="same query", embedding_profile="st_bge_large_zh")
    alternate_request = QueryRequest(query_text="same query", embedding_profile="st_minilm")

    assert orchestrator._make_cache_key(
        default_request,
        _selection(),
    ) != orchestrator._make_cache_key(alternate_request, _selection())
