from fastapi.testclient import TestClient
from src.api import main as api_main
from src.api.main import app
from src.core.rag_orchestrator import QueryResponse


def test_query_contract_fields(monkeypatch):
    # Avoid 429 when Redis-backed limiter shares a key with many earlier /query calls in the suite.
    monkeypatch.setattr(api_main, "_enforce_rate_limit", lambda _client_key: None)
    # The contract test stubs generation, so authentication is outside its scope.
    monkeypatch.setattr(api_main._cfg.api, "auth_enabled", False)

    def _fake_run(_req):
        return QueryResponse(
            query="what is bm25",
            provider="ollama",
            model="qwen2.5:7b",
            answer="BM25 ranks by lexical relevance [Doc chunkA]",
            status="answered",
            refusal_reason=None,
            evidence=[
                {
                    "id": "chunkA",
                    "text": "BM25 ranks by lexical relevance",
                    "metadata": {"filename": "doc.txt"},
                }
            ],
            citations=[
                {
                    "raw_id": "chunkA",
                    "chunk_id": "chunkA",
                    "resolved": True,
                    "title": "doc.txt",
                    "source": ".txt",
                    "filename": "doc.txt",
                    "page_number": None,
                    "section_title": "BM25",
                    "evidence_anchor": "docs.bm25",
                    "claim_text": "BM25 ranks by lexical relevance",
                    "text_preview": "BM25 ranks by lexical relevance",
                    "verification_score": 0.82,
                    "verification": "supported",
                }
            ],
            processing_time_ms=123.0,
        )

    monkeypatch.setattr(api_main._orchestrator, "run", _fake_run)
    client = TestClient(app)
    res = client.post("/query", json={"query": "what is bm25"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["query"] == "what is bm25"
    assert payload["provider"] == "ollama"
    assert isinstance(payload["citations"], list)
    assert "processing_time_ms" in payload
    assert payload["status"] == "answered"
    assert payload["refusal_reason"] is None
    assert payload["evidence"][0]["id"] == "chunkA"
    assert payload["citations"][0]["filename"] == "doc.txt"
