from fastapi.testclient import TestClient
from src.api import main as api_main
from src.api.main import app
from src.core.rag_orchestrator import QueryResponse
from src.core.retrieval_result import RetrievalResult
from src.evaluation.truthfulness import TruthfulnessResult
from starlette.requests import Request


def _request(client_host: str = "127.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "client": (client_host, 1234),
            "server": ("testserver", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_gateway_api_key_client_identifier_is_stable_irreversible_and_distinct():
    secret = "gateway-secret-plaintext"

    first = api_main._resolve_client_key(_request(), secret)
    second = api_main._resolve_client_key(_request(), secret)
    other = api_main._resolve_client_key(_request(), "another-secret")

    assert first == second
    assert first != other
    assert secret not in first
    assert first.startswith("key:sha256:")


def test_client_identifier_keeps_ip_fallback_without_gateway_key():
    assert api_main._resolve_client_key(_request("192.0.2.10"), None) == "ip:192.0.2.10"


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_metrics_endpoint():
    api_main._cfg.api.auth_enabled = False
    client = TestClient(app)
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "available_providers" in res.json()


def test_llm_config_endpoint():
    client = TestClient(app)
    res = client.get("/config/llm")
    assert res.status_code == 200
    data = res.json()
    assert data["default_provider"] == "deepseek"
    assert list(data["allowed_models_by_provider"]) == ["deepseek"]
    assert data["allowed_models_by_provider"]["deepseek"] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert isinstance(data["provider_key_configured"], dict)
    assert "deepseek" in data["provider_key_configured"]
    assert isinstance(data["demo_mode"], bool)


def test_runtime_config_endpoint():
    client = TestClient(app)
    res = client.get("/config/runtime")
    assert res.status_code == 200
    data = res.json()
    assert data["chunking_default_strategy"]
    assert isinstance(data["chunking_allowed_strategies"], list)
    assert data["embedding_default_profile"]
    assert isinstance(data["embedding_profiles"], dict)


def test_query_endpoint(monkeypatch):
    captured = {}

    def _fake_run(request):
        captured["request"] = request
        return QueryResponse(
            query="q",
            provider="ollama",
            model="qwen2.5:7b",
            answer="hello [Doc chunk1]",
            citations=[
                {
                    "raw_id": "chunk1",
                    "chunk_id": "chunk1",
                    "resolved": True,
                    "title": "Doc1",
                    "source": ".txt",
                    "verification_score": 0.8,
                    "verification": "supported",
                }
            ],
            retrieved=[
                RetrievalResult(
                    id="chunk1",
                    text="body",
                    confidence=0.7,
                    cross_encoder_score=0.88,
                    rerank_position=1,
                )
            ],
        )

    monkeypatch.setattr(api_main._orchestrator, "run", _fake_run)
    api_main._cfg.api.auth_enabled = False
    client = TestClient(app)
    res = client.post(
        "/query",
        json={"query": "hello", "retrieval_mode": "bm25"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "ollama"
    assert len(data["citations"]) == 1
    assert "embedding_profile" in data
    assert captured["request"].retrieval_mode == "bm25"
    assert data["retrieved"][0]["cross_encoder_score"] == 0.88
    assert data["retrieved"][0]["rerank_position"] == 1


def test_query_requires_api_key_when_enabled():
    api_main._cfg.api.auth_enabled = True
    api_main._cfg.api.api_keys = ["test-key"]
    client = TestClient(app)
    res = client.post("/query", json={"query": "hello", "provider": "openai", "model": "gpt-4o-mini"})
    assert res.status_code == 401
    res2 = client.post(
        "/query",
        json={"query": "hello", "provider": "openai", "model": "gpt-4o-mini"},
        headers={"X-API-Key": "test-key"},
    )
    assert res2.status_code in (200, 400)


def test_query_stream_endpoint(monkeypatch):
    captured = {}

    class FakeStreamingSession:
        def __init__(self, _orch, _req):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_tokens(self):
            yield "hello "
            yield "world"

        def finalize(self):
            return QueryResponse(
                query="q",
                provider="ollama",
                model="qwen2.5:7b",
                answer="grounded answer",
                status="answered",
                refusal_reason=None,
                evidence=[{"id": "chunk1", "text": "body", "metadata": {}}],
                citations=[],
                processing_time_ms=12.0,
                cached=False,
                validation_issues=[],
                truthfulness=TruthfulnessResult(
                    nli_faithfulness=0.8,
                    citation_groundedness=0.7,
                    uncited_claims=1,
                    score=0.76,
                ),
                step_latencies={
                    "retrieval": 1.0,
                    "reranking": 2.0,
                    "generation": 3.0,
                    "citation_verification": 4.0,
                    "truthfulness_scoring": 5.0,
                },
            )

    monkeypatch.setattr(api_main, "StreamingQuerySession", FakeStreamingSession)
    monkeypatch.setattr(
        api_main._metrics_collector,
        "record_request",
        lambda metrics: captured.setdefault("metrics", metrics),
    )
    api_main._cfg.api.auth_enabled = False
    client = TestClient(app)
    res = client.post("/query/stream", json={"query": "hello", "stream": True})
    assert res.status_code == 200
    assert "token" in res.text
    assert '"processing_time_ms": 12.0' in res.text
    assert '"cached": false' in res.text
    assert '"answer": "grounded answer"' in res.text
    assert '"status": "answered"' in res.text
    assert '"evidence": [{"id": "chunk1"' in res.text
    assert "metrics" in captured
    assert captured["metrics"].truthfulness_latency_ms == 5.0


def test_rate_limit_uses_redis_when_available(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.count = 0

        def incr(self, _key):
            self.count += 1
            return self.count

        def expire(self, _key, _ttl):
            return True

    fake = FakeRedis()
    monkeypatch.setattr(api_main, "_get_redis", lambda: fake)
    api_main._cfg.api.rate_limit_per_minute = 2
    api_main._enforce_rate_limit("clientA")
    api_main._enforce_rate_limit("clientA")
    try:
        api_main._enforce_rate_limit("clientA")
        assert False, "Expected rate limit exception"
    except Exception:
        assert True
