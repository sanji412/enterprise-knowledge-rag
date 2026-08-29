from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from src.core.rag_orchestrator import QueryResponse


def _load_demo_api(monkeypatch, tmp_path):
    monkeypatch.setenv("DOC_PROFILE", "demo")
    monkeypatch.setenv("DOC_DEMO_UPLOADS", "1")
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-contract-secret")
    import src.api.main as api_main

    return importlib.reload(api_main)


def test_enterprise_smoke_contract_with_mocked_provider(monkeypatch, tmp_path):
    api_main = _load_demo_api(monkeypatch, tmp_path)
    monkeypatch.setattr(api_main, "_enforce_rate_limit", lambda _client_key: None)
    monkeypatch.setattr(
        api_main,
        "run_ingest",
        lambda *args, **kwargs: {"processed_files": 3, "status": "ok"},
    )

    def _fake_run(req):
        if "年度奖金" in req.query_text:
            return QueryResponse(
                query=req.query_text,
                provider="deepseek",
                model="deepseek-v4-flash",
                answer="当前知识库中没有足够依据回答该问题",
                status="refused",
                refusal_reason="insufficient_retrieval_evidence",
                processing_time_ms=80.0,
            )
        return QueryResponse(
            query=req.query_text,
            provider="deepseek",
            model="deepseek-v4-flash",
            answer="重启后仍异常时，应长按复位键 8 秒。[Doc chunk-e03]",
            status="answered",
            refusal_reason=None,
            evidence=[
                {
                    "id": "chunk-e03",
                    "text": "仍异常则长按复位键 8 秒",
                    "metadata": {
                        "filename": "product_manual.docx",
                        "evidence_anchor": "product.atlas.e03",
                    },
                }
            ],
            citations=[
                {
                    "raw_id": "chunk-e03",
                    "chunk_id": "chunk-e03",
                    "resolved": True,
                    "title": "E03 故障处理",
                    "source": ".docx",
                    "filename": "product_manual.docx",
                    "page_number": 1,
                    "section_title": "E03 故障处理",
                    "evidence_anchor": "product.atlas.e03",
                    "claim_text": "应长按复位键 8 秒",
                    "text_preview": "仍异常则长按复位键 8 秒",
                    "verification_score": 0.82,
                    "verification": "supported",
                }
            ],
            processing_time_ms=123.0,
        )

    monkeypatch.setattr(api_main._orchestrator, "run", _fake_run)
    client = TestClient(api_main.app)

    health = client.get("/health")
    assert health.status_code == 200

    created = client.post("/sessions")
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    corpus_dir = Path("evals/corpus/generated")
    upload = client.post(
        f"/sessions/{session_id}/documents",
        headers={"X-API-Key": "dev-key-1"},
        files=[
            (
                "files",
                ("employee_handbook.pdf", (corpus_dir / "employee_handbook.pdf").read_bytes(), "application/pdf"),
            ),
            (
                "files",
                (
                    "product_manual.docx",
                    (corpus_dir / "product_manual.docx").read_bytes(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            ),
            (
                "files",
                ("after_sales_faq.md", (corpus_dir / "after_sales_faq.md").read_bytes(), "text/markdown"),
            ),
        ],
        data={
            "chunk_strategy": "zh_structure",
            "embedding_profile": "st_bge_large_zh",
        },
    )
    assert upload.status_code == 200
    assert len(upload.json()["results"]) == 3

    common = {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "session_id": session_id,
        "knowledge_scope": "session",
        "retrieval_mode": "hybrid",
        "use_rerank": True,
        "embedding_profile": "st_bge_large_zh",
    }
    answered = client.post(
        "/query",
        headers={"X-API-Key": "dev-key-1"},
        json={
            **common,
            "query": "ATLAS-X2 出现 E03 且重启后仍异常，要长按复位键多久？",
        },
    )
    refused = client.post(
        "/query",
        headers={"X-API-Key": "dev-key-1"},
        json={**common, "query": "公司今年给每位员工发多少年度奖金？"},
    )

    assert answered.status_code == 200
    assert answered.json()["status"] == "answered"
    assert any(citation["resolved"] for citation in answered.json()["citations"])
    assert refused.status_code == 200
    assert refused.json()["status"] == "refused"
    assert "sk-contract-secret" not in json.dumps(
        [health.json(), created.json(), upload.json(), answered.json(), refused.json()],
        ensure_ascii=False,
    )
