from unittest.mock import MagicMock

from src.core.context_optimizer import OptimizedContext
from src.core.grounding_policy import AnswerStatus, GroundingPolicy
from src.core.rag_orchestrator import QueryRequest, RAGOrchestrator
from src.core.retrieval_result import RetrievalResult
from src.utils.config import load_config


def _retrieval_result():
    return RetrievalResult(
        id="atlas-e03",
        text="E03 时长按复位键 8 秒",
        metadata={"evidence_anchor": "product.atlas.e03"},
        confidence=0.9,
    )


def test_refuses_when_no_evidence():
    decision = GroundingPolicy().before_generation([])

    assert decision.status is AnswerStatus.REFUSED
    assert decision.reason == "no_relevant_evidence"


def test_refuses_unresolved_citations_after_generation():
    decision = GroundingPolicy().after_generation(
        [_retrieval_result()],
        [
            {
                "resolved": False,
                "verification": "unresolved",
                "verification_score": 0.0,
            }
        ],
    )

    assert decision.status is AnswerStatus.REFUSED
    assert decision.reason == "unverified_citations"


def test_answers_with_supported_citation():
    decision = GroundingPolicy().after_generation(
        [_retrieval_result()],
        [
            {
                "resolved": True,
                "verification": "supported",
                "verification_score": 0.8,
            }
        ],
    )

    assert decision.status is AnswerStatus.ANSWERED


def test_orchestrator_does_not_call_provider_without_evidence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = load_config("config.yaml")
    cfg.evaluation.inline_enabled = False
    orchestrator = RAGOrchestrator(cfg)
    monkeypatch.setattr(
        orchestrator,
        "_retrieve_docs_for_query",
        lambda _req, _trace, _latencies: ([], [], "st_bge_large_zh"),
    )
    generate = MagicMock()
    monkeypatch.setattr(orchestrator.provider_router, "generate", generate)

    response = orchestrator.run(QueryRequest(query_text="火星差旅标准是什么"))

    assert response.status == "refused"
    assert response.refusal_reason == "no_relevant_evidence"
    assert response.answer == "当前知识库中没有足够依据回答该问题"
    generate.assert_not_called()


def test_orchestrator_refuses_answer_without_verified_citation(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = load_config("config.yaml")
    cfg.evaluation.inline_enabled = False
    orchestrator = RAGOrchestrator(cfg)
    evidence = _retrieval_result()
    monkeypatch.setattr(
        orchestrator,
        "_retrieve_docs_for_query",
        lambda _req, _trace, _latencies: ([evidence], [evidence], "st_bge_large_zh"),
    )
    monkeypatch.setattr(
        orchestrator.context_optimizer,
        "optimize_context",
        lambda _query, _documents: OptimizedContext(
            documents=[
                {
                    "id": evidence.id,
                    "text": evidence.text,
                    "metadata": evidence.metadata,
                }
            ]
        ),
    )
    generate = MagicMock(return_value="出现 E03 时请更换整机。")
    monkeypatch.setattr(orchestrator.provider_router, "generate", generate)

    response = orchestrator.run(
        QueryRequest(query_text="E03 故障怎么处理", use_rerank=False)
    )

    assert response.status == "refused"
    assert response.refusal_reason == "unverified_citations"
    assert response.answer == "当前知识库中没有足够依据回答该问题"
    assert response.evidence[0]["id"] == "atlas-e03"
    generate.assert_called_once()
