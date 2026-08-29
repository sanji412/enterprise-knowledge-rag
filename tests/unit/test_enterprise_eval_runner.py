from __future__ import annotations

import json
from pathlib import Path

from evals.enterprise_schema import EnterpriseCase
from evals.run_ablation import EXPERIMENTS, run_ablation, write_ablation_report
from evals.run_enterprise_evals import (
    EvaluationConfig,
    evaluate_cases,
    load_benchmark_config,
    percentile,
    write_report,
)
from src.core.rag_orchestrator import QueryResponse
from src.core.retrieval_result import RetrievalResult


def _cases() -> list[EnterpriseCase]:
    difficulties = ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10
    return [
        EnterpriseCase(
            id=f"C{index:02d}",
            difficulty=difficulty,
            category="fixture",
            question=f"问题 {index}",
            answerable=index <= 25,
            required_facts=[[f"事实{index}"]] if index <= 25 else [],
            gold_evidence=[f"anchor.{index}"] if index <= 25 else [],
            reference_answer=f"事实{index}" if index <= 25 else "知识库无依据",
        )
        for index, difficulty in enumerate(difficulties, start=1)
    ]


class FakeOrchestrator:
    def __init__(self, *, fail_case: int | None = None, secret: str = "") -> None:
        self.fail_case = fail_case
        self.secret = secret

    def run(self, request):
        index = int(request.query_text.rsplit(" ", 1)[1])
        if index == self.fail_case:
            raise RuntimeError(f"provider rejected key {self.secret}")
        answerable = index <= 25
        anchor = f"anchor.{index}"
        return QueryResponse(
            query=request.query_text,
            provider=request.provider or "deepseek",
            model=request.model or "deepseek-chat",
            answer=f"结论是事实{index}" if answerable else "当前知识库没有足够依据",
            retrieved=[
                RetrievalResult(
                    id=f"chunk-{index}",
                    text="测试文本",
                    metadata={"evidence_anchor": anchor},
                )
            ],
            citations=([{"evidence_anchor": anchor, "resolved": True}] if answerable else []),
            processing_time_ms=float(index * 10),
            step_latencies={"retrieval": float(index)},
            embedding_profile="st_bge_large_zh",
            status="answered" if answerable else "refused",
            refusal_reason=None if answerable else "insufficient_evidence",
        )


def _config(secret: str = "sk-unit-test-secret") -> EvaluationConfig:
    return EvaluationConfig(
        provider="deepseek",
        model="deepseek-chat",
        retrieval_mode="hybrid",
        use_rerank=True,
        embedding_profile="st_bge_large_zh",
        provider_api_key=secret,
    )


def test_one_case_error_does_not_abort_and_latencies_use_correct_scope():
    secret = "sk-unit-test-secret"
    report = evaluate_cases(
        _cases(),
        FakeOrchestrator(fail_case=30, secret=secret),
        _config(secret),
    )

    assert len(report["cases"]) == 30
    assert report["cases"][-1]["status"] == "execution_error"
    assert report["summary"]["execution_success_rate"] == 29 / 30
    assert report["summary"]["retrieval_latency_p50_ms"] == 15.0
    assert report["summary"]["retrieval_latency_p95_ms"] == 28.0
    assert report["summary"]["end_to_end_latency_p50_ms"] == 150.0
    assert report["summary"]["end_to_end_latency_p95_ms"] == 280.0
    assert report["by_difficulty"]["easy"]["retrieval_latency_p50_ms"] == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert secret not in json.dumps(report, ensure_ascii=False)


def test_report_writers_never_persist_api_key(tmp_path: Path):
    secret = "sk-unit-test-secret"
    report = evaluate_cases(_cases(), FakeOrchestrator(secret=secret), _config(secret))
    json_path, markdown_path = write_report(report, tmp_path / "single.json")

    assert report["summary"]["hit_at_5"] == 1.0
    assert report["summary"]["mrr_at_5"] == 1.0
    assert report["summary"]["citation_resolution_rate"] == 1.0
    assert report["summary"]["citation_accuracy"] == 1.0
    assert report["summary"]["refusal_accuracy"] == 1.0
    assert secret not in json_path.read_text(encoding="utf-8")
    assert secret not in markdown_path.read_text(encoding="utf-8")


def test_ablation_runs_all_four_named_experiments_and_sanitizes_output(tmp_path: Path):
    secret = "sk-unit-test-secret"
    report = run_ablation(
        _cases(),
        orchestrator_factory=lambda: FakeOrchestrator(secret=secret),
        base_config=_config(secret),
    )
    json_path, markdown_path = write_ablation_report(report, tmp_path / "ablation.json")

    assert list(report["experiments"]) == list(EXPERIMENTS)
    assert list(report["experiments"]) == [
        "bm25",
        "vector",
        "hybrid",
        "hybrid_rerank",
    ]
    assert secret not in json_path.read_text(encoding="utf-8")
    assert secret not in markdown_path.read_text(encoding="utf-8")


def test_benchmark_config_disables_unmeasured_inline_nli(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation:\n  inline_enabled: true\n", encoding="utf-8")

    cfg = load_benchmark_config(config_path)

    assert cfg.evaluation.inline_enabled is False
