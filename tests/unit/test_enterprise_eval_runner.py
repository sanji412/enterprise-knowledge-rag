from __future__ import annotations

import json
from pathlib import Path

from evals import run_ablation as ablation_module
from evals.enterprise_schema import EnterpriseCase
from evals.run_ablation import EXPERIMENTS, run_ablation, write_ablation_report
from evals.run_ablation import main as ablation_main
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


def _saved_ablation_report() -> dict:
    answerable = {
        "id": "C01",
        "difficulty": "easy",
        "answerable": True,
        "status": "answered",
        "hit_at_5": 1.0,
        "mrr_at_5": 1.0,
        "fact_coverage": 1.0,
        "citation_resolution_rate": 1.0,
        "citation_accuracy": 1.0,
        "refusal_accuracy": 1.0,
        "step_latencies": {"retrieval": 10.0},
        "processing_time_ms": 20.0,
        "execution_success": 1.0,
    }
    unanswerable = {
        "id": "C02",
        "difficulty": "hard",
        "answerable": False,
        "status": "refused",
        "hit_at_5": 0.0,
        "mrr_at_5": 0.0,
        "fact_coverage": 0.0,
        "citation_resolution_rate": 0.0,
        "citation_accuracy": 0.0,
        "refusal_accuracy": 1.0,
        "step_latencies": {"retrieval": 30.0},
        "processing_time_ms": 40.0,
        "execution_success": 1.0,
    }
    return {
        "generated_at": "2026-08-29T00:00:00+00:00",
        "git_commit": "b9c3cb8",
        "base_configuration": {},
        "experiments": {
            "hybrid": {
                "configuration": {
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "retrieval_mode": "hybrid",
                    "use_rerank": False,
                },
                "summary": {"hit_at_5": -1.0},
                "by_difficulty": {},
                "cases": [answerable, unanswerable],
            }
        },
    }


def test_reaggregate_ablation_uses_saved_cases_and_eligible_denominators(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "frozen.json"
    source.write_text(json.dumps(_saved_ablation_report()), encoding="utf-8")
    monkeypatch.setattr(ablation_module, "_git_commit", lambda: "cf0eb9d")
    monkeypatch.setattr(ablation_module, "_git_dirty", lambda: False)

    report, json_path, markdown_path = ablation_module.reaggregate_ablation_report(
        source,
        tmp_path / "reaggregated.json",
    )

    summary = report["experiments"]["hybrid"]["summary"]
    assert summary["hit_at_5"] == 1.0
    assert summary["citation_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert report["raw_execution_git_commit"] == "b9c3cb8"
    assert report["aggregation_git_commit"] == "cf0eb9d"
    assert report["aggregation_git_dirty"] is False
    assert report["git_commit"] == "cf0eb9d"
    assert json_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "b9c3cb8" in markdown
    assert "cf0eb9d" in markdown
    assert "false" in markdown.lower()
    assert all(line == line.rstrip() for line in markdown.splitlines())


def test_reaggregate_cli_never_loads_dataset_config_or_orchestrator(tmp_path: Path, monkeypatch):
    source = tmp_path / "frozen.json"
    output = tmp_path / "offline.json"
    source.write_text(json.dumps(_saved_ablation_report()), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("live evaluation dependency was called")

    monkeypatch.setattr(ablation_module, "load_enterprise_cases", forbidden)
    monkeypatch.setattr(ablation_module, "load_benchmark_config", forbidden)
    monkeypatch.setattr(ablation_module, "RAGOrchestrator", forbidden)

    assert ablation_main(
        ["--reaggregate-from", str(source), "--output", str(output)]
    ) == 0
    assert output.exists()
