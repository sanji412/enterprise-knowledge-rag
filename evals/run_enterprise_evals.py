"""Run the versioned Chinese enterprise benchmark against one RAG configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from src.core.rag_orchestrator import QueryRequest, RAGOrchestrator
from src.evaluation.enterprise_metrics import (
    citation_accuracy,
    fact_coverage,
    hit_at_k,
    reciprocal_rank_at_k,
    refusal_correct,
)
from src.utils.config import Config, load_config

from evals.enterprise_schema import EnterpriseCase, load_enterprise_cases

SUMMARY_KEYS = (
    "hit_at_5",
    "mrr_at_5",
    "fact_coverage",
    "citation_resolution_rate",
    "citation_accuracy",
    "refusal_accuracy",
    "retrieval_latency_p50_ms",
    "retrieval_latency_p95_ms",
    "end_to_end_latency_p50_ms",
    "end_to_end_latency_p95_ms",
    "execution_success_rate",
)


def load_benchmark_config(config_path: str | Path = "config.yaml") -> Config:
    """Load production retrieval settings without an unmeasured NLI side model."""
    cfg = load_config(str(config_path))
    evaluation = cfg.evaluation.model_copy(update={"inline_enabled": False})
    return cfg.model_copy(update={"evaluation": evaluation})


@dataclass(frozen=True)
class EvaluationConfig:
    provider: str
    model: str
    retrieval_mode: str
    use_rerank: bool
    embedding_profile: str
    provider_api_key: str | None = None
    top_k: int = 5

    def public_dict(self) -> dict[str, Any]:
        """Return report-safe configuration; credentials are deliberately omitted."""
        return {
            "provider": self.provider,
            "model": self.model,
            "retrieval_mode": self.retrieval_mode,
            "use_rerank": self.use_rerank,
            "embedding_profile": self.embedding_profile,
            "top_k": self.top_k,
        }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((len(ordered) - 1) * fraction)),
    )
    return ordered[index]


def _metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        value = item.get("metadata")
    else:
        value = getattr(item, "metadata", None)
    return value if isinstance(value, dict) else {}


def _ranked_anchors(retrieved: Iterable[Any]) -> list[str]:
    anchors: list[str] = []
    for item in retrieved:
        anchor = _metadata(item).get("evidence_anchor")
        if isinstance(anchor, str) and anchor:
            anchors.append(anchor)
    return anchors


def _citation_anchors(citations: Iterable[dict[str, Any]]) -> list[str]:
    return [anchor for citation in citations if isinstance((anchor := citation.get("evidence_anchor")), str) and anchor]


def _citation_resolution_rate(citations: Sequence[dict[str, Any]]) -> float:
    if not citations:
        return 0.0
    return sum(bool(citation.get("resolved")) for citation in citations) / len(citations)


def _safe_error(exc: Exception, secret: str | None) -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return f"{type(exc).__name__}: {message}"


def _failed_row(case: EnterpriseCase, exc: Exception, config: EvaluationConfig) -> dict[str, Any]:
    return {
        "id": case.id,
        "difficulty": case.difficulty,
        "category": case.category,
        "question": case.question,
        "answerable": case.answerable,
        "status": "execution_error",
        "refusal_reason": None,
        "error": _safe_error(exc, config.provider_api_key),
        "provider": config.provider,
        "model": config.model,
        "retrieved_anchors": [],
        "cited_anchors": [],
        "hit_at_5": 0.0,
        "mrr_at_5": 0.0,
        "fact_coverage": 0.0,
        "citation_resolution_rate": 0.0,
        "citation_accuracy": 0.0,
        "refusal_accuracy": 0.0,
        "step_latencies": {},
        "processing_time_ms": 0.0,
        "execution_success": 0.0,
    }


def _successful_row(case: EnterpriseCase, response: Any) -> dict[str, Any]:
    retrieved = list(getattr(response, "retrieved", []) or [])
    citations = list(getattr(response, "citations", []) or [])
    ranked_anchors = _ranked_anchors(retrieved)
    cited_anchors = _citation_anchors(citations)
    gold = set(case.gold_evidence)
    status = str(getattr(response, "status", "answered"))
    answer = str(getattr(response, "answer", ""))
    step_latencies = dict(getattr(response, "step_latencies", {}) or {})
    return {
        "id": case.id,
        "difficulty": case.difficulty,
        "category": case.category,
        "question": case.question,
        "answerable": case.answerable,
        "status": status,
        "refusal_reason": getattr(response, "refusal_reason", None),
        "error": None,
        "provider": str(getattr(response, "provider", "")),
        "model": str(getattr(response, "model", "")),
        "retrieved_anchors": ranked_anchors,
        "cited_anchors": cited_anchors,
        "hit_at_5": hit_at_k(ranked_anchors, gold, 5),
        "mrr_at_5": reciprocal_rank_at_k(ranked_anchors, gold, 5),
        "fact_coverage": fact_coverage(answer, case.required_facts),
        "citation_resolution_rate": _citation_resolution_rate(citations),
        "citation_accuracy": citation_accuracy(cited_anchors, gold),
        "refusal_accuracy": refusal_correct(case.answerable, status),
        "step_latencies": step_latencies,
        "processing_time_ms": float(getattr(response, "processing_time_ms", 0.0)),
        "execution_success": 1.0,
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in SUMMARY_KEYS}
    successful = [row for row in rows if row["execution_success"] == 1.0]
    retrieval_latencies = [float(row["step_latencies"].get("retrieval", 0.0)) for row in successful]
    end_to_end_latencies = [float(row["processing_time_ms"]) for row in successful]
    metric_names = (
        "hit_at_5",
        "mrr_at_5",
        "fact_coverage",
        "citation_resolution_rate",
        "citation_accuracy",
        "refusal_accuracy",
    )
    summary = {name: mean(float(row[name]) for row in rows) for name in metric_names}
    summary.update(
        {
            "retrieval_latency_p50_ms": percentile(retrieval_latencies, 0.50),
            "retrieval_latency_p95_ms": percentile(retrieval_latencies, 0.95),
            "end_to_end_latency_p50_ms": percentile(end_to_end_latencies, 0.50),
            "end_to_end_latency_p95_ms": percentile(end_to_end_latencies, 0.95),
            "execution_success_rate": mean(float(row["execution_success"]) for row in rows),
        }
    )
    return summary


def evaluate_cases(
    cases: Sequence[EnterpriseCase],
    orchestrator: RAGOrchestrator,
    config: EvaluationConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        request = QueryRequest(
            query_text=case.question,
            top_k=config.top_k,
            provider=config.provider,
            model=config.model,
            provider_api_key=config.provider_api_key,
            include_citations=True,
            use_rerank=config.use_rerank,
            embedding_profile=config.embedding_profile,
            retrieval_mode=config.retrieval_mode,  # type: ignore[arg-type]
        )
        try:
            rows.append(_successful_row(case, orchestrator.run(request)))
        except Exception as exc:  # one provider failure must not abort the benchmark
            rows.append(_failed_row(case, exc, config))

    difficulties = ("easy", "medium", "hard")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": config.public_dict(),
        "case_count": len(rows),
        "summary": summarize_rows(rows),
        "by_difficulty": {
            difficulty: summarize_rows([row for row in rows if row["difficulty"] == difficulty])
            for difficulty in difficulties
        },
        "cases": rows,
    }


def _report_paths(output: str | Path, stem: str) -> tuple[Path, Path]:
    output_path = Path(output)
    if output_path.suffix.lower() == ".json":
        json_path = output_path
    else:
        json_path = output_path / f"{stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return json_path, json_path.with_suffix(".md")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _summary_markdown(summary: dict[str, float]) -> str:
    return _markdown_table(
        ["指标", "结果"],
        [[key, f"{summary[key]:.4f}"] for key in SUMMARY_KEYS],
    )


def report_markdown(report: dict[str, Any]) -> str:
    config = report["configuration"]
    difficulty_rows = [
        [
            difficulty,
            f"{summary['hit_at_5']:.4f}",
            f"{summary['mrr_at_5']:.4f}",
            f"{summary['fact_coverage']:.4f}",
            f"{summary['refusal_accuracy']:.4f}",
            f"{summary['execution_success_rate']:.4f}",
        ]
        for difficulty, summary in report["by_difficulty"].items()
    ]
    failures = [row for row in report["cases"] if row["status"] == "execution_error"]
    failure_lines = "\n".join(f"- {row['id']}: {row['error']}" for row in failures) if failures else "- 无"
    return (
        "# 企业知识库 RAG 评测报告\n\n"
        f"生成时间：{report['generated_at']}\n\n"
        "## 实际配置\n\n"
        f"- Provider / Model：{config['provider']} / {config['model']}\n"
        f"- 检索模式：{config['retrieval_mode']}\n"
        f"- 重排：{config['use_rerank']}\n"
        f"- Embedding：{config['embedding_profile']}\n\n"
        "## 总体结果\n\n"
        f"{_summary_markdown(report['summary'])}\n\n"
        "## 难度分层\n\n"
        + _markdown_table(
            ["难度", "Hit@5", "MRR@5", "事实覆盖", "拒答准确率", "执行成功率"],
            difficulty_rows,
        )
        + "\n\n## 执行失败\n\n"
        + failure_lines
        + "\n"
    )


def write_report(report: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    json_path, markdown_path = _report_paths(output, "enterprise_eval")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/datasets/enterprise_30.jsonl")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--retrieval-mode",
        choices=("bm25", "vector", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--use-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--embedding-profile", default="st_bge_large_zh")
    parser.add_argument("--output", default="evals/reports/enterprise_eval.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_enterprise_cases(args.dataset)
    orchestrator = RAGOrchestrator(load_benchmark_config("config.yaml"))
    report = evaluate_cases(
        cases,
        orchestrator,
        EvaluationConfig(
            provider=args.provider,
            model=args.model,
            retrieval_mode=args.retrieval_mode,
            use_rerank=args.use_rerank,
            embedding_profile=args.embedding_profile,
        ),
    )
    json_path, markdown_path = write_report(report, args.output)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
