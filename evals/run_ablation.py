"""Run the four deterministic enterprise retrieval ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from src.core.rag_orchestrator import RAGOrchestrator

from evals.enterprise_schema import EnterpriseCase, load_enterprise_cases
from evals.run_enterprise_evals import (
    EvaluationConfig,
    _markdown_table,
    _report_paths,
    evaluate_cases,
    load_benchmark_config,
    summarize_rows,
)

EXPERIMENTS = {
    "bm25": {"retrieval_mode": "bm25", "use_rerank": False},
    "vector": {"retrieval_mode": "vector", "use_rerank": False},
    "hybrid": {"retrieval_mode": "hybrid", "use_rerank": False},
    "hybrid_rerank": {"retrieval_mode": "hybrid", "use_rerank": True},
}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                ":(exclude).superpowers/**",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True


def run_ablation(
    cases: Sequence[EnterpriseCase],
    *,
    orchestrator_factory: Callable[[], RAGOrchestrator],
    base_config: EvaluationConfig,
) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    for name, experiment in EXPERIMENTS.items():
        config = replace(base_config, **experiment)
        reports[name] = evaluate_cases(cases, orchestrator_factory(), config)
    commit = _git_commit()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "raw_execution_git_commit": commit,
        "aggregation_git_commit": commit,
        "aggregation_git_dirty": _git_dirty(),
        "base_configuration": base_config.public_dict(),
        "experiments": reports,
    }


def _comparison_rows(report: dict[str, Any], difficulty: str | None = None) -> list[list[str]]:
    rows: list[list[str]] = []
    for name, experiment in report["experiments"].items():
        summary = experiment["summary"] if difficulty is None else experiment["by_difficulty"][difficulty]
        config = experiment["configuration"]
        rows.append(
            [
                name,
                f"{config['provider']} / {config['model']}",
                config["retrieval_mode"],
                str(config["use_rerank"]),
                f"{summary['hit_at_5']:.4f}",
                f"{summary['mrr_at_5']:.4f}",
                f"{summary['fact_coverage']:.4f}",
                f"{summary['citation_accuracy']:.4f}",
                f"{summary['refusal_accuracy']:.4f}",
                f"{summary['end_to_end_latency_p95_ms']:.2f}",
            ]
        )
    return rows


def ablation_markdown(report: dict[str, Any]) -> str:
    headers = [
        "实验",
        "Provider / Model",
        "检索",
        "重排",
        "Hit@5",
        "MRR@5",
        "事实覆盖",
        "引用准确率",
        "拒答准确率",
        "端到端 P95(ms)",
    ]
    execution_commit = report.get("raw_execution_git_commit", report.get("git_commit", "unknown"))
    aggregation_commit = report.get("aggregation_git_commit", report.get("git_commit", "unknown"))
    aggregation_dirty = bool(report.get("aggregation_git_dirty", False))
    sections = [
        "# 企业知识库 RAG 消融实验\n",
        f"生成时间：{report['generated_at']}  ",
        f"原始执行 Commit：`{execution_commit}`  ",
        f"聚合 Commit：`{aggregation_commit}`  ",
        f"聚合工作区 Dirty：`{str(aggregation_dirty).lower()}`\n",
        "## 总体对比\n",
        _markdown_table(headers, _comparison_rows(report)),
    ]
    labels = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
    for difficulty, label in labels.items():
        sections.extend(
            [
                f"\n## {label} 对比\n",
                _markdown_table(headers, _comparison_rows(report, difficulty)),
            ]
        )

    failures: list[str] = []
    for name, experiment in report["experiments"].items():
        for row in experiment["cases"]:
            if row["status"] == "execution_error":
                failures.append(f"- {name} / {row['id']}: {row['error']}")
    sections.extend(["\n## 单条执行失败\n", "\n".join(failures) if failures else "- 无"])
    return "\n".join(sections) + "\n"


def write_ablation_report(
    report: dict[str, Any],
    output: str | Path,
) -> tuple[Path, Path]:
    json_path, markdown_path = _report_paths(output, "enterprise_ablation")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(ablation_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def reaggregate_ablation_report(
    source: str | Path,
    output: str | Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Recompute saved case aggregates without constructing an orchestrator."""
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    for experiment in report["experiments"].values():
        rows = list(experiment["cases"])
        experiment["case_count"] = len(rows)
        experiment["summary"] = summarize_rows(rows)
        experiment["by_difficulty"] = {
            difficulty: summarize_rows(
                [row for row in rows if row["difficulty"] == difficulty]
            )
            for difficulty in ("easy", "medium", "hard")
        }

    aggregation_commit = _git_commit()
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["raw_execution_git_commit"] = report.get(
        "raw_execution_git_commit",
        report.get("git_commit", "unknown"),
    )
    report["aggregation_git_commit"] = aggregation_commit
    report["aggregation_git_dirty"] = _git_dirty()
    report["git_commit"] = aggregation_commit
    json_path, markdown_path = write_ablation_report(report, output)
    return report, json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/datasets/enterprise_30.jsonl")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--embedding-profile", default="st_bge_large_zh")
    parser.add_argument("--output", default="evals/reports/enterprise_ablation.json")
    parser.add_argument(
        "--reaggregate-from",
        help="Recompute metrics from a saved ablation JSON without any model calls",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reaggregate_from:
        _report, json_path, markdown_path = reaggregate_ablation_report(
            args.reaggregate_from,
            args.output,
        )
        print(f"JSON: {json_path}")
        print(f"Markdown: {markdown_path}")
        return 0
    cases = load_enterprise_cases(args.dataset)
    cfg = load_benchmark_config("config.yaml")
    report = run_ablation(
        cases,
        orchestrator_factory=lambda: RAGOrchestrator(cfg),
        base_config=EvaluationConfig(
            provider=args.provider,
            model=args.model,
            retrieval_mode="hybrid",
            use_rerank=True,
            embedding_profile=args.embedding_profile,
        ),
    )
    json_path, markdown_path = write_ablation_report(report, args.output)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
