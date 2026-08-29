from collections import Counter
from pathlib import Path

from evals.enterprise_schema import load_enterprise_cases
from src.core.document_processor import DocumentProcessor

DATASET_PATH = Path("evals/datasets/enterprise_30.jsonl")


def test_enterprise_dataset_has_required_distribution():
    cases = load_enterprise_cases(DATASET_PATH)

    assert len(cases) == 30
    assert Counter(case.difficulty for case in cases) == {
        "easy": 10,
        "medium": 10,
        "hard": 10,
    }
    assert sum(not case.answerable for case in cases) == 5
    assert len({case.id for case in cases}) == 30
    assert all(case.required_facts or not case.answerable for case in cases)
    assert all(case.gold_evidence or not case.answerable for case in cases)


def test_answerable_cases_reference_declared_corpus_anchors():
    cases = load_enterprise_cases(DATASET_PATH)
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("evals/corpus/source").glob("*.md"))
    )

    answerable_anchors = {anchor for case in cases if case.answerable for anchor in case.gold_evidence}

    assert answerable_anchors
    assert all(f"{{#{anchor}}}" in source_text for anchor in answerable_anchors)


def test_generated_corpus_manifest_covers_all_formats():
    generated = Path("evals/corpus/generated")

    assert (generated / "employee_handbook.pdf").is_file()
    assert (generated / "product_manual.docx").is_file()
    assert (generated / "after_sales_faq.md").is_file()
    assert (generated / "manifest.json").is_file()


def test_generated_chinese_pdf_preserves_text_and_evidence_anchors():
    processor = DocumentProcessor(chunk_strategy="zh_structure")

    units = processor._extract_source_units(
        "evals/corpus/generated/employee_handbook.pdf",
    )

    assert len(units) == 6
    assert [unit.evidence_anchor for unit in units] == [
        "handbook.attendance.late",
        "handbook.leave.annual",
        "handbook.leave.sick",
        "handbook.expense.deadline",
        "handbook.expense.hotel",
        "handbook.onboarding.security",
    ]
    assert "09:10 后完成打卡" in units[0].text
    assert units[0].section_title == "考勤与迟到"
