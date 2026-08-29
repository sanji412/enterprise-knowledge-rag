from src.evaluation.enterprise_metrics import (
    citation_accuracy,
    fact_coverage,
    hit_at_k,
    reciprocal_rank_at_k,
    refusal_correct,
)


def test_retrieval_metrics_use_gold_evidence_anchors():
    ranked = ["product.atlas.power", "product.atlas.e03"]
    assert hit_at_k(ranked, {"product.atlas.e03"}, 5) == 1.0
    assert reciprocal_rank_at_k(ranked, {"product.atlas.e03"}, 5) == 0.5


def test_fact_coverage_accepts_alias_groups():
    groups = [["8 秒", "八秒"], ["复位键", "重置键"]]
    assert fact_coverage("请长按重置键八秒", groups) == 1.0


def test_refusal_and_citation_accuracy():
    assert refusal_correct(answerable=False, status="refused") == 1.0
    assert citation_accuracy(["faq.return.window"], {"faq.return.window"}) == 1.0
