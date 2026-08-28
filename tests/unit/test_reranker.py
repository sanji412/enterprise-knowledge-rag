from src.core.reranker import CrossEncoderReranker
from src.core.retrieval_result import RetrievalResult
from src.utils.config import load_config


class _FakeCrossEncoder:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [0.2, 0.9, 0.6]


def test_chinese_reranker_sorts_and_copies_scores_without_download():
    cfg = load_config("config.yaml")
    reranker = CrossEncoderReranker(
        model_name=cfg.reranker.model,
        batch_size=8,
        score_threshold=0.0,
    )
    fake_model = _FakeCrossEncoder()
    reranker._model = fake_model
    documents = [
        RetrievalResult(id="a", text="证据 A"),
        RetrievalResult(id="b", text="证据 B"),
        RetrievalResult(id="c", text="证据 C"),
    ]

    rows = reranker.rerank("怎么处理故障", documents, top_k=2)

    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert fake_model.pairs == [
        ("怎么处理故障", "证据 A"),
        ("怎么处理故障", "证据 B"),
        ("怎么处理故障", "证据 C"),
    ]
    assert [row.result.id for row in rows] == ["b", "c"]
    assert [row.cross_encoder_score for row in rows] == [0.9, 0.6]
    assert [row.result.cross_encoder_score for row in rows] == [0.9, 0.6]
    assert [row.result.rerank_position for row in rows] == [1, 2]
