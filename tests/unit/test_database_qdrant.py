from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from src.utils.config import EmbeddingProfile
from src.utils.database import VectorDatabase


@pytest.fixture
def fake_qdrant_db():
    profile = EmbeddingProfile(
        provider="sentence_transformers",
        framework="sentence_transformers",
        model="BAAI/bge-large-zh-v1.5",
        dimension=1024,
        options={"normalize_embeddings": True},
    )
    db = VectorDatabase(
        mode="prod",
        qdrant_url="http://qdrant.invalid:6333",
        embedding_profile_name="st_bge_large_zh",
        embedding_profile=profile,
    )
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=1024)),
        )
    )
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="4f980c70-207b-5ab0-9437-4c82dd7bea3b",
                payload={
                    "chunk_id": "handbook__leave__chunk0",
                    "text": "年假需提前申请",
                    "metadata": {"filename": "员工手册.pdf"},
                },
                score=0.91,
            )
        ]
    )
    db._qdrant_client = client
    db.generate_embedding = lambda text: [0.1] * 1024
    return db


def test_qdrant_upsert_keeps_text_and_original_chunk_id(fake_qdrant_db):
    fake_qdrant_db.generate_embeddings_batch = lambda texts: [
        [0.1] * 1024 for _ in texts
    ]
    fake_qdrant_db.add_documents(
        "documents",
        [
            {
                "id": "handbook__leave__chunk0",
                "text": "年假需提前申请",
                "filename": "员工手册.pdf",
            }
        ],
    )

    point = fake_qdrant_db.qdrant_client.upsert.call_args.kwargs["points"][0]
    assert str(point.id) != "handbook__leave__chunk0"
    assert point.payload["chunk_id"] == "handbook__leave__chunk0"
    assert point.payload["text"] == "年假需提前申请"
    assert point.payload["metadata"] == {"filename": "员工手册.pdf"}


def test_qdrant_query_restores_retrieval_shape(fake_qdrant_db):
    rows = fake_qdrant_db.query_documents("documents", "怎么请年假", top_k=5)

    assert rows[0]["id"] == "handbook__leave__chunk0"
    assert rows[0]["text"] == "年假需提前申请"
    assert rows[0]["metadata"] == {"filename": "员工手册.pdf"}
    assert rows[0]["score"] == pytest.approx(0.91)
    assert rows[0]["distance"] == pytest.approx(0.09)


def test_qdrant_rejects_existing_collection_with_wrong_dimension(fake_qdrant_db):
    fake_qdrant_db.qdrant_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=384)),
        )
    )

    with pytest.raises(ValueError, match="expected 1024, got 384"):
        fake_qdrant_db.create_collection("documents")


def test_qdrant_filters_nested_metadata(fake_qdrant_db):
    fake_qdrant_db.query_documents(
        "documents",
        "怎么请年假",
        filters={"filename": "员工手册.pdf"},
    )

    query_filter = fake_qdrant_db.qdrant_client.query_points.call_args.kwargs[
        "query_filter"
    ]
    assert query_filter.must[0].key == "metadata.filename"


def test_sentence_transformer_batch_applies_profile_normalization(fake_qdrant_db):
    model = MagicMock()
    model.encode.return_value = [
        [0.1] * 1024,
        [0.2] * 1024,
    ]
    fake_qdrant_db._st_model_cache["BAAI/bge-large-zh-v1.5"] = model

    vectors = fake_qdrant_db.generate_embeddings_batch(["第一段", "第二段"])

    assert len(vectors) == 2
    model.encode.assert_called_once_with(
        ["第一段", "第二段"],
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def test_qdrant_deletes_exact_session_collection_and_profile_variants(fake_qdrant_db):
    fake_qdrant_db.qdrant_client.get_collections.return_value = SimpleNamespace(
        collections=[
            SimpleNamespace(name="documents"),
            SimpleNamespace(name="sess_abc123"),
            SimpleNamespace(name="sess_abc123__st_bge_large_zh"),
            SimpleNamespace(name="sess_other"),
        ]
    )

    deleted = fake_qdrant_db.delete_collection_family("sess_abc123")

    assert deleted == ["sess_abc123", "sess_abc123__st_bge_large_zh"]
    assert [
        call.kwargs["collection_name"]
        for call in fake_qdrant_db.qdrant_client.delete_collection.call_args_list
    ] == deleted


def test_chroma_deletes_exact_session_collection_and_profile_variants():
    db = VectorDatabase(mode="dev")
    client = MagicMock()
    client.list_collections.return_value = [
        SimpleNamespace(name="documents"),
        SimpleNamespace(name="sess_abc123"),
        SimpleNamespace(name="sess_abc123__st_bge_large_zh"),
        SimpleNamespace(name="sess_other"),
    ]
    db._chroma_client = client

    deleted = db.delete_collection_family("sess_abc123")

    assert deleted == ["sess_abc123", "sess_abc123__st_bge_large_zh"]
    assert [call.kwargs["name"] for call in client.delete_collection.call_args_list] == deleted
