from __future__ import annotations

from src.utils.config import Config
from src.utils.database import VectorDatabase


def build_vector_database(
    cfg: Config,
    embedding_profile_name: str,
    collection_name: str | None = None,
) -> VectorDatabase:
    profile = cfg.embeddings.resolve_profile(embedding_profile_name)
    db = VectorDatabase(
        mode="prod" if cfg.vector_store.backend == "qdrant" else "dev",
        qdrant_url=cfg.vector_store.qdrant_url,
        chroma_path=cfg.vector_store.chroma_path,
        embedding_profile_name=embedding_profile_name,
        embedding_profile=profile,
    )
    if collection_name:
        db.create_collection(collection_name)
    return db
