'''
- ChromaDB for development
- Qdrant for production scaling
- Embedding generation via Ollama
- Batch operations for efficiency
- Metadata filtering capabilities
'''
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

import chromadb
import ollama
from src.utils.config import EmbeddingProfile

# qdrant_client is imported lazily inside _init_qdrant() so the module can be
# imported on environments where qdrant-client is not installed (e.g. HF Spaces
# running in Chroma-only / dev mode).

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def _qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"enterprise-rag:{chunk_id}"))


class VectorDatabase:
    def __init__(
        self,
        mode: str = "dev",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        chroma_path: str = "./chroma_db",
        qdrant_url: Optional[str] = None,
        embedding_profile_name: str = "ollama_nomic",
        embedding_profile: Optional[EmbeddingProfile] = None,
    ):
        self.mode = mode
        self._chroma_path = chroma_path
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._qdrant_url = qdrant_url or f"http://{qdrant_host}:{qdrant_port}"
        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._qdrant_client: Optional[Any] = None  # QdrantClient, imported lazily
        self._ollama_client = ollama.Client(host=self._resolve_ollama_host())
        self._st_model_cache: Dict[str, Any] = {}
        self.embedding_profile_name = embedding_profile_name
        self.embedding_profile = embedding_profile or EmbeddingProfile(
            provider=os.getenv("DOC_EMBEDDING_PROVIDER", "ollama").lower(),
            framework=os.getenv("DOC_EMBEDDING_PROVIDER", "ollama").lower(),
            model="nomic-embed-text",
            dimension=768,
        )

    # --- client accessors (lazy init) ---

    @property
    def chroma_client(self) -> chromadb.ClientAPI:
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=self._chroma_path)
            logger.info("ChromaDB initialized at %s", self._chroma_path)
        return self._chroma_client

    @property
    def qdrant_client(self) -> Any:
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient  # noqa: PLC0415

            self._qdrant_client = QdrantClient(url=self._qdrant_url)
            logger.info("Qdrant initialized at %s", self._qdrant_url)
        return self._qdrant_client

    # --- embedding ---

    @staticmethod
    def _resolve_ollama_host() -> str:
        return (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        )

    def _generate_st_embeddings(
        self,
        texts: List[str],
        model_name: str,
    ) -> List[List[float]]:
        if model_name not in self._st_model_cache:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._st_model_cache[model_name] = SentenceTransformer(model_name)
            logger.info("SentenceTransformer loaded: %s", model_name)
        options = dict(self.embedding_profile.options or {})
        normalize = bool(options.get("normalize_embeddings", False))
        vectors = self._st_model_cache[model_name].encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    def generate_embedding(self, text: str) -> List[float]:
        provider = self.embedding_profile.provider.strip().lower()
        if provider == "sentence_transformers":
            return self._generate_st_embeddings(
                [text],
                self.embedding_profile.model,
            )[0]
        attempts = 3
        last_error: Exception | None = None
        for idx in range(attempts):
            try:
                response = self._ollama_client.embeddings(model=self.embedding_profile.model, prompt=text)
                return response["embedding"]  # type: ignore[return-value]
            except Exception as exc:
                last_error = exc
                if idx == attempts - 1:
                    raise
                time.sleep(0.35 * (idx + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected Ollama embedding retry state")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        provider = self.embedding_profile.provider.strip().lower()
        if provider == "sentence_transformers":
            return self._generate_st_embeddings(texts, self.embedding_profile.model)
        return [self.generate_embedding(t) for t in texts]

    # --- collection management ---
    def collection_name_for_profile(self, base_collection_name: str) -> str:
        default_aliases = {"documents"}
        if self.embedding_profile_name in {"", "ollama_nomic"} and base_collection_name in default_aliases:
            return base_collection_name
        return f"{base_collection_name}__{self.embedding_profile_name}"

    def _assert_embedding_dimension(self, embedding: List[float]) -> None:
        expected = int(self.embedding_profile.dimension)
        if len(embedding) != expected:
            raise ValueError(
                f"Embedding dimension mismatch for profile {self.embedding_profile_name!r}: "
                f"expected {expected}, got {len(embedding)}"
            )

    def create_collection(self, collection_name: str) -> None:
        profile_collection_name = self.collection_name_for_profile(collection_name)
        if self.mode == "dev":
            self.chroma_client.get_or_create_collection(name=profile_collection_name)
            logger.info("ChromaDB collection %r ready", profile_collection_name)
        else:
            from qdrant_client.http.models import Distance, VectorParams  # noqa: PLC0415

            if self.qdrant_client.collection_exists(profile_collection_name):
                info = self.qdrant_client.get_collection(profile_collection_name)
                actual_size = self._qdrant_collection_vector_size(info)
                expected_size = int(self.embedding_profile.dimension)
                if actual_size != expected_size:
                    raise ValueError(
                        f"Qdrant collection {profile_collection_name!r} vector dimension mismatch: "
                        f"expected {expected_size}, got {actual_size}"
                    )
                return
            self.qdrant_client.create_collection(
                collection_name=profile_collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_profile.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Qdrant collection %r created", profile_collection_name)

    @staticmethod
    def _qdrant_collection_vector_size(info: Any) -> int:
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            if "size" in vectors:
                return int(vectors["size"])
            if len(vectors) == 1:
                only = next(iter(vectors.values()))
                if hasattr(only, "size"):
                    return int(only.size)
                if isinstance(only, dict) and "size" in only:
                    return int(only["size"])
        raise ValueError("Unable to determine Qdrant collection vector dimension")

    # --- insert ---

    def add_documents(self, collection_name: str, documents: List[Dict]) -> None:
        """Insert documents with embeddings generated from document['text'].

        Each document dict must have 'id' and 'text'; all other keys go into metadata.
        """
        self.create_collection(collection_name)
        profile_collection_name = self.collection_name_for_profile(collection_name)
        for batch_start in range(0, len(documents), BATCH_SIZE):
            batch = documents[batch_start: batch_start + BATCH_SIZE]
            texts = [doc["text"] for doc in batch]
            embeddings = self.generate_embeddings_batch(texts)
            for emb in embeddings:
                self._assert_embedding_dimension(emb)

            if self.mode == "dev":
                collection = self.chroma_client.get_or_create_collection(name=profile_collection_name)
                metadatas = [
                    ({k: v for k, v in doc.items() if k not in ("id", "text")} or None)
                    for doc in batch
                ]
                collection.upsert(  # type: ignore[arg-type]
                    ids=[str(doc["id"]) for doc in batch],
                    documents=texts,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    metadatas=metadatas,  # type: ignore[arg-type]
                )
            else:
                from qdrant_client.http.models import PointStruct  # noqa: PLC0415
                points = [
                    PointStruct(
                        id=_qdrant_point_id(str(doc["id"])),
                        vector=embedding,
                        payload={
                            "chunk_id": str(doc["id"]),
                            "text": str(doc["text"]),
                            "metadata": {
                                k: v
                                for k, v in doc.items()
                                if k not in ("id", "text")
                            },
                        },
                    )
                    for doc, embedding in zip(batch, embeddings)
                ]
                self.qdrant_client.upsert(collection_name=profile_collection_name, points=points)

            logger.info("Upserted batch of %d documents into %r", len(batch), profile_collection_name)

    # --- query ---

    def query_documents(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for similar documents, optionally filtered by metadata key/value pairs."""
        profile_collection_name = self.collection_name_for_profile(collection_name)
        query_embedding = self.generate_embedding(query_text)
        self._assert_embedding_dimension(query_embedding)

        if self.mode == "dev":
            collection = self.chroma_client.get_or_create_collection(name=profile_collection_name)
            where = ({k: v for k, v in filters.items()} if filters else None)
            results = collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=top_k,
                where=where,
            )
            ids: List[str] = (results["ids"] or [[]])[0]
            docs: List[str] = (results["documents"] or [[]])[0]
            metas: List[Dict] = (results["metadatas"] or [[]])[0]  # type: ignore[assignment]
            dists: List[float] = (results["distances"] or [[]])[0]
            return [
                {"id": id_, "text": doc, "metadata": meta, "distance": dist}
                for id_, doc, meta, dist in zip(ids, docs, metas, dists)
            ]
        else:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415
            search_filter: Optional[Any] = None
            if filters:
                conditions: Sequence[FieldCondition] = [
                    FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
                search_filter = Filter(must=list(conditions))

            response = self.qdrant_client.query_points(
                collection_name=profile_collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )
            rows: List[Dict] = []
            for hit in response.points:
                payload = dict(hit.payload or {})
                score = float(hit.score)
                rows.append(
                    {
                        "id": str(payload["chunk_id"]),
                        "text": str(payload.get("text", "")),
                        "metadata": dict(payload.get("metadata") or {}),
                        "score": score,
                        "distance": 1.0 - score,
                    }
                )
            return rows
