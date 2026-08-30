"""Session-scoped corpus storage for demo uploads."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable
from uuid import uuid4

SESSION_ROOT = Path("/tmp/doc-ingest-sessions")
SESSION_TTL_SECONDS = int(1800)
JANITOR_MAX_BYTES = 1024 * 1024 * 1024

_LOCK = threading.RLock()
logger = logging.getLogger(__name__)
VectorCleanup = Callable[[str], None]


def _session_root() -> Path:
    return Path(os.getenv("DOC_DEMO_SESSION_ROOT", str(SESSION_ROOT)))


def _session_ttl_seconds() -> int:
    raw = os.getenv("DOC_DEMO_SESSION_TTL", str(SESSION_TTL_SECONDS))
    return max(1, int(raw))


@dataclass
class SessionCorpus:
    session_id: str
    upload_dir: Path
    # Deprecated compatibility path. Qdrant session isolation uses collection_name.
    chroma_path: Path
    bm25_index_path: Path
    collection_name: str
    created_at: float


def new_session_id() -> str:
    return uuid4().hex[:12]


def _session_dir(sid: str) -> Path:
    return _session_root() / sid


def _touched_path(sid: str) -> Path:
    return _session_dir(sid) / ".touched"


def _materialize_session(sid: str) -> SessionCorpus:
    root = _session_dir(sid)
    uploads = root / "uploads"
    chroma = root / "chroma"
    bm25 = root / "bm25_index.json"
    touched = _touched_path(sid)
    root.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    chroma.mkdir(parents=True, exist_ok=True)
    touched.touch(exist_ok=True)
    now = time.time()
    os.utime(touched, (now, now))
    return SessionCorpus(
        session_id=sid,
        upload_dir=uploads,
        chroma_path=chroma,
        bm25_index_path=bm25,
        collection_name=f"sess_{sid}",
        created_at=root.stat().st_ctime if root.exists() else now,
    )


def get_or_create(sid: str) -> SessionCorpus:
    with _LOCK:
        return _materialize_session(sid)


def touch(sid: str) -> None:
    with _LOCK:
        touched = _touched_path(sid)
        if not touched.exists():
            _materialize_session(sid)
            return
        now = time.time()
        os.utime(touched, (now, now))


def total_bytes(session: SessionCorpus) -> int:
    total = 0
    for file_path in session.upload_dir.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def list_active_sessions() -> list[SessionCorpus]:
    root = _session_root()
    if not root.exists():
        return []
    sessions: list[SessionCorpus] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            sessions.append(
                SessionCorpus(
                    session_id=child.name,
                    upload_dir=child / "uploads",
                    chroma_path=child / "chroma",
                    bm25_index_path=child / "bm25_index.json",
                    collection_name=f"sess_{child.name}",
                    created_at=child.stat().st_ctime,
                )
            )
    return sessions


@lru_cache(maxsize=1)
def _default_vector_cleanup() -> VectorCleanup:
    """Build the vector dependency only when a session is actually deleted."""
    from src.utils.config import load_config  # noqa: PLC0415
    from src.utils.vector_factory import build_vector_database  # noqa: PLC0415

    cfg = load_config()
    profile_name = cfg.embeddings.resolve_profile_name(None)
    db = build_vector_database(cfg, profile_name)
    return db.delete_collection_family


def delete_session(sid: str, *, vector_cleanup: VectorCleanup | None = None) -> None:
    with _LOCK:
        path = _session_dir(sid)
        cleanup = vector_cleanup or _default_vector_cleanup()
        try:
            cleanup(f"sess_{sid}")
        except Exception:
            logger.exception(
                "Vector cleanup failed for session %s; preserving local state for retry",
                sid,
            )
            raise
        if path.exists():
            shutil.rmtree(path)


def janitor_sweep(
    now: float | None = None,
    *,
    vector_cleanup: VectorCleanup | None = None,
) -> int:
    with _LOCK:
        ts = time.time() if now is None else now
        ttl = _session_ttl_seconds()
        deleted = 0
        sessions = list_active_sessions()
        for session in sessions:
            touched = _touched_path(session.session_id)
            last_touch = touched.stat().st_mtime if touched.exists() else session.created_at
            if ts - last_touch > ttl:
                delete_session(session.session_id, vector_cleanup=vector_cleanup)
                deleted += 1

        # If disk grows beyond cap, evict oldest touched sessions.
        root = _session_root()
        if root.exists():
            entries = [p for p in root.rglob("*") if p.is_file()]
            total_size = sum(p.stat().st_size for p in entries)
            if total_size > JANITOR_MAX_BYTES:
                ordered = sorted(
                    list_active_sessions(),
                    key=lambda s: _touched_path(s.session_id).stat().st_mtime
                    if _touched_path(s.session_id).exists()
                    else s.created_at,
                )
                for session in ordered:
                    if total_size <= JANITOR_MAX_BYTES:
                        break
                    before = sum(
                        p.stat().st_size
                        for p in _session_dir(session.session_id).rglob("*")
                        if p.is_file()
                    )
                    delete_session(session.session_id, vector_cleanup=vector_cleanup)
                    deleted += 1
                    total_size -= before
        return deleted
