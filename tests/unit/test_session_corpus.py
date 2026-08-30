from __future__ import annotations

import os
import threading
import time

import pytest
from src.web import session_corpus


def test_get_or_create_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path))
    sid = "abc123def456"
    a = session_corpus.get_or_create(sid)
    b = session_corpus.get_or_create(sid)
    assert a.session_id == sid
    assert b.upload_dir == a.upload_dir
    assert a.upload_dir.exists()
    assert a.chroma_path.exists()


def test_delete_session_without_local_files_still_cleans_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path))
    cleaned: list[str] = []

    session_corpus.delete_session("missing", vector_cleanup=cleaned.append)

    assert cleaned == ["sess_missing"]
    assert not (tmp_path / "missing").exists()


def test_delete_session_cleans_vectors_before_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path))
    session = session_corpus.get_or_create("abc123")
    cleanup_calls: list[tuple[str, bool]] = []

    def cleanup(collection_name: str) -> None:
        cleanup_calls.append((collection_name, session.upload_dir.parent.exists()))

    session_corpus.delete_session("abc123", vector_cleanup=cleanup)

    assert cleanup_calls == [("sess_abc123", True)]
    assert not session.upload_dir.parent.exists()


def test_delete_session_preserves_local_state_when_vector_cleanup_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path))
    session = session_corpus.get_or_create("abc123")

    def cleanup(_collection_name: str) -> None:
        raise RuntimeError("qdrant unavailable")

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        session_corpus.delete_session("abc123", vector_cleanup=cleanup)

    assert session.upload_dir.parent.exists()


def test_janitor_sweep_evicts_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path))
    monkeypatch.setenv("DOC_DEMO_SESSION_TTL", "1")
    sid = "ttl001"
    s = session_corpus.get_or_create(sid)
    old = time.time() - 10
    os.utime(s.upload_dir.parent / ".touched", (old, old))
    cleaned: list[str] = []
    deleted = session_corpus.janitor_sweep(
        now=time.time(),
        vector_cleanup=cleaned.append,
    )
    assert deleted >= 1
    assert cleaned == ["sess_ttl001"]
    assert not s.upload_dir.parent.exists()


def test_concurrent_get_or_create_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_DEMO_SESSION_ROOT", str(tmp_path))
    sid = "concur123456"
    errors: list[Exception] = []

    def _run():
        try:
            session_corpus.get_or_create(sid)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert (tmp_path / sid / "uploads").exists()
