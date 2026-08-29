import subprocess
from pathlib import Path

import pytest
from scripts.cache_hf_model import _download_with_resume, cache_snapshot


def test_cache_snapshot_writes_standard_huggingface_layout(tmp_path: Path):
    requested: list[str] = []

    def fake_download(url: str, destination: Path) -> None:
        requested.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"downloaded:{destination.name}", encoding="utf-8")

    snapshot = cache_snapshot(
        repo_id="BAAI/example-model",
        revision="abc123",
        files=["config.json", "1_Pooling/config.json"],
        cache_root=tmp_path,
        endpoint="https://mirror.example",
        downloader=fake_download,
    )

    assert snapshot == (
        tmp_path
        / "hub"
        / "models--BAAI--example-model"
        / "snapshots"
        / "abc123"
    )
    assert (snapshot / "config.json").read_text() == "downloaded:config.json"
    assert (snapshot / "1_Pooling/config.json").read_text() == "downloaded:config.json"
    assert (snapshot.parent.parent / "refs/main").read_text() == "abc123"
    assert requested == [
        "https://mirror.example/BAAI/example-model/resolve/abc123/config.json",
        "https://mirror.example/BAAI/example-model/resolve/abc123/1_Pooling/config.json",
    ]


def test_cache_snapshot_rejects_paths_outside_snapshot(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsafe model file path"):
        cache_snapshot(
            repo_id="BAAI/example-model",
            revision="abc123",
            files=["../secret"],
            cache_root=tmp_path,
            endpoint="https://mirror.example",
        )


def test_download_uses_curl_resume_then_promotes_partial(tmp_path: Path, monkeypatch):
    destination = tmp_path / "model.bin"
    partial = tmp_path / "model.bin.incomplete"
    partial.write_bytes(b"partial-")
    captured: list[str] = []

    def fake_run(command: list[str], *, check: bool):
        captured.extend(command)
        assert check is True
        output = Path(command[command.index("--output") + 1])
        with output.open("ab") as handle:
            handle.write(b"complete")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    _download_with_resume("https://mirror.example/model.bin", destination)

    assert destination.read_bytes() == b"partial-complete"
    assert not partial.exists()
    assert "--http1.1" in captured
    assert "--continue-at" in captured
    assert captured[captured.index("--continue-at") + 1] == "-"
