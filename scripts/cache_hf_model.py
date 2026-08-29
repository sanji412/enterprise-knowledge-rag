from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

Downloader = Callable[[str, Path], None]


def _download_with_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.incomplete")
    subprocess.run(  # noqa: S603
        [
            "curl",
            "--fail",
            "--location",
            "--http1.1",
            "--silent",
            "--show-error",
            "--retry",
            "10",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    os.replace(partial, destination)


def _safe_relative_path(model_file: str) -> PurePosixPath:
    relative = PurePosixPath(model_file)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError(f"Unsafe model file path: {model_file}")
    return relative


def cache_snapshot(
    *,
    repo_id: str,
    revision: str,
    files: Sequence[str],
    cache_root: Path,
    endpoint: str,
    downloader: Downloader = _download_with_resume,
) -> Path:
    repo_cache = cache_root / "hub" / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)

    base_url = endpoint.rstrip("/")
    for model_file in files:
        relative = _safe_relative_path(model_file)
        destination = snapshot.joinpath(*relative.parts)
        if not destination.exists():
            url = f"{base_url}/{repo_id}/resolve/{revision}/{relative.as_posix()}"
            downloader(url, destination)

    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")
    return snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a pinned Hugging Face model snapshot")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--file", action="append", dest="files", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_snapshot(
        repo_id=args.repo_id,
        revision=args.revision,
        files=args.files,
        cache_root=args.cache_root,
        endpoint=args.endpoint,
    )


if __name__ == "__main__":
    main()
