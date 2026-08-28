"""Export the FastAPI OpenAPI document for frontend type generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.api.main import app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
