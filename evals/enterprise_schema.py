from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EnterpriseCase(BaseModel):
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    category: str
    question: str
    answerable: bool
    required_facts: list[list[str]] = Field(default_factory=list)
    gold_evidence: list[str] = Field(default_factory=list)
    reference_answer: str


def load_enterprise_cases(path: str | Path) -> list[EnterpriseCase]:
    rows = [
        EnterpriseCase.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Enterprise case ids must be unique")
    return rows
