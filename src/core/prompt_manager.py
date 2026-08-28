"""Template-based RAG prompts with optional filesystem overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from src.core.context_optimizer import OptimizedContext
from src.core.query_processor import QueryIntent

FACTUAL_TEMPLATE = """你是企业知识库问答助手。只能依据下方证据回答。

证据：
{context}

用户问题：{query}

每个关键事实后必须使用 [Doc chunk_id] 标注证据；证据不足时明确拒答。

回答：
"""

EXPLORATORY_TEMPLATE = """你是企业知识库问答助手。只能依据下方证据分析。

证据：
{context}

用户问题：{query}

需要比较时分别引用每个结论，格式为 [Doc chunk_id]；证据不足时明确拒答。

回答：
"""

DEFAULT_TEMPLATE = FACTUAL_TEMPLATE


def _format_context(ctx: OptimizedContext) -> str:
    parts = []
    for doc in ctx.documents:
        did = doc.get("id", "")
        text = doc.get("text", "")
        parts.append(f"[Doc {did}]\n{text}")
    return "\n\n---\n\n".join(parts)


class PromptManager:
    """Build user prompts from templates; optional YAML overrides under template_path."""

    def __init__(self, template_path: str | Path | None = None) -> None:
        self.template_path = (
            Path(template_path)
            if template_path
            else Path(__file__).resolve().parents[2] / "config" / "prompts"
        )
        self.templates: Dict[str, str] = {
            "factual": FACTUAL_TEMPLATE,
            "exploratory": EXPLORATORY_TEMPLATE,
            "default": DEFAULT_TEMPLATE,
        }
        self._load_dir_templates()

    def _load_dir_templates(self) -> None:
        root = self.template_path
        if not root.is_dir():
            return
        for name in ("factual", "exploratory", "default"):
            p = root / f"{name}.yaml"
            if p.is_file():
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    body = data.get("template")
                    if isinstance(body, str):
                        self.templates[name] = body

    @staticmethod
    def intent_to_query_type(intent: QueryIntent) -> str:
        if intent is QueryIntent.EXPLORATORY or intent is QueryIntent.COMPARATIVE:
            return "exploratory"
        return "factual"

    def get_system_prompt(self, query_type: str) -> str:
        """Short system line for chat APIs that support a system role."""
        if query_type == "exploratory":
            return "仅依据给定证据进行分析，并分别使用 [Doc id] 标注结论。"
        return "仅依据给定证据准确回答，并使用 [Doc id] 标注关键事实。"

    def build_prompt(self, query: str, context: OptimizedContext, query_type: str = "factual") -> str:
        tpl = self.templates.get(query_type) or self.templates["default"]
        ctx_block = _format_context(context)
        return tpl.format(context=ctx_block, query=query)
