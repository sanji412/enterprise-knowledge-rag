from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import jieba

_ASCII_TERM = re.compile(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*")
_PUNCT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_.-]+")
_ONLY_SEPARATOR = re.compile(r"^[-_.]+$")
_STOPWORDS = {"的", "了", "和", "是", "在", "怎么", "如何", "请问", "一下"}


@lru_cache(maxsize=1)
def _load_domain_terms() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[2] / "config" / "domain_terms.txt"
    if not path.is_file():
        return ()
    terms = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    for term in terms:
        jieba.add_word(term)
    return terms


def tokenize_search_text(text: str) -> list[str]:
    _load_domain_terms()
    normalized = _PUNCT.sub(" ", text).strip().lower()
    tokens: list[str] = []

    def append_chinese_segment(segment: str) -> None:
        for raw_token in jieba.lcut(segment, cut_all=False):
            token = raw_token.strip().lower()
            if not token or token in _STOPWORDS or _ONLY_SEPARATOR.fullmatch(token):
                continue
            tokens.append(token)

    cursor = 0
    for match in _ASCII_TERM.finditer(normalized):
        append_chinese_segment(normalized[cursor:match.start()])
        tokens.append(match.group(0).lower())
        cursor = match.end()
    append_chinese_segment(normalized[cursor:])
    return tokens
