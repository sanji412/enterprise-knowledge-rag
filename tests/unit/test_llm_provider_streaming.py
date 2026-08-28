import os

import src.core.llm_provider as llm_provider
from src.core.llm_provider import AnthropicProvider, GeminiProvider, OpenAIProvider


class _FakeResp:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


class _FakeJsonResp:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_deepseek_provider_uses_own_key_and_disables_thinking(monkeypatch):
    provider_class = getattr(llm_provider, "OpenAICompatibleProvider", None)
    assert provider_class is not None

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return _FakeJsonResp({"choices": [{"message": {"content": "答案"}}]})

    monkeypatch.setattr("src.core.llm_provider.requests.post", fake_post)
    provider = provider_class(
        "https://api.deepseek.com",
        10,
        provider_name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        extra_body={"thinking": {"type": "disabled"}},
    )

    assert provider.generate("问题", "deepseek-v4-flash") == "答案"
    assert captured["headers"]["Authorization"] == "Bearer test-deepseek-key"
    assert captured["json"]["thinking"] == {"type": "disabled"}


def test_deepseek_stream_parses_openai_sse(monkeypatch):
    provider_class = getattr(llm_provider, "OpenAICompatibleProvider", None)
    assert provider_class is not None

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    lines = [
        'data: {"choices":[{"delta":{"content":"你"}}]}',
        'data: {"choices":[{"delta":{"content":"好"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        "src.core.llm_provider.requests.post",
        lambda *args, **kwargs: _FakeResp(lines),
    )
    provider = provider_class(
        "https://api.deepseek.com",
        10,
        provider_name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        extra_body={"thinking": {"type": "disabled"}},
    )

    assert "".join(provider.stream("问题", "deepseek-v4-flash")) == "你好"


def test_openai_stream_parses_sse(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "k"
    lines = [
        'data: {"choices":[{"delta":{"content":"hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _FakeResp(lines))
    p = OpenAIProvider("https://api.openai.com/v1", 10)
    assert "".join(p.stream("q", "gpt")) == "hello world"


def test_anthropic_stream_parses_sse(monkeypatch):
    os.environ["ANTHROPIC_API_KEY"] = "k"
    lines = [
        'data: {"type":"content_block_delta","delta":{"text":"hi"}}',
        'data: {"type":"content_block_delta","delta":{"text":" there"}}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _FakeResp(lines))
    p = AnthropicProvider("https://api.anthropic.com/v1", 10)
    assert "".join(p.stream("q", "claude")) == "hi there"


def test_gemini_stream_parses_sse(monkeypatch):
    os.environ["GEMINI_API_KEY"] = "k"
    lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"a"}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"b"}]}}]}',
    ]
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _FakeResp(lines))
    p = GeminiProvider("https://generativelanguage.googleapis.com/v1beta", 10)
    assert "".join(p.stream("q", "gemini-1.5-flash")) == "ab"
