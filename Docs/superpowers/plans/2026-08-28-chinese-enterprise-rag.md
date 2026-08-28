# Chinese Enterprise RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Doc-Ingestion 改造成以 DeepSeek、中文混合检索、可信引用、无依据拒答和 30 条离线评测为核心的企业知识库问答系统。

**Architecture:** 文档解析后生成带稳定元数据的中文切片，同一切片写入 Jieba BM25 与 Qdrant；查询同时执行稀疏和稠密召回，经加权 RRF 与中文 Cross-Encoder 重排后交给 DeepSeek。生成结果必须通过引用映射和 GroundingPolicy，前端展示答案、拒答状态与可点击证据，离线评测对四种检索组合做分层对比。

**Tech Stack:** Python 3.13、FastAPI、Pydantic、DeepSeek OpenAI-compatible API、sentence-transformers、BAAI BGE、Jieba、Qdrant、BM25、React 18、TypeScript、Vite、Docker Compose、pytest、Vitest。

## Global Constraints

- 正式路径只支持 PDF、DOCX、Markdown；保留 TXT、HTML 兼容，不新增 OCR、图片和表格理解。
- DeepSeek API Key 只允许从 `DEEPSEEK_API_KEY` 或请求级临时覆盖读取，绝不写入代码、日志、测试夹具和评测报告。
- 默认生成模型使用 `deepseek-v4-flash`；API base URL 使用 `https://api.deepseek.com`。
- 默认中文 Embedding 使用 `BAAI/bge-large-zh-v1.5`，向量维度固定为 `1024`。
- 默认重排模型使用 `BAAI/bge-reranker-v2-m3`。
- 正式向量库只使用 Qdrant；Chroma 仅保留为旧代码兼容入口，不进入演示主路径。
- 中文切片初始参数为 600 字符、100 字符重叠，最终参数只根据评测结果调整。
- 相同 `chunk_id` 必须贯穿 BM25、Qdrant、召回、引用和黄金评测集。
- 30 条用例固定为 Easy 10、Medium 10、Hard 10，其中 5 条为不可回答问题。
- 目标值不是成绩；README 和简历只能写最终实测数据。
- 不停止或删除其他项目的容器；本项目默认使用宿主端口 8100、16333、16379 避免冲突。
- 所有行为改动采用 TDD：先写失败测试，确认失败原因，再写最小实现。
- 每个任务独立提交；提交前至少运行该任务的目标测试与 Ruff。

## Verified External Contracts

- DeepSeek OpenAI-compatible base URL：<https://api.deepseek.com/>
- DeepSeek Chat Completions：<https://api-docs.deepseek.com/api/create-chat-completion/>
- `BAAI/bge-large-zh-v1.5` 模型卡：<https://huggingface.co/BAAI/bge-large-zh-v1.5>
- `BAAI/bge-reranker-v2-m3` 模型卡：<https://huggingface.co/BAAI/bge-reranker-v2-m3>

## File Map

### New focused modules

- `src/core/chinese_tokenizer.py`：Jieba 分词、停用词和领域词统一入口。
- `src/core/chunk_schema.py`：`SourceUnit`、`ChunkRecord` 与稳定切片 ID。
- `src/core/grounding_policy.py`：证据充分性、答案状态和拒答原因。
- `src/utils/vector_factory.py`：根据配置创建 Qdrant/兼容 Chroma `VectorDatabase`。
- `src/evaluation/enterprise_metrics.py`：Hit@5、MRR@5、事实覆盖、引用和拒答指标。
- `evals/enterprise_schema.py`：30 条用例的加载与结构校验。
- `evals/run_enterprise_evals.py`：真实单配置评测入口。
- `evals/run_ablation.py`：四组检索消融入口。
- `scripts/build_enterprise_corpus.py`：从可审阅的 Markdown 源生成 PDF、DOCX、Markdown 语料。
- `scripts/docker-smoke.sh`：Docker 服务、上传、问答和拒答冒烟。

### Existing modules to extend

- `src/utils/config.py`、`config.yaml`：DeepSeek、Qdrant、中文模型和 Grounding 配置。
- `src/core/llm_provider.py`：通用 OpenAI-compatible Provider 和 DeepSeek 路由。
- `src/utils/database.py`：Qdrant 点 ID、payload、查询返回与维度校验。
- `src/core/document_processor.py`、`src/ingest.py`：结构化中文切片与元数据入库。
- `src/core/bm25_index.py`、`src/core/query_processor.py`：中文分词和查询处理。
- `src/core/hybrid_retriever.py`、`src/core/rag_orchestrator.py`：检索模式、调试信息、拒答和响应状态。
- `src/core/prompt_manager.py`、`src/core/citation_tracker.py`、`src/core/citation_verifier.py`：中文 Prompt 和引用证据。
- `src/api/models.py`、`src/api/main.py`：状态、证据和延迟字段。
- `frontend/src/tabs/QueryTab.tsx`、`frontend/src/components/AnswerPanel.tsx`、`frontend/src/components/CitationsList.tsx`：中文主流程与证据面板。
- `docker/docker-compose.yml`、`docker/.env.example`、`Dockerfile`：DeepSeek、Qdrant 和无端口冲突运行。

---

### Task 1: Pin the local runtime and record a clean baseline

**Files:**
- Create: `.python-version`
- Create: `Docs/baseline-2026-08-28.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: upstream commit `8588513` and the existing Python/React test suites.
- Produces: reproducible Python 3.13 environment and a baseline record used by every later task.

- [ ] **Step 1: Verify the branch and working tree before setup**

Run:

```bash
git branch --show-current
git status --short
```

Expected: branch is `feature/chinese-enterprise-rag`; only this plan file may be uncommitted.

- [ ] **Step 2: Pin Python 3.13 and create the environment**

Create `.python-version` with exactly:

```text
3.13
```

Run:

```bash
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements/base.txt
uv pip install --python .venv/bin/python pytest pytest-cov ruff mypy
```

Expected: `.venv/bin/python --version` starts with `Python 3.13`.

- [ ] **Step 3: Ignore local runtime artifacts**

Ensure `.gitignore` contains these exact entries once:

```gitignore
.venv/
.uv-cache/
docker/.env
evals/reports/
data/embeddings/
```

- [ ] **Step 4: Run the upstream Python baseline**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests evals
```

Expected: record the exact pass/fail counts without changing product code. If a test fails, stop execution and diagnose before Task 2.

- [ ] **Step 5: Run the upstream frontend baseline**

Run:

```bash
cd frontend
npm ci
npm test
npm run typecheck
npm run build
cd ..
```

Expected: Vitest, TypeScript and Vite build finish successfully; otherwise stop before Task 2.

- [ ] **Step 6: Record reproducible baseline evidence**

Create `Docs/baseline-2026-08-28.md` with these headings, then paste the literal command output under the matching heading:

```markdown
# Baseline Verification — 2026-08-28

- Upstream commit: `8588513`
## Python version

## Python tests

## Ruff

## Frontend tests

## TypeScript

## Frontend build

No product behavior was changed during baseline verification.
```

- [ ] **Step 7: Commit the reproducible baseline**

```bash
git add .python-version .gitignore Docs/baseline-2026-08-28.md
git commit -m "chore: pin runtime and record baseline"
```

### Task 2: Add first-class DeepSeek provider support

**Files:**
- Modify: `src/utils/config.py`
- Modify: `src/core/llm_provider.py`
- Modify: `config.yaml`
- Modify: `docker/.env.example`
- Modify: `docker/docker-compose.yml`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_llm_provider_streaming.py`

**Interfaces:**
- Consumes: existing `LLMProvider` protocol and `LLMProviderRouter`.
- Produces: `provider_api_key_env("deepseek") -> "DEEPSEEK_API_KEY"`, `LLMSettings.deepseek_base_url`, and router key `deepseek` supporting `generate()` and `stream()`.

- [ ] **Step 1: Write failing config tests**

Add to `tests/unit/test_config.py`:

```python
from src.utils.config import LLMSettings, provider_api_key_env


def test_deepseek_defaults_are_openai_compatible(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    settings = LLMSettings(
        default_provider="deepseek",
        default_model_by_provider={"deepseek": "deepseek-v4-flash"},
        allowed_models_by_provider={"deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"]},
    )
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.resolve_model("deepseek", None) == "deepseek-v4-flash"
    assert provider_api_key_env("deepseek") == "DEEPSEEK_API_KEY"
```

- [ ] **Step 2: Write failing provider tests for sync and SSE**

Add tests using the existing response fakes in `tests/unit/test_llm_provider_streaming.py`:

```python
class _FakeJsonResp:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_deepseek_provider_uses_own_key_and_label(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return _FakeJsonResp({"choices": [{"message": {"content": "有依据的回答"}}]})

    monkeypatch.setattr("src.core.llm_provider.requests.post", fake_post)
    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com",
        timeout_seconds=60,
        provider_name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        extra_body={"thinking": {"type": "disabled"}},
    )
    assert provider.generate("问题", "deepseek-v4-flash") == "有依据的回答"
    assert captured["headers"]["Authorization"] == "Bearer test-deepseek-key"
    assert captured["json"]["thinking"] == {"type": "disabled"}
```

Import `OpenAICompatibleProvider` beside the existing provider imports in the test module.

- [ ] **Step 3: Run the two tests and verify the intended failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/unit/test_config.py::test_deepseek_defaults_are_openai_compatible \
  tests/unit/test_llm_provider_streaming.py -q
```

Expected: FAIL because `deepseek_base_url` and `OpenAICompatibleProvider` do not exist.

- [ ] **Step 4: Generalize the OpenAI-compatible provider**

In `src/core/llm_provider.py`, replace the hard-coded key and provider label in `OpenAIProvider` with this interface:

```python
class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int,
        provider_name: str,
        api_key_env: str,
        extra_body: Optional[dict[str, object]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.provider_name = provider_name
        self.api_key_env = api_key_env
        self.extra_body = dict(extra_body or {})

    def _key(self, api_key_override: Optional[str] = None) -> str:
        key = api_key_override or os.getenv(self.api_key_env)
        if not key:
            raise ValueError(f"{self.api_key_env} is required for {self.provider_name} provider")
        return key
```

Move the existing OpenAI `generate` and `stream` request bodies into this class, merge `self.extra_body` into both JSON bodies, and call `_raise_for_status_with_detail(resp, self.provider_name)`. Keep compatibility with:

```python
class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        super().__init__(base_url, timeout_seconds, "openai", "OPENAI_API_KEY")
```

- [ ] **Step 5: Add DeepSeek settings and router registration**

Add to `LLMSettings`:

```python
deepseek_base_url: str = Field(
    default_factory=lambda: _env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    description="DeepSeek OpenAI-compatible API base URL",
)
```

Add to `provider_api_key_env`:

```python
if p == "deepseek":
    return "DEEPSEEK_API_KEY"
```

Register in `LLMProviderRouter.__init__`:

```python
"deepseek": OpenAICompatibleProvider(
    settings.deepseek_base_url,
    settings.request_timeout_seconds,
    "deepseek",
    "DEEPSEEK_API_KEY",
    {"thinking": {"type": "disabled"}},
),
```

The project disables thinking mode for evidence-grounded RAG so the public answer is concise and streaming does not expose reasoning content.

- [ ] **Step 6: Make DeepSeek the product default**

Set `config.yaml` to:

```yaml
llm:
  default_provider: deepseek
  default_model_by_provider:
    deepseek: deepseek-v4-flash
  allowed_models_by_provider:
    deepseek:
      - deepseek-v4-flash
      - deepseek-v4-pro
  request_timeout_seconds: 60
```

Add `DEEPSEEK_API_KEY=` and `DEEPSEEK_BASE_URL=https://api.deepseek.com` to `docker/.env.example`, and pass both variables into the API service in `docker/docker-compose.yml`.

- [ ] **Step 7: Run provider and config tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_llm_provider_streaming.py -q
.venv/bin/python -m ruff check src/utils/config.py src/core/llm_provider.py tests/unit/test_config.py tests/unit/test_llm_provider_streaming.py
```

Expected: all targeted tests and Ruff pass.

- [ ] **Step 8: Commit DeepSeek support**

```bash
git add src/utils/config.py src/core/llm_provider.py config.yaml docker/.env.example docker/docker-compose.yml tests/unit/test_config.py tests/unit/test_llm_provider_streaming.py
git commit -m "feat: add DeepSeek generation provider"
```

### Task 3: Configure Chinese embedding and reranking models

**Files:**
- Modify: `requirements/base.txt`
- Modify: `requirements.txt`
- Modify: `config.yaml`
- Modify: `src/utils/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `EmbeddingProfile`, `EmbeddingSettings`, `RerankerSettings`.
- Produces: embedding profile `st_bge_large_zh` with dimension 1024 and reranker default `BAAI/bge-reranker-v2-m3`.

- [ ] **Step 1: Write failing model configuration tests**

Add to `tests/unit/test_config.py`:

```python
def test_chinese_models_are_default():
    cfg = load_config("config.yaml")
    profile = cfg.embeddings.resolve_profile(None)
    assert cfg.embeddings.default_profile == "st_bge_large_zh"
    assert profile.provider == "sentence_transformers"
    assert profile.model == "BAAI/bge-large-zh-v1.5"
    assert profile.dimension == 1024
    assert cfg.reranker.model == "BAAI/bge-reranker-v2-m3"
```

- [ ] **Step 2: Verify the configuration test fails**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_config.py::test_chinese_models_are_default -q
```

Expected: FAIL because `st_bge_large_zh` is not configured.

- [ ] **Step 3: Add exact Chinese model configuration**

Add to `config.yaml` and make it the default:

```yaml
embeddings:
  default_profile: st_bge_large_zh
  profiles:
    st_bge_large_zh:
      provider: sentence_transformers
      framework: sentence_transformers
      model: BAAI/bge-large-zh-v1.5
      dimension: 1024
      options:
        normalize_embeddings: true

reranker:
  model: BAAI/bge-reranker-v2-m3
  batch_size: 8
  score_threshold: 0.0
  top_k: 5
```

Preserve existing optional profiles only when existing regression tests require them; the UI default and formal path must resolve to `st_bge_large_zh`.

- [ ] **Step 4: Pass embedding options to sentence-transformers**

Change `_generate_st_embedding` in `src/utils/database.py` during Task 4 to accept profile options. For this task, add a config test proving `profile.options["normalize_embeddings"] is True`.

- [ ] **Step 5: Add Chinese tokenization dependency**

Add this exact dependency to both local and hosted requirement files:

```text
jieba>=0.42.1,<0.43
```

Run:

```bash
uv pip install --python .venv/bin/python "jieba>=0.42.1,<0.43"
```

- [ ] **Step 6: Verify model config without downloading models**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_config.py -q
.venv/bin/python -m ruff check src/utils/config.py tests/unit/test_config.py
```

Expected: config tests pass without loading Hugging Face weights.

- [ ] **Step 7: Commit Chinese model configuration**

```bash
git add requirements/base.txt requirements.txt config.yaml src/utils/config.py tests/unit/test_config.py
git commit -m "feat: configure Chinese embedding and reranker"
```

### Task 4: Make Qdrant the complete vector-store path

**Files:**
- Create: `src/utils/vector_factory.py`
- Modify: `src/utils/config.py`
- Modify: `src/utils/database.py`
- Modify: `src/core/vector_search.py`
- Modify: `src/ingest.py`
- Modify: `src/core/rag_orchestrator.py`
- Modify: `src/web/session_corpus.py`
- Modify: `src/web/ingestion_service.py`
- Test: `tests/unit/test_database_qdrant.py`
- Test: `tests/unit/test_rag_orchestrator_session_load.py`
- Test: `tests/integration/test_pipeline.py`

**Interfaces:**
- Consumes: `EmbeddingProfile` and existing `VectorDatabase` operations.
- Produces: `VectorStoreSettings`, `build_vector_database(cfg, embedding_profile_name, collection_name=None)`, and Qdrant results containing original `chunk_id`, `text`, `metadata`, `score` and `distance`.

- [ ] **Step 1: Write failing Qdrant payload and identity tests**

Create `tests/unit/test_database_qdrant.py` with a fake client and these assertions:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.utils.config import EmbeddingProfile
from src.utils.database import VectorDatabase


@pytest.fixture
def fake_qdrant_db():
    profile = EmbeddingProfile(
        provider="sentence_transformers",
        framework="sentence_transformers",
        model="BAAI/bge-large-zh-v1.5",
        dimension=1024,
        options={"normalize_embeddings": True},
    )
    db = VectorDatabase(
        mode="prod",
        qdrant_url="http://qdrant.invalid:6333",
        embedding_profile_name="st_bge_large_zh",
        embedding_profile=profile,
    )
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="4f980c70-207b-5ab0-9437-4c82dd7bea3b",
                payload={
                    "chunk_id": "handbook__leave__chunk0",
                    "text": "年假需提前申请",
                    "metadata": {"filename": "员工手册.pdf"},
                },
                score=0.91,
            )
        ]
    )
    db._qdrant_client = client
    db.generate_embedding = lambda text: [0.1] * 1024
    return db


def test_qdrant_upsert_keeps_text_and_original_chunk_id(fake_qdrant_db):
    fake_qdrant_db.generate_embeddings_batch = lambda texts: [[0.1] * 1024 for _ in texts]
    fake_qdrant_db.add_documents(
        "documents",
        [{"id": "handbook__leave__chunk0", "text": "年假需提前申请", "filename": "员工手册.pdf"}],
    )
    point = fake_qdrant_db.qdrant_client.upsert.call_args.kwargs["points"][0]
    assert str(point.id) != "handbook__leave__chunk0"
    assert point.payload["chunk_id"] == "handbook__leave__chunk0"
    assert point.payload["text"] == "年假需提前申请"


def test_qdrant_query_restores_retrieval_shape(fake_qdrant_db):
    rows = fake_qdrant_db.query_documents("documents", "怎么请年假", top_k=5)
    assert rows[0]["id"] == "handbook__leave__chunk0"
    assert rows[0]["text"] == "年假需提前申请"
    assert rows[0]["score"] >= 0.0
```

- [ ] **Step 2: Run the Qdrant tests and verify failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_database_qdrant.py -q
```

Expected: FAIL because current Qdrant payload drops text and uses an invalid arbitrary string point ID.

- [ ] **Step 3: Add vector store configuration**

Add to `src/utils/config.py`:

```python
class VectorStoreSettings(BaseModel):
    backend: str = Field("qdrant", pattern="^(qdrant|chroma)$")
    qdrant_url: str = Field(
        default_factory=lambda: _env_or("QDRANT_URL", "http://localhost:16333")
    )
    chroma_path: str = Field("data/embeddings/chroma")


class Config(BaseModel):
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
```

Add to `config.yaml`:

```yaml
vector_store:
  backend: qdrant
  qdrant_url: http://localhost:16333
  chroma_path: data/embeddings/chroma
```

- [ ] **Step 4: Implement a single vector database factory**

Create `src/utils/vector_factory.py`:

```python
from src.utils.config import Config
from src.utils.database import VectorDatabase


def build_vector_database(cfg: Config, embedding_profile_name: str) -> VectorDatabase:
    profile = cfg.embeddings.resolve_profile(embedding_profile_name)
    return VectorDatabase(
        mode="prod" if cfg.vector_store.backend == "qdrant" else "dev",
        qdrant_url=cfg.vector_store.qdrant_url,
        chroma_path=cfg.vector_store.chroma_path,
        embedding_profile_name=embedding_profile_name,
        embedding_profile=profile,
    )
```

Update `VectorDatabase.__init__` to accept `qdrant_url: str` and initialize `QdrantClient(url=qdrant_url)`.

- [ ] **Step 5: Preserve text and stable chunk identity in Qdrant**

Add this helper to `src/utils/database.py`:

```python
from uuid import NAMESPACE_URL, uuid5


def _qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"enterprise-rag:{chunk_id}"))
```

For every Qdrant point, use `_qdrant_point_id(str(doc["id"]))` as `PointStruct.id` and use this payload contract:

```python
payload = {
    "chunk_id": str(doc["id"]),
    "text": str(doc["text"]),
    "metadata": {k: v for k, v in doc.items() if k not in ("id", "text")},
}
```

Map query hits back with:

```python
payload = dict(hit.payload or {})
rows.append(
    {
        "id": str(payload["chunk_id"]),
        "text": str(payload.get("text", "")),
        "metadata": dict(payload.get("metadata") or {}),
        "score": float(hit.score),
        "distance": 1.0 - float(hit.score),
    }
)
```

When a collection already exists, call `get_collection` and compare its vector size to `embedding_profile.dimension`; raise `ValueError` before insertion when they differ. For metadata filters, use Qdrant payload paths such as `metadata.filename` rather than the old flat Chroma key.

- [ ] **Step 6: Apply embedding options and batch encoding**

Change sentence-transformers generation to:

```python
options = dict(self.embedding_profile.options or {})
normalize = bool(options.get("normalize_embeddings", False))
vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=normalize)
return [vector.tolist() for vector in vectors]
```

Keep `_assert_embedding_dimension` on every vector before any upsert.

- [ ] **Step 7: Replace direct `VectorDatabase(mode="dev")` construction**

Use `build_vector_database` in `src/ingest.py` and `src/core/rag_orchestrator.py`. Session collections remain isolated through their unique `sess_{sid}` collection names; remove `chroma_path` as a requirement for deciding whether a session corpus exists.

Update `SessionCorpus` to expose `bm25_index_path` and `collection_name`; keep `chroma_path` only as a deprecated compatibility field until existing session tests are migrated.

- [ ] **Step 8: Run unit and integration tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/unit/test_database_qdrant.py \
  tests/unit/test_rag_orchestrator_session_load.py \
  tests/integration/test_pipeline.py -q
.venv/bin/python -m ruff check src/utils src/core/vector_search.py src/ingest.py src/core/rag_orchestrator.py src/web tests/unit/test_database_qdrant.py
```

Expected: all targeted tests pass with no live Qdrant dependency; fake clients prove the payload contract.

- [ ] **Step 9: Commit the Qdrant path**

```bash
git add src/utils/config.py src/utils/database.py src/utils/vector_factory.py src/core/vector_search.py src/ingest.py src/core/rag_orchestrator.py src/web/session_corpus.py src/web/ingestion_service.py config.yaml tests/unit/test_database_qdrant.py tests/unit/test_rag_orchestrator_session_load.py tests/integration/test_pipeline.py
git commit -m "feat: make Qdrant the primary vector store"
```

### Task 5: Add shared Chinese tokenization for indexing and queries

**Files:**
- Create: `src/core/chinese_tokenizer.py`
- Create: `config/domain_terms.txt`
- Modify: `src/core/bm25_index.py`
- Modify: `src/core/query_processor.py`
- Test: `tests/unit/test_chinese_tokenizer.py`
- Test: `tests/unit/test_bm25_index.py`

**Interfaces:**
- Consumes: raw Chinese or mixed-language text.
- Produces: `tokenize_search_text(text: str) -> list[str]` shared by BM25 indexing and query processing.

- [ ] **Step 1: Write failing tokenizer and retrieval tests**

Create `tests/unit/test_chinese_tokenizer.py`:

```python
from src.core.chinese_tokenizer import tokenize_search_text


def test_tokenize_chinese_and_product_codes():
    tokens = tokenize_search_text("星云网关 ATLAS-X2 出现 E03 怎么重置？")
    assert "星云网关" in tokens
    assert "atlas-x2" in tokens
    assert "e03" in tokens
    assert "怎么" not in tokens
```

Add to `tests/unit/test_bm25_index.py`:

```python
def test_bm25_retrieves_chinese_exact_term():
    index = BM25Index()
    index.add_document("atlas-e03", "ATLAS-X2 出现 E03 时长按复位键 8 秒", {})
    rows = index.score("星云网关 E03 如何复位", top_k=5)
    assert [row["id"] for row in rows] == ["atlas-e03"]
```

- [ ] **Step 2: Verify tests fail with the ASCII-only tokenizer**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_chinese_tokenizer.py tests/unit/test_bm25_index.py::test_bm25_retrieves_chinese_exact_term -q
```

Expected: FAIL because Chinese terms are removed.

- [ ] **Step 3: Implement one tokenizer for documents and queries**

Create `src/core/chinese_tokenizer.py`:

```python
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import jieba

_ASCII_TERM = re.compile(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*")
_PUNCT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_.-]+")
_STOPWORDS = {"的", "了", "和", "是", "在", "怎么", "如何", "请问", "一下"}


@lru_cache(maxsize=1)
def _load_domain_terms() -> tuple[str, ...]:
    path = Path("config/domain_terms.txt")
    if not path.is_file():
        return ()
    terms = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    for term in terms:
        jieba.add_word(term)
    return terms


def tokenize_search_text(text: str) -> list[str]:
    _load_domain_terms()
    normalized = _PUNCT.sub(" ", text).strip().lower()
    ascii_terms = [match.group(0).lower() for match in _ASCII_TERM.finditer(normalized)]
    chinese_terms = [
        token.strip().lower()
        for token in jieba.lcut(normalized, cut_all=False)
        if token.strip() and token.strip() not in _STOPWORDS and not token.isspace()
    ]
    return list(dict.fromkeys(chinese_terms + ascii_terms))
```

Create `config/domain_terms.txt`:

```text
星云网关
极光终端
售后工单
退换货
年假
病假
差旅报销
```

- [ ] **Step 4: Wire the tokenizer into BM25 and QueryProcessor**

Replace `BM25Index._tokenize` with:

```python
@staticmethod
def _tokenize(text: str) -> List[str]:
    return tokenize_search_text(text)
```

Make `QueryProcessor.normalize` preserve Chinese characters and make `_tokenize` call `tokenize_search_text`. English synonym expansion remains enabled only for tokens present in `_SYNONYMS`.

- [ ] **Step 5: Version the persisted BM25 index**

Write `"tokenizer": "jieba-v1"` in `BM25Index.save`. In `load`, reject an explicit non-`jieba-v1` tokenizer with:

```python
tokenizer = data.get("tokenizer")
if tokenizer not in (None, "jieba-v1"):
    raise ValueError(f"Unsupported BM25 tokenizer: {tokenizer}")
```

Legacy indexes without the field may load for regression tests, but the runbook must require re-ingestion.

- [ ] **Step 6: Run tokenizer and BM25 tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_chinese_tokenizer.py tests/unit/test_bm25_index.py tests/unit/test_hybrid_retriever.py -q
.venv/bin/python -m ruff check src/core/chinese_tokenizer.py src/core/bm25_index.py src/core/query_processor.py tests/unit/test_chinese_tokenizer.py
```

Expected: Chinese terms and product codes are retrieved; existing English tests remain green.

- [ ] **Step 7: Commit Chinese sparse retrieval**

```bash
git add src/core/chinese_tokenizer.py src/core/bm25_index.py src/core/query_processor.py config/domain_terms.txt tests/unit/test_chinese_tokenizer.py tests/unit/test_bm25_index.py
git commit -m "feat: add Jieba BM25 tokenization"
```

### Task 6: Preserve page, section, and stable chunk metadata

**Files:**
- Create: `src/core/chunk_schema.py`
- Modify: `src/core/document_processor.py`
- Modify: `src/ingest.py`
- Test: `tests/unit/test_document_processor.py`
- Test: `tests/integration/test_pipeline.py`

**Interfaces:**
- Consumes: PDF, DOCX, Markdown, TXT or HTML path.
- Produces: `ChunkRecord(text: str, metadata: dict[str, object])` with `document_id`, `filename`, nullable `page_number`, nullable `section_title`, `chunk_index`, `content_hash` and stable `chunk_id`.

- [ ] **Step 1: Write failing structured metadata tests**

Add tests for Markdown and PDF page metadata to `tests/unit/test_document_processor.py`:

```python
def test_markdown_chunks_keep_section_and_stable_id(tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("# 员工手册\n## 年假\n员工转正后可申请年假。\n## 报销\n发票须在 30 天内提交。", encoding="utf-8")
    processor = DocumentProcessor(chunk_size=600, overlap=100, tokenizer_name="gpt2", chunk_strategy="zh_structure")
    result = processor.process_document(str(path))
    records = result["chunk_records"]
    annual = next(record for record in records if record["metadata"]["section_title"] == "年假")
    assert annual["metadata"]["filename"] == "handbook.md"
    assert annual["metadata"]["chunk_id"] == annual["id"]
    fresh = DocumentProcessor(chunk_size=600, overlap=100, tokenizer_name="gpt2", chunk_strategy="zh_structure")
    again = fresh.process_document(str(path))
    assert annual["id"] in {record["id"] for record in again["chunk_records"]}
```

- [ ] **Step 2: Verify structured metadata tests fail**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_document_processor.py::test_markdown_chunks_keep_section_and_stable_id -q
```

Expected: FAIL because `zh_structure` and `chunk_records` do not exist.

- [ ] **Step 3: Define the chunk schema and stable identity**

Create `src/core/chunk_schema.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceUnit:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    evidence_anchor: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any]


def stable_chunk_id(document_id: str, section: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    safe_section = "".join(ch.lower() if ch.isalnum() else "-" for ch in section).strip("-") or "body"
    return f"{document_id}__{safe_section}__{chunk_index}__{digest}"
```

Also add `parse_heading_anchor(title: str) -> tuple[str, str | None]` that recognizes a final `{#handbook.leave.annual}` marker, returns the clean Chinese title and stores the marker value as `evidence_anchor`.

- [ ] **Step 4: Add `zh_structure` extraction and chunking**

Add `zh_structure` to allowed strategies. Implement source units as follows:

- PDF: iterate `PdfReader.pages` and emit one `SourceUnit` per page with one-based `page_number`.
- DOCX: update `section_title` when paragraph style starts with `Heading`; emit paragraph units.
- Markdown: update `section_title` on `#` through `######` headings; emit paragraph units.
- TXT/HTML: emit paragraph units with no page or section.

For PDF, DOCX and Markdown headings, run `parse_heading_anchor` and copy the anchor to every child chunk's metadata. Ordinary user documents without `{#anchor}` keep `evidence_anchor=None`.

Within each source unit, split first on paragraphs, then on `。！？；`, then use the existing token window only if a sentence exceeds the 600-character target. Carry the final 100 characters into the next chunk without splitting a product code.

- [ ] **Step 5: Return backward-compatible records**

`process_document` must return both old and new fields:

```python
return {
    "text": text,
    "metadata": document_metadata,
    "chunks": [record.text for record in records],
    "chunk_records": [
        {"id": record.id, "text": record.text, "metadata": record.metadata}
        for record in records
    ],
}
```

- [ ] **Step 6: Ingest the same records into both indexes**

In `src/ingest.py`, remove separately reconstructed IDs. For each `chunk_record`, pass the exact `id`, `text` and `metadata` to BM25 and Qdrant. Prefix the indexed text with `section_title` through `compose_index_text`, but keep the stored retrieval text unchanged.

- [ ] **Step 7: Run document and ingestion tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_document_processor.py tests/integration/test_pipeline.py -q
.venv/bin/python -m ruff check src/core/chunk_schema.py src/core/document_processor.py src/ingest.py tests/unit/test_document_processor.py
```

Expected: stable IDs match in both index paths; PDF pages and Markdown/DOCX sections survive into retrieval metadata.

- [ ] **Step 8: Commit structured chunking**

```bash
git add src/core/chunk_schema.py src/core/document_processor.py src/ingest.py src/utils/config.py config.yaml tests/unit/test_document_processor.py tests/integration/test_pipeline.py
git commit -m "feat: preserve structured document metadata"
```

### Task 7: Expose deterministic retrieval modes and Chinese reranking

**Files:**
- Modify: `src/core/hybrid_retriever.py`
- Modify: `src/core/retrieval_result.py`
- Modify: `src/core/reranker.py`
- Modify: `src/core/rag_orchestrator.py`
- Modify: `src/api/models.py`
- Test: `tests/unit/test_hybrid_retriever.py`
- Test: `tests/unit/test_rag_orchestrator_session_load.py`
- Test: `tests/unit/test_api_main.py`

**Interfaces:**
- Consumes: BM25 hits and Qdrant hits with common chunk IDs.
- Produces: `RetrievalMode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]`, per-leg ranks, RRF score and optional cross-encoder score for evaluation.

- [ ] **Step 1: Write failing retrieval-mode tests**

Add to `tests/unit/test_hybrid_retriever.py`:

```python
def test_bm25_mode_does_not_call_vector():
    bm25 = MagicMock(spec=BM25Search)
    bm25.search.return_value = [{"id": "e03", "text": "长按复位键 8 秒", "metadata": {}, "score": 2.0}]
    vector = MagicMock(spec=VectorSearch)
    retriever = HybridRetriever(bm25, vector, enable_cache=False)
    rows = retriever.retrieve("E03", "E03", k=5, mode="bm25")
    assert rows[0].sources == ["bm25"]
    vector.search.assert_not_called()


def test_vector_mode_does_not_call_bm25():
    bm25 = MagicMock(spec=BM25Search)
    vector = MagicMock(spec=VectorSearch)
    vector.search.return_value = [
        {"id": "e03", "text": "长按复位键 8 秒", "metadata": {}, "distance": 0.1}
    ]
    retriever = HybridRetriever(bm25, vector, enable_cache=False)
    rows = retriever.retrieve("如何复位", "如何复位", k=5, mode="vector")
    assert rows[0].sources == ["vector"]
    bm25.search.assert_not_called()
```

Add a test proving `hybrid` executes both legs and RRF, while `hybrid_rerank` is represented as hybrid retrieval plus `use_rerank=True` in `QueryRequest`.

- [ ] **Step 2: Verify retrieval-mode tests fail**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_hybrid_retriever.py -q
```

Expected: new mode tests fail because `retrieve` has no `mode` parameter.

- [ ] **Step 3: Add exact retrieval mode type and behavior**

In `src/core/hybrid_retriever.py`:

```python
from typing import Literal

RetrievalMode = Literal["bm25", "vector", "hybrid"]
```

Extend `retrieve` with `mode: RetrievalMode = "hybrid"`. For `bm25`, call only `_run_bm25`; for `vector`, call only `_run_vector`; for `hybrid`, keep the existing parallel two-leg RRF. Single-leg results must still be converted to `RetrievalResult` with their native rank and source.

- [ ] **Step 4: Preserve cross-encoder score in the public result**

Add to `RetrievalResult`:

```python
cross_encoder_score: Optional[float] = None
rerank_position: Optional[int] = None
```

When `CrossEncoderReranker.rerank` returns `RankedResult`, copy these values onto the display result before API serialization. Do not overwrite `fusion_score`.

- [ ] **Step 5: Map eval modes into QueryRequest**

Add to `QueryRequest`:

```python
retrieval_mode: str = "hybrid"
```

Validate the value at the orchestrator boundary:

```python
if req.retrieval_mode not in {"bm25", "vector", "hybrid"}:
    raise ValueError(f"Unsupported retrieval_mode: {req.retrieval_mode}")
```

Pass it into `HybridRetriever.retrieve`. The four experiment combinations are:

```python
EXPERIMENTS = {
    "bm25": {"retrieval_mode": "bm25", "use_rerank": False},
    "vector": {"retrieval_mode": "vector", "use_rerank": False},
    "hybrid": {"retrieval_mode": "hybrid", "use_rerank": False},
    "hybrid_rerank": {"retrieval_mode": "hybrid", "use_rerank": True},
}
```

Expose `retrieval_mode` in `QueryRequestModel` only for authenticated debug/evaluation calls; the UI always sends `hybrid` with reranking enabled.

- [ ] **Step 6: Test the Chinese reranker without model download**

Add a unit test that assigns a fake object to `reranker._model` and verifies query/chunk pairs, descending score order, `top_k=5`, and copied scores. The test must assert the configured model name is `BAAI/bge-reranker-v2-m3` but must not download it.

- [ ] **Step 7: Run retrieval and API tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/unit/test_hybrid_retriever.py \
  tests/unit/test_rag_orchestrator_session_load.py \
  tests/unit/test_api_main.py -q
.venv/bin/python -m ruff check src/core/hybrid_retriever.py src/core/retrieval_result.py src/core/reranker.py src/core/rag_orchestrator.py src/api/models.py
```

Expected: all four experiment configurations have deterministic, separately testable behavior.

- [ ] **Step 8: Commit retrieval modes and reranking metadata**

```bash
git add src/core/hybrid_retriever.py src/core/retrieval_result.py src/core/reranker.py src/core/rag_orchestrator.py src/api/models.py tests/unit/test_hybrid_retriever.py tests/unit/test_rag_orchestrator_session_load.py tests/unit/test_api_main.py
git commit -m "feat: add retrieval modes and rerank traces"
```

### Task 8: Add grounded Chinese answers, verified citations, and refusal status

**Files:**
- Create: `src/core/grounding_policy.py`
- Create: `config/prompts/factual.yaml`
- Create: `config/prompts/exploratory.yaml`
- Modify: `src/utils/config.py`
- Modify: `src/core/prompt_manager.py`
- Modify: `src/core/citation_tracker.py`
- Modify: `src/core/citation_verifier.py`
- Modify: `src/core/rag_orchestrator.py`
- Modify: `src/api/models.py`
- Modify: `src/api/main.py`
- Test: `tests/unit/test_grounding_policy.py`
- Test: `tests/unit/test_citation_pipeline.py`
- Test: `tests/integration/test_api_query_contract.py`

**Interfaces:**
- Consumes: retrieved chunks, generated text and verified citations.
- Produces: `AnswerStatus`, `GroundingDecision`, response `status`, `refusal_reason`, enriched `citations`, `evidence` and step latencies.

- [ ] **Step 1: Write failing refusal-policy tests**

Create `tests/unit/test_grounding_policy.py`:

```python
from src.core.grounding_policy import AnswerStatus, GroundingPolicy
from src.core.retrieval_result import RetrievalResult


def _retrieval_result():
    return RetrievalResult(
        id="atlas-e03",
        text="E03 时长按复位键 8 秒",
        metadata={"evidence_anchor": "product.atlas.e03"},
        confidence=0.9,
    )


def test_refuses_when_no_evidence():
    decision = GroundingPolicy().before_generation([])
    assert decision.status is AnswerStatus.REFUSED
    assert decision.reason == "no_relevant_evidence"


def test_refuses_unresolved_citations_after_generation():
    decision = GroundingPolicy().after_generation(
        [_retrieval_result()],
        [{"resolved": False, "verification": "unresolved", "verification_score": 0.0}],
    )
    assert decision.status is AnswerStatus.REFUSED
    assert decision.reason == "unverified_citations"


def test_answers_with_supported_citation():
    decision = GroundingPolicy().after_generation(
        [_retrieval_result()],
        [{"resolved": True, "verification": "supported", "verification_score": 0.8}],
    )
    assert decision.status is AnswerStatus.ANSWERED
```

- [ ] **Step 2: Verify policy tests fail**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_grounding_policy.py -q
```

Expected: FAIL because `grounding_policy.py` does not exist.

- [ ] **Step 3: Implement the explicit grounding policy**

Create `src/core/grounding_policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from src.core.retrieval_result import RetrievalResult


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass(frozen=True)
class GroundingDecision:
    status: AnswerStatus
    reason: str | None = None


class GroundingPolicy:
    def __init__(self, min_retrieval_confidence: float = 0.05, min_citation_score: float = 0.55) -> None:
        self.min_retrieval_confidence = min_retrieval_confidence
        self.min_citation_score = min_citation_score

    def before_generation(self, retrieved: Sequence[RetrievalResult]) -> GroundingDecision:
        if not retrieved or max((row.confidence for row in retrieved), default=0.0) < self.min_retrieval_confidence:
            return GroundingDecision(AnswerStatus.REFUSED, "no_relevant_evidence")
        return GroundingDecision(AnswerStatus.ANSWERED)

    def after_generation(
        self,
        retrieved: Sequence[RetrievalResult],
        citations: Sequence[dict[str, Any]],
    ) -> GroundingDecision:
        initial = self.before_generation(retrieved)
        if initial.status is AnswerStatus.REFUSED:
            return initial
        supported = [
            citation
            for citation in citations
            if citation.get("resolved")
            and citation.get("verification") == "supported"
            and float(citation.get("verification_score", 0.0)) >= self.min_citation_score
        ]
        if not supported:
            return GroundingDecision(AnswerStatus.REFUSED, "unverified_citations")
        return GroundingDecision(AnswerStatus.ANSWERED)
```

- [ ] **Step 4: Create the Chinese evidence-only prompt**

Create `config/prompts/factual.yaml`:

```yaml
template: |
  你是企业知识库问答助手。只能依据下方“证据”回答，禁止使用模型记忆补充制度、参数或承诺。

  证据：
  {context}

  用户问题：{query}

  回答要求：
  1. 每个关键事实后标注对应证据，格式必须是 [Doc chunk_id]。
  2. 不得引用证据列表之外的编号。
  3. 若证据不足，回答“当前知识库中没有足够依据回答该问题”。
  4. 使用简洁、准确的中文，不输出推理过程。

  回答：
```

Create `config/prompts/exploratory.yaml` with the same grounding contract and one additional instruction：需要比较时分别引用每个结论的证据。

- [ ] **Step 5: Enrich citation metadata**

Make `CitationTracker.map_citations` return these fields from the matched document metadata:

```python
{
    "raw_id": raw_id,
    "chunk_id": chunk_id,
    "resolved": resolved,
    "filename": metadata.get("filename"),
    "page_number": metadata.get("page_number"),
    "section_title": metadata.get("section_title"),
    "source": metadata.get("source") or metadata.get("file_type"),
    "evidence_anchor": metadata.get("evidence_anchor"),
    "text_preview": str((doc or {}).get("text", ""))[:240],
}
```

Update `CitationTracker` to attach the Chinese sentence containing each `[Doc id]` marker as `claim_text`. Update `CitationVerifier._tokenize` to call `tokenize_search_text` and score `claim_text` against the cited chunk, falling back to the whole answer only when `claim_text` is absent. This avoids penalizing one valid citation for unrelated claims elsewhere in a multi-source answer.

- [ ] **Step 6: Add status and evidence to core/API response models**

Add to `QueryResponse` and `QueryResponseModel`:

```python
status: str = "answered"
refusal_reason: Optional[str] = None
evidence: List[Dict[str, Any]] = Field(default_factory=list)
```

For the dataclass, use `field(default_factory=list)` instead of Pydantic `Field`.

Add matching fields to `CitationModel`: `filename`, `page_number`, `section_title`, `evidence_anchor`, `claim_text`, `text_preview`.

- [ ] **Step 7: Apply policy before and after generation**

In `RAGOrchestrator.run`:

1. Run `before_generation(display_items)` immediately after retrieval.
2. If refused, return the standard Chinese refusal answer without calling DeepSeek.
3. After generation and citation verification, run `after_generation`.
4. If refused after generation, replace the public answer with the standard refusal answer and preserve retrieved evidence for inspection.
5. Populate `evidence` from the final five retrieved chunks.

For SSE, send `status`, `refusal_reason`, final `answer`, `citations` and `evidence` in the final event. The React client must replace any buffered draft with `final.answer` when the status is refused.

- [ ] **Step 8: Run grounding and API contract tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/unit/test_grounding_policy.py \
  tests/unit/test_citation_pipeline.py \
  tests/integration/test_api_query_contract.py \
  tests/unit/test_api_main.py -q
.venv/bin/python -m ruff check src/core/grounding_policy.py src/core/prompt_manager.py src/core/citation_tracker.py src/core/citation_verifier.py src/core/rag_orchestrator.py src/api
```

Expected: no-evidence queries do not call the provider; unsupported citations produce `status="refused"`; supported citations include resolvable evidence.

- [ ] **Step 9: Commit trusted answering**

```bash
git add src/core/grounding_policy.py config/prompts/factual.yaml config/prompts/exploratory.yaml src/utils/config.py src/core/prompt_manager.py src/core/citation_tracker.py src/core/citation_verifier.py src/core/rag_orchestrator.py src/api/models.py src/api/main.py tests/unit/test_grounding_policy.py tests/unit/test_citation_pipeline.py tests/integration/test_api_query_contract.py tests/unit/test_api_main.py
git commit -m "feat: add grounded answers and safe refusal"
```

### Task 9: Build the Chinese document and evidence user experience

**Files:**
- Modify: `frontend/src/api/generated.ts` via OpenAPI generation
- Modify: `frontend/src/lib/streamQuery.ts`
- Modify: `frontend/src/tabs/QueryTab.tsx`
- Modify: `frontend/src/components/AnswerPanel.tsx`
- Modify: `frontend/src/components/CitationsList.tsx`
- Modify: `frontend/src/components/RetrievedChunks.tsx`
- Modify: `frontend/src/components/Uploader.tsx`
- Modify: `frontend/src/tabs/OverviewTab.tsx`
- Modify: `frontend/src/index.css`
- Test: `frontend/src/lib/streamQuery.test.ts`
- Test: `frontend/src/components/CitationsList.test.tsx`
- Test: `frontend/src/components/AnswerPanel.test.tsx`

**Interfaces:**
- Consumes: API `status`, `refusal_reason`, `citations`, `evidence` and final SSE answer.
- Produces: Chinese upload/query flow, visible answer/refusal state, and clickable evidence details.

- [ ] **Step 1: Write failing final-event and evidence-panel tests**

Add a stream test proving a refused final event replaces draft text:

```typescript
it('uses the server refusal answer instead of streamed draft text', () => {
  const final = {
    type: 'final',
    status: 'refused',
    answer: '当前知识库中没有足够依据回答该问题',
    refusal_reason: 'no_relevant_evidence',
    citations: [],
    evidence: [],
    provider: 'deepseek',
    model: 'deepseek-v4-flash',
  } as const
  expect(testInternals.resolveFinalAnswer('模型正在生成的草稿', final)).toBe(
    '当前知识库中没有足够依据回答该问题',
  )
})
```

Export this helper through `testInternals` and use it from `QueryTab`:

```typescript
function resolveFinalAnswer(draft: string, event: Extract<StreamEvent, { type: 'final' }>) {
  return event.answer?.trim() ? event.answer : draft
}
```

Create `CitationsList.test.tsx` with a citation fixture containing `filename: '员工手册.pdf'`, `page_number: 2`, `section_title: '年假'`, and `text_preview: '工作满 12 个月可享 5 天年假'`. Render the component, click the citation button with `userEvent.click`, and assert all four strings are visible.

- [ ] **Step 2: Verify frontend tests fail**

Run:

```bash
cd frontend
npm test -- src/lib/streamQuery.test.ts src/components/CitationsList.test.tsx
cd ..
```

Expected: FAIL because final status/evidence fields and clickable details are absent.

- [ ] **Step 3: Regenerate TypeScript API types**

Start the FastAPI app with a temporary test configuration, then run:

```bash
cd frontend
npm run gen:api
npm run typecheck
cd ..
```

Expected: generated `QueryResponseModel` includes `status`, `refusal_reason`, `evidence` and enriched citation fields. Commit generated output; do not hand-edit the schema.

- [ ] **Step 4: Make the query flow DeepSeek-only and Chinese-first**

In `QueryTab.tsx`:

- Default to the sole server provider/model; hide provider and model selectors when only one option exists.
- Keep an optional session-only API key input but label it `DeepSeek API Key（仅本次会话使用）` and default persistence to off.
- Use the sample prompts from the 30-case corpus.
- Send `retrieval_mode: 'hybrid'`, `use_rerank: true`, `embedding_profile: 'st_bge_large_zh'`.
- On SSE final, replace `answerRef.current` with `final.answer` and render the final status.

- [ ] **Step 5: Render answer and refusal as different product states**

`AnswerPanel` must render:

```tsx
const refused = response?.status === 'refused'
const stateClass = refused ? 'border-amber-300 bg-amber-50' : 'border-emerald-200 bg-white'
const stateLabel = refused ? '知识库暂无依据' : '已根据知识库回答'
```

Do not display a truthfulness percentage as a guarantee. Keep it under a collapsible technical details block labeled `评测信号`.

- [ ] **Step 6: Implement clickable evidence details**

Each citation button displays `文件名 · 页码 · 章节`. Selecting it opens an inline evidence card with `text_preview`, verification label and score. Use the citation `chunk_id` as the stable React key and never infer filenames from upload names on the client.

- [ ] **Step 7: Localize the core product surface**

Use these product strings consistently:

```text
企业知识库可信问答
上传企业文档
向知识库提问
答案依据
检索证据
当前知识库中没有足够依据回答该问题
```

Keep developer-only metrics and IDs in English where they are code concepts.

- [ ] **Step 8: Run frontend verification**

Run:

```bash
cd frontend
npm test
npm run typecheck
npm run lint
npm run build
cd ..
```

Expected: tests, typecheck, lint and production build pass.

- [ ] **Step 9: Commit the Chinese trusted-answer UI**

```bash
git add frontend/src
git commit -m "feat: add Chinese evidence-first RAG UI"
```

### Task 10: Create versioned enterprise corpus and 30 golden cases

**Files:**
- Create: `evals/corpus/source/employee_handbook.md`
- Create: `evals/corpus/source/product_manual.md`
- Create: `evals/corpus/source/after_sales_faq.md`
- Create: `scripts/build_enterprise_corpus.py`
- Create: `evals/corpus/generated/employee_handbook.pdf`
- Create: `evals/corpus/generated/product_manual.docx`
- Create: `evals/corpus/generated/after_sales_faq.md`
- Create: `evals/corpus/generated/manifest.json`
- Create: `evals/datasets/enterprise_30.jsonl`
- Create: `evals/enterprise_schema.py`
- Modify: `requirements/base.txt`
- Test: `tests/unit/test_enterprise_dataset.py`

**Interfaces:**
- Consumes: three human-readable Markdown sources.
- Produces: deterministic PDF/DOCX/Markdown corpus and exactly 30 validated cases with difficulty, answerability, required facts and evidence anchors.

- [ ] **Step 1: Write failing schema and distribution tests**

Create `tests/unit/test_enterprise_dataset.py`:

```python
from collections import Counter

from evals.enterprise_schema import load_enterprise_cases


def test_enterprise_dataset_has_required_distribution():
    cases = load_enterprise_cases("evals/datasets/enterprise_30.jsonl")
    assert len(cases) == 30
    assert Counter(case.difficulty for case in cases) == {"easy": 10, "medium": 10, "hard": 10}
    assert sum(not case.answerable for case in cases) == 5
    assert len({case.id for case in cases}) == 30
    assert all(case.required_facts or not case.answerable for case in cases)
    assert all(case.gold_evidence or not case.answerable for case in cases)
```

- [ ] **Step 2: Verify the dataset test fails**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_enterprise_dataset.py -q
```

Expected: FAIL because schema and dataset do not exist.

- [ ] **Step 3: Define the validated case schema**

Create `evals/enterprise_schema.py`:

```python
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


def load_enterprise_cases(path: str) -> list[EnterpriseCase]:
    rows = [
        EnterpriseCase.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Enterprise case ids must be unique")
    return rows
```

`required_facts` uses alias groups：每个内层列表满足任意一个短语即算命中，例如 `["8 秒", "八秒"]`。

- [ ] **Step 4: Author the exact enterprise facts**

The three source documents must contain these unique, non-sensitive facts. Every level-two heading ends in its marker, for example `## 年假 {#handbook.leave.annual}`; Task 6 removes the marker from the visible title and stores it as `evidence_anchor`:

| Evidence anchor | Required fact |
| --- | --- |
| `handbook.attendance.late` | 09:10 后打卡计为迟到 |
| `handbook.leave.annual` | 转正且工作满 12 个月可享 5 天年假 |
| `handbook.leave.sick` | 连续病假超过 2 天需二级以上医院证明 |
| `handbook.expense.deadline` | 发票需在费用发生后 30 个自然日内提交 |
| `handbook.expense.hotel` | 一线城市住宿上限 500 元/晚 |
| `handbook.onboarding.security` | 入职 7 天内完成信息安全培训后开通知识库权限 |
| `product.atlas.power` | 星云网关 ATLAS-X2 输入 12V/2A |
| `product.atlas.e03` | E03 时断电 10 秒后重启；仍异常则长按复位键 8 秒 |
| `product.atlas.temperature` | 工作温度 -10℃ 至 45℃ |
| `product.orbit.power` | 极光终端 ORBIT-M1 输入 5V/3A |
| `product.orbit.e17` | E17 表示传感器连接异常，应重新插拔并运行自检 |
| `product.orbit.temperature` | 工作温度 0℃ 至 40℃ |
| `product.warranty.atlas` | ATLAS-X2 整机保修 24 个月 |
| `product.warranty.orbit` | ORBIT-M1 整机保修 12 个月 |
| `faq.return.window` | 未激活且包装完整可在签收后 7 天内申请退货 |
| `faq.exchange.window` | 质量问题可在签收后 15 天内申请换货 |
| `faq.shipping.delay` | 物流超过 48 小时无更新可提交物流核查工单 |
| `faq.invoice.correction` | 电子发票开具后 30 天内可申请一次抬头更正 |
| `faq.repair.materials` | 维修需订单号、序列号和故障视频 |
| `faq.exclusion.consumables` | 线材和包装等耗材不在整机保修范围 |

- [ ] **Step 5: Generate three file formats deterministically**

Add `reportlab>=4.2,<5` to `requirements/base.txt`. `scripts/build_enterprise_corpus.py` must:

1. Read the three UTF-8 Markdown sources.
2. Generate `employee_handbook.pdf` with ReportLab `UnicodeCIDFont("STSong-Light")`, one anchored level-two section per page, and `invariant=1`.
3. Generate `product_manual.docx` with Heading 1/2 styles using `python-docx`.
4. Copy `after_sales_faq.md` byte-for-byte.
5. Re-extract the generated PDF/DOCX/Markdown text, normalize whitespace, and write source plus extracted-text SHA-256 values to `manifest.json`.

Run the script twice and verify the normalized extracted-text hashes remain identical. Binary DOCX hashes are not the determinism criterion because ZIP entry timestamps may differ.

- [ ] **Step 6: Author exactly 30 cases**

Use these fixed IDs and intents in `enterprise_30.jsonl`:

```text
E01 迟到判定时间
E02 年假资格与天数
E03 病假证明条件
E04 报销发票期限
E05 一线城市住宿标准
E06 知识库权限开通条件
E07 ATLAS-X2 电源参数
E08 ATLAS-X2 E03 复位时长
E09 ORBIT-M1 E17 含义
E10 七天退货条件
M01 “星云设备”同义表达查询 ATLAS-X2 工作温度
M02 “极光机”同义表达查询 ORBIT-M1 电源
M03 ATLAS-X2 与 ORBIT-M1 保修期对比
M04 E03 的完整两步处理流程
M05 E17 的处理动作与后续自检
M06 物流两天没有变化如何处理
M07 换货窗口与必要前提
M08 电子发票抬头更正次数和期限
M09 维修工单所需三项材料
M10 报销期限与住宿上限组合查询
H01 比较两款设备电源与工作温度
H02 ATLAS-X2 故障仍存在时结合保修期给出处理建议
H03 ORBIT-M1 传感器异常并需要送修时列出操作与材料
H04 判断未激活商品第 8 天能否无理由退货以及可选售后路径
H05 判断线材损坏是否属于整机保修
H06 查询员工年度奖金金额（不可回答）
H07 查询客户张三的收货地址（不可回答）
H08 查询竞品 NOVA-Z9 的电源参数（不可回答）
H09 查询公司下一轮调薪比例（不可回答）
H10 要求给出劳动仲裁法律结论（不可回答）
```

Each JSONL row must include the exact evidence anchors from Step 4 and Chinese alias groups for every required fact.

- [ ] **Step 7: Build corpus and validate all cases**

Run:

```bash
uv pip install --python .venv/bin/python "reportlab>=4.2,<5"
PYTHONPATH=. .venv/bin/python scripts/build_enterprise_corpus.py
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_enterprise_dataset.py -q
```

Expected: three generated files exist; exactly 30 cases pass schema and distribution validation.

- [ ] **Step 8: Commit corpus and golden cases**

```bash
git add evals/corpus evals/datasets/enterprise_30.jsonl evals/enterprise_schema.py scripts/build_enterprise_corpus.py requirements/base.txt tests/unit/test_enterprise_dataset.py
git commit -m "test: add Chinese enterprise RAG corpus"
```

### Task 11: Implement deterministic enterprise metrics and ablation runs

**Files:**
- Create: `src/evaluation/enterprise_metrics.py`
- Create: `evals/run_enterprise_evals.py`
- Create: `evals/run_ablation.py`
- Modify: `evals/enterprise_schema.py`
- Test: `tests/unit/test_enterprise_metrics.py`
- Test: `tests/unit/test_enterprise_eval_runner.py`

**Interfaces:**
- Consumes: `EnterpriseCase`, `QueryResponse`, retrieval mode and wall-clock measurements.
- Produces: per-case rows and JSON/Markdown summaries containing Hit@5, MRR@5, fact coverage, citation resolution/accuracy, refusal accuracy, P50/P95 and difficulty breakdown.

- [ ] **Step 1: Write failing metric tests**

Create `tests/unit/test_enterprise_metrics.py`:

```python
from src.evaluation.enterprise_metrics import (
    citation_accuracy,
    fact_coverage,
    hit_at_k,
    reciprocal_rank_at_k,
    refusal_correct,
)


def test_retrieval_metrics_use_gold_evidence_anchors():
    ranked = ["product.atlas.power", "product.atlas.e03"]
    assert hit_at_k(ranked, {"product.atlas.e03"}, 5) == 1.0
    assert reciprocal_rank_at_k(ranked, {"product.atlas.e03"}, 5) == 0.5


def test_fact_coverage_accepts_alias_groups():
    groups = [["8 秒", "八秒"], ["复位键", "重置键"]]
    assert fact_coverage("请长按重置键八秒", groups) == 1.0


def test_refusal_and_citation_accuracy():
    assert refusal_correct(answerable=False, status="refused") == 1.0
    assert citation_accuracy(["faq.return.window"], {"faq.return.window"}) == 1.0
```

- [ ] **Step 2: Verify metric tests fail**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_enterprise_metrics.py -q
```

Expected: FAIL because the enterprise metric module does not exist.

- [ ] **Step 3: Implement deterministic metrics**

Create `src/evaluation/enterprise_metrics.py` with these signatures:

```python
def hit_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    return float(any(item in relevant for item in ranked[:k]))


def reciprocal_rank_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def fact_coverage(answer: str, required_facts: list[list[str]]) -> float:
    if not required_facts:
        return 1.0
    normalized = answer.lower().replace(" ", "")
    hits = sum(any(alias.lower().replace(" ", "") in normalized for alias in group) for group in required_facts)
    return hits / len(required_facts)


def citation_accuracy(cited_anchors: list[str], gold: set[str]) -> float:
    if not cited_anchors:
        return 0.0
    return sum(anchor in gold for anchor in cited_anchors) / len(cited_anchors)


def refusal_correct(answerable: bool, status: str) -> float:
    return float((answerable and status == "answered") or (not answerable and status == "refused"))
```

Map retrieved chunks and citations to evidence anchors through metadata key `evidence_anchor`; never parse it from display text.

- [ ] **Step 4: Implement one-config evaluation runner**

`evals/run_enterprise_evals.py` must accept:

```text
--dataset
--provider
--model
--retrieval-mode
--use-rerank / --no-rerank
--embedding-profile
--output
```

For each case, run `RAGOrchestrator.run`, collect the deterministic metrics, record `difficulty`, `category`, `status`, `refusal_reason`, `step_latencies`, `processing_time_ms`, and continue after a single case failure while marking it as `execution_error`.

- [ ] **Step 5: Aggregate global, difficulty and latency summaries**

Use the standard library only:

```python
from statistics import mean


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]
```

Summary keys must be:

```text
hit_at_5
mrr_at_5
fact_coverage
citation_resolution_rate
citation_accuracy
refusal_accuracy
retrieval_latency_p50_ms
retrieval_latency_p95_ms
end_to_end_latency_p50_ms
end_to_end_latency_p95_ms
execution_success_rate
```

- [ ] **Step 6: Implement the four-run ablation controller**

`evals/run_ablation.py` imports the shared evaluator and runs the exact `EXPERIMENTS` mapping from Task 7. It writes one JSON file and one Markdown file containing:

- overall comparison table;
- Easy/Medium/Hard comparison tables;
- per-case failures;
- actual configuration and model names;
- timestamp and Git commit hash.

- [ ] **Step 7: Test runners with a fake orchestrator**

Create `tests/unit/test_enterprise_eval_runner.py` with 30 fake responses. Assert:

- one case error does not abort the report;
- P50/P95 are computed from the correct scopes;
- the four experiment names are present;
- no API key value appears in JSON or Markdown output.

- [ ] **Step 8: Run evaluation unit tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/unit/test_enterprise_metrics.py \
  tests/unit/test_enterprise_eval_runner.py \
  tests/unit/test_evals_runner.py -q
.venv/bin/python -m ruff check src/evaluation/enterprise_metrics.py evals/run_enterprise_evals.py evals/run_ablation.py tests/unit/test_enterprise_metrics.py tests/unit/test_enterprise_eval_runner.py
```

Expected: deterministic metrics and report generation pass without DeepSeek or live Qdrant.

- [ ] **Step 9: Commit enterprise evaluation**

```bash
git add src/evaluation/enterprise_metrics.py evals/run_enterprise_evals.py evals/run_ablation.py evals/enterprise_schema.py tests/unit/test_enterprise_metrics.py tests/unit/test_enterprise_eval_runner.py
git commit -m "feat: add enterprise RAG evaluation harness"
```

### Task 12: Package a conflict-free Docker stack and real smoke test

**Files:**
- Modify: `docker/docker-compose.yml`
- Modify: `docker/.env.example`
- Modify: `Dockerfile`
- Create: `scripts/docker-smoke.sh`
- Modify: `tests/integration/test_api_query_contract.py`

**Interfaces:**
- Consumes: Docker, a user-provided `DEEPSEEK_API_KEY`, generated enterprise corpus.
- Produces: API/React on host 8100, Qdrant on 16333, Redis on 16379, health checks and a two-query smoke result.

- [ ] **Step 1: Write the smoke contract before the script**

The script must verify these exact behaviors:

```text
GET /health returns 200
POST /sessions returns a session_id
POST /sessions/{sid}/documents accepts one PDF, one DOCX and one Markdown file
POST /query answers E08 and returns status=answered with at least one resolved citation
POST /query refuses H06 and returns status=refused
No response contains DEEPSEEK_API_KEY
```

Add an integration test proving the API response JSON contains the same contract with the provider mocked.

- [ ] **Step 2: Make host ports configurable and non-conflicting**

Use this Compose mapping:

```yaml
services:
  api:
    ports:
      - "${APP_PORT:-8100}:8000"
  redis:
    ports:
      - "${REDIS_HOST_PORT:-16379}:6379"
  qdrant:
    ports:
      - "${QDRANT_HOST_PORT:-16333}:6333"
```

Inside Compose set `QDRANT_URL=http://qdrant:6333` and `REDIS_URL=redis://redis:6379/0`. Pass `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` to the API container.

- [ ] **Step 3: Add dependency health checks**

Add Compose health checks for Qdrant `/healthz`, Redis `redis-cli ping`, and API `/health`. Make API depend on healthy Qdrant and Redis.

Replace the Dockerfile's old `cross-encoder/ms-marco-MiniLM-L-6-v2` warmup with `BAAI/bge-reranker-v2-m3`, and warm `BAAI/bge-large-zh-v1.5` through `SentenceTransformer`. Keep the Hugging Face cache volume so rebuilds reuse downloaded weights.

- [ ] **Step 4: Implement the real smoke script**

Create executable `scripts/docker-smoke.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

api_url="${APP_URL:-http://127.0.0.1:8100}"
api_key="${DOC_API_KEY:-dev-key-1}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required for the real smoke test}"

curl -fsS "${api_url}/health" >/dev/null
session_json="$(curl -fsS -X POST "${api_url}/sessions")"
session_id="$(printf '%s' "$session_json" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')"

curl -fsS -X POST "${api_url}/sessions/${session_id}/documents" \
  -H "X-API-Key: ${api_key}" \
  -F "files=@evals/corpus/generated/employee_handbook.pdf" \
  -F "files=@evals/corpus/generated/product_manual.docx" \
  -F "files=@evals/corpus/generated/after_sales_faq.md" \
  -F "chunk_strategy=zh_structure" \
  -F "embedding_profile=st_bge_large_zh" >/tmp/enterprise-rag-upload.json
```

Append two query calls and parse them with `.venv/bin/python`; assert `answered` plus a resolved citation for E08 and `refused` for H06. Never enable shell tracing and never print the key.

- [ ] **Step 5: Run Docker stack and smoke test**

Run:

```bash
cp docker/.env.example docker/.env
# Set DEEPSEEK_API_KEY in docker/.env using a local editor; do not print it.
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
./scripts/docker-smoke.sh
```

Expected: all services healthy; both smoke assertions pass. If the 16 GB machine approaches its Docker memory limit, ask before stopping containers from the first project.

- [ ] **Step 6: Run regression verification**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/integration/test_api_query_contract.py -q
cd frontend && npm run build && cd ..
```

- [ ] **Step 7: Commit deployment and smoke tooling**

```bash
git add docker/docker-compose.yml docker/.env.example Dockerfile scripts/docker-smoke.sh tests/integration/test_api_query_contract.py
git commit -m "chore: package enterprise RAG Docker stack"
```

### Task 13: Run real evaluation, freeze results, and finish project documentation

**Files:**
- Create: `evals/reports/enterprise-final.json` only if reports are intentionally versioned
- Create: `evals/reports/enterprise-final.md` only if reports are intentionally versioned
- Modify: `README.md`
- Create: `Docs/ARCHITECTURE-ZH.md`
- Modify: `Docs/RUNBOOK.md`
- Create: `Docs/EVALUATION-ZH.md`
- Create: `Docs/INTERVIEW-NOTES-ZH.md`
- Modify: `.gitignore` if final report files are committed selectively

**Interfaces:**
- Consumes: healthy Docker stack, 30-case dataset and four experiment configurations.
- Produces: reproducible final metrics, documentation, screenshots checklist and interview-ready project evidence.

- [ ] **Step 1: Ingest the frozen corpus once**

List this project's Qdrant collections and record their point counts. Ingest the frozen three-file corpus into `documents__st_bge_large_zh`; stable chunk IDs make the operation idempotent, so no existing collection or user data is deleted.

Expected: BM25 and Qdrant contain identical chunk IDs; record the chunk count.

- [ ] **Step 2: Run all four real experiment configurations**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m evals.run_ablation \
  --dataset evals/datasets/enterprise_30.jsonl \
  --provider deepseek \
  --model deepseek-v4-flash \
  --embedding-profile st_bge_large_zh \
  --output evals/reports/final
```

Expected: 120 case executions produce one JSON and one Markdown report; failures are recorded per case rather than omitted.

- [ ] **Step 3: Evaluate against acceptance targets without rewriting history**

Check:

```text
Hit@5 >= 0.90
MRR@5 >= 0.75
Citation resolution rate == 1.00
Citation accuracy >= 0.90
Refusal accuracy >= 0.80
Execution success rate == 1.00
```

If a target is missed, inspect per-case evidence and change only one justified parameter at a time. Re-run all four configurations after any change and keep the final report generated from the committed configuration.

- [ ] **Step 4: Write documentation from measured facts**

`README.md` must include:

- the Chinese business problem;
- end-to-end architecture diagram;
- exact quickstart and DeepSeek environment variable names;
- three supported formal file types;
- one answer screenshot and one refusal screenshot;
- final measured results table;
- upstream open-source attribution and this project's changes.

`Docs/EVALUATION-ZH.md` explains every metric, 30-case distribution, four experiments, failures and limitations. `Docs/INTERVIEW-NOTES-ZH.md` explains the full chain in the student's own words and lists likely interviewer follow-ups.

- [ ] **Step 5: Run complete verification before any completion claim**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests evals scripts
cd frontend
npm test
npm run typecheck
npm run lint
npm run build
cd ..
docker compose -f docker/docker-compose.yml ps
./scripts/docker-smoke.sh
git diff --check
git status --short
```

Expected: every automated check passes; Docker services are healthy; only intentional final documentation/report files are uncommitted.

- [ ] **Step 6: Scan for secrets and accidental generated data**

Run:

```bash
rg -n --hidden --glob '!.git/**' --glob '!docker/.env' 'sk-[A-Za-z0-9_-]{12,}|DEEPSEEK_API_KEY=' .
git status --short
```

Expected: only environment-variable examples with blank values appear; no secret values or local indexes are tracked.

- [ ] **Step 7: Commit the measured documentation**

```bash
git add README.md Docs/ARCHITECTURE-ZH.md Docs/RUNBOOK.md Docs/EVALUATION-ZH.md Docs/INTERVIEW-NOTES-ZH.md
git add -f evals/reports/final/enterprise-final.json evals/reports/final/enterprise-final.md
git commit -m "docs: publish enterprise RAG results and runbook"
```

Use the actual generated report filenames if the runner includes timestamps; rename the selected frozen pair to `enterprise-final.json` and `enterprise-final.md` before the commit, leaving raw timestamped reports ignored.

- [ ] **Step 8: Prepare branch handoff**

Run:

```bash
git log --oneline --decorate --max-count=20
git status --short --branch
```

Expected: clean feature branch with task-sized commits, ready to push to the user's GitHub fork and open a PR.

## Final Definition of Done

- A new user can follow the runbook and start the stack without using the first project's ports.
- PDF、DOCX、Markdown all ingest through the visible React flow.
- BM25 and Qdrant share stable chunk IDs and Chinese metadata.
- DeepSeek sync and streaming paths work with no key leakage.
- Answered responses have resolvable, supported citations and visible source text.
- Out-of-scope questions return the standard refusal state.
- The 30-case, four-configuration report is reproducible from committed code and corpus.
- Full Python, frontend, Docker and smoke verification passes.
- README distinguishes the upstream baseline from the implemented Chinese enterprise enhancements.
