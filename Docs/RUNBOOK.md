# 企业知识库 RAG 运行手册

本文给出从空环境启动、入库、验证、评测、停机和排障的完整命令。所有命令都在仓库根目录执行。

## 1. 环境要求

- macOS、Linux 或 WSL
- Docker Desktop / Docker Engine + Compose 插件
- Git
- 约 12 GB 可用磁盘空间
- 建议 Docker 可用内存至少 8 GB；16 GB 内存的电脑不要同时运行多个重型 AI/Java 栈
- DeepSeek API Key（真实生成与评测需要）

默认宿主机端口：

| 组件 | 端口 |
| --- | ---: |
| React + FastAPI | 8100 |
| Qdrant | 16333 |
| Redis | 16379 |

## 2. 首次配置

```bash
cp docker/.env.example docker/.env
```

在本机编辑 `docker/.env`，至少确认：

```dotenv
BIND_HOST=127.0.0.1
APP_PORT=8100
REDIS_HOST_PORT=16379
QDRANT_HOST_PORT=16333
DOC_PROFILE=demo
DOC_API_KEYS=dev-key-1
DOC_API_KEY=dev-key-1
DEEPSEEK_API_KEY=你的真实 Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

注意：

- `docker/.env` 不提交 Git。
- 不要在终端执行 `echo $DEEPSEEK_API_KEY`，也不要把真实 Key 放进截图。
- `DOC_PROFILE=demo` 会绕过查询和上传的网关鉴权；本地演示的安全边界是 `BIND_HOST=127.0.0.1`，不是 `dev-key-1`。
- `DOC_API_KEYS` 只在非 demo profile 生效。不要在保留 demo 绕过时把 `BIND_HOST` 改成 `0.0.0.0` 或局域网地址，否则其他客户端可消耗服务端 DeepSeek Key。

## 3. 构建与启动

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --build
docker compose --env-file docker/.env -f docker/docker-compose.yml ps
```

首次构建会把以下 Hugging Face 模型固化进镜像/缓存：

- `BAAI/bge-large-zh-v1.5`
- `BAAI/bge-reranker-v2-m3`

构建完成后的正常状态是 `api`、`qdrant`、`redis` 均为 `healthy`。检查：

```bash
curl -fsS http://127.0.0.1:8100/health
curl -fsS http://127.0.0.1:16333/healthz
docker compose --env-file docker/.env -f docker/docker-compose.yml exec redis redis-cli ping
```

然后访问 `http://127.0.0.1:8100`。

## 4. 入库冻结企业语料

### 4.1 从宿主机入库

先创建本地 Python 环境：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/base.txt -r requirements/eval.txt
```

因为宿主机配置连接 Qdrant 的 `16333` 端口，可以直接执行：

```bash
PYTHONPATH=. .venv/bin/python -m src.ingest \
  --docs evals/corpus/generated \
  --chunk-strategy zh_structure \
  --embedding-profile st_bge_large_zh
```

### 4.2 从 API 容器入库

也可以在容器内部执行：

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml exec api \
  python -m src.ingest \
  --docs evals/corpus/generated \
  --chunk-strategy zh_structure \
  --embedding-profile st_bge_large_zh
```

期望结果：

- 3 个文件处理成功；
- BM25 保存到 `data/embeddings/bm25_index.json`；
- Qdrant collection 为 `documents__st_bge_large_zh`；
- 冻结语料共 27 个稳定切片；
- 重复入库不会随机增加重复切片。

## 5. 浏览器验收

1. 打开 `http://127.0.0.1:8100`。
2. 创建演示会话。
3. 在“企业文档”上传 PDF、DOCX、Markdown，选择 `zh_structure` 和 `st_bge_large_zh`。
4. 切换到“可信问答”。
5. 提问：`ATLAS-X2 出现 E03 且重启后仍异常，要长按复位键多久？`
6. 期望：状态为已回答，答案包含 `8 秒`，引用指向 `product_manual.docx` 的 `ATLAS-X2 E03 故障处理`。
7. 提问：`公司今年给每位员工发多少年度奖金？`
8. 期望：状态为拒答，答案为 `当前知识库中没有足够依据回答该问题`。

## 6. 自动冒烟测试

冒烟脚本会创建临时会话、上传三种文件，真实调用 DeepSeek 执行一条回答和一条拒答，并断言响应不包含真实 Key。

把本地 env 临时加载到当前 shell，再运行脚本：

```bash
set -a
source docker/.env
set +a
./scripts/docker-smoke.sh
```

期望最后一行：

```text
Enterprise RAG Docker smoke test passed: health, 3-file upload, answered citation, refusal.
```

脚本产生的数据位于临时目录，退出时自动删除；显式删除、TTL 过期或容量淘汰会先删除 Qdrant 中 `sess_<sid>` 及所有 embedding-profile 变体，再删除本地上传和 BM25 状态，且不会删除全局 collection。向量删除失败时本地状态会保留并抛出/记录错误，供后续重试。

## 7. API 调用

### 7.1 创建会话

```bash
curl -fsS -X POST http://127.0.0.1:8100/sessions
```

### 7.2 非流式查询

```bash
curl -fsS -X POST http://127.0.0.1:8100/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-key-1' \
  -d '{
    "query":"ATLAS-X2 出现 E03 且重启后仍异常，要长按复位键多久？",
    "provider":"deepseek",
    "model":"deepseek-v4-flash",
    "knowledge_scope":"global",
    "retrieval_mode":"hybrid",
    "use_rerank":false,
    "embedding_profile":"st_bge_large_zh"
  }'
```

在默认 `DOC_PROFILE=demo` 下，示例中的 `X-API-Key` 不参与鉴权；它仅用于展示切换到非 demo profile 后的调用形态。

关注返回字段：

```text
status / answer / refusal_reason
citations[] / retrieved[]
truthfulness / processing_time_ms
```

### 7.3 SSE 流式查询

把路径改为 `/query/stream`，并在请求体加入 `"stream": true`：

```bash
curl -N -X POST http://127.0.0.1:8100/query/stream \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-key-1' \
  -d '{
    "query":"员工工作满多久可以享受年假？",
    "provider":"deepseek",
    "model":"deepseek-v4-flash",
    "knowledge_scope":"global",
    "retrieval_mode":"hybrid",
    "use_rerank":false,
    "embedding_profile":"st_bge_large_zh",
    "stream":true
  }'
```

## 8. 运行 30 条 × 4 组真实评测

确保 Qdrant 健康、冻结语料已入库，并在当前 shell 设置 `DEEPSEEK_API_KEY`：

```bash
set -a
source docker/.env
set +a

PYTHONPATH=. .venv/bin/python -m evals.run_ablation \
  --dataset evals/datasets/enterprise_30.jsonl \
  --provider deepseek \
  --model deepseek-v4-flash \
  --embedding-profile st_bge_large_zh \
  --output evals/reports/final
```

这会执行 120 次真实请求。不要并行运行第二份评测，也不要在执行中途修改配置。最终冻结一对报告：

```bash
cp evals/reports/final/enterprise_ablation.json evals/reports/final/enterprise-final.json
cp evals/reports/final/enterprise_ablation.md evals/reports/final/enterprise-final.md
```

只有明确要发布的 `enterprise-final.json/md` 应强制加入 Git，其他临时报告保持忽略。

### 8.1 从冻结逐用例记录离线重新聚合

指标代码或分母口径变化、但原始 120 条 `cases` 仍有效时，不要再次调用 DeepSeek。先提交聚合代码并保持工作区干净，再运行：

```bash
PYTHONPATH=. .venv/bin/python -m evals.run_ablation \
  --reaggregate-from evals/reports/final/enterprise-final.json \
  --output evals/reports/final/enterprise-reaggregated.json
```

该分支不加载数据集、benchmark 配置或 `RAGOrchestrator`，只调用当前 `summarize_rows` 重算 `summary` 和 `by_difficulty`。JSON/Markdown 同时记录 `raw_execution_git_commit`、`aggregation_git_commit`、`aggregation_git_dirty`；兼容字段 `git_commit` 表示本次聚合 commit。

## 9. 开发验证

### Python

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check src tests evals scripts
```

### React

```bash
cd frontend
npm install
npm test
npm run typecheck
npm run lint
npm run build
cd ..
```

### 提交前检查

```bash
git diff --check
rg -n --hidden --glob '!.git/**' --glob '!docker/.env' \
  'sk-[A-Za-z0-9_-]{12,}|DEEPSEEK_API_KEY=' .
git status --short
```

允许出现环境变量名、`.env.example` 占位值和测试夹具中的明确假值，不允许出现真实 Key；每个命中都要人工核对来源。

## 10. 停止与恢复

停止容器但保留 Qdrant、Redis 和模型缓存卷：

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml down
```

再次启动：

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
```

不要随意加 `-v`；它会删除本项目命名卷。需要清空数据时，应先确认目标卷并备份。

## 11. 常见问题

### API 一直不健康

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml ps -a
docker compose --env-file docker/.env -f docker/docker-compose.yml logs --tail=200 api
```

常见原因是模型缓存不完整、Qdrant 未就绪、内存不足或环境变量错误。

### Embedding / Reranker 提示 offline 且找不到模型

镜像构建没有成功缓存模型。确认网络和镜像源后重新构建；不要简单关闭 offline 标志然后在每次运行时临时下载。

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml build --no-cache api
```

### DeepSeek 返回鉴权或模型错误

检查 `docker/.env` 中：

```text
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
```

然后只重建 API 容器：

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --force-recreate api
```

不要打印 Key。可用健康接口和一条最小查询判断问题是否仍在。

### 页面能打开但请求发到 8000 端口

生产构建应使用同源 API，即浏览器访问 `8100` 时请求仍发往 `8100`。确认使用了包含 `resolveApiBaseUrlForRuntime` 修复的前端镜像，然后重新构建 `api`。

### Qdrant 维度不匹配

`st_bge_large_zh` 固定为 1024 维，collection 名为 `documents__st_bge_large_zh`。不要把其他 embedding profile 写进该 collection；更换模型应使用新的 profile/collection 并重新入库。

### Docker 内存不足或容器退出 137

退出码 137 且 `OOMKilled=true` 通常是内存不足。先查看：

```bash
docker stats --no-stream
docker inspect enterprise-rag-api-1 --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}}'
```

停止当前不需要的其他重型项目容器，再重新启动本栈；不要在未确认目标的情况下删除卷或容器数据。

### 本可回答的问题被拒答

依次检查：

1. 检索轨迹是否包含黄金证据。
2. chunk ID 是否能解析到本次候选。
3. DeepSeek 是否按要求输出引用。
4. 引用验证分数是否低于 `grounding.min_citation_score`。
5. 召回置信度是否低于 `grounding.min_retrieval_confidence`。

一次只调整一个参数，并用 30 条完整评测验证，避免为了修一个问题破坏整体拒答能力。
