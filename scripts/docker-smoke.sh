#!/usr/bin/env bash
set -euo pipefail

api_url="${APP_URL:-http://127.0.0.1:8100}"
api_key="${DOC_API_KEY:-dev-key-1}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required for the real smoke test}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}" >&2
  exit 1
fi

smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/enterprise-rag-smoke.XXXXXX")"
trap 'rm -rf "${smoke_dir}"' EXIT

health_path="${smoke_dir}/health.json"
session_path="${smoke_dir}/session.json"
upload_path="${smoke_dir}/upload.json"
answered_path="${smoke_dir}/answered.json"
refused_path="${smoke_dir}/refused.json"

health_attempts="${HEALTH_ATTEMPTS:-24}"
health_ready=0
for ((attempt = 1; attempt <= health_attempts; attempt++)); do
  if curl --connect-timeout 2 --max-time 5 -fsS "${api_url}/health" >"${health_path}"; then
    health_ready=1
    break
  fi
  sleep 5
done
if [[ "${health_ready}" -ne 1 ]]; then
  echo "API did not become healthy after ${health_attempts} attempts: ${api_url}" >&2
  exit 1
fi

session_json="$(curl -fsS -X POST "${api_url}/sessions")"
printf '%s' "${session_json}" >"${session_path}"
session_id="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["session_id"])' "${session_path}")"

curl -fsS -X POST "${api_url}/sessions/${session_id}/documents" \
  -H "X-API-Key: ${api_key}" \
  -F "files=@evals/corpus/generated/employee_handbook.pdf" \
  -F "files=@evals/corpus/generated/product_manual.docx" \
  -F "files=@evals/corpus/generated/after_sales_faq.md" \
  -F "chunk_strategy=zh_structure" \
  -F "embedding_profile=st_bge_large_zh" >"${upload_path}"

run_query() {
  local question="$1"
  local output_path="$2"
  QUERY_TEXT="${question}" SESSION_ID="${session_id}" "${python_bin}" -c '
import json
import os

print(json.dumps({
    "query": os.environ["QUERY_TEXT"],
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "session_id": os.environ["SESSION_ID"],
    "knowledge_scope": "session",
    "retrieval_mode": "hybrid",
    "use_rerank": True,
    "embedding_profile": "st_bge_large_zh",
}, ensure_ascii=False))
' | curl -fsS -X POST "${api_url}/query" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${api_key}" \
    --data-binary @- >"${output_path}"
}

run_query "ATLAS-X2 出现 E03 且重启后仍异常，要长按复位键多久？" "${answered_path}"
run_query "公司今年给每位员工发多少年度奖金？" "${refused_path}"

"${python_bin}" - "${health_path}" "${session_path}" "${upload_path}" "${answered_path}" "${refused_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

health_path, session_path, upload_path, answered_path, refused_path = map(Path, sys.argv[1:])
response_paths = (health_path, session_path, upload_path, answered_path, refused_path)
raw_payloads = [path.read_text(encoding="utf-8") for path in response_paths]
secret = os.environ["DEEPSEEK_API_KEY"]
assert all(secret not in raw for raw in raw_payloads), "API key leaked into a response"

health = json.loads(health_path.read_text(encoding="utf-8"))
session = json.loads(session_path.read_text(encoding="utf-8"))
upload = json.loads(upload_path.read_text(encoding="utf-8"))
answered = json.loads(answered_path.read_text(encoding="utf-8"))
refused = json.loads(refused_path.read_text(encoding="utf-8"))

assert health["status"] == "ok"
assert session["session_id"]
assert len(upload["results"]) == 3
assert all(row["status"] in {"queued", "skipped"} for row in upload["results"])
assert answered["status"] == "answered"
assert any(citation.get("resolved") for citation in answered.get("citations", []))
assert refused["status"] == "refused"
PY

echo "Enterprise RAG Docker smoke test passed: health, 3-file upload, answered citation, refusal."
