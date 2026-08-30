from pathlib import Path

import yaml


def test_api_uses_the_huggingface_hub_as_sentence_transformers_cache():
    compose = yaml.safe_load(Path("docker/docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["api"]["environment"]

    assert "HF_HOME=/app/.cache/huggingface" in environment
    assert "SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/hub" in environment
    assert "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" in environment
    assert "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}" in environment


def test_compose_uses_conflict_free_host_ports_and_internal_service_urls():
    compose = yaml.safe_load(Path("docker/docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["api"]["ports"] == ["${BIND_HOST:-127.0.0.1}:${APP_PORT:-8100}:8000"]
    assert services["redis"]["ports"] == [
        "${BIND_HOST:-127.0.0.1}:${REDIS_HOST_PORT:-16379}:6379"
    ]
    assert services["qdrant"]["ports"] == [
        "${BIND_HOST:-127.0.0.1}:${QDRANT_HOST_PORT:-16333}:6333"
    ]
    assert "REDIS_URL=redis://redis:6379/0" in services["api"]["environment"]
    assert "QDRANT_URL=http://qdrant:6333" in services["api"]["environment"]


def test_env_example_preserves_compose_conflict_free_defaults():
    env_example = Path("docker/.env.example").read_text(encoding="utf-8")

    assert "BIND_HOST=127.0.0.1" in env_example
    assert "APP_PORT=8100" in env_example
    assert "REDIS_HOST_PORT=16379" in env_example
    assert "QDRANT_HOST_PORT=16333" in env_example
    assert "HF_HUB_OFFLINE=1" in env_example
    assert "TRANSFORMERS_OFFLINE=1" in env_example


def test_smoke_scans_session_response_and_retries_health_check():
    smoke = Path("scripts/docker-smoke.sh").read_text(encoding="utf-8")

    assert 'session_path="${smoke_dir}/session.json"' in smoke
    assert "(health_path, session_path, upload_path, answered_path, refused_path)" in smoke
    assert 'health_attempts="${HEALTH_ATTEMPTS:-24}"' in smoke
