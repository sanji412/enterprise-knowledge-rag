from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hugging_face_sync_is_optional_without_repository_configuration():
    workflow = (REPO_ROOT / ".github/workflows/sync-to-spaces.yml").read_text()

    assert "Skip Hugging Face sync when configuration is missing" in workflow
    assert "env.HF_TOKEN == ''" in workflow
    assert "env.HF_USERNAME == ''" in workflow
    assert "env.HF_SPACE_NAME == ''" in workflow
    assert "env.HF_TOKEN != ''" in workflow
    assert "env.HF_USERNAME != ''" in workflow
    assert "env.HF_SPACE_NAME != ''" in workflow
