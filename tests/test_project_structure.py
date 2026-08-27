from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_project_files_exist():
    required = [
        "README.md",
        "RESEARCH_PROTOCOL.md",
        "requirements.txt",
        "config/default.yaml",
        "docs/model.md",
        "docs/dataset.md",
        "docs/methodology.md",
        "scripts/predict.py",
        "scripts/evaluate.py",
        "scripts/audit_dataset.py",
        "app/api/main.py",
    ]
    for relative in required:
        assert (ROOT / relative).exists(), f"Missing: {relative}"
