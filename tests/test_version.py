from pathlib import Path
import tomllib

from otuformer import __version__


def test_package_and_project_versions_match():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "0.7.0"
    assert project["project"]["version"] == __version__


def test_version_documentation_mentions_patch_loss_migration():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "README.cn.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "0.7.0" in text
