from pathlib import Path
import tomllib

from otuformer import __version__


def test_package_and_project_versions_match():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "0.6.0"
    assert project["project"]["version"] == __version__
