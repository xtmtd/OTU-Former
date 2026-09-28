from pathlib import Path
import tomllib

from otuformer import __version__


def test_package_and_project_versions_match():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "0.10.0"
    assert project["project"]["version"] == __version__


def test_version_documentation_mentions_patch_loss_migration():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "README.cn.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "0.7.0" in text


def test_v080_readmes_state_subcenters_range():
    """Both Sub-center modes share the documented K range."""
    from otuformer.constants import MAX_SUBCENTERS, MIN_SUBCENTERS

    root = Path(__file__).resolve().parents[1]

    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_cn = (root / "README.cn.md").read_text(encoding="utf-8")

    assert f"an integer from {MIN_SUBCENTERS} to {MAX_SUBCENTERS}" in readme
    assert f"{MIN_SUBCENTERS} 到 {MAX_SUBCENTERS} 的整数" in readme_cn


def test_v080_readmes_document_the_metric_loss_modes():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "README.cn.md"):
        text = (root / name).read_text(encoding="utf-8")
        for mode in (
            "arcface",
            "supcon",
            "subcenter-arcface",
            "subcenter-arcface-compact",
        ):
            assert mode in text, f"{name} is missing loss mode {mode}"
        for flag in (
            "--subcenters",
            "--compact-weight",
            "--supcon-temperature",
            "--trace-batch-ids",
        ):
            assert flag in text, f"{name} is missing {flag}"
        for token in ("0.8.0", "loss_diagnostics.finetune.csv", "weight decay"):
            assert token in text, f"{name} is missing {token}"


def test_v100_readmes_document_optional_registers():
    root = Path(__file__).resolve().parents[1]

    for name in ("README.md", "README.cn.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "0.10.0" in text, f"{name} is missing the v0.10.0 note"
        assert "--register-tokens" in text, f"{name} is missing --register-tokens"
        assert "num_prefix_tokens" in text, f"{name} is missing the patch policy"


def test_v080_readmes_state_raw_cls_open_set_and_training_only_centers():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_cn = (root / "README.cn.md").read_text(encoding="utf-8")

    assert "raw CLS" in readme
    assert "open-set" in readme
    assert "training-only" in readme
    assert "zero weight decay on the L2-normalized prototypes" in readme

    assert "原始 CLS" in readme_cn
    assert "开集" in readme_cn
    assert "仅训练期" in readme_cn
    assert "零 weight decay" in readme_cn
