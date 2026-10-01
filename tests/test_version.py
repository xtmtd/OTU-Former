from pathlib import Path
import tomllib

from otuformer import __version__

from test_docs import cn_docs, en_docs


def _corpus(pairs) -> str:
    return "\n".join(text for _, text in pairs)


def test_package_and_project_versions_match():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "0.11.0"
    assert project["project"]["version"] == __version__


def test_version_documentation_mentions_patch_loss_migration():
    """The patch-objective migration note lives in docs/commands/pretrain.md."""
    assert "0.7.0" in _corpus(en_docs())
    assert "0.7.0" in _corpus(cn_docs())


def test_v080_readmes_state_subcenters_range():
    """Both Sub-center modes share the documented K range."""
    from otuformer.constants import MAX_SUBCENTERS, MIN_SUBCENTERS

    assert f"an integer from {MIN_SUBCENTERS} to {MAX_SUBCENTERS}" in _corpus(en_docs())
    assert f"{MIN_SUBCENTERS} 到 {MAX_SUBCENTERS} 的整数" in _corpus(cn_docs())


def test_v080_readmes_document_the_metric_loss_modes():
    for text in (_corpus(en_docs()), _corpus(cn_docs())):
        for mode in (
            "arcface",
            "supcon",
            "subcenter-arcface",
            "subcenter-arcface-compact",
        ):
            assert mode in text, f"is missing loss mode {mode}"
        for flag in (
            "--subcenters",
            "--compact-weight",
            "--supcon-temperature",
            "--trace-batch-ids",
        ):
            assert flag in text, f"is missing {flag}"
        for token in ("0.8.0", "loss_diagnostics.finetune.csv", "weight decay"):
            assert token in text, f"is missing {token}"


def test_v100_readmes_document_optional_registers():
    for text in (_corpus(en_docs()), _corpus(cn_docs())):
        assert "0.10.0" in text, "is missing the v0.10.0 note"
        assert "--register-tokens" in text, "is missing --register-tokens"
        assert "num_prefix_tokens" in text, "is missing the patch policy"


def test_v080_readmes_state_raw_cls_open_set_and_training_only_centers():
    en = _corpus(en_docs())
    cn = _corpus(cn_docs())

    assert "raw CLS" in en
    assert "open-set" in en
    assert "training-only" in en
    assert "zero weight decay on the L2-normalized prototypes" in en

    assert "原始 CLS" in cn
    assert "开集" in cn
    assert "仅训练期" in cn
    assert "零 weight decay" in cn
