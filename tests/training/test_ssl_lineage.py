"""Fine-tune SSL lineage: classification, source pairs and consumer hand-off."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app

import otuformer.training.finetune_loop as finetune_loop_module
from otuformer.training import trainer
import otuformer.training.finetune_support as finetune_support_module

MODEL = "vit_tiny_patch16_224"
SIZE = 64
runner = CliRunner()
SHA = "0123456789abcdef" * 4


def _ssl_checkpoint(ssl_identity: tuple[str, str] | None, out_dim: int = 8) -> dict:
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name=MODEL, out_dim=out_dim, pretrained=False, img_size=SIZE
    )
    config: dict = {"model_name": MODEL, "out_dim": out_dim}
    if ssl_identity is not None:
        config["ssl_method"] = ssl_identity[0]
        config["ssl_head"] = ssl_identity[1]
    return {
        "model_state_dict": encoder.state_dict(),
        "teacher": encoder.state_dict(),
        "student": encoder.state_dict(),
        "config": config,
        "args": {"global_crop_size": SIZE},
    }


def _metric_checkpoint(
    out_dim: int = 8, source: tuple[str, str] | None = None, sha: str | None = SHA
) -> dict:
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name=MODEL, out_dim=out_dim, pretrained=False, img_size=SIZE
    )
    head = ArcFaceEmbeddingHead(encoder.backbone.num_features, out_dim)
    state = {
        key: value
        for key, value in encoder.state_dict().items()
        if not key.startswith("projector.")
    }
    state.update({f"projector.{k}": v for k, v in head.state_dict().items()})
    config: dict = {
        "model_name": MODEL,
        "out_dim": out_dim,
        "embedding_head": "arcface_mlp_512",
        "loss": "arcface",
    }
    payload: dict = {
        "model_state_dict": state,
        "config": config,
        # A valid v0.8.0 fine-tune payload also carries its loss head, so each
        # test below can break exactly one target field.
        "loss_state_dict": {"head.weight": torch.zeros(2, out_dim)},
    }
    if source is not None:
        config["ssl_source_method"] = source[0]
        config["ssl_source_head"] = source[1]
    if sha is not None:
        config["ssl_initialization_checkpoint_sha256"] = sha
    return payload


def _write_dataset(root: Path, rows: int = 4) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels = []
    for index in range(rows):
        name = f"img_{index}.png"
        Image.new("RGB", (SIZE, SIZE), (index * 40, 10, 10)).save(images / name)
        labels.append({"image": name, "label": "classA" if index < rows // 2 else "classB"})
    frame = pd.DataFrame(labels)
    path = root / "labels.csv"
    frame.to_csv(path, index=False)
    return images, path


def _finetune_args(root: Path, checkpoint: Path, **overrides) -> argparse.Namespace:
    values = dict(
        seed=42,
        cpus=0,
        # CPU: this run uses smaller local crops, whose positional
        # interpolation needs the antialiased bicubic backward that is
        # unimplemented on MPS (device policy: retained CPU exception).
        device="cpu",
        out_dir=str(root / "ft_out"),
        checkpoint=str(checkpoint),
        resume="",
        train_data=str(root / "labels.csv"),
        input_images_dir=str(root / "images"),
        model_name=MODEL,
        metric_embed_dim=8,
        finetune_lr=1e-4,
        loss="arcface",
        freeze_ratio=0.7,
        extract_size=None,
        compute_embedding_metrics=False,
        augmentation="none",
        orientation_policy=None,
        finetune_epochs=1,
        log_every_n_steps=1,
        save_every_epochs=1,
        keep_last_checkpoints=1,
        visualize_data="",
        metrics_sample_size=100,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=5,
        batch_size=2,
        num_workers=0,
        subcenters=None,
        compact_weight=None,
        supcon_temperature=None,
        long_tail="none",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("identity", "kind"),
    [
        (("relational", "projection_mlp_2048"), "ssl"),
        (("dino", "dino_mlp_2048"), "ssl"),
        (None, "ssl"),  # an unmarked backbone-only payload keeps legacy ssl
    ],
)
def test_ssl_identity_matches_finetune_classification(identity, kind):
    checkpoint = _ssl_checkpoint(identity)

    assert finetune_loop_module._classify_finetune_source(checkpoint) == kind


def test_dino_source_is_ssl_and_not_head_only():
    checkpoint = _ssl_checkpoint(("dino", "dino_mlp_2048"))

    assert finetune_loop_module._classify_finetune_source(checkpoint) == "ssl"
    assert finetune_loop_module._select_finetune_embedding_head(checkpoint) == "arcface_mlp_512"
    assert finetune_loop_module._resolve_finetune_ssl_source(checkpoint) == (
        "dino",
        "dino_mlp_2048",
    )


def test_metric_payload_exposes_only_its_source_pair():
    assert finetune_loop_module._resolve_finetune_ssl_source(_metric_checkpoint()) is None
    assert finetune_loop_module._resolve_finetune_ssl_source(
        _metric_checkpoint(source=("dino", "dino_mlp_2048"))
    ) == ("dino", "dino_mlp_2048")


def test_sha_only_metric_resume_does_not_reopen_ssl(tmp_path):
    checkpoint = _metric_checkpoint(source=None, sha=SHA)

    # No pair is readable from a hash alone, and no file is opened for it.
    assert finetune_loop_module._resolve_finetune_ssl_source(checkpoint) is None


def test_legacy_relational_source_is_recorded(tmp_path):
    _write_dataset(tmp_path)
    checkpoint = _ssl_checkpoint(None, out_dim=8)
    path = tmp_path / "ssl.pth"
    torch.save(checkpoint, path)

    finetune_loop_module.run_finetune(_finetune_args(tmp_path, path))

    saved = torch.load(tmp_path / "ft_out" / "finetune_latest.pth", weights_only=False)
    config = saved["config"]
    assert config["ssl_source_method"] == "relational"
    assert config["ssl_source_head"] == "projection_mlp_2048"
    assert "ssl_method" not in config
    assert "ssl_head" not in config
    assert config["embedding_head"] == "arcface_mlp_512"


def test_dino_finetune_installs_fresh_metric_head(tmp_path):
    _write_dataset(tmp_path)
    checkpoint = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    path = tmp_path / "dino_ssl.pth"
    torch.save(checkpoint, path)

    finetune_loop_module.run_finetune(_finetune_args(tmp_path, path))

    saved = torch.load(tmp_path / "ft_out" / "finetune_latest.pth", weights_only=False)
    config = saved["config"]
    assert config["ssl_source_method"] == "dino"
    assert config["ssl_source_head"] == "dino_mlp_2048"
    assert config["embedding_head"] == "arcface_mlp_512"
    # The prototype objective is never part of a fine-tune payload.
    assert "dino_objective" not in saved
    assert not any("prototype" in key for key in saved["model_state_dict"])


def test_finetune_resume_preserves_ssl_lineage(tmp_path):
    _write_dataset(tmp_path)
    checkpoint = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    path = tmp_path / "dino_ssl.pth"
    torch.save(checkpoint, path)
    finetune_loop_module.run_finetune(_finetune_args(tmp_path, path, out_dir=str(tmp_path / "ft")))

    resume_args = _finetune_args(
        tmp_path,
        path,
        out_dir=str(tmp_path / "ft"),
        resume=str(tmp_path / "ft" / "finetune_latest.pth"),
        finetune_epochs=2,
    )
    resume_args.checkpoint = ""
    finetune_loop_module.run_finetune(resume_args)

    saved = torch.load(tmp_path / "ft" / "finetune_latest.pth", weights_only=False)
    assert saved["config"]["ssl_source_method"] == "dino"
    assert saved["config"]["ssl_source_head"] == "dino_mlp_2048"
    assert saved["config"]["ssl_initialization_checkpoint_sha256"]


def test_finetune_preflight_rejects_alias_conflict(tmp_path):
    checkpoint = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    checkpoint["model_state_dict"] = dict(checkpoint["model_state_dict"])
    checkpoint["model_state_dict"]["projector.net.4.weight"] = torch.zeros(9, 2048)

    with pytest.raises(ValueError):
        finetune_loop_module.preflight_finetune_source(checkpoint, MODEL)


def test_resolve_pseudo_ssl_source_contract():
    original = _ssl_checkpoint(("dino", "dino_mlp_2048"))

    # No recorded pair: the original identity (or None) passes through.
    assert finetune_loop_module.resolve_pseudo_ssl_source(original, {}) == (
        "dino",
        "dino_mlp_2048",
    )
    # An unmarked backbone-only payload has no recognizable SSL identity.
    unknown = _ssl_checkpoint(None)
    unknown["model_state_dict"] = {
        key: value
        for key, value in unknown["model_state_dict"].items()
        if not key.startswith("projector.")
    }
    assert finetune_loop_module.resolve_pseudo_ssl_source(unknown, {}) is None

    # A matching pair is accepted; a contradicting or unverifiable one is not.
    recorded = {
        "ssl_source_method": "dino",
        "ssl_source_head": "dino_mlp_2048",
        "ssl_initialization_checkpoint_sha256": SHA,
    }
    assert finetune_loop_module.resolve_pseudo_ssl_source(original, recorded) == (
        "dino",
        "dino_mlp_2048",
    )
    with pytest.raises(ValueError, match="unknown|contradict"):
        finetune_loop_module.resolve_pseudo_ssl_source(unknown, recorded)
    contradicts = dict(recorded, ssl_source_method="relational", ssl_source_head="projection_mlp_2048")
    with pytest.raises(ValueError, match="contradict"):
        finetune_loop_module.resolve_pseudo_ssl_source(original, contradicts)
    unanchored = {
        "ssl_source_method": "dino",
        "ssl_source_head": "dino_mlp_2048",
    }
    with pytest.raises(ValueError, match="SHA|source|identity"):
        finetune_loop_module.resolve_pseudo_ssl_source(original, unanchored)


def test_direct_finetune_lineage_rejection_creates_no_output(tmp_path):
    _write_dataset(tmp_path)
    checkpoint = _metric_checkpoint(
        out_dim=8, source=("dino", "dino_mlp_2048"), sha="not-a-sha"
    )
    path = tmp_path / "bad_lineage.pth"
    torch.save(checkpoint, path)
    out_dir = tmp_path / "rejected"

    with pytest.raises(ValueError):
        finetune_loop_module.run_finetune(_finetune_args(tmp_path, path, out_dir=str(out_dir)))
    assert not out_dir.exists()


def test_pseudo_round_lineage_conflict_fails_before_output(tmp_path):
    _write_dataset(tmp_path)
    original = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    path = tmp_path / "dino_ssl.pth"
    torch.save(original, path)
    source_config = {
        "ssl_source_method": "relational",
        "ssl_source_head": "projection_mlp_2048",
        "ssl_initialization_checkpoint_sha256": SHA,
    }
    out_dir = tmp_path / "pseudo_out"
    args = _finetune_args(tmp_path, path, out_dir=str(out_dir))
    args.pseudo_round_data = {
        "rows": [],
        "source_config": source_config,
        "ssl_checkpoint": str(path),
        "source_checkpoint": str(path),
    }

    with pytest.raises(ValueError, match="contradict|unknown"):
        finetune_loop_module.run_finetune(args, source_checkpoint=original)
    assert not out_dir.exists()


def _checkpoint_with_identity(identity: dict, out_dim: int = 8) -> dict:
    payload = _metric_checkpoint(out_dim=out_dim)
    payload["config"].update(identity)
    return payload


def test_finetune_identity_preflight_preserves_overwrite_sentinel(tmp_path):
    images, labels = _write_dataset(tmp_path)
    checkpoint = _metric_checkpoint(
        out_dim=8, source=("dino", "dino_mlp_2048"), sha=None
    )  # a source pair with no SHA anchor
    path = tmp_path / "bad.pth"
    torch.save(checkpoint, path)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_bytes(b"original")

    result = runner.invoke(
        app,
        [
            "finetune",
            "--checkpoint",
            str(path),
            "--train-data",
            str(labels),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out_dir),
            "--overwrite",
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ],
        env={"COLUMNS": "400"},
    )

    assert result.exit_code != 0
    assert "SSL source identity" in result.output
    assert sentinel.read_bytes() == b"original"
    assert not (out_dir / "logs" / "finetune.log").exists()


def test_finetune_actual_alias_conflict_preserves_overwrite_sentinel(tmp_path):
    images, labels = _write_dataset(tmp_path)
    # Valid DINO SSL teacher/student states, but the actual metric alias lost a
    # projector parameter: identity resolution prefers the teacher, so only the
    # alias check can catch this.
    ssl = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    checkpoint = _metric_checkpoint(
        out_dim=8,
        source=("dino", "dino_mlp_2048"),
        sha=_sha(tmp_path / "placeholder.pth") if False else SHA,
    )
    checkpoint["teacher"] = ssl["teacher"]
    checkpoint["student"] = ssl["student"]
    checkpoint["config"]["ssl_source_method"] = "dino"
    checkpoint["config"]["ssl_source_head"] = "dino_mlp_2048"
    checkpoint["model_state_dict"] = dict(checkpoint["model_state_dict"])
    checkpoint["model_state_dict"].pop("projector.net.2.weight")
    path = tmp_path / "alias.pth"
    torch.save(checkpoint, path)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_bytes(b"original")

    result = runner.invoke(
        app,
        [
            "finetune",
            "--checkpoint",
            str(path),
            "--train-data",
            str(labels),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out_dir),
            "--overwrite",
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ],
        env={"COLUMNS": "400"},
    )

    assert result.exit_code != 0
    # The rejection must name the damaged alias, not some unrelated validation.
    assert "projector.net.2.weight" in result.output
    assert "incomplete" in result.output
    assert "Malformed v0.8.0" not in result.output
    assert sentinel.read_bytes() == b"original"
    assert not (out_dir / "logs" / "finetune.log").exists()


def test_finetune_resume_identity_preflight_preserves_logs(tmp_path):
    images, labels = _write_dataset(tmp_path)
    ssl = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    ssl_path = tmp_path / "dino_ssl.pth"
    torch.save(ssl, ssl_path)
    out_dir = tmp_path / "ft"
    finetune_loop_module.run_finetune(_finetune_args(tmp_path, ssl_path, out_dir=str(out_dir)))

    # Corrupt the resumed payload's own lineage anchor: a pair without a valid
    # SHA is contradictory for every consumer.
    saved = torch.load(out_dir / "finetune_latest.pth", weights_only=False)
    saved["config"]["ssl_source_method"] = "dino"
    saved["config"]["ssl_source_head"] = "dino_mlp_2048"
    saved["config"]["ssl_initialization_checkpoint_sha256"] = "not-a-sha"
    bad = tmp_path / "bad_resume.pth"
    torch.save(saved, bad)

    # The direct run above wrote no CLI log, so seed one to prove a rejected
    # resume leaves existing log bytes untouched and appends nothing.
    log_path = out_dir / "logs" / "finetune.log"
    log_path.write_bytes(b"previous finetune log\n")
    log_bytes = log_path.read_bytes()
    logs_before = set(p.name for p in (out_dir / "logs").iterdir())

    result = runner.invoke(
        app,
        [
            "finetune",
            "--resume",
            str(bad),
            "--train-data",
            str(labels),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert log_path.read_bytes() == log_bytes
    assert set(p.name for p in (out_dir / "logs").iterdir()) == logs_before


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_old_pseudo_source_gains_validated_ssl_lineage(tmp_path):
    """A finetune#1 without a pair still yields a finetune#2 that records one."""
    images, labels = _write_dataset(tmp_path)
    Image.new("RGB", (SIZE, SIZE), (5, 200, 5)).save(images / "extra.png")
    ssl = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    ssl_path = tmp_path / "dino_ssl.pth"
    torch.save(ssl, ssl_path)
    finetune_loop_module.run_finetune(
        _finetune_args(tmp_path, ssl_path, out_dir=str(tmp_path / "ft1"))
    )
    ft1_path = tmp_path / "ft1" / "finetune_latest.pth"
    ft1 = torch.load(ft1_path, weights_only=False)
    source_cfg = dict(ft1["config"])
    # Model a historical finetune#1: no recorded source pair.
    source_cfg.pop("ssl_source_method")
    source_cfg.pop("ssl_source_head")
    source_cfg["pseudo_source_eligible"] = True
    source_cfg["pseudo_round"] = 0

    rows = [
        {"image": "img_0.png", "label": "classA", "source": "expert"},
        {"image": "img_1.png", "label": "classA", "source": "expert"},
        {"image": "img_2.png", "label": "classB", "source": "expert"},
        {"image": "img_3.png", "label": "classB", "source": "expert"},
        {"image": "extra.png", "label": "classB", "source": "known-pseudo"},
    ]
    args = _finetune_args(tmp_path, ssl_path, out_dir=str(tmp_path / "ft2"))
    args.checkpoint = ""
    args.pseudo_round_data = {
        "rows": rows,
        "class_labels": ["classA", "classB"],
        "diagnostics": [{"image": "extra.png", "decision": "accepted"}],
        "summary": {
            "candidate_count": 1,
            "accepted_count": 1,
            "thresholds": {},
        },
        "source_config": source_cfg,
        "ssl_checkpoint": str(ssl_path),
        "source_checkpoint": str(ft1_path),
    }

    finetune_loop_module.run_finetune(args, source_checkpoint=ssl)

    config = torch.load(
        tmp_path / "ft2" / "finetune_latest.pth", weights_only=False
    )["config"]
    assert "ssl_source_method" not in source_cfg  # historical absence
    assert config["ssl_source_method"] == "dino"
    assert config["ssl_source_head"] == "dino_mlp_2048"
    # The three SHA roles stay distinct and correct.
    assert config["initialization_checkpoint_sha256"] == _sha(ssl_path)
    assert config["ssl_initialization_checkpoint_sha256"] == _sha(ssl_path)
    assert config["pseudo_source_checkpoint_sha256"] == _sha(ft1_path)
    assert config["pseudo_round"] == 1


def _pseudo_args(tmp_path, ssl_path, source_cfg, source_ckpt, out_dir):
    args = _finetune_args(tmp_path, ssl_path, out_dir=str(out_dir))
    args.checkpoint = ""
    args.pseudo_round_data = {
        "rows": [
            {"image": "img_0.png", "label": "classA", "source": "expert"},
            {"image": "img_1.png", "label": "classA", "source": "expert"},
            {"image": "img_2.png", "label": "classB", "source": "expert"},
            {"image": "img_3.png", "label": "classB", "source": "expert"},
        ],
        "class_labels": ["classA", "classB"],
        "diagnostics": [],
        "summary": {"candidate_count": 0, "accepted_count": 0, "thresholds": {}},
        "source_config": source_cfg,
        "ssl_checkpoint": str(ssl_path),
        "source_checkpoint": str(source_ckpt),
    }
    return args


def test_pseudo_resume_preserves_lineage_without_reopening_ssl(
    tmp_path, monkeypatch
):
    import shutil as _shutil

    _write_dataset(tmp_path)
    ssl = _ssl_checkpoint(("dino", "dino_mlp_2048"), out_dim=8)
    ssl_path = tmp_path / "dino_ssl.pth"
    torch.save(ssl, ssl_path)
    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path, ssl_path, out_dir=str(tmp_path / "ft1"), finetune_epochs=2
        )
    )
    ft1_path = tmp_path / "ft1" / "finetune_latest.pth"
    source_cfg = dict(torch.load(ft1_path, weights_only=False)["config"])
    source_cfg["pseudo_source_eligible"] = True
    source_cfg["pseudo_round"] = 0

    # Interrupt finetune#2 after its first published epoch.
    class _Interrupted(Exception):
        pass

    real_copy = _shutil.copy
    seen = {"count": 0}

    def copy_spy(source, destination):
        seen["count"] += 1
        result = real_copy(source, destination)
        if str(destination).endswith("finetune_latest.pth") and seen["count"] == 1:
            raise _Interrupted
        return result

    args = _pseudo_args(tmp_path, ssl_path, source_cfg, ft1_path, tmp_path / "ft2")
    args.finetune_epochs = 2
    monkeypatch.setattr(shutil, "copy", copy_spy)
    with pytest.raises(_Interrupted):
        finetune_loop_module.run_finetune(args, source_checkpoint=ssl)
    monkeypatch.setattr(shutil, "copy", real_copy)

    # The original SSL file is gone before the resume: a resume that reopened it
    # would fail here.
    ssl_path.unlink()
    resume_args = _pseudo_args(
        tmp_path, ssl_path, source_cfg, ft1_path, tmp_path / "ft2"
    )
    resume_args.finetune_epochs = 2
    resume_args.resume = str(tmp_path / "ft2" / "finetune_latest.pth")
    # A real resume carries no fresh round payload: accepted rows are rebuilt
    # from the checkpoint and no new pseudo artifacts are written.
    resume_args.pseudo_round_data = None
    writes = []
    real_write = finetune_loop_module._write_pseudo_outputs

    def write_spy(out_dir, diagnostics, summary):
        writes.append(Path(out_dir))
        return real_write(out_dir, diagnostics, summary)

    monkeypatch.setattr(finetune_loop_module, "_write_pseudo_outputs", write_spy)
    finetune_loop_module.run_finetune(resume_args)
    assert writes == []

    config = torch.load(
        tmp_path / "ft2" / "finetune_latest.pth", weights_only=False
    )["config"]
    assert config["ssl_source_method"] == "dino"
    assert config["ssl_source_head"] == "dino_mlp_2048"


@pytest.mark.parametrize("entry", ["cli", "direct"])
@pytest.mark.parametrize("corruption", [
    "actual_alias", "first_width", "missing_parameter", "unknown_identity",
])
def test_pseudo_original_identity_rejected_before_output(
    tmp_path, monkeypatch, entry, corruption
):
    from otuformer.embedding import pseudo_label

    images, labels = _write_dataset(tmp_path)
    Image.new("RGB", (SIZE, SIZE), (5, 200, 5)).save(images / "extra.png")
    ssl = _ssl_checkpoint(("dino", "dino_mlp_2048"))
    ssl_path = tmp_path / "ssl.pth"
    torch.save(ssl, ssl_path)
    finetune_loop_module.run_finetune(_finetune_args(tmp_path, ssl_path))
    ft1_path = tmp_path / "ft_out" / "finetune_latest.pth"
    ft1 = torch.load(ft1_path, weights_only=False)
    if corruption != "unknown_identity":
        ssl["model_state_dict"] = dict(ssl["model_state_dict"])
        if corruption == "actual_alias":
            ssl["model_state_dict"]["projector.net.4.weight"] = torch.zeros(512, 2048)
        elif corruption == "first_width":
            original = ssl["model_state_dict"]["projector.net.0.weight"]
            ssl["model_state_dict"]["projector.net.0.weight"] = torch.zeros(512, original.shape[1])
        else:
            ssl["model_state_dict"].pop("projector.net.2.weight")
    else:
        ssl["config"].pop("ssl_method")
        ssl["config"].pop("ssl_head")
        ssl["model_state_dict"] = {
            k: v for k, v in ssl["model_state_dict"].items()
            if not k.startswith("projector.")
        }
    torch.save(ssl, ssl_path)
    source_cfg = ft1["config"]
    source_cfg["ssl_initialization_checkpoint_sha256"] = _sha(ssl_path)
    ft1["config"] = source_cfg
    torch.save(ft1, ft1_path)
    # Only replace costly neighbour inference; provenance/SHA/CLI checks are real.
    monkeypatch.setattr(pseudo_label, "generate_pseudo_rows", lambda **kwargs: (
        [{"image": "extra.png", "label": "classB"}], [], {"thresholds": {}}
    ))
    out_dir = tmp_path / "ft2"
    message = "projector|width|identity|unknown|SSL"
    if entry == "direct":
        with pytest.raises(ValueError, match=message):
            finetune_loop_module.run_finetune(
                _pseudo_args(tmp_path, ssl_path, source_cfg, ft1_path, out_dir),
                source_checkpoint=ssl,
            )
        assert not out_dir.exists()
    else:
        out_dir.mkdir()
        sentinel = out_dir / "keep.txt"
        sentinel.write_bytes(b"original")
        result = runner.invoke(app, [
            "finetune", "--pseudo-label-from", str(ft1_path),
            "--checkpoint", str(ssl_path), "--train-data", str(labels),
            "--input-images-dir", str(images), "--out-dir", str(out_dir),
            "--overwrite", "--device", "cpu", "--num-workers", "0",
        ], env={"COLUMNS": "400"})
        assert result.exit_code != 0
        assert ("projector" in result.output or "identity" in result.output)
        assert sentinel.read_bytes() == b"original"
        assert not (out_dir / "logs").exists()


@pytest.mark.parametrize("entry", ["cli", "direct"])
def test_pseudo_resume_identity_rejected_before_output(tmp_path, entry):
    images, labels = _write_dataset(tmp_path)
    ssl = _ssl_checkpoint(("dino", "dino_mlp_2048"))
    ssl_path = tmp_path / "ssl.pth"
    torch.save(ssl, ssl_path)
    finetune_loop_module.run_finetune(_finetune_args(tmp_path, ssl_path))
    ft1_path = tmp_path / "ft_out" / "finetune_latest.pth"
    source = torch.load(ft1_path, weights_only=False)
    out_dir = tmp_path / "ft2"
    finetune_loop_module.run_finetune(
        _pseudo_args(tmp_path, ssl_path, source["config"], ft1_path, out_dir),
        source_checkpoint=ssl,
    )
    path = out_dir / "finetune_latest.pth"
    saved = torch.load(path, weights_only=False)
    assert saved["config"]["pseudo_round"] == 1
    saved["config"]["ssl_source_head"] = "projection_mlp_2048"
    torch.save(saved, path)
    logs = out_dir / "logs"
    log = logs / "finetune.log"
    log.write_bytes(b"previous pseudo log\n")
    before = {p.name for p in logs.iterdir()}
    if entry == "direct":
        args = _finetune_args(tmp_path, ssl_path, out_dir=str(out_dir),
                              resume=str(path), finetune_epochs=2)
        with pytest.raises(ValueError, match="SSL|source|pair"):
            finetune_loop_module.run_finetune(args)
    else:
        result = runner.invoke(app, [
            "finetune", "--resume", str(path), "--train-data", str(labels),
            "--input-images-dir", str(images), "--out-dir", str(out_dir),
            "--finetune-epochs", "2", "--device", "cpu", "--num-workers", "0",
        ])
        assert result.exit_code != 0
        assert "SSL" in result.output or "source" in result.output
    assert log.read_bytes() == b"previous pseudo log\n"
    assert {p.name for p in logs.iterdir()} == before


@pytest.mark.parametrize("layout", ["backbone_only", "unmarked_512"])
def test_unknown_ssl_pseudo_source_omits_the_pair(tmp_path, layout):
    _write_dataset(tmp_path)
    original = _ssl_checkpoint(None, out_dim=8)
    if layout == "backbone_only":
        original["model_state_dict"] = {
            key: value
            for key, value in original["model_state_dict"].items()
            if not key.startswith("projector.")
        }
    else:
        original["model_state_dict"] = _metric_checkpoint(
            out_dim=8, source=None, sha=None
        )["model_state_dict"]
        assert original["model_state_dict"]["projector.net.0.weight"].shape[0] == 512
    assert finetune_loop_module._classify_finetune_source(original) == "ssl"
    path = tmp_path / "unknown_ssl.pth"
    torch.save(original, path)
    _write_dataset(tmp_path)
    ft1 = _metric_checkpoint(
        out_dim=8, source=None, sha=_sha(path)
    )
    ft1_path = tmp_path / "ft1.pth"
    torch.save(ft1, ft1_path)
    source_cfg = dict(ft1["config"])
    source_cfg["pseudo_source_eligible"] = True
    source_cfg["pseudo_round"] = 0
    source_cfg["ssl_initialization_checkpoint_sha256"] = _sha(path)

    finetune_loop_module.run_finetune(
        _pseudo_args(
            tmp_path, path, source_cfg, ft1_path, tmp_path / "ft2"
        ),
        source_checkpoint=original,
    )

    config = torch.load(
        tmp_path / "ft2" / "finetune_latest.pth", weights_only=False
    )["config"]
    # Unknown original identity plus no recorded pair stays legacy-eligible and
    # writes no fabricated provenance.
    assert "ssl_source_method" not in config
    assert config["ssl_initialization_checkpoint_sha256"] == _sha(path)
