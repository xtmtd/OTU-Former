"""Unit tests for the shared checkpoint architecture resolver."""

import pytest
import torch

from otuformer.training.model import ArcFaceEmbeddingHead
from otuformer.utils.checkpoint import (
    ARCFACE_EMBEDDING_HEAD,
    PROJECTION_EMBEDDING_HEAD,
    normalize_legacy_projector_keys,
    resolve_checkpoint,
    resolve_checkpoint_embedding_head,
    resolve_checkpoint_state_dict,
    resolve_projector_out_dim,
)


def _arcface_weights(embed_dim: int = 192, out_dim: int = 256) -> dict:
    head = ArcFaceEmbeddingHead(embed_dim, out_dim)
    return {
        "projector.net.0.weight": head.net[0].weight.detach(),
        "projector.net.2.weight": head.net[2].weight.detach(),
    }


def _projection_weights(embed_dim: int = 192, out_dim: int = 256) -> dict:
    return {
        "projector.net.0.weight": torch.zeros(2048, embed_dim),
        "projector.net.4.weight": torch.zeros(out_dim, 2048),
    }


def test_head_is_inferred_from_projector_width_when_metadata_is_absent():
    assert resolve_checkpoint_embedding_head({}, _arcface_weights()) == (
        ARCFACE_EMBEDDING_HEAD
    )
    assert resolve_checkpoint_embedding_head({}, _projection_weights()) == (
        PROJECTION_EMBEDDING_HEAD
    )
    # No projector weights at all: historical default is ProjectionHead.
    assert resolve_checkpoint_embedding_head({}, {}) == PROJECTION_EMBEDDING_HEAD


def test_declared_head_wins_even_when_shapes_disagree():
    assert resolve_checkpoint_embedding_head(
        {"config": {"embedding_head": ARCFACE_EMBEDDING_HEAD}}, _projection_weights()
    ) == ARCFACE_EMBEDDING_HEAD


def test_unknown_declared_head_and_unknown_projector_width_are_rejected():
    with pytest.raises(ValueError, match="Unsupported embedding head"):
        resolve_checkpoint_embedding_head(
            {"config": {"embedding_head": "mystery_head"}}, _projection_weights()
        )
    with pytest.raises(ValueError, match="Unrecognised projector width"):
        resolve_checkpoint_embedding_head(
            {}, {"projector.net.0.weight": torch.zeros(768, 192)}
        )


def test_resolve_checkpoint_prefers_student_only_when_requested():
    student = _arcface_weights()
    checkpoint = {
        "teacher": _projection_weights(),
        "student": student,
        "model_state_dict": {},
    }
    assert resolve_checkpoint_state_dict(checkpoint, prefer_student=True)[1] == "student"
    assert resolve_checkpoint_state_dict(checkpoint)[1] == "teacher"


def test_prefer_student_still_falls_back_for_ref_script_sft():
    """``--use-student`` must have no effect on checkpoints that hold no student."""
    ref_script_sft = {"model": {"projector.0.weight": torch.zeros(512, 192)}}

    state_dict, source_key = resolve_checkpoint_state_dict(
        ref_script_sft, prefer_student=True
    )

    assert source_key == "model"
    assert state_dict is ref_script_sft["model"]


def test_apply_checkpoint_weights_rejects_an_incompatible_backbone():
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import (
        CheckpointArchitecture,
        apply_checkpoint_weights,
    )

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    architecture = CheckpointArchitecture(
        model_name="vit_tiny_patch16_224",
        embedding_dim=8,
        embedding_head=PROJECTION_EMBEDDING_HEAD,
        state_dict={"backbone.not_a_real_layer.weight": torch.zeros(2, 2)},
        source_key="model_state_dict",
    )

    with pytest.raises(ValueError, match="pass the correct --model-name"):
        apply_checkpoint_weights(model, architecture)


def test_apply_checkpoint_weights_blames_metadata_when_it_conflicts_with_weights():
    """A recorded model name outranks --model-name, so the hint must say so."""
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import (
        CheckpointArchitecture,
        apply_checkpoint_weights,
    )

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    architecture = CheckpointArchitecture(
        model_name="vit_tiny_patch16_224",
        embedding_dim=8,
        embedding_head=PROJECTION_EMBEDDING_HEAD,
        state_dict={"backbone.not_a_real_layer.weight": torch.zeros(2, 2)},
        source_key="model_state_dict",
        model_name_from_metadata=True,
    )

    with pytest.raises(ValueError, match="precedence over --model-name"):
        apply_checkpoint_weights(model, architecture)


def test_apply_checkpoint_weights_accepts_a_matching_backbone():
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import (
        CheckpointArchitecture,
        apply_checkpoint_weights,
    )

    source = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    architecture = CheckpointArchitecture(
        model_name="vit_tiny_patch16_224",
        embedding_dim=8,
        embedding_head=PROJECTION_EMBEDDING_HEAD,
        state_dict=source.state_dict(),
        source_key="model_state_dict",
    )

    apply_checkpoint_weights(model, architecture)

    assert torch.equal(
        model.backbone.cls_token, source.backbone.cls_token
    )


def test_resolve_checkpoint_uses_legacy_args_when_config_is_absent():
    checkpoint = {
        "teacher": _projection_weights(out_dim=256),
        "args": {"model_name": "vit_small_patch16_224", "out_dim": 256},
    }

    resolved = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")

    assert resolved.model_name == "vit_small_patch16_224"
    assert resolved.embedding_dim == 256
    assert resolved.embedding_head == PROJECTION_EMBEDDING_HEAD
    assert resolved.source_key == "teacher"


def test_resolve_checkpoint_infers_dim_and_head_from_ref_script_arcface_weights():
    head = ArcFaceEmbeddingHead(192, 64)
    checkpoint = {
        "model": {
            "projector.0.weight": head.net[0].weight.detach(),
            "projector.2.weight": head.net[2].weight.detach(),
        }
    }

    resolved = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")

    assert resolved.source_key == "model"
    assert resolved.model_name == "vit_tiny_patch16_224"  # no config, no args
    assert resolved.embedding_dim == 64
    assert resolved.embedding_head == ARCFACE_EMBEDDING_HEAD
    assert set(normalize_legacy_projector_keys(resolved.state_dict)) == {
        "projector.net.0.weight",
        "projector.net.2.weight",
    }


def test_normalize_legacy_projector_keys_leaves_current_names_untouched():
    state = {
        "backbone.cls_token": torch.zeros(1, 1, 1),
        "projector.net.0.weight": torch.zeros(2, 2),
    }
    assert normalize_legacy_projector_keys(state) is state


def test_projector_out_dim_is_none_without_a_projector():
    assert resolve_projector_out_dim({"backbone.cls_token": torch.zeros(1)}) is None
