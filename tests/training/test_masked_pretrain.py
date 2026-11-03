"""Focused tests for v0.7.0 masked pretraining.

Covers effective patch configuration, mask generation, the masked training
forward, patch objectives, named EMA, resume/RNG handling, and diagnostics.
"""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn

from otuformer.training import trainer
from otuformer.training.model import OTUFormerEncoder, PatchObjective

# Snapshot the real implementations once so repeated helper calls never chain
# their spies through a previously installed monkeypatch.
_REAL_FORWARD_PRETRAIN = OTUFormerEncoder.forward_pretrain
_REAL_PATCH_OBJECTIVE = PatchObjective
_REAL_UPDATE_TEACHER = trainer.update_teacher
_REAL_SAMPLE_PATCH_MASKS = trainer._sample_patch_masks
_REAL_UPDATE_PATCH_CENTER = trainer._update_patch_center
_REAL_IBOT_PATCH_LOSS = trainer.ibot_patch_loss


def _patch_args(**overrides):
    base = {
        "patch_loss": "consistency",
        "masking_strategy": "random",
        "mask_ratio": "auto",
        "ibot_prototypes": 512,
        "lambda_mask": 1.0,
        "patch_loss_explicit": False,
        "masking_strategy_explicit": False,
        "mask_ratio_explicit": False,
        "ibot_prototypes_explicit": False,
        "lambda_mask_explicit": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    ("mode", "strategy", "expected_requested"),
    [
        ("none", "random", None),
        ("consistency", "random", "auto"),
        ("masked-feature", "random", "auto"),
        ("masked-feature", "blockwise", "auto"),
        ("masked-feature", "hybrid", "auto"),
        ("ibot", "random", "auto"),
        ("ibot", "blockwise", "auto"),
        ("ibot", "hybrid", "auto"),
    ],
)
def test_patch_config_resolves_new_run_defaults(
    mode, strategy, expected_requested
):
    config = trainer._resolve_patch_config(
        _patch_args(patch_loss=mode, masking_strategy=strategy), None
    )

    assert config["patch_loss"] == mode
    assert config["mask_ratio_requested"] == expected_requested
    expected = None if mode == "none" else {"blockwise": 0.20}.get(strategy, 0.30)
    assert config["mask_ratio_resolved"] == expected
    if mode in {"masked-feature", "ibot"}:
        assert config["masking_strategy"] == strategy
    else:
        assert config["masking_strategy"] is None
    if mode == "masked-feature":
        assert config["patch_target_layers"] == 4
        assert config["patch_center_momentum"] is None
        assert config["ibot_prototypes"] is None
    elif mode == "ibot":
        assert config["patch_target_layers"] is None
        assert config["patch_center_momentum"] == 0.9
        assert config["ibot_prototypes"] == 512
    else:
        assert config["patch_target_layers"] is None
        assert config["patch_center_momentum"] is None
        assert config["ibot_prototypes"] is None


def test_patch_config_rejects_explicit_ratio_out_of_range():
    for value in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="mask_ratio"):
            trainer._resolve_patch_config(
                _patch_args(mask_ratio=value, mask_ratio_explicit=True), None
            )


def test_patch_config_honours_explicit_ratio():
    config = trainer._resolve_patch_config(
        _patch_args(
            patch_loss="consistency", mask_ratio=0.5, mask_ratio_explicit=True
        ),
        None,
    )

    assert config["mask_ratio_requested"] == 0.5
    assert config["mask_ratio_resolved"] == 0.5


def test_patch_config_rejects_explicit_inapplicable_options():
    with pytest.raises(ValueError, match="mask_ratio"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="none", mask_ratio=0.3, mask_ratio_explicit=True),
            None,
        )
    with pytest.raises(ValueError, match="masking_strategy"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="none", masking_strategy_explicit=True), None
        )
    with pytest.raises(ValueError, match="ibot_prototypes"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="none", ibot_prototypes_explicit=True), None
        )
    with pytest.raises(ValueError, match="lambda_mask"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="none", lambda_mask=0.5, lambda_mask_explicit=True),
            None,
        )
    with pytest.raises(ValueError, match="masking_strategy"):
        trainer._resolve_patch_config(
            _patch_args(
                patch_loss="consistency",
                masking_strategy="blockwise",
                masking_strategy_explicit=True,
            ),
            None,
        )
    with pytest.raises(ValueError, match="ibot_prototypes"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="consistency", ibot_prototypes_explicit=True), None
        )
    with pytest.raises(ValueError, match="ibot_prototypes"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="masked-feature", ibot_prototypes_explicit=True),
            None,
        )


def test_patch_config_rejects_unknown_modes_and_strategies():
    with pytest.raises(ValueError, match="patch_loss"):
        trainer._resolve_patch_config(_patch_args(patch_loss="mystery"), None)
    with pytest.raises(ValueError, match="masking_strategy"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="ibot", masking_strategy="mystery"), None
        )


@pytest.mark.parametrize("prototypes", [2, 3, 256, 999, 4096, 65536])
def test_patch_config_accepts_a_range_of_ibot_prototype_counts(prototypes):
    config = trainer._resolve_patch_config(
        _patch_args(patch_loss="ibot", ibot_prototypes=prototypes), None
    )

    assert config["ibot_prototypes"] == prototypes


@pytest.mark.parametrize("prototypes", [0, 1, -5])
def test_patch_config_rejects_non_positive_ibot_prototype_counts(prototypes):
    with pytest.raises(ValueError, match="ibot_prototypes"):
        trainer._resolve_patch_config(
            _patch_args(patch_loss="ibot", ibot_prototypes=prototypes), None
        )


# --- Task 2: mask generation -------------------------------------------------


@pytest.mark.parametrize("strategy", ["random", "blockwise", "hybrid"])
@pytest.mark.parametrize("grid_size", [(8, 8), (8, 14)])
@pytest.mark.parametrize("ratio", [0.15, 0.30, 0.75])
def test_mask_generation_exact_count_and_determinism(
    strategy, grid_size, ratio
):
    patch_count = grid_size[0] * grid_size[1]
    expected = trainer._target_patch_count(ratio, patch_count)

    torch.manual_seed(0)
    first = trainer._sample_patch_masks(
        3, grid_size, ratio, strategy, torch.device("cpu")
    )
    torch.manual_seed(0)
    second = trainer._sample_patch_masks(
        3, grid_size, ratio, strategy, torch.device("cpu")
    )

    assert first.shape == (3, patch_count)
    assert first.dtype == torch.bool
    assert torch.equal(first, second)
    assert torch.all(first.sum(dim=1) == expected)


def test_mask_covers_whole_grid_and_leaves_a_visible_patch():
    for ratio in (0.0001, 0.5, 0.9999):
        mask = trainer._sample_patch_masks(
            2, (8, 8), ratio, "random", torch.device("cpu")
        )
        counts = mask.sum(dim=1)
        assert torch.all(counts >= 1)
        assert torch.all(counts <= 63)


@pytest.mark.parametrize("strategy", ["random", "blockwise", "hybrid"])
def test_mask_views_are_independently_sampled(strategy):
    torch.manual_seed(1)
    view_a = trainer._sample_patch_masks(
        2, (8, 8), 0.3, strategy, torch.device("cpu")
    )
    view_b = trainer._sample_patch_masks(
        2, (8, 8), 0.3, strategy, torch.device("cpu")
    )
    assert view_a is not view_b
    assert not torch.equal(view_a, view_b)
    # Different batch rows are independently sampled too.
    assert not torch.equal(view_a[0], view_a[1])


def test_blockwise_geometry_limits_are_respected():
    shapes = trainer._legal_block_shapes((8, 8))
    assert shapes
    max_area = (20 * 64) // 100
    for height, width in shapes:
        assert height >= 2 and width >= 2
        assert 0.5 <= height / width <= 2.0
        assert height <= 8 and width <= 8
        assert height * width <= max_area


def test_blockwise_rejects_four_by_four_grid_area_limit():
    with pytest.raises(ValueError, match="blockwise"):
        trainer._validate_blockwise_grid((4, 4))


def test_blockwise_rejects_narrow_grid_without_a_two_patch_side():
    with pytest.raises(ValueError, match="blockwise"):
        trainer._validate_blockwise_grid((1, 32))


def test_blockwise_random_fill_happens_only_after_a_legal_rectangle(monkeypatch):
    calls: list[str] = []

    def tiny_rectangle(shapes, grid_size, device):
        calls.append("rectangle")
        mask = torch.zeros(
            grid_size[0] * grid_size[1], dtype=torch.bool, device=device
        )
        mask[:4] = True
        return mask

    real_fill = trainer._random_positions_mask

    def traced_fill(*args, **kwargs):
        calls.append("fill")
        return real_fill(*args, **kwargs)

    monkeypatch.setattr(trainer, "_sample_rectangle", tiny_rectangle)
    monkeypatch.setattr(trainer, "_random_positions_mask", traced_fill)

    mask = trainer._sample_patch_masks(
        1, (8, 8), 0.5, "blockwise", torch.device("cpu")
    )

    assert int(mask.sum()) == trainer._target_patch_count(0.5, 64)
    assert calls[0] == "rectangle"
    assert "fill" in calls


def test_hybrid_starts_from_blockwise_positions(monkeypatch):
    calls: list[str] = []

    def tracked_rectangle(shapes, grid_size, device):
        calls.append("rectangle")
        mask = torch.zeros(
            grid_size[0] * grid_size[1], dtype=torch.bool, device=device
        )
        mask[8:12] = True
        return mask

    monkeypatch.setattr(trainer, "_sample_rectangle", tracked_rectangle)

    mask = trainer._sample_patch_masks(
        1, (8, 8), 0.5, "hybrid", torch.device("cpu")
    )

    assert calls and calls[0] == "rectangle"
    assert int(mask.sum()) == trainer._target_patch_count(0.5, 64)
    # The blockwise share is exactly floor(target / 2), so part of the seeded
    # rectangle survives into the final mask.
    assert bool(mask[0, 8:12].any())


def test_sample_patch_masks_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="masking_strategy"):
        trainer._sample_patch_masks(
            1, (8, 8), 0.3, "mystery", torch.device("cpu")
        )


# --- Task 3: single-pass masked training forward -----------------------------


class _CountingPatchEmbed(nn.Module):
    def __init__(self, patch_size, dim):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.proj(x).flatten(2).transpose(1, 2)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return x + torch.tanh(self.fc(self.norm(x)))


class _TinyViT(nn.Module):
    """Minimal timm-VisionTransformer-shaped double for deterministic tests."""

    def __init__(
        self, img_size=32, patch_size=8, dim=16, depth=4, prefix_tokens=1
    ):
        super().__init__()
        grid = img_size // patch_size
        self.patch_embed = _CountingPatchEmbed(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.reg_token = nn.Parameter(torch.randn(1, prefix_tokens - 1, dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, prefix_tokens + grid * grid, dim)
        )
        self.blocks = nn.ModuleList([_TinyBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.norm_pre = nn.Identity()
        self.patch_drop = nn.Identity()
        self.num_prefix_tokens = prefix_tokens
        self.num_features = dim

    def _build_prefix(self, batch_size):
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        if self.reg_token.shape[1] > 0:
            tokens.append(self.reg_token.expand(batch_size, -1, -1))
        return torch.cat(tokens, dim=1)

    def _pos_embed(self, x):
        prefix = self._build_prefix(x.shape[0])
        return torch.cat([prefix, x], dim=1) + self.pos_embed

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def _tiny_encoder(monkeypatch, **kwargs):
    import otuformer.training.model as model_module

    double = _TinyViT(**kwargs)
    monkeypatch.setattr(model_module.timm, "create_model", lambda *a, **k: double)
    encoder = model_module.OTUFormerEncoder(
        model_name="tiny-vit", out_dim=8, return_patch_tokens=True, pretrained=False
    )
    return encoder, double


def test_pretrain_forward_returns_grid_and_final_four_from_one_execution(monkeypatch):
    encoder, backbone = _tiny_encoder(monkeypatch)
    encoder.validate_pretrain_backbone(require_intermediates=True)
    x = torch.randn(2, 3, 32, 32)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :3] = True
    mask_token = torch.randn(1, 1, 16)

    before = backbone.patch_embed.calls
    out = encoder.forward_pretrain(
        x, mask=mask, mask_token=mask_token, return_intermediates=True
    )

    assert backbone.patch_embed.calls - before == 1
    assert out.grid_size == (4, 4)
    assert len(out.final_four) == 4
    assert out.tokens.shape == (2, 1 + 16, 16)
    assert out.tokens.shape == out.final_four[-1].shape
    assert out.cls.shape == (2, 8)


def test_masked_forward_replaces_only_selected_patch_embeddings(monkeypatch):
    encoder, backbone = _tiny_encoder(monkeypatch)
    encoder.validate_pretrain_backbone()
    x = torch.randn(2, 3, 32, 32)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, [0, 5, 9]] = True
    mask_token = torch.full((1, 1, 16), 0.25)

    with torch.no_grad():
        raw_patches = backbone.patch_embed(x).clone()

    captured = {}
    real_pos_embed = backbone._pos_embed

    def spy(tokens):
        captured["tokens"] = tokens.detach().clone()
        return real_pos_embed(tokens)

    monkeypatch.setattr(backbone, "_pos_embed", spy)
    encoder.forward_pretrain(x, mask=mask, mask_token=mask_token)

    tokens = captured["tokens"]
    # Only patch tokens reach _pos_embed, so prefix tokens can never be masked.
    assert tokens.shape == (2, 16, 16)
    for row in range(2):
        for col in range(16):
            if mask[row, col]:
                assert torch.allclose(tokens[row, col], mask_token[0, 0])
                # Original content at the selected position is gone.
                assert not torch.allclose(tokens[row, col], raw_patches[row, col])
            else:
                assert torch.allclose(tokens[row, col], raw_patches[row, col])


def test_masked_positions_keep_their_own_positional_encoding(monkeypatch):
    encoder, backbone = _tiny_encoder(monkeypatch)
    encoder.validate_pretrain_backbone()
    x = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[0, 7] = True
    mask_token = torch.randn(1, 1, 16)

    captured = {}
    real_pos_embed = backbone._pos_embed

    def spy(tokens):
        embedded = real_pos_embed(tokens)
        captured["out"] = embedded.detach().clone()
        return embedded

    monkeypatch.setattr(backbone, "_pos_embed", spy)
    encoder.forward_pretrain(x, mask=mask, mask_token=mask_token)

    prefix = backbone.num_prefix_tokens
    expected = mask_token[0, 0] + backbone.pos_embed[0, prefix + 7]
    assert torch.allclose(captured["out"][0, prefix + 7], expected, atol=1e-6)
    # Prefix tokens are never masked.
    assert torch.allclose(
        captured["out"][0, :prefix],
        (backbone.cls_token + backbone.pos_embed[0, :prefix]),
        atol=1e-6,
    )


def test_flattened_mask_indices_map_to_row_and_column(monkeypatch):
    encoder, backbone = _tiny_encoder(monkeypatch)
    encoder.validate_pretrain_backbone()
    x = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[0, 2 * 4 + 3] = True  # row 2, column 3 on a 4 x 4 grid
    mask_token = torch.fill(torch.zeros(1, 1, 16), 99.0)

    captured = {}
    real_pos_embed = backbone._pos_embed

    def spy(tokens):
        captured["tokens"] = tokens.detach().clone()
        return real_pos_embed(tokens)

    monkeypatch.setattr(backbone, "_pos_embed", spy)
    encoder.forward_pretrain(x, mask=mask, mask_token=mask_token)

    assert torch.allclose(captured["tokens"][0, 2 * 4 + 3], mask_token[0, 0])
    assert not torch.allclose(captured["tokens"][0, 3 * 4 + 2], mask_token[0, 0])


def test_validate_pretrain_backbone_names_the_missing_capability(monkeypatch):
    import types

    import otuformer.training.model as model_module

    broken = types.SimpleNamespace(num_features=16)
    monkeypatch.setattr(model_module.timm, "create_model", lambda *a, **k: broken)
    encoder = model_module.OTUFormerEncoder(
        model_name="broken-backbone", out_dim=8, pretrained=False
    )

    with pytest.raises(ValueError) as excinfo:
        encoder.validate_pretrain_backbone()

    message = str(excinfo.value)
    assert "broken-backbone" in message
    assert "patch_embed" in message


def test_unmasked_pretrain_forward_matches_backbone_forward_features():
    """The training forward must reproduce the normal ViT sequence exactly."""
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    encoder.validate_pretrain_backbone(require_intermediates=True)
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        tokens = encoder.forward_pretrain(x).tokens
        reference = encoder.backbone.forward_features(x)

    assert tokens.shape == reference.shape
    assert torch.allclose(tokens, reference, atol=1e-6)


# --- v0.10.0 optional register tokens -----------------------------------------


def _real_tiny_encoder(monkeypatch, native_regs=0, class_token=True,
                       no_embed_class=False, reg_tokens=None):
    """Build an encoder over a real tiny timm VisionTransformer.

    ``native_regs`` is the count the stand-in pretrained model natively has; a
    ``reg_tokens`` kwarg on the create_model call selects the requested count,
    exactly as timm would receive it.
    """
    import otuformer.training.model as model_module
    from timm.models.vision_transformer import VisionTransformer

    def factory(_model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        count = native_regs if requested is None else int(requested)
        torch.manual_seed(0)
        return VisionTransformer(
            img_size=32, patch_size=16, in_chans=3, embed_dim=32, depth=2,
            num_heads=2, mlp_ratio=2.0, num_classes=0, global_pool="",
            class_token=class_token, reg_tokens=count,
            no_embed_class=no_embed_class, dynamic_img_size=True,
        )

    monkeypatch.setattr(model_module.timm, "create_model", factory)
    encoder = model_module.OTUFormerEncoder(
        model_name="tiny-real", out_dim=8, return_patch_tokens=True,
        img_size=32, pretrained=False, reg_tokens=reg_tokens,
    )
    return encoder, encoder.backbone


@pytest.mark.parametrize("size,grid", [(32, 2), (224, 14)])
def test_registered_unmasked_pretrain_matches_backbone_at_two_sizes(
    monkeypatch, size, grid
):
    for native_regs, reg_tokens, prefix in ((0, 4, 5), (4, None, 5), (0, None, 1)):
        encoder, backbone = _real_tiny_encoder(
            monkeypatch, native_regs=native_regs, reg_tokens=reg_tokens
        )
        encoder.eval()
        encoder.validate_pretrain_backbone()
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            tokens = encoder.forward_pretrain(x).tokens
            reference = backbone.forward_features(x)

        assert backbone.num_prefix_tokens == prefix
        assert tokens.shape == (2, grid * grid + prefix, 32)
        assert reference.shape == tokens.shape
        assert torch.allclose(tokens, reference, atol=1e-6)


def test_registered_masked_forward_masks_only_image_patches(monkeypatch):
    encoder, backbone = _real_tiny_encoder(monkeypatch, reg_tokens=4)
    encoder.validate_pretrain_backbone()
    x = torch.randn(2, 3, 32, 32)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    replacement = torch.zeros(1, 1, 32)

    seen = {}
    original = encoder._replace_patches

    def spy(tokens, patch_mask, mask_token, patch_count):
        seen["patch_count"] = patch_count
        seen["shape"] = tuple(tokens.shape)
        return original(tokens, patch_mask, mask_token, patch_count)

    monkeypatch.setattr(encoder, "_replace_patches", spy)
    out = encoder.forward_pretrain(x, mask=mask, mask_token=replacement)

    assert seen == {"patch_count": 4, "shape": (2, 4, 32)}
    assert out.tokens.shape == (2, 4 + backbone.num_prefix_tokens, 32)
    assert out.grid_size == (2, 2)

    # A patch-output loss must reach the registers through self-attention.
    out.tokens[:, backbone.num_prefix_tokens:].square().sum().backward()
    assert backbone.reg_token.grad is not None
    assert backbone.reg_token.grad.abs().sum() > 0


def _encoder(**kwargs):
    import otuformer.training.model as model_module

    kwargs.setdefault("out_dim", 8)
    kwargs.setdefault("return_patch_tokens", True)
    kwargs.setdefault("img_size", 64)
    kwargs.setdefault("pretrained", False)
    return model_module.OTUFormerEncoder(model_name=kwargs.pop("model_name", "tiny-vit"), **kwargs)


def _record_encoder_constructions(monkeypatch, calls):
    """Record every encoder construction, including the helper's TaggedEncoder.

    ``_run_tiny_pretrain`` replaces ``trainer.OTUFormerEncoder`` with a subclass
    of ``model_module.OTUFormerEncoder``, so subclassing that attribute records
    both the resume probes and the training encoders.
    """
    import otuformer.training.model as model_module

    class RecordingEncoder(model_module.OTUFormerEncoder):
        def __init__(self, *args, **kwargs):
            calls.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(model_module, "OTUFormerEncoder", RecordingEncoder)
    return RecordingEncoder


def test_registered_new_run_records_the_added_registers(tmp_path, monkeypatch):
    calls: list[dict] = []
    _record_encoder_constructions(monkeypatch, calls)
    observations = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="reg4", register_tokens=4
    )
    saved = torch.load(
        observations["out_dir"] / "SSL_latest.pth", map_location="cpu", weights_only=False
    )

    assert saved["config"]["register_tokens"] == 4
    assert saved["args"]["register_tokens"] == 4
    assert saved["model_state_dict"]["backbone.reg_token"].shape == (1, 4, 16)
    assert "resume_state" not in saved["args"]
    registered = [
        call
        for call in calls
        if call.get("reg_tokens") == 4 and call.get("out_dim") != 256
    ]
    assert len(registered) == 2
    assert all(call.get("pretrained", True) is True for call in registered)
    # Pre-training equality is asserted by the dedicated synchronization test;
    # after one optimizer step plus EMA the two copies legitimately differ a
    # little, so only closeness is required here.
    assert torch.allclose(
        saved["student"]["backbone.reg_token"],
        saved["teacher"]["backbone.reg_token"],
        atol=5e-3,
    )


def test_registered_resume_rebuilds_without_pretrained_weights(tmp_path, monkeypatch):
    observations = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="reg4src", register_tokens=4
    )
    saved = torch.load(
        observations["out_dir"] / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    resume_path = tmp_path / "registered_resume.pth"
    torch.save(saved, resume_path)

    calls: list[dict] = []
    _record_encoder_constructions(monkeypatch, calls)
    _run_tiny_pretrain(
        tmp_path, monkeypatch, label="reg4resume", register_tokens=4,
        resume=str(resume_path), max_epochs=2,
    )

    registered = [call for call in calls if call.get("reg_tokens") == 4]
    assert len(registered) >= 2
    assert all(call["pretrained"] is False for call in registered)
    assert all(call["pretrained"] is False for call in calls)


def test_registered_resume_rejects_a_conflicting_register_option(tmp_path, monkeypatch):
    observations = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="reg4src2", register_tokens=4
    )
    saved = torch.load(
        observations["out_dir"] / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    resume_path = tmp_path / "registered_resume2.pth"
    torch.save(saved, resume_path)

    with pytest.raises(ValueError, match="Cannot resume a four-register checkpoint"):
        _run_tiny_pretrain(
            tmp_path, monkeypatch, label="reg4bad", register_tokens=0,
            resume=str(resume_path), max_epochs=2,
        )


def test_legacy_pretrain_resume_keeps_its_permissive_partial_load(monkeypatch):
    """The design excludes legacy zero-register resume from the new check.

    Design section 3: "Do not apply this new complete-backbone requirement to
    legacy zero-register resume." The registered paths carry the check; this
    path keeps its historical ``strict=False`` behaviour so an older checkpoint
    with deliberately tolerated missing keys still resumes.
    """
    import timm as timm_module

    monkeypatch.setattr(timm_module, "create_model", _tiny_vit_factory)
    state = {
        f"backbone.{key}": value
        for key, value in timm_module.create_model("tiny-vit").state_dict().items()
    }
    state.pop("backbone.blocks.0.attn.qkv.weight")
    state.pop("backbone.cls_token")

    assert trainer._resume_register_plan(
        {"model_state_dict": state}, "tiny-vit", 64, 8
    ) == ("legacy", 0)


def test_malformed_recorded_register_count_is_rejected_by_every_entry_point(
    monkeypatch,
):
    """config.register_tokens is validated before any layout decision."""
    import timm as timm_module
    from timm.models.vision_transformer import VisionTransformer

    import otuformer.utils.checkpoint as checkpoint_module

    monkeypatch.setattr(timm_module, "create_model", _tiny_vit_factory)
    for recorded in (-1, False, "4"):
        checkpoint = {"model_state_dict": {}, "config": {"register_tokens": recorded}}
        with pytest.raises(ValueError, match="register_tokens"):
            trainer._resume_register_plan(checkpoint, "tiny-vit", 64, 8)
        with pytest.raises(ValueError, match="register_tokens"):
            trainer._finetune_register_plan(checkpoint, "tiny-vit", 64, 8)

    backbone = VisionTransformer(
        img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
        num_heads=2, num_classes=0, global_pool="", dynamic_img_size=True,
    )
    with pytest.raises(ValueError, match="register_tokens"):
        checkpoint_module.resolve_register_layout(
            {"config": {"register_tokens": False}}, {}, backbone
        )
    # A legitimate zero count still reads as zero.
    assert checkpoint_module.recorded_register_count({"config": {}}) is None
    assert checkpoint_module.recorded_register_count(
        {"config": {"register_tokens": 0}}
    ) == 0


def test_registered_resume_requires_student_and_teacher(tmp_path, monkeypatch):
    observations = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="reg4src3", register_tokens=4
    )
    saved = torch.load(
        observations["out_dir"] / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    del saved["student"]
    resume_path = tmp_path / "registered_no_student.pth"
    torch.save(saved, resume_path)

    with pytest.raises(ValueError, match="Registered resume requires explicit 'student'"):
        _run_tiny_pretrain(
            tmp_path, monkeypatch, label="reg4nostudent", register_tokens=None,
            resume=str(resume_path), max_epochs=2,
        )


def test_omitted_option_on_native_register_model_keeps_its_own_registers(
    tmp_path, monkeypatch
):
    observations = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="native4", model_name="tiny-vit-native-reg4",
        register_tokens=None,
    )
    saved = torch.load(
        observations["out_dir"] / "SSL_latest.pth", map_location="cpu", weights_only=False
    )

    assert saved["config"]["register_tokens"] == 4
    assert saved["args"]["register_tokens"] is None
    assert saved["model_state_dict"]["backbone.reg_token"].shape == (1, 4, 16)


def test_explicit_register_option_rejected_on_no_cls_pretrain(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no CLS token"):
        _run_tiny_pretrain(
            tmp_path, monkeypatch, label="gap", model_name="tiny-vit-gap",
            register_tokens=0,
        )


def test_native_register_no_cls_pretrain_rejected_even_when_omitted(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no CLS token"):
        _run_tiny_pretrain(
            tmp_path, monkeypatch, label="gapreg", model_name="tiny-vit-gap-reg4",
            register_tokens=None,
        )


def test_unsupported_registered_model_rejected_without_a_download(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="standard timm VisionTransformer"):
        _run_tiny_pretrain(
            tmp_path, monkeypatch, label="eva",
            model_name="vit_small_patch16_dinov3.lvd1689m", register_tokens=None,
        )


def test_validate_pretrain_backbone_requires_four_blocks_for_intermediates(
    monkeypatch,
):
    import otuformer.training.model as model_module

    encoder, backbone = _tiny_encoder(
        monkeypatch, depth=3, prefix_tokens=1, dim=16
    )
    encoder.validate_pretrain_backbone(require_intermediates=False)

    with pytest.raises(ValueError, match="four"):
        encoder.validate_pretrain_backbone(require_intermediates=True)


def test_validate_pretrain_backbone_rejects_non_callable_patch_drop(monkeypatch):
    """EVA sets ``patch_drop = None``; that must fail validation, not forward."""
    encoder, backbone = _tiny_encoder(monkeypatch)
    backbone.patch_drop = None

    with pytest.raises(ValueError, match="patch_drop"):
        encoder.validate_pretrain_backbone()


def test_validate_pretrain_backbone_rejects_non_callable_norm_pre(monkeypatch):
    encoder, backbone = _tiny_encoder(monkeypatch)
    backbone.norm_pre = None

    with pytest.raises(ValueError, match="norm_pre"):
        encoder.validate_pretrain_backbone()


# --- Task 5: patch objective module and named EMA ----------------------------


@pytest.mark.parametrize(
    ("mode", "expect_mask", "expect_predictor", "expect_ibot"),
    [
        ("none", False, False, False),
        ("consistency", False, False, False),
        ("masked-feature", True, True, False),
        ("ibot", True, False, True),
    ],
)
def test_patch_objective_state_matches_mode(
    mode, expect_mask, expect_predictor, expect_ibot
):
    from otuformer.training.model import PatchObjective

    objective = PatchObjective(mode, hidden_dim=8, ibot_prototypes=4)

    assert objective.mode == mode
    assert (objective.mask_token is not None) is expect_mask
    assert (objective.predictor is not None) is expect_predictor
    assert (objective.student_ibot_head is not None) is expect_ibot
    assert (objective.teacher_ibot_head is not None) is expect_ibot
    assert (objective.patch_center is not None) is expect_ibot
    if expect_mask:
        assert objective.mask_token.shape == (1, 1, 8)
        # Truncated-normal initialization: small, finite, non-degenerate.
        peak = float(objective.mask_token.detach().abs().max())
        assert 0.0 < peak < 0.5
    if expect_predictor:
        layers = objective.predictor
        assert isinstance(layers[0], nn.LayerNorm)
        assert isinstance(layers[1], nn.Linear)
        assert isinstance(layers[2], nn.GELU)
        assert isinstance(layers[3], nn.Linear)
        out = objective.predict_masked_features(torch.randn(3, 8))
        assert out.shape == (3, 8)
        assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)
    if expect_ibot:
        assert objective.student_ibot_head[-1].out_features == 4
        assert objective.teacher_ibot_head[-1].out_features == 4
        assert objective.patch_center.shape == (1, 4)
        student_state = objective.student_ibot_head.state_dict()
        for name, value in objective.teacher_ibot_head.state_dict().items():
            assert torch.allclose(value, student_state[name])
        assert all(
            not p.requires_grad for p in objective.teacher_ibot_head.parameters()
        )


def test_patch_objective_supports_large_prototype_dictionaries():
    from otuformer.training.model import PatchObjective

    objective = PatchObjective("ibot", hidden_dim=8, ibot_prototypes=4096)

    assert objective.patch_center.shape == (1, 4096)
    assert objective.student_ibot_head[-1].out_features == 4096
    assert objective.teacher_ibot_head[-1].out_features == 4096


def test_patch_objective_trainable_parameters_are_mode_specific():
    from otuformer.training.model import PatchObjective

    assert (
        PatchObjective("none", hidden_dim=8, ibot_prototypes=4).trainable_parameters()
        == []
    )
    assert (
        PatchObjective(
            "consistency", hidden_dim=8, ibot_prototypes=4
        ).trainable_parameters()
        == []
    )
    masked = PatchObjective("masked-feature", hidden_dim=8, ibot_prototypes=4)
    masked_ids = {id(p) for p in masked.trainable_parameters()}
    assert id(masked.mask_token) in masked_ids
    assert {id(p) for p in masked.predictor.parameters()} <= masked_ids

    ibot = PatchObjective("ibot", hidden_dim=8, ibot_prototypes=4)
    ibot_ids = {id(p) for p in ibot.trainable_parameters()}
    assert id(ibot.mask_token) in ibot_ids
    assert {id(p) for p in ibot.student_ibot_head.parameters()} <= ibot_ids
    assert not ({id(p) for p in ibot.teacher_ibot_head.parameters()} & ibot_ids)


def _module_with(order, value):
    module = nn.Module()
    for name in order:
        setattr(module, name, nn.Linear(2, 2))
    for param in module.parameters():
        with torch.no_grad():
            param.fill_(value)
    return module


def test_named_ema_matches_by_name_regardless_of_registration_order():
    student = _module_with(["first", "second"], 1.0)
    teacher = _module_with(["second", "first"], 0.0)

    trainer.update_teacher(student, teacher, 0.9)

    for name, param in teacher.named_parameters():
        assert torch.allclose(param, torch.full_like(param, 0.1)), name


def test_named_ema_raises_when_a_teacher_name_has_no_student_counterpart():
    student = nn.Module()
    student.shared = nn.Linear(2, 2)
    teacher = nn.Module()
    teacher.shared = nn.Linear(2, 2)
    teacher.orphan = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="orphan"):
        trainer.update_teacher(student, teacher, 0.9)


def test_named_ema_ignores_student_only_parameters():
    student = nn.Module()
    student.shared = nn.Linear(2, 2)
    student.mask_token = nn.Parameter(torch.ones(1, 1, 2))
    for param in student.shared.parameters():
        with torch.no_grad():
            param.fill_(1.0)
    teacher = nn.Module()
    teacher.shared = nn.Linear(2, 2)
    for param in teacher.parameters():
        with torch.no_grad():
            param.zero_()

    trainer.update_teacher(student, teacher, 0.5)

    assert torch.allclose(teacher.shared.weight, torch.full_like(teacher.shared.weight, 0.5))


def test_named_ema_updates_the_decorrelated_ibot_teacher_head():
    from otuformer.training.model import PatchObjective

    objective = PatchObjective("ibot", hidden_dim=4, ibot_prototypes=3)
    for param in objective.student_ibot_head.parameters():
        with torch.no_grad():
            param.fill_(1.0)
    for param in objective.teacher_ibot_head.parameters():
        with torch.no_grad():
            param.zero_()

    trainer.update_teacher(
        objective.student_ibot_head, objective.teacher_ibot_head, 0.75
    )

    for param in objective.teacher_ibot_head.parameters():
        assert torch.allclose(param, torch.full_like(param, 0.25))
    # The center never enters EMA.
    assert torch.equal(objective.patch_center, torch.zeros(1, 3))


def test_optimizer_membership_is_mode_specific():
    from otuformer.training.model import PatchObjective

    def encoder():
        model = nn.Module()
        model.backbone = nn.Linear(4, 4)
        model.projector = nn.Linear(4, 4)
        return model

    for mode in ("none", "consistency", "masked-feature", "ibot"):
        student = encoder()
        objective = PatchObjective(mode, hidden_dim=4, ibot_prototypes=8)
        params = trainer._student_optimizer_parameters(student, objective)
        ids = {id(p) for p in params}

        assert {id(p) for p in student.parameters()} <= ids
        assert ids == {id(p) for p in student.parameters()} | {
            id(p) for p in objective.trainable_parameters()
        }
        if objective.teacher_ibot_head is not None:
            assert not (
                {id(p) for p in objective.teacher_ibot_head.parameters()} & ids
            )
            assert id(objective.patch_center) not in ids


def test_optimizer_membership_accepts_an_absent_objective():
    student = nn.Linear(4, 4)
    params = trainer._student_optimizer_parameters(student, None)
    assert {id(p) for p in params} == {id(p) for p in student.parameters()}


# --- Task 6: patch-mode integration ------------------------------------------

# Native register/CLS profile per tiny stand-in model name. ``reg_tokens`` in
# the create_model kwargs always wins, so an explicit option is observable.
_TINY_NATIVE_REGISTERS = {
    "tiny-vit": 0,
    "tiny-vit-native-reg4": 4,
    "tiny-vit-native-reg2": 2,
    "tiny-vit-gap": 0,
    "tiny-vit-gap-reg4": 4,
}
_TINY_CLASS_TOKEN = {"tiny-vit-gap": False, "tiny-vit-gap-reg4": False}


class _VitNamedButNotAVit(nn.Module):
    """Reports native registers without being a timm VisionTransformer."""

    def __init__(self) -> None:
        super().__init__()
        self.reg_token = nn.Parameter(torch.zeros(1, 4, 16))
        self.num_reg_tokens = 4
        self.num_prefix_tokens = 5
        self.num_features = 16


def _tiny_vit_factory(*args, **kwargs):
    from timm.models.vision_transformer import VisionTransformer

    model_name = args[0] if args else kwargs.pop("model_name", "tiny-vit")
    if model_name == "vit_small_patch16_dinov3.lvd1689m":
        return _VitNamedButNotAVit()

    requested = kwargs.get("reg_tokens")
    native = _TINY_NATIVE_REGISTERS.get(model_name, 0)
    count = native if requested is None else int(requested)
    return VisionTransformer(
        img_size=64,
        patch_size=8,
        embed_dim=16,
        depth=4,
        num_heads=2,
        num_classes=0,
        global_pool="",
        class_token=_TINY_CLASS_TOKEN.get(model_name, True),
        reg_tokens=count,
        dynamic_img_size=True,
    )


def _run_tiny_pretrain(tmp_path, monkeypatch, label="", on_first_step=None, **overrides):
    """Run one epoch of a tiny CPU pretrain with call tracing."""
    import pandas as pd
    import timm as timm_module
    from PIL import Image

    import otuformer.training.model as model_module

    image_names = []
    for index in range(4):
        path = tmp_path / f"img_{index}.jpg"
        Image.new("RGB", (64, 64), color=(index * 40, 30, 20)).save(path)
        image_names.append(path.name)
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": image_names}).to_csv(csv_path, index=False)

    monkeypatch.setattr(timm_module, "create_model", _tiny_vit_factory)

    observations: dict[str, object] = {
        "objective": None,
        "forwards": [],
        "masks": [],
        "events": [],
        "ema_modules": [],
        "center_inputs": [],
        "ibot_logits": [],
    }

    real_objective_cls = _REAL_PATCH_OBJECTIVE

    class RecordingObjective(real_objective_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            observations["objective"] = self

    monkeypatch.setattr(trainer, "PatchObjective", RecordingObjective)

    created = []

    class TaggedEncoder(model_module.OTUFormerEncoder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if k.get("out_dim") == 256:
                # Throwaway architecture probe; not a training encoder.
                self.tag = "probe"
                return
            self.tag = "student" if not created else "teacher"
            created.append(self)

    monkeypatch.setattr(trainer, "OTUFormerEncoder", TaggedEncoder)

    real_forward = _REAL_FORWARD_PRETRAIN

    def spy_forward(self, x, *, mask=None, mask_token=None, return_intermediates=False):
        observations["forwards"].append(
            {
                "tag": getattr(self, "tag", "?"),
                "masked": mask is not None,
                "intermediates": return_intermediates,
                "batch": int(x.shape[0]),
                "mask": None if mask is None else mask.clone(),
            }
        )
        return real_forward(
            self,
            x,
            mask=mask,
            mask_token=mask_token,
            return_intermediates=return_intermediates,
        )

    monkeypatch.setattr(
        model_module.OTUFormerEncoder, "forward_pretrain", spy_forward
    )

    real_sample = _REAL_SAMPLE_PATCH_MASKS

    def spy_sample(batch_size, grid_size, ratio, strategy, device):
        mask = real_sample(batch_size, grid_size, ratio, strategy, device)
        observations["masks"].append(
            {
                "strategy": strategy,
                "ratio": ratio,
                "grid": grid_size,
                "mask": mask.clone(),
            }
        )
        return mask

    monkeypatch.setattr(trainer, "_sample_patch_masks", spy_sample)

    real_teacher_center = trainer._update_teacher_center

    def spy_teacher_center(center, inputs):
        if on_first_step is not None and "first_step_done" not in observations:
            observations["first_step_done"] = True
            on_first_step(created[0], created[1])
        return real_teacher_center(center, inputs)

    monkeypatch.setattr(trainer, "_update_teacher_center", spy_teacher_center)

    real_update = _REAL_UPDATE_TEACHER

    def spy_update(student, teacher, momentum):
        observations["ema_modules"].append(type(teacher).__name__)
        observations["events"].append("ema")
        return real_update(student, teacher, momentum)

    monkeypatch.setattr(trainer, "update_teacher", spy_update)

    real_center = _REAL_UPDATE_PATCH_CENTER

    def spy_center(center, logits):
        observations["center_inputs"].append(logits.detach().clone())
        observations["events"].append("center")
        return real_center(center, logits)

    monkeypatch.setattr(trainer, "_update_patch_center", spy_center)

    real_ibot = _REAL_IBOT_PATCH_LOSS

    def spy_ibot(student_logits, teacher_logits, mask, center, s_temp, t_temp):
        observations["events"].append("ibot_loss")
        observations["ibot_logits"].append(teacher_logits[mask].detach().clone())
        observations["ibot_center"] = center.detach().clone()
        return real_ibot(student_logits, teacher_logits, mask, center, s_temp, t_temp)

    monkeypatch.setattr(trainer, "ibot_patch_loss", spy_ibot)

    real_step = torch.optim.AdamW.step

    def spy_step(self, *a, **k):
        observations["events"].append("optimizer_step")
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy_step)

    out_dir = tmp_path / f"out{label}"
    settings = {
        "patch_loss": "consistency",
        "masking_strategy": "random",
        "mask_ratio": "auto",
        "mask_ratio_explicit": False,
        "patch_loss_explicit": False,
        "masking_strategy_explicit": False,
        "ibot_prototypes": 512,
        "ibot_prototypes_explicit": False,
        "lambda_mask": 1.0,
        "lambda_mask_explicit": False,
        "lambda_local": 1.0,
        "global_crop_size": 64,
        "local_crop_size": 16,
        "local_crops": 1,
        "max_epochs": 1,
        "save_every_epochs": 1,
        "resume": "",
        # Keep the historical ordinary/full recipe for these orchestration
        # assertions; the relational routing matrix owns weak/cross coverage.
        "relational_teacher_views": "ordinary",
        "relational_center_source": None,
        "relational_pairing": "full",
        "teacher_temp_warmup_ratio": 0.7,
        "teacher_temp_start": 0.04,
        "teacher_temp_end": 0.07,
    }
    settings.update(overrides)
    args = argparse.Namespace(
        train_data=str(csv_path),
        input_images_dir=str(tmp_path),
        out_dir=str(out_dir),
        overwrite=False,
        model_name=settings.pop("model_name", "tiny-vit"),
        out_dim=8,
        lr=1e-3,
        weight_decay=0.0,
        warmup_epochs=0,
        augmentation="legacy",
        orientation_policy="sensitive",
        teacher_momentum=0.9,
        teacher_momentum_end=0.99,
        student_temp=0.1,
        log_every_n_steps=1,
        keep_last_checkpoints=1,
        visualize_data="",
        extract_size=None,
        metrics_sample_size=100,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=20,
        compute_embedding_metrics=False,
        batch_size=4,
        num_workers=0,
        cpus=2,
        device="cpu",
        seed=42,
        **settings,
    )

    observations["encoders"] = created
    trainer.run_pretrain(args)
    observations["out_dir"] = out_dir
    observations["args"] = args
    return observations


@pytest.mark.parametrize(
    ("mode", "strategy"),
    [
        ("none", "random"),
        ("consistency", "random"),
        ("masked-feature", "random"),
        ("masked-feature", "blockwise"),
        ("masked-feature", "hybrid"),
        ("ibot", "random"),
        ("ibot", "blockwise"),
        ("ibot", "hybrid"),
    ],
)
def test_pretrain_one_step_smoke_and_orchestration(
    tmp_path, monkeypatch, mode, strategy
):
    obs = _run_tiny_pretrain(
        tmp_path, monkeypatch, patch_loss=mode, masking_strategy=strategy
    )
    forwards = obs["forwards"]
    teacher_forwards = [f for f in forwards if f["tag"] == "teacher"]
    student_masked = [f for f in forwards if f["tag"] == "student" and f["masked"]]
    student_unmasked = [
        f for f in forwards if f["tag"] == "student" and not f["masked"]
    ]

    # One teacher forward per global view, always unmasked.
    assert len(teacher_forwards) == 2
    assert all(not f["masked"] for f in teacher_forwards)
    # The unmasked global student forwards stay in the global/local SSL path.
    assert len(student_unmasked) == 2

    if mode == "none":
        assert obs["masks"] == []
        assert student_masked == []
    elif mode == "consistency":
        assert student_masked == []
        assert len(obs["masks"]) == 2
        assert all(m["strategy"] == "random" for m in obs["masks"])
    else:
        assert len(student_masked) == 2
        assert len(obs["masks"]) == 2
        assert all(m["strategy"] == strategy for m in obs["masks"])
        first, second = obs["masks"][0]["mask"], obs["masks"][1]["mask"]
        assert not torch.equal(first, second)
        assert not torch.equal(first[0], first[1])

    # Losses are finite and the weighted patch term composes the total.
    import pandas as pd

    log = pd.read_csv(obs["out_dir"] / "logs" / "instant_metrics.pretrain.csv")
    row = log.iloc[-1]
    for column in ("loss", "global_loss", "local_loss", "mask_loss"):
        assert pd.notna(row[column])
        assert abs(float(row[column])) < 1e6
    assert float(row["loss"]) == pytest.approx(
        float(row["global_loss"])
        + float(obs["args"].lambda_local) * float(row["local_loss"])
        + float(obs["args"].lambda_mask) * float(row["mask_loss"]),
        abs=1e-4,
    )

    objective = obs["objective"]
    if mode == "masked-feature":
        assert objective.mask_token.grad is not None
        assert all(
            p.grad is not None for p in objective.predictor.parameters()
        )
    elif mode == "ibot":
        assert objective.mask_token.grad is not None
        assert all(
            p.grad is not None for p in objective.student_ibot_head.parameters()
        )
        assert all(
            p.grad is None for p in objective.teacher_ibot_head.parameters()
        )
        assert float(objective.patch_center.abs().sum()) > 0.0
    else:
        assert objective.trainable_parameters() == []


def test_ibot_center_uses_pre_ema_teacher_logits_once_per_step(tmp_path, monkeypatch):
    obs = _run_tiny_pretrain(tmp_path, monkeypatch, patch_loss="ibot")

    assert len(obs["center_inputs"]) == 1
    concatenated = torch.cat(obs["ibot_logits"], dim=0)
    assert obs["center_inputs"][0].shape == concatenated.shape
    assert torch.allclose(obs["center_inputs"][0], concatenated)

    objective = obs["objective"]
    expected_center = 0.1 * concatenated.mean(dim=0, keepdim=True)
    assert torch.allclose(objective.patch_center, expected_center, atol=1e-6)

    # Order: loss with the old center, optimizer step, EMA, then center update.
    events = obs["events"]
    assert events[0] == "ibot_loss"
    assert events.count("ibot_loss") == 2
    assert events.index("optimizer_step") < events.index("ema")
    assert events.index("ema") < events.index("center")
    assert sorted(obs["ema_modules"]) == ["Sequential", "TaggedEncoder"]


def test_pretrain_rejects_blockwise_on_area_limited_grid(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="blockwise"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            patch_loss="masked-feature",
            masking_strategy="blockwise",
            global_crop_size=32,
            local_crop_size=16,
        )


def test_pretrain_rejects_blockwise_on_single_patch_grid(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="blockwise"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            patch_loss="ibot",
            masking_strategy="hybrid",
            global_crop_size=8,
            local_crops=0,
        )


# --- Task 7: strict resume, separate state, and RNG round-tripping -----------


def _load_ckpt(obs):
    return torch.load(
        obs["out_dir"] / "SSL_epoch_0001.pth", map_location="cpu", weights_only=False
    )


def _save_ckpt(checkpoint, path):
    torch.save(checkpoint, path)


@pytest.mark.parametrize(
    ("mode", "strategy"),
    [
        ("none", "random"),
        ("consistency", "random"),
        ("masked-feature", "random"),
        ("ibot", "random"),
    ],
)
def test_v07_checkpoint_records_patch_state_and_config(tmp_path, monkeypatch, mode, strategy):
    obs = _run_tiny_pretrain(
        tmp_path, monkeypatch, patch_loss=mode, masking_strategy=strategy
    )
    ckpt = _load_ckpt(obs)
    config = ckpt["config"]

    assert config["patch_loss"] == mode
    assert config["mask_ratio_requested"] == (
        None if mode == "none" else "auto"
    )
    assert config["mask_ratio_resolved"] == (
        None if mode == "none" else 0.30
    )
    assert config["masking_strategy"] == (
        strategy if mode in ("masked-feature", "ibot") else None
    )
    assert config["ibot_prototypes"] == (512 if mode == "ibot" else None)
    assert config["patch_target_layers"] == (
        4 if mode == "masked-feature" else None
    )
    assert config["patch_center_momentum"] == (0.9 if mode == "ibot" else None)

    # Training-only state lives at the checkpoint root, not in student/teacher.
    assert "patch_objective" in ckpt
    objective_state = ckpt["patch_objective"]
    if mode in ("none", "consistency"):
        assert objective_state == {}
    else:
        assert "mask_token" in objective_state
    if mode == "masked-feature":
        assert any(k.startswith("predictor.") for k in objective_state)
    if mode == "ibot":
        assert any(k.startswith("student_ibot_head.") for k in objective_state)
        assert any(k.startswith("teacher_ibot_head.") for k in objective_state)
        assert "patch_center" in objective_state

    assert "mask_token" not in ckpt["student"]
    assert "mask_token" not in ckpt["teacher"]
    assert ckpt["rng_state"]["python"] is not None
    assert ckpt["rng_state"]["numpy"] is not None
    assert ckpt["rng_state"]["torch"] is not None


def test_v07_resume_restores_objective_and_rng(tmp_path, monkeypatch):
    first = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_a", patch_loss="masked-feature"
    )
    checkpoint = first["out_dir"] / "SSL_epoch_0001.pth"
    second = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_b",
        patch_loss="masked-feature",
        resume=str(checkpoint),
        max_epochs=2,
    )
    assert second["masks"]


def test_v07_resume_rejects_patch_loss_change(tmp_path, monkeypatch):
    first = _run_tiny_pretrain(tmp_path, monkeypatch, label="_a")
    checkpoint = first["out_dir"] / "SSL_epoch_0001.pth"

    with pytest.raises(ValueError, match="patch_loss"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            patch_loss="masked-feature",
            patch_loss_explicit=True,
            resume=str(checkpoint),
            max_epochs=2,
        )


def test_v07_resume_rejects_resolved_ratio_change(tmp_path, monkeypatch):
    first = _run_tiny_pretrain(tmp_path, monkeypatch, label="_a")
    checkpoint = first["out_dir"] / "SSL_epoch_0001.pth"

    with pytest.raises(ValueError, match="mask_ratio"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            mask_ratio=0.45,
            mask_ratio_explicit=True,
            resume=str(checkpoint),
            max_epochs=2,
        )


def test_v07_resume_rejects_strategy_change(tmp_path, monkeypatch):
    first = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_a", patch_loss="masked-feature"
    )
    checkpoint = first["out_dir"] / "SSL_epoch_0001.pth"

    with pytest.raises(ValueError, match="masking_strategy"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            patch_loss="masked-feature",
            masking_strategy="blockwise",
            masking_strategy_explicit=True,
            resume=str(checkpoint),
            max_epochs=2,
        )


def test_v07_resume_rejects_ibot_prototype_change(tmp_path, monkeypatch):
    first = _run_tiny_pretrain(tmp_path, monkeypatch, label="_a", patch_loss="ibot")
    checkpoint = first["out_dir"] / "SSL_epoch_0001.pth"

    with pytest.raises(ValueError, match="ibot_prototypes"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            patch_loss="ibot",
            ibot_prototypes=256,
            ibot_prototypes_explicit=True,
            resume=str(checkpoint),
            max_epochs=2,
        )


@pytest.mark.parametrize("mode", ["masked-feature", "ibot"])
def test_v07_resume_rejects_missing_patch_objective_state(
    tmp_path, monkeypatch, mode
):
    first = _run_tiny_pretrain(tmp_path, monkeypatch, label="_a", patch_loss=mode)
    checkpoint = _load_ckpt(first)
    checkpoint.pop("patch_objective")
    broken = tmp_path / "broken.pth"
    _save_ckpt(checkpoint, broken)

    with pytest.raises(ValueError, match="patch_objective"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            patch_loss=mode,
            resume=str(broken),
            max_epochs=2,
        )


def test_v07_resume_rejects_malformed_patch_objective_state(tmp_path, monkeypatch):
    first = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_a", patch_loss="masked-feature"
    )
    checkpoint = _load_ckpt(first)
    checkpoint["patch_objective"]["mask_token"] = torch.zeros(1, 1, 3)
    broken = tmp_path / "broken.pth"
    _save_ckpt(checkpoint, broken)

    with pytest.raises(ValueError, match="patch_objective"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            patch_loss="masked-feature",
            resume=str(broken),
            max_epochs=2,
        )


def _make_legacy_checkpoint(
    tmp_path, monkeypatch, saved_ratio=0.5, *, mixed_generation=False
):
    """A historical ``main`` payload, or a mixed-generation variant on request.

    ``mixed_generation`` keeps the canonical relational bundle and the new-style
    ``args`` fields to prove that a checkpoint written by the updated code stays
    resumable; the default is a genuine pre-change payload.
    """
    first = _run_tiny_pretrain(tmp_path, monkeypatch, label="_a")
    checkpoint = _load_ckpt(first)
    config = dict(checkpoint["config"])
    for key in (
        "patch_loss",
        "masking_strategy",
        "mask_ratio_requested",
        "mask_ratio_resolved",
        "ibot_prototypes",
        "patch_target_layers",
        "patch_center_momentum",
    ):
        config.pop(key, None)
    if not mixed_generation:
        config.pop(trainer.RELATIONAL_TARGET_CONFIG_KEY, None)
    checkpoint["config"] = config
    checkpoint.pop("patch_objective", None)
    checkpoint.pop("rng_state", None)
    checkpoint["args"] = dict(checkpoint.get("args") or {})
    if not mixed_generation:
        for key in (
            "relational_teacher_views",
            "relational_pairing",
            "relational_center_source",
            "teacher_temp_start",
            "teacher_temp_end",
            "teacher_temp_warmup_ratio",
            "weak_teacher_augmentation_config",
            "weak_relational_teacher_views",
            "relational_teacher_views_explicit",
            "relational_pairing_explicit",
            "relational_center_source_explicit",
            "teacher_temp_start_explicit",
            "teacher_temp_end_explicit",
            "teacher_temp_warmup_ratio_explicit",
        ):
            checkpoint["args"].pop(key, None)
    checkpoint["args"]["disable_cross_view_loss"] = False
    if saved_ratio is None:
        checkpoint["args"].pop("mask_ratio", None)
    else:
        checkpoint["args"]["mask_ratio"] = saved_ratio
    path = tmp_path / "legacy.pth"
    _save_ckpt(checkpoint, path)
    return path


@pytest.mark.parametrize("saved_ratio", [0.5, None])
def test_legacy_resume_inherits_recorded_ratio_and_warns(
    tmp_path, monkeypatch, capsys, saved_ratio
):
    legacy = _make_legacy_checkpoint(tmp_path, monkeypatch, saved_ratio=saved_ratio)

    obs = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_b",
        resume=str(legacy),
        max_epochs=2,
    )
    assert obs["masks"]
    out = capsys.readouterr().out
    assert "random sequence cannot continue exactly" in out

    checkpoint = torch.load(
        obs["out_dir"] / "SSL_epoch_0002.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["config"]["patch_loss"] == "consistency"
    assert checkpoint["config"]["mask_ratio_resolved"] == 0.5


def test_legacy_resume_rejects_conflicting_ratio(tmp_path, monkeypatch):
    legacy = _make_legacy_checkpoint(tmp_path, monkeypatch, saved_ratio=0.5)

    with pytest.raises(ValueError, match="mask-ratio"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            mask_ratio=0.3,
            mask_ratio_explicit=True,
            resume=str(legacy),
            max_epochs=2,
        )


@pytest.mark.parametrize("mode", ["masked-feature", "ibot"])
def test_legacy_resume_rejects_real_masking_modes(tmp_path, monkeypatch, mode):
    legacy = _make_legacy_checkpoint(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="patch_loss"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_b",
            patch_loss=mode,
            patch_loss_explicit=True,
            resume=str(legacy),
            max_epochs=2,
        )


def test_rng_state_round_trips_python_numpy_and_torch():
    import random as random_module

    import numpy as np

    random_module.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = trainer._capture_rng_state()
    expected = (
        random_module.random(),
        float(np.random.rand()),
        float(torch.rand(1)),
    )

    random_module.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    trainer._restore_rng_state(state)

    assert (
        random_module.random(),
        float(np.random.rand()),
        float(torch.rand(1)),
    ) == expected


def test_resume_reproduces_the_uninterrupted_mask_sequence(tmp_path, monkeypatch):
    uninterrupted = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_full", max_epochs=2
    )
    first_leg = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_leg1", max_epochs=1
    )
    checkpoint = first_leg["out_dir"] / "SSL_epoch_0001.pth"
    resumed = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_leg2",
        resume=str(checkpoint),
        max_epochs=2,
    )

    combined = first_leg["masks"] + resumed["masks"]
    assert len(combined) == len(uninterrupted["masks"])
    for expected, actual in zip(uninterrupted["masks"], combined):
        assert torch.equal(expected["mask"], actual["mask"])


def _epoch_masks(observations, views_per_step=2):
    masks = [entry["mask"] for entry in observations["masks"]]
    assert len(masks) % views_per_step == 0
    return [
        masks[index : index + views_per_step]
        for index in range(0, len(masks), views_per_step)
    ]


def test_rng_advances_between_sparse_checkpoint_epochs(tmp_path, monkeypatch):
    """With save_every_epochs > 1 the RNG must not restart from the last save."""
    obs = _run_tiny_pretrain(
        tmp_path, monkeypatch, max_epochs=4, save_every_epochs=2
    )
    epochs = _epoch_masks(obs)
    assert len(epochs) == 4
    # Epochs 3 and 4 must use different random sequences.
    assert not torch.equal(epochs[2][0], epochs[3][0])
    assert not torch.equal(epochs[0][0], epochs[2][0])


def test_resume_with_sparse_checkpoints_reproduces_the_sequence(
    tmp_path, monkeypatch
):
    uninterrupted = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_full", max_epochs=4, save_every_epochs=2
    )
    first_leg = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_leg1", max_epochs=2, save_every_epochs=2
    )
    checkpoint = first_leg["out_dir"] / "SSL_epoch_0002.pth"
    resumed = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_leg2",
        resume=str(checkpoint),
        max_epochs=4,
        save_every_epochs=2,
    )

    combined = first_leg["masks"] + resumed["masks"]
    assert len(combined) == len(uninterrupted["masks"])
    for expected, actual in zip(uninterrupted["masks"], combined):
        assert torch.equal(expected["mask"], actual["mask"])


def test_none_mode_still_validates_the_pretrain_backbone(tmp_path, monkeypatch):
    calls = []
    real_validate = OTUFormerEncoder.validate_pretrain_backbone

    def spy(self, require_intermediates=False):
        calls.append(require_intermediates)
        return real_validate(self, require_intermediates=require_intermediates)

    monkeypatch.setattr(OTUFormerEncoder, "validate_pretrain_backbone", spy)

    _run_tiny_pretrain(tmp_path, monkeypatch, patch_loss="none")

    # The architecture probe and both training encoders are validated.
    assert calls == [False, False, False]


def test_legacy_resume_accepts_explicit_lambda_mask(tmp_path, monkeypatch):
    """lambda_mask weights the consistency patch loss and stays applicable."""
    legacy = _make_legacy_checkpoint(tmp_path, monkeypatch)

    obs = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_b",
        resume=str(legacy),
        max_epochs=2,
        lambda_mask=0.5,
        lambda_mask_explicit=True,
    )

    assert obs["masks"]


def test_read_only_loader_ignores_patch_objective_state(tmp_path, monkeypatch):
    from otuformer.utils.checkpoint import resolve_checkpoint

    obs = _run_tiny_pretrain(
        tmp_path, monkeypatch, patch_loss="masked-feature"
    )
    checkpoint = _load_ckpt(obs)
    resolved = resolve_checkpoint(checkpoint, "tiny-vit")

    assert resolved.source_key in {"teacher", "student", "model_state_dict"}
    assert all(
        not key.startswith(("mask_token", "predictor."))
        for key in resolved.state_dict
    )


# --- Task 8: instant metrics schema and patch diagnostics --------------------

_LEGACY_PRETRAIN_HEADER = [
    "iteration",
    "epoch",
    "step",
    "loss",
    "global_loss",
    "local_loss",
    "mask_loss",
    "lr",
    "teacher_temp",
    "grad_norm",
    "feature_std",
    "embedding_norm_mean",
    "cls_token_norm_mean",
    "teacher_center_norm",
    "cosine_similarity",
]

_NEW_PRETRAIN_FIELDS = [
    "patch_loss_raw",
    "patch_loss_weighted",
    "patch_loss_fraction_of_total",
    "requested_mask_ratio",
    "selected_patch_ratio",
    "actual_mask_ratio_mean",
    "actual_mask_ratio_min",
    "actual_mask_ratio_max",
    "prototype_perplexity",
    "active_prototype_ratio",
    "assignment_entropy",
    "max_prototype_occupancy",
    "patch_center_norm",
]


def test_new_pretrain_csv_has_the_v07_schema(tmp_path, monkeypatch):
    import pandas as pd

    obs = _run_tiny_pretrain(tmp_path, monkeypatch)
    log = pd.read_csv(obs["out_dir"] / "logs" / "instant_metrics.pretrain.csv")

    assert list(log.columns)[: len(_LEGACY_PRETRAIN_HEADER)] == _LEGACY_PRETRAIN_HEADER
    for field in _NEW_PRETRAIN_FIELDS:
        assert field in log.columns


def _write_legacy_csv(path):
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_LEGACY_PRETRAIN_HEADER)
        writer.writerow(list(range(len(_LEGACY_PRETRAIN_HEADER))))
        writer.writerow([value + 1 for value in range(len(_LEGACY_PRETRAIN_HEADER))])


def test_legacy_pretrain_schema_is_migrated_atomically(tmp_path):
    import pandas as pd

    path = tmp_path / "instant_metrics.pretrain.csv"
    _write_legacy_csv(path)

    trainer.InstantMetricsLogger(path, mode="pretrain")

    migrated = pd.read_csv(path)
    assert len(migrated) == 2
    # The v0.7.0 columns are frozen and the CLS-distillation columns are
    # appended after them, so historical values and order are preserved.
    expected_fields = (
        _LEGACY_PRETRAIN_HEADER
        + _NEW_PRETRAIN_FIELDS
        + trainer._PRETRAIN_SSL_SETTING_FIELDS
        + trainer._PRETRAIN_SSL_METRIC_FIELDS
    )
    assert list(migrated.columns) == expected_fields
    assert migrated.iloc[0]["iteration"] == 0
    assert migrated.iloc[1]["iteration"] == 1
    for field in _NEW_PRETRAIN_FIELDS + trainer._PRETRAIN_SSL_SETTING_FIELDS + (
        trainer._PRETRAIN_SSL_METRIC_FIELDS
    ):
        assert migrated[field].isna().all(), field
    assert list(tmp_path.glob("*.tmp")) == []


def test_migrated_legacy_csv_can_be_plotted(tmp_path):
    path = tmp_path / "instant_metrics.pretrain.csv"
    _write_legacy_csv(path)
    logger = trainer.InstantMetricsLogger(path, mode="pretrain")

    logger.plot(tmp_path)

    assert (tmp_path / "training_curves_pretrain.pdf").exists()


def test_current_pretrain_schema_is_appended_without_migration(tmp_path):
    import pandas as pd

    path = tmp_path / "instant_metrics.pretrain.csv"
    logger = trainer.InstantMetricsLogger(path, mode="pretrain")
    before = path.read_bytes()
    logger.log(iteration=0, epoch=0, step=0, loss=1.0)
    logger_again = trainer.InstantMetricsLogger(path, mode="pretrain")
    logger_again.log(iteration=1, epoch=0, step=1, loss=2.0)

    assert path.read_bytes() != before
    assert len(pd.read_csv(path)) == 2


@pytest.mark.parametrize(
    "content",
    [
        "a,b,a\n1,2,3\n",
        "unknown,header\n1,2\n",
        "iteration,epoch,step,loss\n1,2\n",
    ],
)
def test_malformed_pretrain_schema_fails_without_touching_the_file(tmp_path, content):
    path = tmp_path / "instant_metrics.pretrain.csv"
    path.write_text(content, encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError, match="schema"):
        trainer.InstantMetricsLogger(path, mode="pretrain")

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("mode", "strategy"),
    [
        ("none", "random"),
        ("consistency", "random"),
        ("masked-feature", "random"),
        ("masked-feature", "blockwise"),
        ("ibot", "random"),
        ("ibot", "blockwise"),
    ],
)
def test_mode_specific_patch_diagnostics(tmp_path, monkeypatch, mode, strategy):
    import pandas as pd

    obs = _run_tiny_pretrain(
        tmp_path, monkeypatch, patch_loss=mode, masking_strategy=strategy
    )
    log = pd.read_csv(obs["out_dir"] / "logs" / "instant_metrics.pretrain.csv")
    row = log.iloc[-1]

    assert float(row["patch_loss_raw"]) == pytest.approx(float(row["mask_loss"]))
    assert float(row["patch_loss_weighted"]) == pytest.approx(
        float(row["patch_loss_raw"]) * float(obs["args"].lambda_mask)
    )
    if float(row["loss"]) != 0.0:
        assert float(row["patch_loss_fraction_of_total"]) == pytest.approx(
            float(row["patch_loss_weighted"]) / float(row["loss"])
        )
    assert str(row["requested_mask_ratio"]) == (
        "nan" if mode == "none" else "auto"
    )

    actual_fields = (
        "actual_mask_ratio_mean",
        "actual_mask_ratio_min",
        "actual_mask_ratio_max",
    )
    if mode == "consistency":
        assert 0.0 < float(row["selected_patch_ratio"]) < 1.0
        assert all(pd.isna(row[field]) for field in actual_fields)
    elif mode in ("masked-feature", "ibot"):
        assert all(pd.notna(row[field]) for field in actual_fields)
        assert float(row["actual_mask_ratio_min"]) <= float(
            row["actual_mask_ratio_mean"]
        ) <= float(row["actual_mask_ratio_max"])
        assert pd.isna(row["selected_patch_ratio"])
    else:
        assert all(pd.isna(row[field]) for field in actual_fields)
        assert pd.isna(row["selected_patch_ratio"])

    ibot_fields = (
        "prototype_perplexity",
        "active_prototype_ratio",
        "assignment_entropy",
        "max_prototype_occupancy",
        "patch_center_norm",
    )
    if mode == "ibot":
        assert all(pd.notna(row[field]) for field in ibot_fields)
        assert 1.0 <= float(row["prototype_perplexity"]) <= 512.0
        assert 0.0 < float(row["active_prototype_ratio"]) <= 1.0
        assert float(row["patch_center_norm"]) > 0.0
    else:
        assert all(pd.isna(row[field]) for field in ibot_fields)


def test_ibot_diagnostics_match_reference():
    torch.manual_seed(3)
    k = 4
    probs = torch.softmax(torch.randn(6, k), dim=-1)
    center = torch.tensor([[0.5, -0.25, 0.0, 1.0]])

    diagnostics = trainer._ibot_diagnostics(probs, center)

    mean_assignment = probs.mean(dim=0)
    entropy = -(mean_assignment * torch.log(mean_assignment)).sum()
    hard = probs.argmax(dim=-1)
    counts = torch.bincount(hard, minlength=k).float()
    per_position_entropy = -(probs * torch.log(probs)).sum(-1).mean()

    assert diagnostics["prototype_perplexity"] == pytest.approx(
        float(torch.exp(entropy)), rel=1e-5
    )
    assert diagnostics["active_prototype_ratio"] == pytest.approx(
        float((counts > 0).sum()) / k
    )
    assert diagnostics["assignment_entropy"] == pytest.approx(
        float(per_position_entropy), rel=1e-5
    )
    assert diagnostics["max_prototype_occupancy"] == pytest.approx(
        float(counts.max()) / probs.shape[0]
    )
    assert diagnostics["patch_center_norm"] == pytest.approx(float(center.norm()))



# --- relational teacher routing --------------------------------------------

def test_relational_teacher_routing(tmp_path, monkeypatch, capsys):
    obs = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_weak",
        patch_loss="none",
        relational_teacher_views="weak",
    )
    forwards = obs["forwards"]
    teacher_forwards = [f for f in forwards if f["tag"] == "teacher"]
    # 2 weak relational views + ordinary A for the diagnostic cosine
    assert len(teacher_forwards) == 3
    assert all(not f["masked"] for f in teacher_forwards)

    out = capsys.readouterr().out
    assert "Relational target config: " in out
    assert "Teacher temperature schedule: " in out
    assert '"warmup_ratio": 0.7' in out
    assert "Initial ordinary-global cosine similarity:" in out
    assert "Initial cosine similarity:" not in out

    checkpoint = torch.load(
        obs["out_dir"] / "SSL_epoch_0001.pth", map_location="cpu", weights_only=False
    )
    bundle = checkpoint["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY]
    assert bundle["relational_teacher_views"] == "weak"
    assert bundle["relational_center_source"] == "weak"
    assert bundle["relational_pairing"] == "full"
    assert bundle["weak_teacher_augmentation_config"]["normalize"] == {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    assert set(bundle) == set(trainer.RELATIONAL_BUNDLE_KEYS)
    saved_args = checkpoint["args"]
    for key in trainer.RELATIONAL_ARGS_KEYS:
        assert saved_args[key] == bundle[key]
    assert saved_args["teacher_temp_warmup_ratio"] == 0.7
    for key in (
        "include_weak_teacher_views",
        "weak_relational_teacher_views",
        "disable_cross_view_loss",
        "pretrain_recipe",
    ):
        assert key not in saved_args


def test_relational_teacher_routing_weak_with_ordinary_center(tmp_path, monkeypatch):
    calls = []
    real_center = trainer._update_teacher_center

    def spy_center(center, inputs):
        calls.append((center.detach().clone(), inputs.detach().clone()))
        return real_center(center, inputs)

    monkeypatch.setattr(trainer, "_update_teacher_center", spy_center)

    obs = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_wc",
        patch_loss="masked-feature",
        masking_strategy="random",
        relational_teacher_views="weak",
        relational_center_source="ordinary",
    )
    teacher_forwards = [f for f in obs["forwards"] if f["tag"] == "teacher"]
    # 2 weak + 2 ordinary (patch targets and the ordinary center population)
    assert len(teacher_forwards) == 4
    # the weak forwards never need intermediates, so patch targets cannot use them
    assert [f["intermediates"] for f in teacher_forwards].count(True) == 2

    # Exactly one center update, from two ordinary views, using c_k-centred and
    # L2-normalized features (the historical population).
    assert len(calls) == 1
    before, inputs = calls[0]
    assert inputs.shape[0] == 2 * 4
    assert torch.allclose(inputs.norm(dim=-1), torch.ones(inputs.shape[0]), atol=1e-5)

    checkpoint = torch.load(
        obs["out_dir"] / "SSL_epoch_0001.pth", map_location="cpu", weights_only=False
    )
    bundle = checkpoint["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY]
    assert bundle["relational_center_source"] == "ordinary"
    # A disabled center update would leave the saved center at its zero init.
    assert float(checkpoint["center"].abs().sum()) > 0.0


def test_relational_teacher_routing_ordinary_keeps_two_forwards(tmp_path, monkeypatch):
    obs = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_ord",
        patch_loss="consistency",
        relational_teacher_views="ordinary",
    )
    teacher_forwards = [f for f in obs["forwards"] if f["tag"] == "teacher"]
    assert len(teacher_forwards) == 2


def test_relational_teacher_preserves_fixed_patch_inputs(tmp_path, monkeypatch):
    plain = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_p", patch_loss="consistency"
    )
    weak = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_w",
        patch_loss="consistency",
        relational_teacher_views="weak",
    )
    # The ordinary patch path keeps both ordinary views and their masks.
    plain_masks = [m["mask"] for m in plain["masks"]]
    weak_masks = [m["mask"] for m in weak["masks"]]
    assert len(plain_masks) == len(weak_masks) == 2
    assert all(not m["masked"] for m in plain["forwards"] if m["tag"] == "teacher")
    assert all(not m["masked"] for m in weak["forwards"] if m["tag"] == "teacher")
    # teacher temperature still reaches the iBOT target sharpening
    ibot_plain = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_ip", patch_loss="ibot"
    )
    ibot_warm = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_iw",
        patch_loss="ibot",
        teacher_temp_start=0.08,
        teacher_temp_end=0.11,
    )
    import pandas as pd

    plain_rows = pd.read_csv(
        ibot_plain["out_dir"] / "logs" / "instant_metrics.pretrain.csv"
    )
    warm_rows = pd.read_csv(
        ibot_warm["out_dir"] / "logs" / "instant_metrics.pretrain.csv"
    )
    # The teacher temperature reaches iBOT target sharpening, so the patch loss
    # must differ even at identical fixed teacher state and masks.
    assert float(plain_rows["mask_loss"].iloc[-1]) != pytest.approx(
        float(warm_rows["mask_loss"].iloc[-1])
    )
    assert float(plain_rows["teacher_temp"].iloc[-1]) == pytest.approx(0.07)
    assert float(warm_rows["teacher_temp"].iloc[-1]) == pytest.approx(0.11)


def test_relational_fresh_teacher_copies_student_state(tmp_path, monkeypatch):
    """The pre-forward pair must be identical, non-aliased and frozen."""
    snap = {}

    def capture(student, teacher):
        snap["student"] = {
            name: tensor.detach().clone()
            for name, tensor in student.state_dict().items()
        }
        snap["teacher"] = {
            name: tensor.detach().clone()
            for name, tensor in teacher.state_dict().items()
        }
        snap["student_ptr"] = {
            name: tensor.data_ptr() for name, tensor in student.state_dict().items()
        }
        snap["teacher_ptr"] = {
            name: tensor.data_ptr() for name, tensor in teacher.state_dict().items()
        }
        snap["teacher_frozen"] = all(
            not parameter.requires_grad for parameter in teacher.parameters()
        )
        snap["student_trainable"] = any(
            parameter.requires_grad for parameter in student.parameters()
        )

    _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_sync",
        patch_loss="consistency",
        relational_teacher_views="weak",
        on_first_step=capture,
    )

    assert snap["student"].keys() == snap["teacher"].keys()
    for name, tensor in snap["student"].items():
        # Deleting the fresh-run copy makes every independently seeded value differ.
        assert torch.equal(tensor, snap["teacher"][name]), name
        assert snap["student_ptr"][name] != snap["teacher_ptr"][name], name
    assert snap["teacher_frozen"] is True
    assert snap["student_trainable"] is True
    # the projector is copied too, unlike the replaced register-only helper
    assert "projector" in snap["student"] or any(
        name.startswith("projector") for name in snap["student"]
    )


# --- repeated relational resume chains -------------------------------------

# A resume hop must omit the relational requests when the checkpoint's own
# recipe differs from the pinned historical fixture defaults.
_OMITTED_RECIPE = dict(
    relational_teacher_views=None,
    relational_center_source=None,
    relational_pairing=None,
    teacher_temp_start=None,
    teacher_temp_end=None,
)


def _saved_epoch(out_dir, epoch):
    return torch.load(
        out_dir / f"SSL_epoch_{epoch:04d}.pth", map_location="cpu", weights_only=False
    )


def _assert_relational_hop(checkpoint, *, views, pairing, center, start, end, weak):
    config = checkpoint["config"]
    bundle = config[trainer.RELATIONAL_TARGET_CONFIG_KEY]
    assert set(bundle) == set(trainer.RELATIONAL_BUNDLE_KEYS)
    assert bundle["version"] == 1
    assert bundle["relational_teacher_views"] == views
    assert bundle["relational_center_source"] == center
    assert bundle["relational_pairing"] == pairing
    assert bundle["teacher_temp_start"] == pytest.approx(start)
    assert bundle["teacher_temp_end"] == pytest.approx(end)
    if weak:
        assert bundle["weak_teacher_augmentation_config"] is not None
        assert bundle["weak_teacher_augmentation_config"]["normalize"] == {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    else:
        assert bundle["weak_teacher_augmentation_config"] is None
    args = checkpoint["args"]
    for key in trainer.RELATIONAL_ARGS_KEYS:
        assert args[key] == bundle[key], key
    for key in (
        "include_weak_teacher_views",
        "weak_relational_teacher_views",
        "disable_cross_view_loss",
        "pretrain_recipe",
    ):
        assert key not in args, key
    return bundle


def test_relational_legacy_checkpoint_resaves_twice(tmp_path, monkeypatch):
    """Legacy -> resume -> save -> resume -> save keeps the historical recipe."""
    for mixed in (False, True):
        legacy = _make_legacy_checkpoint(
            tmp_path, monkeypatch, mixed_generation=mixed
        )
        first = _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label=f"_hop1_{mixed}",
            resume=str(legacy),
            max_epochs=2,
        )
        hop1 = _saved_epoch(first["out_dir"], 2)
        bundle1 = _assert_relational_hop(
            hop1,
            views="ordinary",
            pairing="full",
            center="ordinary",
            start=0.04,
            end=0.07,
            weak=False,
        )
        assert bundle1 == hop1["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY]

        second = _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label=f"_hop2_{mixed}",
            resume=str(first["out_dir"] / "SSL_epoch_0002.pth"),
            max_epochs=3,
        )
        hop2 = _saved_epoch(second["out_dir"], 3)
        _assert_relational_hop(
            hop2,
            views="ordinary",
            pairing="full",
            center="ordinary",
            start=0.04,
            end=0.07,
            weak=False,
        )
        assert hop2["args"]["teacher_temp_warmup_ratio"] == pytest.approx(0.7)


def test_relational_legacy_same_pairing_survives_two_hops(tmp_path, monkeypatch):
    legacy = _make_legacy_checkpoint(tmp_path, monkeypatch)
    checkpoint = _load_ckpt(
        _run_tiny_pretrain(tmp_path, monkeypatch, label="_pre_same")
    )
    from otuformer.utils.checkpoint import load_checkpoint

    saved = load_checkpoint(legacy)
    saved.pop("rng_state", None)
    saved["config"].pop(trainer.RELATIONAL_TARGET_CONFIG_KEY, None)
    saved["args"]["disable_cross_view_loss"] = True
    _save_ckpt(saved, legacy)
    assert checkpoint is not None

    first = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_same_1",
        resume=str(legacy),
        max_epochs=2,
        **_OMITTED_RECIPE,
    )
    _assert_relational_hop(
        _saved_epoch(first["out_dir"], 2),
        views="ordinary",
        pairing="same",
        center="ordinary",
        start=0.04,
        end=0.07,
        weak=False,
    )
    second = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_same_2",
        resume=str(first["out_dir"] / "SSL_epoch_0002.pth"),
        max_epochs=3,
        **_OMITTED_RECIPE,
    )
    _assert_relational_hop(
        _saved_epoch(second["out_dir"], 3),
        views="ordinary",
        pairing="same",
        center="ordinary",
        start=0.04,
        end=0.07,
        weak=False,
    )


def test_relational_nondefault_ratio_resaves_twice(tmp_path, monkeypatch):
    """A recorded 0.3 survives two real re-saves and stays authoritative."""
    start = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_ratio_a",
        teacher_temp_warmup_ratio=0.3,
        teacher_temp_warmup_ratio_explicit=True,
        relational_teacher_views="weak",
        relational_pairing="cross",
    )
    first = _saved_epoch(start["out_dir"], 1)
    assert first["args"]["teacher_temp_warmup_ratio"] == pytest.approx(0.3)
    bundle = first["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY]
    assert "teacher_temp_warmup_ratio" not in bundle

    hop1 = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_ratio_b",
        resume=str(start["out_dir"] / "SSL_epoch_0001.pth"),
        max_epochs=2,
        **_OMITTED_RECIPE,
    )
    assert _saved_epoch(hop1["out_dir"], 2)["args"][
        "teacher_temp_warmup_ratio"
    ] == pytest.approx(0.3)
    hop2 = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_ratio_c",
        resume=str(hop1["out_dir"] / "SSL_epoch_0002.pth"),
        max_epochs=3,
        teacher_temp_warmup_ratio=0.3,
        teacher_temp_warmup_ratio_explicit=True,
        **_OMITTED_RECIPE,
    )
    assert _saved_epoch(hop2["out_dir"], 3)["args"][
        "teacher_temp_warmup_ratio"
    ] == pytest.approx(0.3)

    with pytest.raises(ValueError, match="teacher-temp-warmup-ratio"):
        _run_tiny_pretrain(
            tmp_path,
            monkeypatch,
            label="_ratio_conflict",
            resume=str(hop1["out_dir"] / "SSL_epoch_0002.pth"),
            max_epochs=4,
            teacher_temp_warmup_ratio=0.7,
            teacher_temp_warmup_ratio_explicit=True,
            **_OMITTED_RECIPE,
        )


def test_relational_resume_preserves_distinct_encoders_twice(tmp_path, monkeypatch):
    """No hop may re-synchronize a resumed student/teacher pair."""
    start = _run_tiny_pretrain(
        tmp_path, monkeypatch, label="_enc_a", relational_teacher_views="weak"
    )
    path = start["out_dir"] / "SSL_epoch_0001.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Make the saved pair deliberately unequal so a copy would be visible.
    student_state = dict(checkpoint["student"])
    teacher_state = dict(checkpoint["teacher"])
    key = next(iter(student_state))
    teacher_state[key] = student_state[key] + 1.0
    checkpoint["teacher"] = teacher_state
    checkpoint["model_state_dict"] = teacher_state
    checkpoint["center"] = checkpoint["center"] + 5.0
    _save_ckpt(checkpoint, path)

    hop1 = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_enc_b",
        resume=str(path),
        max_epochs=2,
        **_OMITTED_RECIPE,
    )
    first = _saved_epoch(hop1["out_dir"], 2)
    assert not torch.equal(first["student"][key], first["teacher"][key])

    hop2 = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_enc_c",
        resume=str(hop1["out_dir"] / "SSL_epoch_0002.pth"),
        max_epochs=3,
        **_OMITTED_RECIPE,
    )
    second = _saved_epoch(hop2["out_dir"], 3)
    assert not torch.equal(second["student"][key], second["teacher"][key])
    assert trainer.RELATIONAL_TARGET_CONFIG_KEY in second["config"]
    assert "initialization" not in second["config"]


def test_relational_out_of_band_endpoint_survives_two_hops(tmp_path, monkeypatch):
    """0.0009 is format-legal: inherited, re-saved unchanged, and never banded."""
    start = _run_tiny_pretrain(tmp_path, monkeypatch, label="_band_a")
    path = start["out_dir"] / "SSL_epoch_0001.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    bundle = checkpoint["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY]
    bundle["teacher_temp_start"] = 0.0009
    checkpoint["args"]["teacher_temp_start"] = 0.0009
    _save_ckpt(checkpoint, path)

    hop1 = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_band_b",
        resume=str(path),
        max_epochs=2,
        **_OMITTED_RECIPE,
    )
    first = _saved_epoch(hop1["out_dir"], 2)
    assert first["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY][
        "teacher_temp_start"
    ] == 0.0009
    assert first["args"]["teacher_temp_start"] == 0.0009

    hop2 = _run_tiny_pretrain(
        tmp_path,
        monkeypatch,
        label="_band_c",
        resume=str(hop1["out_dir"] / "SSL_epoch_0002.pth"),
        max_epochs=3,
        **_OMITTED_RECIPE,
    )
    second = _saved_epoch(hop2["out_dir"], 3)
    assert second["config"][trainer.RELATIONAL_TARGET_CONFIG_KEY][
        "teacher_temp_start"
    ] == 0.0009
    assert second["args"]["teacher_temp_start"] == 0.0009
