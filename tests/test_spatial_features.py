"""Verified complete-block feature grids and bilinear grid sampling."""

from __future__ import annotations

import pytest
import torch

from otuformer.training.model import OTUFormerEncoder
from otuformer.vision.features import (
    FeatureContract,
    LayerSpec,
    SpatialFeatureForward,
    describe_feature_layers,
    get_module_by_name,
    resolve_feature_contract,
    resolve_feature_layers,
    sample_grid_features,
)


def tiny_encoder(reg_tokens: int | None = None, image_size: int = 32) -> OTUFormerEncoder:
    torch.manual_seed(0)
    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=8,
        pretrained=False,
        img_size=image_size,
        reg_tokens=reg_tokens,
    )
    return encoder.eval()


def contract_for(
    encoder: OTUFormerEncoder,
    layers=("blocks.0", "blocks.1"),
    *,
    norm: str = "backbone",
    l2: bool = False,
    image_size: int = 32,
) -> FeatureContract:
    return resolve_feature_contract(
        encoder,
        image_size,
        layers,
        norm=norm,
        l2=l2,
        model_identity={"architecture": "tiny", "weights": "random-test"},
        normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )


def fixed_input(size: int = 32, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator)


def block_grid_expected(
    encoder: OTUFormerEncoder, x: torch.Tensor, index: int, *, norm: str
) -> torch.Tensor:
    """Test oracle: run the real modules and record one block's output."""
    backbone = encoder.backbone
    patches = backbone.patch_embed(x)
    tokens = backbone._pos_embed(patches)
    tokens = backbone.patch_drop(tokens)
    tokens = backbone.norm_pre(tokens)
    captured = None
    for position, block in enumerate(backbone.blocks):
        tokens = block(tokens)
        if position == index:
            captured = tokens
            break
    assert captured is not None
    if norm == "backbone":
        captured = backbone.norm(captured)
    prefix = int(backbone.num_prefix_tokens)
    patch = captured[:, prefix:, :]
    grid_h = int(x.shape[-2]) // int(backbone.patch_embed.patch_size[0])
    grid_w = int(x.shape[-1]) // int(backbone.patch_embed.patch_size[1])
    batch, channels = int(patch.shape[0]), int(patch.shape[-1])
    return patch.reshape(batch, grid_h, grid_w, channels)


def test_selected_blocks_equal_real_module_outputs():
    encoder = tiny_encoder()
    x = fixed_input()
    contract = contract_for(encoder)
    forward = SpatialFeatureForward(encoder, contract)

    with torch.no_grad():
        outputs = forward(x)

    assert len(outputs) == 2
    for order, index in enumerate((0, 1)):
        expected = block_grid_expected(encoder, x, index, norm="backbone")
        assert outputs[order].shape == expected.shape
        assert outputs[order].shape[:2] == (1, 2)  # 32x32 input, patch 16 -> 2x2
        torch.testing.assert_close(outputs[order], expected)
        assert torch.isfinite(outputs[order]).all()


def test_final_block_matches_ordinary_final_spatial_output():
    encoder = tiny_encoder()
    x = fixed_input()
    contract = contract_for(encoder, layers=("blocks.11",))
    forward = SpatialFeatureForward(encoder, contract)
    with torch.no_grad():
        (grid,) = forward(x)
        tokens = encoder.backbone.forward_features(x)
    prefix = int(encoder.backbone.num_prefix_tokens)
    patch = tokens[:, prefix:, :]
    channels = patch.shape[-1]
    expected = patch.reshape(1, 2, 2, channels)
    torch.testing.assert_close(grid, expected)


def test_request_order_is_preserved_and_blocks_run_once():
    encoder = tiny_encoder()
    x = fixed_input()
    calls: list[int] = []
    handles = [
        block.register_forward_hook(
            lambda module, inputs, output, index=index: calls.append(index)
        )
        for index, block in enumerate(encoder.backbone.blocks)
    ]
    try:
        contract = contract_for(encoder, layers=("blocks.1", "blocks.0"))
        with torch.no_grad():
            outputs = SpatialFeatureForward(encoder, contract)(x)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == list(range(12))  # one execution per block, no fusion
    expected_first = block_grid_expected(encoder, x, 1, norm="backbone")
    expected_second = block_grid_expected(encoder, x, 0, norm="backbone")
    torch.testing.assert_close(outputs[0], expected_first)
    torch.testing.assert_close(outputs[1], expected_second)


def test_register_tokens_never_become_pixels():
    encoder = tiny_encoder(reg_tokens=4)
    assert int(encoder.backbone.num_prefix_tokens) == 5
    x = fixed_input()
    contract = contract_for(encoder, layers=("blocks.1",))
    assert contract.prefix_tokens == 5
    with torch.no_grad():
        (grid,) = SpatialFeatureForward(encoder, contract)(x)
    expected = block_grid_expected(encoder, x, 1, norm="backbone")
    assert grid.shape == (1, 2, 2, expected.shape[-1])
    torch.testing.assert_close(grid, expected)


def test_norm_metadata_records_resolved_module_identity():
    encoder = tiny_encoder()
    contract = contract_for(encoder)
    norm = encoder.backbone.norm
    qualname = f"{type(norm).__module__}.{type(norm).__qualname__}"
    assert contract.norm == "backbone"
    assert contract.norm_module == {"path": "norm", "qualname": qualname}

    none_contract = contract_for(encoder, layers=("blocks.0",), norm="none")
    assert none_contract.norm_module is None
    with torch.no_grad():
        (raw,) = SpatialFeatureForward(encoder, none_contract)(fixed_input())
    expected = block_grid_expected(encoder, fixed_input(), 0, norm="none")
    torch.testing.assert_close(raw, expected)


def test_missing_or_incompatible_norm_is_rejected():
    encoder = tiny_encoder()
    encoder.backbone.norm = torch.nn.Identity()
    with pytest.raises(ValueError, match="per-token normalization"):
        contract_for(encoder)

    encoder = tiny_encoder()
    del encoder.backbone.norm
    with pytest.raises(ValueError, match="final norm"):
        contract_for(encoder)


def test_norm_and_l2_formulas_and_zero_vector():
    encoder = tiny_encoder()
    x = fixed_input()
    plain = contract_for(encoder, layers=("blocks.1",), norm="none")
    with torch.no_grad():
        (raw,) = SpatialFeatureForward(encoder, plain)(x)
    normalized = contract_for(encoder, layers=("blocks.1",), norm="none", l2=True)
    with torch.no_grad():
        (unit,) = SpatialFeatureForward(encoder, normalized)(x)
    expected = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-12)
    torch.testing.assert_close(unit, expected)

    # A zero token normalizes to zero rather than NaN/inf.
    zeros = torch.zeros(1, 1, 1, 4)
    with torch.no_grad():
        norms = torch.linalg.vector_norm(zeros, dim=-1, keepdim=True).clamp_min(1e-12)
        normalized_zero = zeros / norms
    assert torch.count_nonzero(normalized_zero) == 0


def test_identity_forward_repeat_is_an_engineering_check():
    encoder = tiny_encoder()
    x = fixed_input()
    contract = contract_for(encoder, layers=("blocks.11",))
    forward = SpatialFeatureForward(encoder, contract)
    with torch.no_grad():
        first = forward(x)[0]
        second = forward(x)[0]
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


def test_unsupported_layer_requests_fail_before_forward():
    encoder = tiny_encoder()
    for bad in ("blocks.99", "blocks", "norm", "blocks.0.norm1", "blocks.0,blocks.0", ""):
        if bad == "":
            with pytest.raises(ValueError, match="empty entry"):
                resolve_feature_layers(encoder.backbone, "blocks.0,")
            continue
        with pytest.raises(ValueError):
            resolve_feature_layers(encoder.backbone, bad)


def test_omitted_layer_list_resolves_to_final_block():
    encoder = tiny_encoder()
    assert resolve_feature_layers(encoder.backbone, None) == ("blocks.11",)
    assert resolve_feature_layers(encoder.backbone, "  ") == ("blocks.11",)
    assert resolve_feature_layers(encoder.backbone, " blocks.0 , blocks.3 ") == (
        "blocks.0",
        "blocks.3",
    )


def test_mismatched_input_size_and_patch_padding_are_rejected():
    encoder = tiny_encoder(image_size=32)
    contract = contract_for(encoder)
    with pytest.raises(ValueError, match="fixes 32x32"):
        SpatialFeatureForward(encoder, contract)(torch.zeros(1, 3, 64, 64))
    with pytest.raises(ValueError, match="not a whole multiple"):
        resolve_feature_contract(encoder, 40, ("blocks.0",), model_identity={})
    encoder.backbone.patch_embed.dynamic_img_pad = True
    with pytest.raises(ValueError, match="dynamic patch padding"):
        contract_for(encoder)


def test_contract_round_trips_through_json():
    encoder = tiny_encoder()
    contract = contract_for(encoder, l2=True)
    payload = contract.to_json()
    restored = FeatureContract.from_json(payload)
    assert restored == contract
    assert restored.layer_paths == ("blocks.0", "blocks.1")
    assert restored.layers[0] == LayerSpec(
        path="blocks.0", block_index=0, grid_hw=(2, 2), channels=192
    )

    tampered = dict(payload)
    tampered["extra_field"] = 1
    with pytest.raises(ValueError, match="unexpected field"):
        FeatureContract.from_json(tampered)
    missing = dict(payload)
    missing.pop("l2")
    with pytest.raises(ValueError, match="missing field"):
        FeatureContract.from_json(missing)


def test_describe_feature_layers_marks_only_complete_blocks():
    encoder = tiny_encoder()
    rows = describe_feature_layers(encoder.backbone)
    eligible = [row["path"] for row in rows if row["spatial_eligible"]]
    assert eligible == [f"blocks.{index}" for index in range(12)]
    assert all(row["block_index"] == index for index, row in enumerate(
        row for row in rows if row["spatial_eligible"]
    ))
    paths = {row["path"] for row in rows}
    assert "blocks.0.norm1" in paths
    assert any(row["path"] == "norm" for row in rows)
    assert get_module_by_name(encoder.backbone, "blocks.3") is encoder.backbone.blocks[3]


def test_sampling_fixture_matches_documented_lattice():
    grid = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
    xy = torch.tensor([[[24.0, 40.0]]])
    values, usable = sample_grid_features(
        grid, xy, patch_hw=(16, 16), valid=torch.ones(1, 4, 4, dtype=torch.bool)
    )
    torch.testing.assert_close(values, torch.tensor([[[9.0]]]))
    assert usable.tolist() == [[True]]


def test_sampling_first_last_centers_and_interior():
    grid = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    # First patch center (8, 8) -> index (0, 0); last center (56, 56) -> (3, 3).
    xy = torch.tensor([[[8.0, 8.0], [56.0, 56.0]]])
    values, usable = sample_grid_features(grid, xy, patch_hw=(16, 16), valid=valid)
    torch.testing.assert_close(values.flatten(), torch.tensor([0.0, 15.0]))
    assert usable.tolist() == [[True, True]]

    # Halfway between cells (2,1) and (2,2) along x only -> mean of 9 and 10.
    xy = torch.tensor([[[32.0, 40.0]]])
    values, _ = sample_grid_features(grid, xy, patch_hw=(16, 16), valid=valid)
    torch.testing.assert_close(values, torch.tensor([[[9.5]]]))
    # A true interior point averages all four contributing cells.
    xy = torch.tensor([[[32.0, 32.0]]])
    values, _ = sample_grid_features(grid, xy, patch_hw=(16, 16), valid=valid)
    torch.testing.assert_close(values, torch.tensor([[[7.5]]]))
    # Exact cell center (2, 2) keeps the stored value.
    xy = torch.tensor([[[40.0, 40.0]]])
    values, _ = sample_grid_features(grid, xy, patch_hw=(16, 16), valid=valid)
    torch.testing.assert_close(values, torch.tensor([[[10.0]]]))


def test_sampling_multichannel_layout_and_invalid_support():
    batch, height, width, channels = 1, 4, 4, 3
    grid = torch.zeros(batch, height, width, channels)
    for row in range(height):
        for column in range(width):
            grid[0, row, column] = torch.tensor(
                [float(row), float(column), float(row * 10 + column)]
            )
    valid = torch.ones(batch, height, width, dtype=torch.bool)
    # Exact center of cell (2, 1): every channel keeps its own axis.
    values, usable = sample_grid_features(
        grid, torch.tensor([[[24.0, 40.0]]]), patch_hw=(16, 16), valid=valid
    )
    torch.testing.assert_close(values, torch.tensor([[[2.0, 1.0, 21.0]]]))
    assert usable.tolist() == [[True]]

    # A nonzero-weight neighbour that is invalid rejects the sample.
    invalid = valid.clone()
    invalid[0, 2, 2] = False
    midpoint = torch.tensor([[[32.0, 32.0]]])
    _, usable = sample_grid_features(grid, midpoint, patch_hw=(16, 16), valid=invalid)
    assert usable.tolist() == [[False]]

    # A zero-weight neighbour outside the grid is harmless.
    _, usable = sample_grid_features(
        grid, torch.tensor([[[56.0, 56.0]]]), patch_hw=(16, 16), valid=valid
    )
    assert usable.tolist() == [[True]]

    # Out-of-bounds support is unavailable, not zero-filled evidence.
    _, usable = sample_grid_features(
        grid, torch.tensor([[[64.0 + 0.5, 8.0]]]), patch_hw=(16, 16), valid=valid
    )
    assert usable.tolist() == [[False]]


def test_sampling_gradients_flow_through_grid_and_points():
    grid = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4, 1).requires_grad_(True)
    xy = torch.tensor([[[32.0, 32.0]]], dtype=torch.float64, requires_grad=True)
    values, _ = sample_grid_features(
        grid, xy, patch_hw=(16, 16), valid=torch.ones(1, 4, 4, dtype=torch.bool)
    )
    values.sum().backward()
    assert grid.grad is not None and xy.grad is not None
    assert torch.isfinite(grid.grad).all()
    torch.testing.assert_close(values, torch.tensor([[[7.5]]], dtype=torch.float64))


@pytest.mark.parametrize("source_kind", ["current", "legacy", "teacher-student"])
def test_checkpoint_sources_rebuild_and_read_full_grids(tmp_path, source_kind):
    """Representative saved sources keep their verified layout through the loader."""
    from otuformer.utils.checkpoint import load_checkpoint_encoder, resolve_checkpoint

    written = tiny_encoder()
    state = written.state_dict()
    config = {
        "model_name": "vit_tiny_patch16_224",
        "out_dim": 8,
        "image_size": 32,
    }
    if source_kind == "current":
        checkpoint = {"model_state_dict": state, "config": config}
    elif source_kind == "legacy":
        checkpoint = {
            "model": state,
            "args": {"model_name": "vit_tiny_patch16_224", "out_dim": 8},
        }
    else:
        checkpoint = {
            "teacher": state,
            "student": state,
            "model_state_dict": state,
            "config": config,
        }

    architecture = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")
    encoder, resolved = load_checkpoint_encoder(
        checkpoint, architecture, image_size=32
    )
    encoder.eval()
    contract = resolve_feature_contract(
        encoder,
        32,
        ("blocks.0", "blocks.11"),
        model_identity={"architecture": source_kind},
    )
    x = fixed_input()
    with torch.no_grad():
        outputs = SpatialFeatureForward(encoder, contract)(x)
    assert len(outputs) == 2
    assert outputs[0].shape == (1, 2, 2, 192)
    assert outputs[1].shape == (1, 2, 2, 192)
    assert resolved.actual_register_tokens == 0


def test_forward_preserves_model_mode_and_rng():
    encoder = tiny_encoder()
    encoder.train()
    torch.manual_seed(11)
    before = torch.random.get_rng_state().clone()
    contract = contract_for(encoder, layers=("blocks.0",))
    forward = SpatialFeatureForward(encoder, contract)
    with torch.no_grad():
        forward(fixed_input())
    assert encoder.training is True
    assert torch.equal(torch.random.get_rng_state(), before)


def test_contract_values_are_validated_not_coerced():
    """A stored contract describes a representation; malformed values are refused."""
    from otuformer.vision.features import FeatureContract

    encoder = tiny_encoder()
    payload = contract_for(encoder, layers=("blocks.0",)).to_json()

    cases = (
        ({"norm": "banana"}, "norm must be one of"),
        ({"epsilon": -1.0}, "epsilon must be finite and positive"),
        ({"l2": "false"}, "l2 must be a Boolean"),
        ({"prefix_tokens": -1}, "prefix_tokens must be non-negative"),
        ({"prefix_tokens": 1.5}, "prefix_tokens must be an integer"),
        ({"patch_hw": [0, 0]}, "patch_hw must be positive"),
        ({"patch_hw": [24, 24]}, "does not tile"),
        ({"norm_module": None}, "needs the resolved norm_module"),
        (
            {"layers": [dict(payload["layers"][0], block_index=9)]},
            "does not match block_index",
        ),
    )
    for patch, match in cases:
        with pytest.raises(ValueError, match=match):
            FeatureContract.from_json({**payload, **patch})


def test_sampling_reports_non_finite_and_unrepresentable_coordinates_unusable():
    grid = torch.zeros(1, 4, 4, 1)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    # Finite in float64 but overflowing the float32 grid.
    xy = torch.tensor(
        [[[float("nan"), 8.0], [float("inf"), 8.0], [1e100, 24.0], [8.0, 8.0]]],
        dtype=torch.float64,
    )
    _values, usable = sample_grid_features(grid, xy, patch_hw=(16, 16), valid=valid)
    assert usable.tolist() == [[False, False, False, True]]


def test_norm_module_identity_is_a_fixed_schema():
    from otuformer.vision.features import FeatureContract

    encoder = tiny_encoder()
    payload = contract_for(encoder, layers=("blocks.0",)).to_json()
    for module, match in (
        ({}, "must hold exactly"),
        ({"path": "norm"}, "must hold exactly"),
        ({"bogus": "value"}, "must hold exactly"),
        ({"path": "", "qualname": "q"}, "non-empty string"),
        ({"path": "norm", "qualname": "q", "extra": "x"}, "must hold exactly"),
    ):
        with pytest.raises(ValueError, match=match):
            FeatureContract.from_json({**payload, "norm_module": module})
    with pytest.raises(ValueError, match="must not record a norm_module"):
        FeatureContract.from_json({**payload, "norm": "none"})
