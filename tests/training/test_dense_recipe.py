"""Dense settings resolution, closed bundles and strict resume validation."""

from __future__ import annotations

import argparse

import pytest
import torch

from otuformer.training import dense_recipe as recipe


def _args(**overrides):
    base = dict(
        coordinate_weight=0.0,
        coordinate_weight_explicit=False,
        ranking_weight=0.0,
        ranking_weight_explicit=False,
        dense_query_points=49,
        dense_query_points_explicit=False,
        ranking_reference_points=16,
        ranking_reference_points_explicit=False,
    )
    for key in list(overrides):
        if key.endswith("_explicit"):
            base[key] = overrides[key]
        else:
            base[key] = overrides[key]
            # Supplying a value in a test means "the operator asked for this".
            base[f"{key}_explicit"] = True
    return argparse.Namespace(**base)


def test_all_off_is_disabled():
    settings = recipe.resolve_dense_settings(_args(), None)
    assert settings.enabled is False
    assert settings.ranking_enabled is False
    assert settings.reference_cap is None


def test_coordinate_only_resolution():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.5), None)
    assert settings.enabled is True and settings.ranking_enabled is False
    assert settings.reference_cap is None
    assert settings.query_cap == 49


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_weights_must_be_finite_and_nonnegative(bad):
    with pytest.raises(ValueError, match="coordinate_weight"):
        recipe.resolve_dense_settings(_args(coordinate_weight=bad), None)


def test_bool_and_float_weights_are_rejected():
    with pytest.raises(ValueError, match="coordinate_weight"):
        recipe.resolve_dense_settings(_args(coordinate_weight=True), None)
    with pytest.raises(ValueError, match="ranking_weight"):
        recipe.resolve_dense_settings(_args(ranking_weight="1.0"), None)


@pytest.mark.parametrize("cap", [0, 50, -1])
def test_query_cap_bounds(cap):
    with pytest.raises(ValueError, match="dense_query_points"):
        recipe.resolve_dense_settings(
            _args(coordinate_weight=1.0, dense_query_points=cap), None
        )


@pytest.mark.parametrize("cap", [1, 50])
def test_reference_cap_bounds(cap):
    with pytest.raises(ValueError, match="ranking_reference_points"):
        recipe.resolve_dense_settings(
            _args(ranking_weight=1.0, ranking_reference_points=cap), None
        )


def test_enabled_run_records_both_weights_and_objectives():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.3), None)
    bundle = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    assert bundle["objectives"] == ["coordinate"]
    assert bundle["ranking_weight"] == 0.0
    assert "ranking_config" not in bundle
    assert bundle["version"] == 1


def test_ranking_bundle_requires_dependency_record():
    settings = recipe.resolve_dense_settings(_args(ranking_weight=0.3), None)
    with pytest.raises(ValueError, match="dependency"):
        recipe.dense_config_bundle(
            settings, batch_size=8, steps_per_epoch=4, dependency=None
        )
    bundle = recipe.dense_config_bundle(
        settings,
        batch_size=8,
        steps_per_epoch=4,
        dependency={
            "name": "diffsort",
            "version": "0.2.0",
            "wheel_sha256": "a" * 64,
            "source_sha256": {"diffsort/x.py": "b" * 64},
        },
    )
    assert bundle["objectives"] == ["ranking"]
    assert bundle["ranking_config"]["reference_cap"] == 16
    assert bundle["ranking_config"]["head"]["hidden_dim"] == 2048


def test_all_off_bundle_is_none():
    settings = recipe.resolve_dense_settings(_args(), None)
    assert recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    ) is None


def test_off_checkpoint_omits_both_bundles():
    settings = recipe.resolve_dense_settings(_args(), None)
    assert recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    ) is None
    runtime = recipe.new_dense_runtime(settings, seed=1)
    assert runtime is None


def test_rng_streams_are_separate_and_deterministic():
    settings = recipe.resolve_dense_settings(_args(ranking_weight=0.1), None)
    first = recipe.new_dense_runtime(settings, seed=7)
    second = recipe.new_dense_runtime(settings, seed=7)
    assert torch.equal(first.query_generator.get_state(), second.query_generator.get_state())
    assert first.reference_generator is not None
    # The two streams must not be the same generator state.
    assert not torch.equal(
        first.query_generator.get_state(), first.reference_generator.get_state()
    )
    other = recipe.new_dense_runtime(settings, seed=8)
    assert not torch.equal(first.query_generator.get_state(), other.query_generator.get_state())


def test_state_bundle_round_trip():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.2), None)
    runtime = recipe.new_dense_runtime(settings, seed=3)
    runtime.query_generator.manual_seed(99)
    state = recipe.dense_state_bundle(runtime, completed_steps=5, ranking=None)
    assert state["version"] == 1 and state["completed_steps"] == 5
    assert "ranking" not in state
    restored = recipe.restore_dense_runtime(settings, state)
    assert torch.equal(restored.query_generator.get_state(), runtime.query_generator.get_state())


def test_state_bundle_rejects_ranking_key_when_disabled():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.2), None)
    runtime = recipe.new_dense_runtime(settings, seed=3)
    state = recipe.dense_state_bundle(runtime, completed_steps=1, ranking=None)
    state["ranking"] = {}
    with pytest.raises(ValueError, match="ranking"):
        recipe.restore_dense_runtime(settings, state)


def test_resume_inherits_omitted_and_rejects_conflict():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    bundle = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    # Omitted request inherits the saved weight.
    inherited = recipe.resolve_dense_settings(_args(), {"config": {"dense_training_config": bundle}})
    assert inherited.coordinate_weight == pytest.approx(0.4)

    with pytest.raises(ValueError, match="coordinate_weight"):
        recipe.resolve_dense_settings(
            _args(coordinate_weight=0.9),
            {"config": {"dense_training_config": bundle}},
        )


def test_resume_cannot_enable_a_disabled_objective():
    off = recipe.resolve_dense_settings(_args(), None)
    assert recipe.dense_config_bundle(
        off, batch_size=8, steps_per_epoch=4, dependency=None
    ) is None
    with pytest.raises(ValueError, match="dense"):
        recipe.resolve_dense_settings(
            _args(coordinate_weight=0.5), {"config": {}, "version": 1}
        )


def test_validate_resume_checks_batch_and_steps():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=3, ranking=None
    )
    recipe.validate_dense_resume(
        config, state, iteration=3, batch_size=8, steps_per_epoch=4
    )
    with pytest.raises(ValueError, match="batch"):
        recipe.validate_dense_resume(
            config, state, iteration=3, batch_size=16, steps_per_epoch=4
        )
    with pytest.raises(ValueError, match="step"):
        recipe.validate_dense_resume(
            config, state, iteration=4, batch_size=8, steps_per_epoch=4
        )


def _dependency(**overrides):
    record = {
        "name": "diffsort",
        "version": "0.2.0",
        "wheel_sha256": "a" * 64,
        "source_sha256": {
            "diffsort/diffsort.py": "b" * 64,
            "diffsort/functional.py": "c" * 64,
            "diffsort/networks.py": "d" * 64,
        },
    }
    record.update(overrides)
    return record


class _Head(torch.nn.Sequential):
    """A shape-faithful three-Linear head producing 0/2/4.* state keys."""

    def __init__(self):
        super().__init__(
            torch.nn.Linear(6, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 3),
        )


class _FakeRanking:
    def __init__(self):
        self.student_head = _Head()
        self.teacher_head = _Head()


def _ranking_pair(*, contributed=2, completed=2, dependency=None):
    settings = recipe.resolve_dense_settings(_args(ranking_weight=0.5), None)
    config = recipe.dense_config_bundle(
        settings,
        batch_size=8,
        steps_per_epoch=4,
        dependency=dependency or _dependency(),
    )
    runtime = recipe.new_dense_runtime(settings, seed=1)
    runtime.ranking_contributed_steps = contributed
    state = recipe.dense_state_bundle(
        runtime, completed_steps=completed, ranking=_FakeRanking()
    )
    return settings, config, state


@pytest.mark.parametrize(
    "name",
    [
        "rng_policy",
        "geometry_policy",
        "support_policy",
        "query_policy",
        "reduction_policy",
        "coordinate_features",
    ],
)
def test_validate_resume_checks_every_policy_identifier(name):
    """An unknown or changed sampling/support/reduction identity must not resume."""
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=3, ranking=None
    )
    recipe.validate_dense_resume(
        config, state, iteration=3, batch_size=8, steps_per_epoch=4
    )
    config[name] = "some-other-policy-v9"
    with pytest.raises(ValueError, match=name):
        recipe.validate_dense_resume(
            config, state, iteration=3, batch_size=8, steps_per_epoch=4
        )


def test_validate_resume_compares_the_loaded_dependency_identity():
    _settings, config, state = _ranking_pair()
    # Matching identity resumes.
    recipe.validate_dense_resume(
        config,
        state,
        iteration=2,
        batch_size=8,
        steps_per_epoch=4,
        dependency=_dependency(),
    )
    # A modified loaded source is not an equivalent recipe, even at the same version.
    modified = _dependency(
        source_sha256={
            "diffsort/diffsort.py": "e" * 64,
            "diffsort/functional.py": "c" * 64,
            "diffsort/networks.py": "d" * 64,
        }
    )
    with pytest.raises(ValueError, match="implementation"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=modified,
        )
    # A different release is rejected on the version too.
    with pytest.raises(ValueError, match="version"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(version="0.3.0"),
        )


def test_validate_resume_rejects_an_impossible_contribution_count():
    _settings, config, state = _ranking_pair(contributed=999, completed=2)
    with pytest.raises(ValueError, match="contributed_steps"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )
    _settings, config, state = _ranking_pair(contributed=-1, completed=2)
    with pytest.raises(ValueError, match="contributed_steps"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )


def test_validate_resume_rejects_malformed_ranking_head_state():
    _settings, config, state = _ranking_pair()
    broken = {
        key: value for key, value in state["ranking"]["student_head"].items()
    }
    broken.pop("4.bias")
    state["ranking"]["student_head"] = broken
    with pytest.raises(ValueError, match="student_head"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )

    _settings, config, state = _ranking_pair()
    nonfinite = dict(state["ranking"]["student_head"])
    nonfinite["0.weight"] = torch.full_like(nonfinite["0.weight"], float("nan"))
    state["ranking"]["student_head"] = nonfinite
    with pytest.raises(ValueError, match="non-finite"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )


def test_validate_resume_rejects_unknown_and_missing_fields():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=1, ranking=None
    )
    broken = dict(config)
    broken["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        recipe.validate_dense_resume(
            broken, state, iteration=1, batch_size=8, steps_per_epoch=4
        )
    missing = dict(config)
    missing.pop("objectives")
    with pytest.raises(ValueError, match="missing"):
        recipe.validate_dense_resume(
            missing, state, iteration=1, batch_size=8, steps_per_epoch=4
        )


def test_validate_resume_rejects_version_mismatch():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=1, ranking=None
    )
    config["version"] = 2
    with pytest.raises(ValueError, match="version"):
        recipe.validate_dense_resume(
            config, state, iteration=1, batch_size=8, steps_per_epoch=4
        )


def test_bool_is_not_accepted_as_a_number():
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    config["query_cap"] = True
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=1, ranking=None
    )
    with pytest.raises(ValueError):
        recipe.validate_dense_resume(
            config, state, iteration=1, batch_size=8, steps_per_epoch=4
        )


def test_explicit_inactive_options_are_rejected():
    with pytest.raises(ValueError, match="dense-query-points"):
        recipe.resolve_dense_settings(_args(dense_query_points=32), None)
    with pytest.raises(ValueError, match="ranking-reference-points"):
        recipe.resolve_dense_settings(
            _args(coordinate_weight=0.5, ranking_reference_points=8), None
        )


def test_omitted_inactive_options_are_fine():
    settings = recipe.resolve_dense_settings(_args(), None)
    assert settings.enabled is False


def test_new_options_share_their_related_help_panels():
    """Dense options sit with the patch objectives; initialization with recovery."""
    import typer.main as typer_main

    from otuformer.cli.main import app

    command = typer_main.get_group(app).commands["pretrain"]
    panel_of: dict[str, str] = {}
    for parameter in command.params:
        for option in getattr(parameter, "opts", []):
            panel_of[option] = getattr(parameter, "rich_help_panel", None)

    for flag in (
        "--coordinate-weight",
        "--dense-query-points",
        "--ranking-weight",
        "--ranking-reference-points",
        "--patch-loss",
    ):
        assert panel_of[flag] == "Patch objectives", flag
    for flag in ("--init-checkpoint", "--init-source", "--resume", "--overwrite"):
        assert panel_of[flag] == "Recovery and safety", flag
    # The two panels this replaced must be gone entirely.
    assert "Dense objectives" not in set(panel_of.values())
    assert "Initialization" not in set(panel_of.values())
    # --ssl-method stays its own axis: it is not a patch-level term.
    assert panel_of["--ssl-method"] == "CLS objective"


def test_every_pretrain_option_has_a_panel():
    """No option may fall out of the task-grouped help inventory."""
    import typer.main as typer_main

    from otuformer.cli.main import app

    command = typer_main.get_group(app).commands["pretrain"]
    missing = [
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", [])
        if not getattr(parameter, "rich_help_panel", None)
    ]
    assert missing == []


def test_head_optimizer_state_matches_the_contribution_count():
    """The live optimizer state must agree with the saved contribution count."""
    head = _Head()
    parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    loss = sum(parameter.sum() for parameter in parameters)
    loss.backward()
    optimizer.step()

    recipe.validate_ranking_head_optimizer_state(
        optimizer, parameters, contributed_steps=1
    )
    # A count that does not match the real step is rejected.
    with pytest.raises(ValueError, match="AdamW step"):
        recipe.validate_ranking_head_optimizer_state(
            optimizer, parameters, contributed_steps=2
        )
    # A never-contributed head must carry no state at all.
    with pytest.raises(ValueError, match="contribution count is 0"):
        recipe.validate_ranking_head_optimizer_state(
            optimizer, parameters, contributed_steps=0
        )
    # A contributing head without state is not resumable.
    fresh_optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    with pytest.raises(ValueError, match="no restored state"):
        recipe.validate_ranking_head_optimizer_state(
            fresh_optimizer, parameters, contributed_steps=3
        )


def test_validate_resume_rejects_unknown_top_level_state_fields():
    """Preflight must reject what restore_dense_runtime would reject later."""
    _settings, config, state = _ranking_pair()
    state["extra_cursor"] = 3
    with pytest.raises(ValueError, match="unexpected field"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )


@pytest.mark.parametrize(
    "corruption",
    ["negative_second_moment", "boolean_step", "fractional_step", "float64_overflow"],
)
def test_head_optimizer_state_rejects_states_that_break_the_next_update(corruption):
    """Only a genuinely loadable head moment may pass the live validator."""
    head = _Head()
    parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    entry = optimizer.state[parameters[0]]
    if corruption == "negative_second_moment":
        entry["exp_avg_sq"] = -entry["exp_avg_sq"]
    elif corruption == "boolean_step":
        entry["step"] = True
    elif corruption == "fractional_step":
        entry["step"] = torch.tensor(1.75)
    else:
        entry["exp_avg_sq"] = torch.full_like(
            entry["exp_avg_sq"], 1e100, dtype=torch.float64
        )
    with pytest.raises(ValueError):
        recipe.validate_ranking_head_optimizer_state(
            optimizer, parameters, contributed_steps=1
        )


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 2])
def test_version_must_be_the_real_integer(version):
    """``True == 1`` and ``1.0 == 1`` must not be accepted as the schema version."""
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=3, ranking=None
    )
    config["version"] = version
    with pytest.raises(ValueError, match="version"):
        recipe.validate_dense_resume(
            config, state, iteration=3, batch_size=8, steps_per_epoch=4
        )


def test_coordinate_only_state_must_not_carry_a_ranking_key_even_as_none():
    """Presence is the signal: ``ranking: null`` is rejected in preflight too."""
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    config = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=3, ranking=None
    )
    state["ranking"] = None
    with pytest.raises(ValueError, match="ranking"):
        recipe.validate_dense_resume(
            config, state, iteration=3, batch_size=8, steps_per_epoch=4
        )


@pytest.mark.parametrize("mode", ["amsgrad", "foreach"])
def test_head_state_rejected_when_the_real_branch_needs_more(mode):
    """The live rule must follow the group's actual AdamW branch."""
    head = _Head()
    parameters = list(head.parameters())
    kwargs = {"amsgrad": True} if mode == "amsgrad" else {"foreach": True}
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, **kwargs)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    entry = optimizer.state[parameters[0]]
    if mode == "amsgrad":
        entry.pop("max_exp_avg_sq", None)
        with pytest.raises(ValueError, match="max_exp_avg_sq"):
            recipe.validate_ranking_head_optimizer_state(
                optimizer, parameters, contributed_steps=1
            )
    else:
        entry["step"] = torch.tensor(1, dtype=torch.int64)
        with pytest.raises(ValueError, match="foreach"):
            recipe.validate_ranking_head_optimizer_state(
                optimizer, parameters, contributed_steps=1
            )


def test_head_weights_must_be_finite_at_the_training_dtype():
    """A float64 head value that overflows float32 is rejected in preflight."""
    _settings, config, state = _ranking_pair()
    overflow = dict(state["ranking"]["student_head"])
    overflow["0.bias"] = overflow["0.bias"].to(torch.float64) * 0.0 + 1e100
    assert bool(torch.isfinite(overflow["0.bias"]).all())
    state["ranking"]["student_head"] = overflow
    with pytest.raises(ValueError, match="training float32"):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )


def test_execution_flags_follow_the_resolved_device_not_the_saved_one():
    """A CUDA/MPS-only capturable step must be rejected when this run is CPU."""
    head = _Head()
    parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    # The non-fused capturable branch asserts CUDA; the CPU target cannot run it.
    optimizer.param_groups[0]["capturable"] = True
    for entry in optimizer.state.values():
        entry["step"] = entry["step"].to("cpu")
    with pytest.raises(ValueError):
        recipe.validate_ranking_head_optimizer_state(
            optimizer, parameters, contributed_steps=1
        )


@pytest.mark.parametrize(
    "mutate",
    [
        "ranking_version_bool",
        "dependency_extra_field",
        "head_layers_float",
        "head_bias_int",
        "sort_steepness_int",
    ],
)
def test_ranking_bundle_rejects_loose_types_and_unknown_fields(mutate):
    """The closed ranking schema is strict at every level."""
    _settings, config, state = _ranking_pair()
    ranking_config = config["ranking_config"]
    if mutate == "ranking_version_bool":
        ranking_config["version"] = True
    elif mutate == "dependency_extra_field":
        ranking_config["dependency"]["extra"] = "x"
    elif mutate == "head_layers_float":
        ranking_config["head"]["layers"] = 3.0
    elif mutate == "head_bias_int":
        ranking_config["head"]["bias"] = 1
    else:
        ranking_config["sort"]["steepness_student"] = 100
    with pytest.raises(ValueError):
        recipe.validate_dense_resume(
            config,
            state,
            iteration=2,
            batch_size=8,
            steps_per_epoch=4,
            dependency=_dependency(),
        )


def test_resume_rejects_an_explicitly_inactive_reference_cap():
    """A Coordinate-only resume must not accept --ranking-reference-points."""
    settings = recipe.resolve_dense_settings(_args(coordinate_weight=0.4), None)
    saved = recipe.dense_config_bundle(
        settings, batch_size=8, steps_per_epoch=4, dependency=None
    )
    state = recipe.dense_state_bundle(
        recipe.new_dense_runtime(settings, seed=1), completed_steps=3, ranking=None
    )
    checkpoint = {"config": {recipe.CONFIG_KEY: saved}, recipe.STATE_KEY: state}
    resume_args = argparse.Namespace(
        coordinate_weight=0.4,
        coordinate_weight_explicit=False,
        ranking_weight=0.0,
        ranking_weight_explicit=False,
        dense_query_points=49,
        dense_query_points_explicit=False,
        ranking_reference_points=16,
        ranking_reference_points_explicit=True,
    )
    with pytest.raises(ValueError, match="ranking-reference-points"):
        recipe.resolve_dense_settings(resume_args, checkpoint)


@pytest.mark.parametrize("kwargs", [{"fused": True}, {"fused": True, "capturable": True}])
def test_fused_on_cpu_is_accepted_because_the_real_step_succeeds(kwargs):
    """torch 2.12.1 steps a fused AdamW on CPU, so the validator must allow it."""
    head = _Head()
    parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, **kwargs)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    # Prove the claim instead of asserting it: the next step really does run.
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in parameters)
    recipe.validate_ranking_head_optimizer_state(
        optimizer, parameters, contributed_steps=2
    )
