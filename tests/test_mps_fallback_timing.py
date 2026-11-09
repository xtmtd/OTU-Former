"""The MPS CPU-fallback switch must be armed before PyTorch initialises.

``PYTORCH_ENABLE_MPS_FALLBACK`` is latched when PyTorch starts, so setting it
from a command function or a training loop is a no-op. These tests run real
subprocesses against the actual missing kernel — the *antialiased* bicubic
resample backward (``_upsample_bicubic2d_aa_backward``), which is what the
augmentation path reaches — under both import orders.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
MPS = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)

# The kernel the augmentation path actually hits. ``antialias=True`` selects the
# antialiased variant; the plain bicubic variant is implemented on MPS and would
# not reproduce the failure.
_AA_BICUBIC_BACKWARD = textwrap.dedent(
    """
    import torch
    x = torch.rand(1, 1, 8, 8, device="mps", requires_grad=True)
    y = torch.nn.functional.interpolate(
        x, size=(16, 16), mode="bicubic", align_corners=False, antialias=True
    )
    y.sum().backward()
    print("BACKWARD_OK", type(x.grad).__name__)
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(SRC.parent),
    )


@MPS
def test_package_import_before_torch_arms_the_fallback():
    """The supported order: importing otuformer first protects the run."""
    result = _run(f"import otuformer\n{_AA_BICUBIC_BACKWARD}")

    assert result.returncode == 0, result.stderr
    assert "BACKWARD_OK" in result.stdout


@MPS
def test_torch_first_is_too_late_and_the_package_says_so():
    """The unsupported order fails, and the package reports it as unknown."""
    result = _run(
        "import torch\n"
        "import otuformer\n"
        "print('STATE', otuformer.MPS_FALLBACK_STATE.value)\n"
        f"{_AA_BICUBIC_BACKWARD}"
    )

    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr
    # The flag must not claim protection that does not exist.
    assert "STATE unknown" in result.stdout


def test_package_first_reports_effective():
    result = _run(
        "import otuformer; "
        "print('STATE', otuformer.MPS_FALLBACK_STATE.value, "
        "otuformer.MPS_FALLBACK_EFFECTIVE)"
    )

    assert result.returncode == 0, result.stderr
    assert "STATE armed True" in result.stdout


def test_torch_first_alone_is_enough_to_report_unknown():
    """No MPS work is needed first: the import order already decides."""
    result = _run(
        "import torch\n"
        "import otuformer\n"
        "print('STATE', otuformer.MPS_FALLBACK_STATE.value, "
        "otuformer.MPS_FALLBACK_EFFECTIVE)"
    )

    assert result.returncode == 0, result.stderr
    assert "STATE unknown False" in result.stdout


def _run_with_env(body: str, **env_overrides) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(SRC.parent),
    )


@MPS
def test_an_operator_preset_is_not_reported_as_missing():
    """Importing torch first is not proof the operator failed to set the switch.

    The operator set the variable before the process started, which is the
    documented way to protect a host that imports ``torch`` first. The real
    kernel succeeds, so the package must not claim the switch is missing.
    """
    result = _run_with_env(
        "import torch\n"
        "import otuformer\n"
        "print('STATE', otuformer.MPS_FALLBACK_STATE.value)\n"
        f"{_AA_BICUBIC_BACKWARD}",
        PYTORCH_ENABLE_MPS_FALLBACK="1",
    )

    assert result.returncode == 0, result.stderr
    assert "BACKWARD_OK" in result.stdout
    # Unknown, not "unset": the package cannot see the operator's own setting.
    assert "STATE unknown" in result.stdout


def test_the_entry_point_warning_does_not_accuse_an_unknown_state(capsys):
    """An unknown state is reported as unknown, never as "you did not set it"."""
    from otuformer import MPSFallbackState
    from otuformer.training import runtime

    class _Device:
        type = "mps"

    for state, expected in (
        (MPSFallbackState.UNKNOWN, "cannot confirm"),
        (MPSFallbackState.DISABLED, "=0 was set explicitly"),
        (MPSFallbackState.ARMED, None),
    ):
        capsys.readouterr()
        original = runtime.mps_fallback_state
        runtime.mps_fallback_state = lambda state=state: state
        try:
            runtime.warn_if_mps_fallback_unarmed(_Device())
        finally:
            runtime.mps_fallback_state = original
        printed = capsys.readouterr().out
        if expected is None:
            assert printed == ""
        else:
            assert expected in printed
        assert "was not set before PyTorch was imported" not in printed


def test_an_explicit_opt_out_is_respected():
    result = _run_with_env(
        "import otuformer; "
        "print('STATE', otuformer.MPS_FALLBACK_STATE.value, "
        "otuformer.MPS_FALLBACK_EFFECTIVE)",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
    )

    assert result.returncode == 0, result.stderr
    assert "STATE disabled False" in result.stdout


def test_entry_points_no_longer_set_the_switch_themselves():
    """A late ``setdefault`` is not a guarantee, so it must not be offered as one."""
    for relative in (
        "training/pretrain_loop.py",
        "training/finetune_loop.py",
        "cli/pretrain.py",
        "cli/finetune.py",
    ):
        text = (SRC / "otuformer" / relative).read_text(encoding="utf-8")
        assert 'setdefault("PYTORCH_ENABLE_MPS_FALLBACK"' not in text, relative
