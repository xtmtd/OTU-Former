import importlib
import subprocess
import sys
import textwrap


def test_main_import_does_not_eager_load_heavy_modules():
    modules_to_clear = [
        "otuformer.cli.main",
        "otuformer.cli.cam",
        "otuformer.cli.cluster",
        "otuformer.vision.cam",
        "scipy.cluster",
        "pytorch_grad_cam",
    ]
    for module_name in modules_to_clear:
        sys.modules.pop(module_name, None)

    importlib.import_module("otuformer.cli.main")

    assert "otuformer.vision.cam" not in sys.modules
    assert "scipy.cluster" not in sys.modules
    assert "pytorch_grad_cam" not in sys.modules


def test_cli_import_and_help_do_not_import_torch():
    script = textwrap.dedent(
        """
        import sys

        import otuformer.cli.pretrain as pretrain
        import otuformer.cli.finetune as finetune
        from typer.testing import CliRunner

        runner = CliRunner()
        for module in (pretrain, finetune):
            result = runner.invoke(module.app, ["--help"])
            assert result.exit_code == 0, result.output

        heavy = [name for name in ("torch", "torchvision") if name in sys.modules]
        print("HEAVY=" + ",".join(heavy))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "HEAVY=" in completed.stdout
    heavy = completed.stdout.split("HEAVY=", 1)[1].strip()
    assert heavy == "", f"CLI import pulled in heavy modules: {heavy}"
