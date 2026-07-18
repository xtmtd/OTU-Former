import json
from types import SimpleNamespace

from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.cli.update import fetch_remote_version

runner = CliRunner()


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_fetch_remote_version_uses_highest_semver_tag(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response([{"name": "v0.1.1"}, {"name": "v0.2.0"}, {"name": "nightly"}]),
    )
    assert fetch_remote_version() == "0.2.0"


def test_update_check_never_runs_pip(monkeypatch):
    monkeypatch.setattr("otuformer.cli.update.fetch_remote_version", lambda: "9.9.9")
    called = []
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: called.append(args))

    result = runner.invoke(app, ["update", "--check"])

    assert result.exit_code == 0
    assert not called


def test_update_yes_installs_exact_tag(monkeypatch):
    monkeypatch.setattr("otuformer.cli.update.fetch_remote_version", lambda: "9.9.9")
    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0),
    )

    result = runner.invoke(app, ["update", "--yes"])

    assert result.exit_code == 0
    assert calls[0][0][-1].endswith("OTU-Former.git@v9.9.9")


def test_update_network_failure_is_nonzero(monkeypatch):
    monkeypatch.setattr("otuformer.cli.update.fetch_remote_version", lambda: (_ for _ in ()).throw(OSError("offline")))

    result = runner.invoke(app, ["update", "--check"])

    assert result.exit_code == 1
    assert "could not reach GitHub" in result.output
