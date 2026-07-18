"""Check for published OTU-Former updates and optionally install one."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from functools import total_ordering

import typer

from otuformer import __version__

_REPO = "xtmtd/OTU-Former"
_TAGS_API_URL = f"https://api.github.com/repos/{_REPO}/tags"
_INSTALL_URL = f"git+https://github.com/{_REPO}.git@v{{version}}"
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")

app = typer.Typer(help="Check for published updates and optionally install the latest version.")


@total_ordering
@dataclass(frozen=True)
class _SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _SemVer):
            return NotImplemented
        core = (self.major, self.minor, self.patch)
        other_core = (other.major, other.minor, other.patch)
        if core != other_core:
            return core < other_core
        if not self.prerelease or not other.prerelease:
            return bool(self.prerelease)
        return self.prerelease < other.prerelease


def _parse_version(version: str) -> _SemVer:
    match = _SEMVER_RE.match(version.strip().lstrip("v"))
    if not match:
        return _SemVer(0, 0, 0, ("invalid",))
    prerelease = match.group(4)
    return _SemVer(
        int(match.group(1)), int(match.group(2)), int(match.group(3)),
        tuple(prerelease.split(".")) if prerelease else (),
    )


def fetch_remote_version(timeout: int = 10) -> str:
    request = urllib.request.Request(
        _TAGS_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "otuformer-updater"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        tags = json.loads(response.read())
    versions = [str(tag["name"]).lstrip("v") for tag in tags]
    versions = [version for version in versions if _SEMVER_RE.match(version)]
    if not versions:
        raise RuntimeError("No Semantic Version tags found in repository.")
    return max(versions, key=_parse_version)


@app.callback(invoke_without_command=True)
def update(
    check: bool = typer.Option(False, "--check", help="Only show version information."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Install without prompting."),
) -> None:
    typer.echo(f"Current version : {__version__}")
    typer.echo("Checking GitHub for updates...")
    try:
        remote_version = fetch_remote_version()
    except Exception as exc:
        typer.echo(f"Error: could not reach GitHub: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Latest version  : {remote_version}")
    if _parse_version(__version__) >= _parse_version(remote_version):
        typer.echo("Already up to date.")
        return
    typer.echo(f"Update available: {__version__} -> {remote_version}")
    if check:
        typer.echo("(Run without --check to install the update.)")
        return
    if not yes and input("Proceed with update? [y/N] ").strip().lower() != "y":
        typer.echo("Update cancelled.")
        return

    typer.echo("Installing latest published version from GitHub...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", _INSTALL_URL.format(version=remote_version)],
        check=False,
    )
    if result.returncode:
        typer.echo("Update failed. Check pip output above.", err=True)
        raise typer.Exit(code=result.returncode)
    typer.echo("Update complete. Restart your shell to use the new version.")
