"""Logging utilities for OTU-Former CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import re
from datetime import datetime


def _project_name_at(root: Path) -> str | None:
    """The ``[project].name`` of ``root``'s ``pyproject.toml``, if readable."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    project = data.get("project")
    return project.get("name") if isinstance(project, dict) else None


def _git_commit(source_dir: Path | None = None) -> str:
    """Short commit of the OTU-Former checkout, or ``unknown``.

    The work tree is only trusted when its ``pyproject.toml`` names this
    project, so a package vendored inside another repository never reports the
    host repository's commit. The ``-dirty`` suffix only flags uncommitted
    changes in the tree: it is a provenance hint, not a hash of the code that
    ran, and a clean tree does not prove the files match the commit.
    """
    start = Path(__file__).resolve().parent if source_dir is None else Path(source_dir)
    try:
        toplevel = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if toplevel.returncode != 0 or not toplevel.stdout.strip():
            return "unknown"
        root = Path(toplevel.stdout.strip())
        if _project_name_at(root) != "otuformer":
            return "unknown"
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return "unknown"
        commit = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_provenance() -> str:
    """One-line version and best-effort commit header for every run log."""
    from otuformer import __version__

    return f"otuformer {__version__} (commit {_git_commit()})"


class TeeLogger:
    """Redirect stdout to both console and file, skipping progress-like writes."""

    PROGRESS_PATTERNS = ("\r", "|█", "|░", "[A")
    ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    def __init__(self, log_path: Path, append: bool = False) -> None:
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a" if append else "w", encoding="utf-8")
        self._last_was_progress = False
        self._at_line_start = True
        # Record the version and a best-effort commit of the OTU-Former
        # checkout. `-dirty` only marks that the tree had uncommitted changes;
        # it does not identify them, so treat the header as a provenance hint.
        self.write(run_provenance() + "\n")

    @staticmethod
    def _timestamp_prefix() -> str:
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "

    def _write_with_timestamp(self, message: str) -> None:
        if not message:
            return
        parts = re.split(r"(\n)", message)
        for part in parts:
            if part == "":
                continue
            if part == "\n":
                self.terminal.write("\n")
                self.log.write("\n")
                self._at_line_start = True
                continue
            if self._at_line_start:
                prefix = self._timestamp_prefix()
                self.terminal.write(prefix)
                self.log.write(prefix)
                self._at_line_start = False
            self.terminal.write(part)
            self.log.write(part)

    def write(self, message: str) -> None:
        clean_message = self.ANSI_PATTERN.sub("", message)
        is_progress = any(pattern in message for pattern in self.PROGRESS_PATTERNS)
        if not is_progress:
            if self._last_was_progress and clean_message.strip():
                self.terminal.write("\n")
                self.log.write("\n")
                self._at_line_start = True
            self._write_with_timestamp(clean_message)
            self.terminal.flush()
            self.log.flush()
            self._last_was_progress = False
        else:
            self.terminal.write(message)
            self.terminal.flush()
            self._last_was_progress = True

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()
