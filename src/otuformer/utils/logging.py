"""Logging utilities for OTU-Former CLI."""

from __future__ import annotations

import sys
from pathlib import Path
import re


class TeeLogger:
    """Redirect stdout to both console and file, skipping progress-like writes."""

    PROGRESS_PATTERNS = ("\r", "|█", "|░", "[A")
    ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    def __init__(self, log_path: Path) -> None:
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w", encoding="utf-8")
        self._last_was_progress = False

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        clean_message = self.ANSI_PATTERN.sub("", message)
        is_progress = any(pattern in message for pattern in self.PROGRESS_PATTERNS)
        if not is_progress:
            if self._last_was_progress and clean_message.strip():
                self.log.write("\n")
            self.log.write(clean_message)
            self.log.flush()
            self._last_was_progress = False
        else:
            self._last_was_progress = True

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()
