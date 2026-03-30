"""Logging utilities for OTU-Former CLI."""

from __future__ import annotations

import sys
from pathlib import Path
import re
from datetime import datetime


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
