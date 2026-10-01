#!/usr/bin/env python3
"""Read-only schema and proposal adapter for the otuformer-workflow Skill.

This script never runs an analysis, never creates an output directory, and never
grants approval. It only imports the installed package (or an explicitly approved
source checkout) and prints JSON:

* schema only:      ``export_cli_schema.py --command pretrain``
* full proposal:    ``export_cli_schema.py --command pretrain --inputs-json plan.json``

With ``--inputs-json`` the adapter also verifies that the launcher that would run
the approved command matches this environment.

Exit codes: 0 success, 2 any error (message on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_ONLY_HINT = "pass --command or --inputs-json"


def _skill_dir() -> Path:
    """Directory of the loaded Skill, independent of the working directory."""
    return Path(__file__).resolve().parent.parent


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Fail on duplicate JSON keys instead of silently keeping the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in inputs JSON: {key!r}")
        result[key] = value
    return result


def _ensure_importable(source_root: Path | None) -> None:
    """Import the installed package, or a verified repository checkout.

    The source fallback applies only when OTU-Former itself is absent. An
    ImportError raised from inside an importable package is a real environment
    failure and must not be masked by a checkout fallback.
    """
    import importlib.util

    if importlib.util.find_spec("otuformer") is not None:
        import otuformer  # noqa: F401

        return

    candidates: list[Path] = []
    if source_root is not None:
        # An explicitly approved checkout is trusted as given.
        candidates.append(Path(source_root).expanduser().resolve())
    # The canonical checkout that contains this Skill (skills/<name>/ -> repo).
    # Only used when this exact Skill really lives inside it: a copied Skill must
    # not guess an unrelated ancestor repository.
    canonical = _skill_dir().parent.parent
    if (canonical / "skills" / _skill_dir().name).resolve() == _skill_dir():
        candidates.append(canonical)

    for candidate in candidates:
        package = candidate / "src" / "otuformer" / "__init__.py"
        if (candidate / "pyproject.toml").is_file() and package.is_file():
            sys.path.insert(0, str(candidate / "src"))
            return
    raise ValueError(
        "otuformer is not importable; install the package or pass --source-root "
        "pointing at a repository checkout (a copied Skill does not guess an "
        "ancestor repository)"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_cli_schema.py",
        description=(
            "Read-only OTU-Former runtime schema and parameter-card adapter. "
            "Never executes an analysis."
        ),
    )
    parser.add_argument("--command", help="Command to inspect.")
    parser.add_argument(
        "--inputs-json",
        help="JSON file with explicit option values; produces a full proposal.",
    )
    parser.add_argument(
        "--executable",
        default="otuformer",
        help="Launcher that would run an approved proposal (default: otuformer).",
    )
    parser.add_argument(
        "--protected-path",
        action="append",
        default=[],
        help="Path that outputs must not overlap (repeatable).",
    )
    parser.add_argument(
        "--source-root",
        help="Repository checkout to import when the package is not installed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _ensure_importable(Path(args.source_root) if args.source_root else None)
        from otuformer.cli_schema import (
            build_proposal,
            export_cli_schema,
            verify_launcher,
        )

        payload: dict[str, Any]
        if args.inputs_json:
            if not args.command:
                raise ValueError("--inputs-json requires --command")
            inputs_path = Path(args.inputs_json).expanduser()
            explicit = json.loads(
                inputs_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
            if not isinstance(explicit, dict):
                raise ValueError("inputs JSON must be an object of option names to values")
            payload = build_proposal(
                args.command,
                explicit,
                executable=[args.executable],
                protected_paths=[Path(p) for p in args.protected_path],
            )
        else:
            payload = export_cli_schema(args.command)
        if args.inputs_json:
            launcher = verify_launcher(
                [args.executable], source_runtime=payload["runtime"]
            )
            prefix = list(launcher["argv_prefix"])
            if prefix != [args.executable]:
                # Rebuild so the reviewed argv uses the verified launch prefix.
                payload = build_proposal(
                    args.command,
                    explicit,
                    executable=prefix,
                    protected_paths=[Path(p) for p in args.protected_path],
                )
            payload["launcher"] = launcher
    except Exception as error:  # noqa: BLE001 - single CLI error boundary
        print(f"error: {error}", file=sys.stderr)
        return 2

    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
