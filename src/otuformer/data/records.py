"""Data records shared by the v2 ingestion reader.

Stdlib only: importing this module must never pull torch, pandas or the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, Union

JsonValue: TypeAlias = Union[
    str, int, float, bool, None, list["JsonValue"], dict[str, "JsonValue"]
]

SEVERITIES = ("error", "warning", "information")


@dataclass(frozen=True)
class FileSnapshot:
    """Resolved local access path plus byte identity. The path is not a key."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CocoImageRef:
    """A role-qualified COCO image identity; IDs are not global across files."""

    source_id: str
    image_id: int


@dataclass(frozen=True)
class ImageRecord:
    image_key: str
    access_path: Path
    output_id: str
    selector_reference: str | None
    width: int
    height: int
    image_file: FileSnapshot
    metadata: dict[str, str] = field(default_factory=dict)
    coco_images: tuple[CocoImageRef, ...] = ()
    sample: str | None = None
    sample_origin: str = "none"


@dataclass(frozen=True)
class SourceIndex:
    source_id: str
    file: FileSnapshot
    payload: dict[str, Any]
    images_by_id: dict[int, dict[str, Any]]
    categories_by_id: dict[int, dict[str, Any]]
    annotations_by_image: dict[int, tuple[dict[str, Any], ...]]
    image_keys: dict[int, str]
    mask_files: dict[int, FileSnapshot] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    source_id: str | None = None
    reference: str | int | None = None
    image_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity!r}")


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    polygon_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def by_severity(self, severity: str) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == severity]


class DataValidationError(ValueError):
    """Raised when a bundle carries errors and must not reach inference."""

    def __init__(self, report: ValidationReport) -> None:
        errors = report.by_severity("error")
        first = errors[0].message if errors else "invalid data"
        super().__init__(f"{len(errors)} data validation error(s): {first}")
        self.report = report


@dataclass(frozen=True)
class DataBundle:
    images_dir: Path
    records: tuple[ImageRecord, ...]
    sources: dict[str, SourceIndex] = field(default_factory=dict)
    selector: FileSnapshot | None = None
    selector_columns: tuple[str, ...] = ()
    selection_rule: str = "directory_scan"
    validation: ValidationReport = field(default_factory=ValidationReport)

    def raise_for_errors(self) -> None:
        if not self.validation.valid:
            raise DataValidationError(self.validation)

    def annotations_for(
        self, image_key: str, source_id: str
    ) -> tuple[dict[str, Any], ...]:
        source = self.sources.get(source_id)
        if source is None:
            return ()
        for record in self.records:
            if record.image_key != image_key:
                continue
            for ref in record.coco_images:
                if ref.source_id == source_id:
                    return source.annotations_by_image.get(ref.image_id, ())
        return ()
