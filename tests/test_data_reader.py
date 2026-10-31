import hashlib
import os
from pathlib import Path

import pytest
from PIL import Image

from otuformer.data.provenance import (
    SnapshotDriftError,
    read_verified_bytes,
    snapshot_bytes,
)
from otuformer.data.reader import read_dataset


def write_png(path: Path, size=(8, 6), color=(10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_csv(path: Path, header, rows) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_coco(path: Path, images) -> Path:
    import json

    path.write_text(
        json.dumps(
            {
                "images": images,
                "categories": [{"id": 4, "name": "Head"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reader_preserves_raw_csv_and_selection(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    write_png(images / "b" / "001.png")
    selector = write_csv(
        tmp_path / "figs.csv",
        ["image", "label", "sex", "specimen_id", "plate_id", "duplicate_group_id"],
        [
            ["a/001.png", "NA", "", "001", "paperA:plate01", "dup01"],
            ["b/001.png", "001", "unknown", "", "", ""],
        ],
    )

    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == ["a/001.png", "b/001.png"]
    assert [r.output_id for r in bundle.records] == ["a/001.png", "b/001.png"]
    assert [r.selector_reference for r in bundle.records] == ["a/001.png", "b/001.png"]
    assert [r.metadata["label"] for r in bundle.records] == ["NA", "001"]
    assert bundle.records[0].metadata["specimen_id"] == "001"
    assert bundle.records[0].metadata["sex"] == ""  # present but empty
    assert bundle.records[1].metadata["sex"] == "unknown"
    assert (bundle.records[0].width, bundle.records[0].height) == (8, 6)
    assert bundle.records[0].image_file.sha256 == sha256(images / "a" / "001.png")
    assert bundle.selector is not None and bundle.selector.sha256 == sha256(selector)
    assert bundle.selector_columns[:2] == ("image", "label")
    assert bundle.selection_rule == "csv_rows"

    # An absent column is not an empty cell.
    narrow = write_csv(tmp_path / "narrow.csv", ["image"], [["a/001.png"]])
    narrow_bundle = read_dataset(images, manifest=narrow)
    narrow_bundle.raise_for_errors()
    assert narrow_bundle.records[0].metadata == {}
    assert "sex" not in narrow_bundle.records[0].metadata


def test_reader_requires_no_label_and_keeps_sample_origin(tmp_path):
    images = tmp_path / "images"
    write_png(images / "plate1" / "x.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["plate1/x.png"]])

    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    record = bundle.records[0]
    assert record.sample == "plate1"
    assert record.sample_origin == "first_directory"

    explicit = write_csv(
        tmp_path / "explicit.csv", ["image", "sample"], [["plate1/x.png", "custom"]]
    )
    explicit_bundle = read_dataset(images, manifest=explicit)
    explicit_bundle.raise_for_errors()
    assert explicit_bundle.records[0].sample == "custom"
    assert explicit_bundle.records[0].sample_origin == "explicit"


def test_directory_selection_is_sorted_and_uses_root_relative_keys(tmp_path):
    images = tmp_path / "images"
    write_png(images / "b" / "2.png")
    write_png(images / "a" / "1.png")

    bundle = read_dataset(images)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == ["a/1.png", "b/2.png"]
    assert [r.output_id for r in bundle.records] == ["a/1.png", "b/2.png"]
    assert bundle.selection_rule == "directory_scan"
    assert bundle.selector is None


@pytest.mark.parametrize(
    "reference, expected_key",
    [
        ("a/001.png", "a/001.png"),
        ("./a/001.png", "a/001.png"),
        ("a//001.png", "a/001.png"),
        ("a/./001.png", "a/001.png"),
        (".hidden/001.png", ".hidden/001.png"),
        (".../001.png", ".../001.png"),
    ],
)
def test_path_identity_and_containment(tmp_path, reference, expected_key):
    images = tmp_path / "images"
    for key in ["a/001.png", ".hidden/001.png", ".../001.png"]:
        write_png(images / key)
    selector = write_csv(tmp_path / "sel.csv", ["image"], [[reference]])

    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == [expected_key]
    assert bundle.records[0].output_id == reference


def test_leading_dot_directories_are_not_stripped(tmp_path):
    """A character-set lstrip('./') implementation would damage these keys."""
    images = tmp_path / "images"
    write_png(images / ".hidden" / "001.png")
    write_png(images / "..." / "001.png")
    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [[".hidden/001.png"], [".../001.png"]]
    )

    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == [
        ".hidden/001.png",
        ".../001.png",
    ]


def test_normalizing_references_that_collide_are_a_duplicate_key(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [["a/001.png"], ["./a/001.png"]]
    )

    bundle = read_dataset(images, manifest=selector)

    assert not bundle.validation.valid
    codes = [issue.code for issue in bundle.validation.by_severity("error")]
    assert "duplicate_image_key" in codes


def test_basename_fallback_is_unique_or_ambiguous(tmp_path):
    images = tmp_path / "images"
    write_png(images / "nested" / "deep" / "001.png")
    unique = write_csv(tmp_path / "unique.csv", ["image"], [["001.png"]])
    bundle = read_dataset(images, manifest=unique)
    bundle.raise_for_errors()
    assert [r.image_key for r in bundle.records] == ["nested/deep/001.png"]
    assert bundle.records[0].output_id == "001.png"

    write_png(images / "other" / "001.png")
    ambiguous = read_dataset(images, manifest=unique)
    assert not ambiguous.validation.valid
    assert "ambiguous_reference" in {
        issue.code for issue in ambiguous.validation.by_severity("error")
    }


def test_basename_fallback_is_not_used_for_directory_selection(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")

    bundle = read_dataset(images)

    assert [r.image_key for r in bundle.records] == ["a/001.png"]


def test_absolute_reference_must_stay_inside_the_root(tmp_path):
    images = tmp_path / "images"
    inside = write_png(images / "a" / "001.png")
    outside = write_png(tmp_path / "outside" / "002.png")

    ok = read_dataset(images, manifest=write_csv(tmp_path / "ok.csv", ["image"], [[str(inside)]]))
    ok.raise_for_errors()
    assert [r.image_key for r in ok.records] == ["a/001.png"]

    escaped = read_dataset(
        images, manifest=write_csv(tmp_path / "bad.csv", ["image"], [[str(outside)]])
    )
    assert not escaped.validation.valid
    assert "root_escape" in {i.code for i in escaped.validation.by_severity("error")}


def test_dotdot_reference_cannot_escape_the_root(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_png(tmp_path / "secret.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["../secret.png"]])

    bundle = read_dataset(images, manifest=selector)

    assert not bundle.validation.valid
    assert "root_escape" in {i.code for i in bundle.validation.by_severity("error")}


def test_case_and_unicode_spelling_are_preserved(tmp_path):
    images = tmp_path / "images"
    write_png(images / "A" / "Beetle_Ö.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["A/Beetle_Ö.png"]])

    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    assert bundle.records[0].image_key == "A/Beetle_Ö.png"
    assert bundle.records[0].output_id == "A/Beetle_Ö.png"


def _symlink_or_skip(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")


def test_internal_symlink_keeps_lexical_key(tmp_path):
    images = tmp_path / "images"
    real = write_png(images / "real" / "001.png")
    link = images / "alias" / "001.png"
    link.parent.mkdir(parents=True)
    _symlink_or_skip(real, link)

    bundle = read_dataset(images)
    bundle.raise_for_errors()

    by_key = {r.image_key: r for r in bundle.records}
    assert set(by_key) == {"real/001.png", "alias/001.png"}
    assert by_key["alias/001.png"].access_path == real.resolve()
    aliases = [i for i in bundle.validation.issues if i.code == "physical_alias"]
    assert len(aliases) == 1 and aliases[0].severity == "warning"


def test_symlink_escape_is_rejected(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    outside = write_png(tmp_path / "outside" / "001.png")
    link = images / "001.png"
    _symlink_or_skip(outside, link)

    bundle = read_dataset(images)

    assert not bundle.validation.valid
    assert "root_escape" in {i.code for i in bundle.validation.by_severity("error")}


def test_hardlink_alias_warns_without_merging(tmp_path):
    images = tmp_path / "images"
    original = write_png(images / "a" / "001.png")
    copy = images / "b" / "001.png"
    copy.parent.mkdir(parents=True)
    try:
        os.link(original, copy)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("hardlinks are not available on this platform")

    bundle = read_dataset(images)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == ["a/001.png", "b/001.png"]
    warnings = [i for i in bundle.validation.issues if i.code == "physical_alias"]
    assert len(warnings) == 1 and warnings[0].severity == "warning"


def test_root_relocation_preserves_keys(tmp_path):
    first = tmp_path / "run1" / "images"
    write_png(first / "a" / "001.png")
    relocated = tmp_path / "run2" / "images"
    relocated.parent.mkdir(parents=True, exist_ok=True)
    os.replace(first, relocated)

    bundle = read_dataset(relocated)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == ["a/001.png"]


def test_reader_reports_input_failures_without_side_effects(tmp_path, monkeypatch):
    import otuformer.data.reader as reader_module

    images = tmp_path / "images"
    write_png(images / "good.png")
    corrupt = images / "broken.png"
    corrupt.write_bytes(b"not an image")

    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [["good.png"], ["broken.png"]]
    )
    bundle = read_dataset(images, manifest=selector)

    assert not bundle.validation.valid
    failure = next(i for i in bundle.validation.issues if i.code == "image_decode_failed")
    assert failure.severity == "error"
    assert failure.image_key == "broken.png"
    assert failure.reference == 3
    assert len(bundle.records) == 1

    # The reader never writes: the fixture tree is unchanged.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["images", "sel.csv"]

    missing_root = read_dataset(tmp_path / "nope")
    assert "invalid_images_root" in {
        i.code for i in missing_root.validation.by_severity("error")
    }

    duplicate = write_csv(
        tmp_path / "dup.csv", ["image"], [["good.png"], ["good.png"]]
    )
    assert "duplicate_image_key" in {
        i.code for i in read_dataset(images, manifest=duplicate).validation.issues
    }

    header_only = write_csv(tmp_path / "empty.csv", ["image"], [])
    assert "empty_selection" in {
        i.code for i in read_dataset(images, manifest=header_only).validation.issues
    }

    # Hash and decode must use one byte buffer, not two reads.
    seen: list[str] = []
    real_open = reader_module.Image.open

    def spy(stream, *args, **kwargs):
        seen.append(hashlib.sha256(stream.getvalue()).hexdigest())
        return real_open(stream, *args, **kwargs)

    monkeypatch.setattr(reader_module.Image, "open", spy)
    clean = read_dataset(images, manifest=write_csv(tmp_path / "ok.csv", ["image"], [["good.png"]]))
    clean.raise_for_errors()
    assert seen == [clean.records[0].image_file.sha256]


def test_verified_bytes_detects_same_size_replacement(tmp_path):
    target = write_png(tmp_path / "001.png", size=(8, 6), color=(1, 2, 3))
    _data, snapshot = snapshot_bytes(target)
    original = target.read_bytes()
    target.write_bytes(bytes(reversed(original)))
    assert target.stat().st_size == snapshot.size_bytes

    with pytest.raises(SnapshotDriftError):
        read_verified_bytes(snapshot)

    target.write_bytes(original)
    assert read_verified_bytes(snapshot) == original


def test_header_names_are_preserved_exactly(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(
        tmp_path / "sel.csv", ["image", "custom column"], [["a/001.png", "x"]]
    )

    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    # Source column names are part of the metadata schema; trimming them rewrites it.
    assert bundle.selector_columns == ("image", "custom column")
    assert bundle.records[0].metadata == {"custom column": "x"}


def test_absolute_inside_symlink_keeps_one_identity(tmp_path):
    images = tmp_path / "images"
    real = write_png(images / "real" / "001.png")

    link = images / "alias" / "001.png"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    selector = write_csv(
        tmp_path / "sel.csv",
        ["image"],
        [["real/001.png"], [str(real)]],
    )
    bundle = read_dataset(images, manifest=selector)

    # The same file named twice under the same lexical key is a duplicate, not a
    # second alias.
    codes = {issue.code for issue in bundle.validation.by_severity("error")}
    assert "duplicate_image_key" in codes


def test_escaping_symlink_is_rejected_when_consumed(tmp_path):
    from otuformer.data.paths import ReferenceError, require_contained

    images = tmp_path / "images"
    target = write_png(images / "a" / "001.png")
    outside = write_png(tmp_path / "outside.png")
    target.unlink()
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    with pytest.raises(ReferenceError) as error:
        require_contained(images, target, "a/001.png")
    assert error.value.code == "root_escape"


def test_unselected_source_reference_cannot_escape_via_symlink(tmp_path):
    import json

    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    outside_dir = tmp_path / "elsewhere"
    write_png(outside_dir / "002.png")

    link = images / "ghost"
    try:
        link.symlink_to(outside_dir)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    source = tmp_path / "coco.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 17, "file_name": "a/001.png", "width": 8, "height": 6},
                    {"id": 18, "file_name": "ghost/002.png", "width": 8, "height": 6},
                ],
                "categories": [{"id": 4, "name": "Head"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    bundle = read_dataset(images, manifest=selector, coco_instances=source)

    escapes = [
        issue
        for issue in bundle.validation.by_severity("error")
        if issue.code == "root_escape"
    ]
    # The escaping reference is named, and it is a source-wide contract: the
    # reference is rejected before any record can be built from the wrong file.
    assert [issue.reference for issue in escapes] == [18]
    assert not bundle.records


def test_availability_states_are_reported_separately(tmp_path):
    import json

    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    write_png(images / "b" / "002.png")
    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [["a/001.png"], ["b/002.png"]]
    )
    # Image 17 is in the source with no objects; image 18 is absent from the source.
    source = tmp_path / "coco.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 17, "file_name": "a/001.png", "width": 8, "height": 6}
                ],
                "categories": [{"id": 4, "name": "Head"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    bundle = read_dataset(images, manifest=selector, coco_instances=source)
    bundle.raise_for_errors()

    # "No objects" and "no source entry" are different states.
    assert bundle.validation.counts["images_without_annotations"] == 2
    assert bundle.validation.counts["images_without_source_reference"] == 1
    assert bundle.validation.scope["unselected_file_and_geometry"] == "not_performed"
    assert (
        bundle.validation.scope["custom_ignore_protocol"]["status"] == "not_implemented"
    )


def test_basename_fallback_does_not_resolve_a_path_reference(tmp_path):
    images = tmp_path / "images"
    write_png(images / "b" / "one.png")
    # The CSV asks for a path that does not exist; b/one.png must not be substituted.
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/one.png"]])

    bundle = read_dataset(images, manifest=selector)

    assert not bundle.validation.valid
    assert "missing_image" in {issue.code for issue in bundle.validation.by_severity("error")}


def test_symlinked_component_cannot_be_folded_out_of_the_root(tmp_path):
    images = tmp_path / "images"
    write_png(images / "one.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    write_png(outside / "one.png", color=(9, 9, 9))
    try:
        (images / "link").symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # Folding ".." lexically would silently read images/one.png instead.
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["link/../one.png"]])
    bundle = read_dataset(images, manifest=selector)

    assert not bundle.validation.valid
    assert "root_escape" in {issue.code for issue in bundle.validation.by_severity("error")}


def test_absolute_alias_keeps_its_own_identity(tmp_path):
    images = tmp_path / "images"
    real = write_png(images / "plate1" / "000.png")
    alias = images / "alias.png"
    try:
        alias.symlink_to(real)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    selector = write_csv(
        tmp_path / "sel.csv",
        ["image"],
        [["plate1/000.png"], [str(alias)]],
    )
    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    # Each spelling keeps its key; the shared physical file is reported, not merged.
    assert [record.image_key for record in bundle.records] == [
        "plate1/000.png",
        "alias.png",
    ]
    warnings = {issue.code for issue in bundle.validation.by_severity("warning")}
    assert "physical_alias" in warnings


def test_symlink_target_dotdot_reads_the_file_that_exists(tmp_path):
    images = tmp_path / "images"
    write_png(images / "one.png", color=(1, 1, 1))
    target = images / "a" / "one.png"
    write_png(target, color=(200, 100, 50))
    (images / "a" / "b").mkdir(parents=True)
    try:
        (images / "link").symlink_to(images / "a" / "b")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # link -> a/b, so link/../one.png is a/one.png on disk, not images/one.png.
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["link/../one.png"]])
    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    record = bundle.records[0]
    assert record.image_key == "a/one.png"
    assert (
        record.image_file.sha256
        == hashlib.sha256(target.read_bytes()).hexdigest()
    )


def test_coco_file_name_resolves_as_written(tmp_path):
    images = tmp_path / "images"
    write_png(images / "one.png", color=(1, 1, 1))
    write_png(images / "a" / "one.png", color=(200, 100, 50))
    (images / "a" / "b").mkdir(parents=True)
    try:
        (images / "link").symlink_to(images / "a" / "b")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    source = write_coco(
        tmp_path / "coco.json",
        [{"id": 1, "file_name": "link/../one.png", "width": 8, "height": 6}],
    )
    bundle = read_dataset(images, coco_instances=source)
    bundle.raise_for_errors()

    # Taking the lexical key would have selected images/one.png instead.
    assert bundle.records[0].image_key == "a/one.png"


def test_coco_file_name_cannot_escape_through_a_symlink(tmp_path):
    images = tmp_path / "images"
    write_png(images / "one.png")
    outside = tmp_path / "outside"
    write_png(outside / "one.png", color=(9, 9, 9))
    try:
        (images / "escape").symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    source = write_coco(
        tmp_path / "coco.json",
        [{"id": 1, "file_name": "escape/../one.png", "width": 8, "height": 6}],
    )
    bundle = read_dataset(images, coco_instances=source)

    assert not bundle.validation.valid
    assert "root_escape" in {issue.code for issue in bundle.validation.by_severity("error")}
    assert not bundle.records


def test_root_symlink_spelling_keeps_alias_identity(tmp_path):
    physical = tmp_path / "physical"
    real = write_png(physical / "real.png")
    alias = physical / "alias.png"
    linked = tmp_path / "linked"
    try:
        alias.symlink_to(real)
        linked.symlink_to(physical)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    selector = write_csv(
        tmp_path / "sel.csv",
        ["image"],
        [["alias.png"], [str(linked / "alias.png")]],
    )
    bundle = read_dataset(linked, manifest=selector)

    # Both spellings address the same alias, so they collide as one key rather than
    # splitting into alias.png and real.png.
    codes = {issue.code for issue in bundle.validation.by_severity("error")}
    assert "duplicate_image_key" in codes


def test_absolute_reference_through_symlink_dotdot_keeps_a_honest_key(tmp_path):
    images = tmp_path / "images"
    write_png(images / "one.png", color=(1, 1, 1))
    target = images / "a" / "one.png"
    write_png(target, color=(200, 100, 50))
    (images / "a" / "b").mkdir(parents=True)
    try:
        (images / "link").symlink_to(images / "a" / "b")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # Absolute spelling: link -> a/b, so link/../one.png is a/one.png on disk while
    # the folded key says one.png.
    reference = str(images / "link" / ".." / "one.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [[reference]])
    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()

    record = bundle.records[0]
    assert record.image_key == "a/one.png"
    assert record.image_file.sha256 == sha256(target)


def test_coco_absolute_reference_through_symlink_dotdot_is_not_redirected(tmp_path):
    images = tmp_path / "images"
    write_png(images / "one.png", color=(1, 1, 1))
    target = images / "a" / "one.png"
    write_png(target, color=(200, 100, 50))
    (images / "a" / "b").mkdir(parents=True)
    try:
        (images / "link").symlink_to(images / "a" / "b")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    source = write_coco(
        tmp_path / "coco.json",
        [
            {
                "id": 1,
                "file_name": str(images / "link" / ".." / "one.png"),
                "width": 8,
                "height": 6,
            }
        ],
    )
    bundle = read_dataset(images, coco_instances=source)
    bundle.raise_for_errors()

    # A folded key would make the reader rebuild images/one.png here.
    record = bundle.records[0]
    assert record.image_key == "a/one.png"
    assert record.image_file.sha256 == sha256(target)


def test_relative_root_spelling_keeps_alias_identity(tmp_path, monkeypatch):
    physical = tmp_path / "physical"
    real = write_png(physical / "real.png")
    alias = physical / "alias.png"
    linked = tmp_path / "linked"
    try:
        alias.symlink_to(real)
        linked.symlink_to(physical)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    selector = write_csv(
        tmp_path / "sel.csv",
        ["image"],
        [["alias.png"], [str(linked / "alias.png")]],
    )
    monkeypatch.chdir(tmp_path)
    bundle = read_dataset(Path("linked"), manifest=selector)

    # The root is written relatively here; the alias must still be one identity.
    codes = {issue.code for issue in bundle.validation.by_severity("error")}
    assert "duplicate_image_key" in codes
    assert [record.image_key for record in bundle.records] == ["alias.png"]
