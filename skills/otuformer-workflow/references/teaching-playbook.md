# Teaching playbook

Teaching mode is opt-in. Offer it after environment inspection, while explaining
CSV/input expectations, or when input problems block progress. Never switch away
from user data without an explicit choice, and state that a demonstration does not
replace analysis of the user's samples.

## Assets

- `examples/Epidorcus/images/`: 230 JPEGs under two label directories
  (`Epidorcus_gracilis` 102, `Epidorcus_tonkinensis` 128);
- `examples/Epidorcus/figs.csv`: 230 `image,label` rows using **basenames** while
  the images are nested;
- no bundled checkpoint, embeddings, assignments, or correction table.

Use **all 230 images**, not a reduced subset. Find `examples/` relative to the
repository that contains this Skill, or accept an explicit examples root when the
Skill was copied. Do not search unrelated directories or download data.

## Prerequisites per step

| Step | Teaching input |
|---|---|
| `doctor` | The actual environment. |
| `pretrain` | All example images; the CSV optionally selects training/visualization inputs. |
| `finetune` | Example labels/images plus a compatible checkpoint from an approved preceding step or a selected existing file. |
| `extract` | All example images plus a compatible checkpoint or ONNX model; labels optional. |
| `cluster` | The embeddings just produced; optionally an ID-matched label table. |
| `annotate` | The selected partition plus an approved teaching correction table. |
| `diversity` | Selected assignments or an actual OTU table. |
| `cam` | Compatible checkpoint plus the example image directory. |
| `export` | Compatible checkpoint. |
| `update` | Explain or inspect `--check` only; never install merely to demonstrate. |

Missing prerequisite: offer an individually approved preceding step, a selected
existing artifact, or an explanation without execution. Never invent a
random-weight checkpoint and present it as trained.

## Label and correction tables

The CLI resolves image basenames to nested files. Extraction emits `id` values
that are either image **file names** (image-directory runs) or the **original
reference strings** from the input CSV (CSV-driven runs, including any directory
parts), and downstream labels must be mapped to those emitted IDs rather than to
training class names. The following tested recipe writes an `id,label` table
before clustering, or a **label-informed** `id,cluster` teaching correction table
before annotation: it keeps emitted IDs verbatim, corrects only the IDs whose
cluster disagrees with the majority cluster of their label, and enforces the
all-230-image teaching contract by default. It performs no analysis and refuses to
write outside the approved demo root.

<!-- teaching-recipe:start -->
```python
def prepare_teaching_table(
    *,
    examples_root,
    embeddings_csv,
    output_csv,
    demo_root,
    assignments_csv=None,
    require_all_examples=True,
    label_to_cluster=None,
):
    """Write a teaching table from the real example assets; return its row count.

    ``examples_root`` is the image directory (``.../Epidorcus/images``), so the
    label CSV is read from its parent. Without ``assignments_csv`` this writes the
    ``id,label`` table used before clustering (one row per emitted id). With it,
    this writes label-informed teaching corrections (``id,cluster``) for the ids
    whose cluster disagrees with the majority cluster of their label: only rows
    that actually change are written, so the table is a real correction proposal
    rather than a copy of the partition. Ids are preserved exactly as extraction
    emitted them. With ``require_all_examples`` (default) the demo must cover
    every example image, matching the all-230-image teaching contract.

    Performs no analysis: destination safety, duplicate/unmatched/ambiguous id
    checks, and field validation all run before anything is written.
    """
    import csv
    import math
    from collections import Counter
    from pathlib import Path

    import pandas as pd

    from otuformer.delineation.annotate import (
        canonicalize_corrections,
        validate_corrections,
        validate_corrections_against_assignments,
    )
    from otuformer.training.dataset import (
        _build_recursive_index,
        _resolve_image_path,
    )

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def resolved(path):
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve()

    def basename(value):
        """Matching key only; never used to rewrite an emitted id."""
        return Path(str(value)).name

    examples = resolved(examples_root)
    demo = resolved(demo_root)
    destination = resolved(output_csv)
    if not destination.is_relative_to(demo):
        raise ValueError(f"refusing to write {destination} outside the demo root {demo}")
    read_only = [examples]
    # When the label CSV sits next to the image directory, that directory is the
    # read-only teaching dataset and must stay untouched as well.
    if (examples.parent / "figs.csv").is_file():
        read_only.append(examples.parent)
    # A repository ``examples/`` tree is read-only in full, not only its images.
    read_only.extend(
        ancestor for ancestor in examples.parents if ancestor.name == "examples"
    )
    if any(
        destination == root or destination.is_relative_to(root) for root in read_only
    ):
        raise ValueError(
            "refusing to write into the read-only examples tree: "
            + ", ".join(str(root) for root in read_only)
        )
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing file {destination}")

    images = {}
    for path in sorted(examples.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            images.setdefault(path.name, []).append(path)
    for name, paths in images.items():
        if len(paths) != 1:
            shown = ", ".join(str(item.relative_to(examples)) for item in paths[:3])
            raise ValueError(
                f"ambiguous image basename {name!r}: {len(paths)} files ({shown}); "
                "an emitted id could not be mapped to one image"
            )

    label_csv = examples.parent / "figs.csv"
    labels = {}
    with label_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = basename(row["image"])
            labels.setdefault(name, set()).add(str(row["label"]))
    for name, found in labels.items():
        if name not in images:
            raise ValueError(f"label CSV references a missing image: {name}")
        on_disk = images[name][0].parent.name
        if len(found) != 1 or found != {on_disk}:
            raise ValueError(
                f"label mismatch for {name}: on disk in {on_disk!r} vs CSV {sorted(found)}"
            )
    total_images = sum(len(paths) for paths in images.values())
    if len(labels) != total_images:
        raise ValueError(
            f"label CSV covers {len(labels)} of {total_images} example images"
        )
    label_of = {name: next(iter(found)) for name, found in labels.items()}

    frame = pd.read_csv(embeddings_csv)
    if "id" not in frame.columns:
        raise ValueError("embeddings CSV has no 'id' column")
    ids = [str(value) for value in frame["id"].tolist()]
    if len(set(ids)) != len(ids):
        raise ValueError("embeddings CSV contains duplicate ids")
    # Resolve every id the way extraction does and require a real file inside the
    # read-only example root. Matching only the basename would accept fabricated
    # ids such as /nonexistent/<example name>.
    index = _build_recursive_index(examples)
    for value in ids:
        try:
            image_path = _resolve_image_path(examples, value, *index)
        except FileNotFoundError as error:
            raise ValueError(
                f"id does not resolve to one example image: {value!r} ({error})"
            ) from None
        if not image_path.exists():
            raise ValueError(
                f"id does not resolve to a real example image: {value!r}"
            )
        if not image_path.resolve().is_relative_to(examples):
            raise ValueError(
                f"id {value!r} resolves outside the example image root "
                f"{examples}: {image_path}"
            )
    unmatched = [value for value in ids if basename(value) not in label_of]
    if unmatched:
        raise ValueError(f"ids without an image/label match: {unmatched[:5]}")
    if require_all_examples:
        covered = {basename(value) for value in ids}
        if len(ids) != total_images or covered != set(label_of):
            raise ValueError(
                f"teaching mode uses all {total_images} example images, but this "
                f"run covers {len(ids)} ids and {len(covered)} example images"
            )

    if assignments_csv is None:
        rows = [
            {"id": value, "label": label_of[basename(value)]} for value in ids
        ]
        fieldnames = ["id", "label"]
    else:
        assignments = pd.read_csv(assignments_csv)
        key_column = (
            "id"
            if "id" in assignments.columns
            else ("image" if "image" in assignments.columns else None)
        )
        if key_column is None or "cluster" not in assignments.columns:
            raise ValueError(
                "assignments CSV needs 'id' (or 'image') and 'cluster' columns"
            )
        emitted_ids = [str(value) for value in ids]
        emitted_set = set(emitted_ids)
        cluster_of = {}
        for _, row in assignments.iterrows():
            raw_id = row[key_column]
            raw_cluster = row["cluster"]
            for field, value in ((key_column, raw_id), ("cluster", raw_cluster)):
                missing = value is None or (
                    isinstance(value, float) and math.isnan(value)
                ) or (not isinstance(value, (int, float)) and str(value).strip() == "")
                if missing:
                    raise ValueError(
                        f"assignments CSV has a missing {field!r} value"
                        + ("" if field == key_column else f" for {raw_id!r}")
                    )
            raw = str(raw_id)
            cluster = str(raw_cluster)
            if raw in cluster_of:
                raise ValueError(
                    f"assignments CSV lists {raw!r} more than once; the partition "
                    "must have exactly one row per emitted id"
                )
            if raw not in emitted_set:
                raise ValueError(
                    f"partition id {raw!r} is not an id emitted by extraction; the "
                    "partition must reference the extraction ids exactly, not a "
                    "renamed or re-based copy"
                )
            cluster_of[raw] = cluster
        missing = sorted(emitted_set - set(cluster_of))
        if missing:
            raise ValueError(f"ids missing from the selected partition: {missing[:5]}")

        available = {
            label: {
                cluster_of[value]
                for value in emitted_ids
                if label_of[basename(value)] == label
            }
            for label in set(label_of.values())
        }
        majority = {}
        if label_to_cluster is not None:
            if not isinstance(label_to_cluster, dict):
                raise ValueError(
                    "label_to_cluster must be a mapping of label to cluster"
                )
            unknown = sorted(set(label_to_cluster) - set(label_of.values()))
            if unknown:
                raise ValueError(
                    f"label_to_cluster names labels that are not in the example "
                    f"labels: {unknown}"
                )
            for label, cluster in label_to_cluster.items():
                if not isinstance(cluster, str) or not cluster.strip():
                    raise ValueError(
                        f"label_to_cluster[{label!r}] must be a non-empty cluster "
                        f"name, got {cluster!r}"
                    )
                if cluster not in available.get(label, set()):
                    raise ValueError(
                        f"label_to_cluster[{label!r}] = {cluster!r} is not a cluster "
                        f"of that label in the selected partition "
                        f"({sorted(available.get(label, set()))})"
                    )
                # keep the reviewed cluster ID byte-for-byte
                majority[label] = cluster
        for label in sorted(set(label_of.values())):
            if label in majority:
                continue
            tally = Counter(
                cluster_of[value]
                for value in emitted_ids
                if label_of[basename(value)] == label
            )
            if not tally:
                continue
            ranked = tally.most_common()
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                spread = ", ".join(f"{cluster}={count}" for cluster, count in ranked)
                raise ValueError(
                    f"ambiguous label-to-cluster mapping for {label!r} ({spread}); "
                    "review the mapping and pass label_to_cluster to choose"
                )
            majority[label] = ranked[0][0]

        rows = sorted(
            (
                {"id": value, "cluster": majority[label_of[basename(value)]]}
                for value in emitted_ids
                if majority.get(label_of[basename(value)]) is not None
                and cluster_of[value] != majority[label_of[basename(value)]]
            ),
            key=lambda row: row["id"],
        )
        fieldnames = ["id", "cluster"]
        partition = pd.DataFrame(
            [{"id": value, "cluster": cluster_of[value]} for value in emitted_ids]
        )
        corrections = canonicalize_corrections(
            pd.DataFrame(rows, columns=["id", "cluster"])
        )
        validate_corrections(corrections)
        validate_corrections_against_assignments(partition, corrections)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
```
<!-- teaching-recipe:end -->

Ask for approval of source, mapping, and destination before running it. A
correction run that writes no data rows means no ID deviates from its label's
reviewed cluster at that cutoff; it does **not** prove the label classes are
separated, so report it that way instead of claiming a correction was applied or
that the partition is correct. A tied label-to-cluster mapping raises until the
operator passes an explicitly reviewed `label_to_cluster`. The cutoff is
user-selected; two supplied classes do not authorize declaring an optimal
partition, and a partition this recipe accepts is a demonstration of mechanics,
not an expert audit.

## Isolation and return to user data

Write demonstrations under an approved root such as `runs/run001/demo/`, with
separate directories per step. Keep `examples/` read-only; derived CSVs go under
the demo root. No image subset is created.

After each demonstration, explain artifacts and limitations, then ask whether to
show another step or repeat on user data. Before switching back, restate and
confirm the user's input, checkpoint, label, and output paths and obtain a fresh
full-card approval. Never silently reuse demo artifacts in the user's analysis.
