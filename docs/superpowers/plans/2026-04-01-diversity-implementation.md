# Diversity Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `otuformer diversity` to support OTU table inputs, per-sample outputs, and improved phylo/MPD handling with clearer help text.

**Architecture:** Keep CLI orchestration in `src/otuformer/cli/diversity.py` and move input validation + metrics into `src/otuformer/delineation/diversity.py`. Introduce small, testable helpers for assignments/OTU-table parsing and MPD computation from counts.

**Tech Stack:** Python, pandas, numpy, typer, scikit-bio (optional for MPD).

---

## File Structure

- Modify: `src/otuformer/cli/diversity.py`
  - CLI options, mutual exclusion, help text, logging, per-sample output routing.
- Modify: `src/otuformer/cli/main.py`
  - Guard for removed `--no-phylo` with guidance.
- Modify: `src/otuformer/delineation/diversity.py`
  - Add OTU table parsing helpers, per-sample table construction, MPD from counts, warnings.
- Modify: `tests/test_diversity.py`
  - Add unit tests for new parsing and per-sample gating logic (no scikit-bio dependency).

---

### Task 1: Add OTU table parsing helpers

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests for OTU table header detection**

```python
import pandas as pd
import pytest

from otuformer.delineation.diversity import detect_otu_table_header


def test_detect_header_by_non_numeric_otu_ids():
    df = pd.DataFrame(
        [
            ["sample", "OTU_1", "OTU_2"],
            ["s1", 3, 0],
        ]
    )
    assert detect_otu_table_header(df) is True


def test_detect_header_numeric_without_flag_errors():
    df = pd.DataFrame(
        [
            ["s1", 1, 2],
            ["s2", 0, 3],
        ]
    )
    with pytest.raises(ValueError):
        detect_otu_table_header(df)


def test_detect_header_numeric_with_flag_allows():
    df = pd.DataFrame(
        [
            ["sample", "1", "2"],
            ["s1", 1, 0],
        ]
    )
    assert detect_otu_table_header(df, has_header=True) is True


def test_is_number_trims_whitespace():
    assert _is_number(" 3 ") is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_diversity.py::test_detect_header_by_non_numeric_otu_ids -v`
Expected: FAIL with import error or missing function.

- [ ] **Step 3: Implement minimal header detection helper**

```python
def detect_otu_table_header(raw: pd.DataFrame, has_header: bool = False) -> bool:
    if raw.empty or raw.shape[1] < 2:
        raise ValueError("OTU table must have at least 2 columns.")
    if has_header:
        return True
    first_row = raw.iloc[0, 1:]
    any_non_numeric = any(
        pd.isna(v) is False and str(v).strip() != "" and not _is_number(v)
        for v in first_row
    )
    if any_non_numeric:
        return True
    raise ValueError("OTU IDs required: provide header row or --otu-table-has-header.")


def _is_number(value: object) -> bool:
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_diversity.py::test_detect_header_by_non_numeric_otu_ids -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "test: add otutable header detection" 
```

---

### Task 2: Parse OTU tables into counts and validate headers

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests for OTU table parsing**

```python
import pandas as pd
import pytest

from otuformer.delineation.diversity import parse_otu_table


def test_parse_otu_table_with_header_and_empty_sample_header():
    df = pd.DataFrame(
        [
            ["", "OTU_1", "OTU_2"],
            ["s1", 2, 0],
            ["s2", 1, 3],
        ]
    )
    otu = parse_otu_table(df, has_header=False)
    assert list(otu.columns) == ["OTU_1", "OTU_2"]
    assert list(otu.index) == ["s1", "s2"]


def test_parse_otu_table_empty_otu_header_errors():
    df = pd.DataFrame(
        [
            ["", "", "OTU_2"],
            ["s1", 2, 0],
        ]
    )
    with pytest.raises(ValueError):
        parse_otu_table(df, has_header=True)


def test_parse_otu_table_numeric_otu_ids_requires_flag():
    df = pd.DataFrame(
        [
            ["sample", "1", "2"],
            ["s1", 2, 0],
        ]
    )
    otu = parse_otu_table(df, has_header=True)
    assert list(otu.columns) == ["1", "2"]


def test_parse_otu_table_with_bom_and_quotes():
    raw = pd.DataFrame(
        [
            ["", "OTU_1", "OTU_2"],
            ["s1", "2", "0"],
        ]
    )
    otu = parse_otu_table(raw, has_header=False)
    assert list(otu.columns) == ["OTU_1", "OTU_2"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_diversity.py::test_parse_otu_table_with_header_and_empty_sample_header -v`
Expected: FAIL with missing function.

- [ ] **Step 3: Implement minimal OTU parsing**

```python
def parse_otu_table(raw: pd.DataFrame, has_header: bool) -> pd.DataFrame:
    header = detect_otu_table_header(raw, has_header=has_header)
    if not header:
        raise ValueError("OTU table requires header row.")
    otu_ids = raw.iloc[0, 1:].astype(str).str.strip().tolist()
    if any(otu == "" for otu in otu_ids):
        raise ValueError("OTU header cells (columns 2..N) must be non-empty.")
    data = raw.iloc[1:, :].copy()
    data.columns = ["sample"] + otu_ids
    data["sample"] = data["sample"].astype(str)
    data = data.set_index("sample")
    return data
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_diversity.py::test_parse_otu_table_with_header_and_empty_sample_header -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat: parse otutable inputs" 
```

---

### Task 3: Add per-sample gating and file naming helpers

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests for sample validity and sanitization**

```python
import pandas as pd

from otuformer.delineation.diversity import has_valid_samples, sanitize_sample_name


def test_has_valid_samples_false_on_empty():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"], "sample": [" "]})
    assert has_valid_samples(df) is False


def test_sanitize_sample_name_replaces_spaces():
    assert sanitize_sample_name("A/B C") == "A_B_C"


def test_sanitize_sample_name_dedup_suffix():
    names = ["s1", "s1"]
    assert dedupe_sample_names(names) == ["s1", "s1_2"]


def test_split_assignments_by_sample():
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "cluster": ["OTU_1", "OTU_1"],
            "sample": ["s1", "s2"],
        }
    )
    parts = split_assignments_by_sample(df)
    assert set(parts.keys()) == {"s1", "s2"}


def test_build_per_sample_paths_dedupe():
    samples = ["s1", "s1"]
    paths = build_per_sample_paths(samples, Path("out"))
    assert list(paths.values())[0].name == "s1.csv"
    assert list(paths.values())[1].name == "s1_2.csv"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_diversity.py::test_has_valid_samples_false_on_empty -v`
Expected: FAIL with missing functions.

- [ ] **Step 3: Implement minimal helpers**

```python
def has_valid_samples(assignments: pd.DataFrame) -> bool:
    if "sample" not in assignments.columns:
        return False
    cleaned = assignments["sample"].astype(str).str.strip()
    return cleaned.ne("").all()


def sanitize_sample_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    return cleaned.replace("/", "_").replace("\\", "_")


def dedupe_sample_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        output.append(name if count == 1 else f"{name}_{count}")
    return output


def split_assignments_by_sample(assignments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped = {}
    for sample, chunk in assignments.groupby("sample"):
        grouped[str(sample)] = chunk.copy()
    return grouped


def build_per_sample_paths(samples: list[str], out_dir: Path) -> dict[str, Path]:
    sanitized = [sanitize_sample_name(s) for s in samples]
    deduped = dedupe_sample_names(sanitized)
    return {s: out_dir / f"{name}.csv" for s, name in zip(samples, deduped)}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_diversity.py::test_has_valid_samples_false_on_empty -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat: add sample validation helpers" 
```

---

### Task 4: Update MPD computation and warnings

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`

- [ ] **Step 1: Write failing test for MPD helper signature (no scikit-bio call)**

```python
from otuformer.delineation.diversity import build_mpd_inputs


def test_build_mpd_inputs_aligns_otu_ids():
    otu_ids = ["OTU_1", "OTU_2"]
    counts = [3, 1]
    ids, arr = build_mpd_inputs(otu_ids, counts)
    assert ids == otu_ids
    assert arr.tolist() == [3, 1]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_diversity.py::test_build_mpd_inputs_aligns_otu_ids -v`
Expected: FAIL with missing function.

- [ ] **Step 3: Implement MPD helpers and update compute_mpd**

```python
def build_mpd_inputs(otu_ids: list[str], counts: list[int]) -> tuple[list[str], np.ndarray]:
    return otu_ids, np.array(counts, dtype=int)


def compute_mpd_from_counts(
    otu_ids: list[str], counts: np.ndarray, tree_newick_path: Path
) -> float:
    import warnings

    from skbio import TreeNode
    from skbio.diversity import alpha_diversity

    tree = TreeNode.read(str(tree_newick_path))
    tip_names = set(tree.subset_nonempty_tips().names)
    otu_set = set(otu_ids)
    missing_from_tree = otu_set - tip_names
    extra_in_tree = tip_names - otu_set
    if missing_from_tree:
        warnings.warn(f"OTUs missing from tree: {missing_from_tree}; dropped for MPD.")
    if extra_in_tree:
        warnings.warn(f"Extra tree tips not in OTU table: {extra_in_tree}; ignored for MPD.")
    overlap = [oid for oid in otu_ids if oid in tip_names]
    if not overlap:
        return float("nan")
    overlap_counts = np.array([counts[otu_ids.index(oid)] for oid in overlap])
    tree_pruned = tree.shear(overlap)
    result = alpha_diversity("mpd", overlap_counts.reshape(1, -1), overlap, tree=tree_pruned)
    return float(result[0])
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_diversity.py::test_build_mpd_inputs_aligns_otu_ids -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat: add mpd helpers" 
```

---

### Task 5: Update assignments normalization, per-sample outputs, and CLI flow

**Files:**
- Modify: `src/otuformer/cli/diversity.py`
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests for assignments parsing and per-sample gating**

```python
import pandas as pd
import pytest

from otuformer.delineation.diversity import normalize_assignments


def test_normalize_assignments_rejects_id_and_image():
    df = pd.DataFrame({"id": ["a"], "image": ["b"], "cluster": ["OTU"]})
    with pytest.raises(ValueError):
        normalize_assignments(df)


def test_assignments_with_empty_sample_disables_per_sample():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"], "sample": [""]})
    assert has_valid_samples(df) is False


def test_assignments_without_sample_global_only():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"]})
    assert has_valid_samples(df) is False


def test_detect_removed_no_phylo():
    assert detect_removed_no_phylo(["otuformer", "diversity", "--no-phylo"]) is True


def test_validate_input_sources():
    with pytest.raises(ValueError):
        validate_input_sources(None, None)
    with pytest.raises(ValueError):
        validate_input_sources(Path("a.csv"), Path("b.csv"))


def test_build_diversity_tables_from_otu_table_thresholds():
    otu = pd.DataFrame({"OTU_1": [2, 0], "OTU_2": [1, 3]}, index=["s1", "s2"])
    global_table, per_sample = build_diversity_tables_from_otu_table(otu, [0, 2], None)
    assert set(global_table.columns) == {"min_abundance_0", "min_abundance_2"}
    assert all(set(t.columns) == {"min_abundance_0", "min_abundance_2"} for t in per_sample.values())
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_diversity.py::test_normalize_assignments_rejects_id_and_image -v`
Expected: FAIL with missing function.

- [ ] **Step 3: Implement normalization + CLI integration**

```python
# delineation/diversity.py
def normalize_assignments(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.strip().lower(): c for c in df.columns}
    if "id" in cols and "image" in cols:
        raise ValueError("Assignments cannot contain both id and image columns.")
    if "id" in cols:
        df = df.rename(columns={cols["id"]: "id"})
    elif "image" in cols:
        df = df.rename(columns={cols["image"]: "id"})
    else:
        raise ValueError("Assignments must include id or image column.")
    if "cluster" not in cols:
        raise ValueError("Assignments must include cluster column.")
    df = df.rename(columns={cols["cluster"]: "cluster"})
    if "sample" in cols:
        df = df.rename(columns={cols["sample"]: "sample"})
    return df
```

```python
# cli/diversity.py
def validate_input_sources(assignments: Path | None, otu_table_csv: Path | None) -> None:
    if (assignments is None) == (otu_table_csv is None):
        raise ValueError("Provide exactly one of --assignments or --otu-table-csv")
```

```python
# cli/diversity.py (outline)
validate_input_sources(assignments, otu_table_csv)
if phylo and tree is None:
    typer.echo("Warning: --phylo set but no --tree provided; MPD skipped.")
if tree is not None and not phylo:
    typer.echo("Warning: --tree provided without --phylo; tree ignored.")
if assignments is not None:
    assignments_df = normalize_assignments(read_csv(assignments, encoding="utf-8-sig"))
    if has_valid_samples(assignments_df):
        per_sample_dir = out_dir / "per-sample"
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        parts = split_assignments_by_sample(assignments_df)
        paths = build_per_sample_paths(list(parts.keys()), per_sample_dir)
        for sample, subset in parts.items():
            table = diversity_table(subset, min_values, tree_newick_path=tree if phylo else None)
            table.reset_index().rename(columns={"index": "metric"}).to_csv(paths[sample], index=False)
    else:
        typer.echo("Warning: sample column missing or empty; per-sample outputs skipped.")
else:
    otu_raw = read_csv(otu_table_csv, encoding="utf-8-sig")
    otu_table = parse_otu_table(otu_raw, has_header=otu_table_has_header)
    global_table, per_sample = build_diversity_tables_from_otu_table(
        otu_table, min_values, tree_newick_path=tree if phylo else None
    )
    global_table.reset_index().rename(columns={"index": "metric"}).to_csv(
        out_dir / "diversity_indices.csv", index=False
    )
    per_sample_dir = out_dir / "per-sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    paths = build_per_sample_paths(list(per_sample.keys()), per_sample_dir)
    for sample, table in per_sample.items():
        table.reset_index().rename(columns={"index": "metric"}).to_csv(paths[sample], index=False)
```

```python
# delineation/diversity.py
def build_diversity_tables_from_otu_table(
    otu_table: pd.DataFrame,
    min_values: list[int],
    tree_newick_path: Optional[Path],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    global_counts = otu_table.sum(axis=0)
    global_assignments = pd.DataFrame(
        {"cluster": global_counts.index.repeat(global_counts.values.astype(int))}
    )
    global_table = diversity_table(global_assignments, min_values, tree_newick_path)
    per_sample = {}
    for sample, row in otu_table.iterrows():
        assignments = pd.DataFrame(
            {"cluster": row.index.repeat(row.values.astype(int))}
        )
        per_sample[str(sample)] = diversity_table(assignments, min_values, tree_newick_path)
    return global_table, per_sample
```

```python
# cli/main.py
def detect_removed_no_phylo(argv: list[str]) -> bool:
    return "diversity" in argv and "--no-phylo" in argv


if detect_removed_no_phylo(sys.argv[1:]):
    typer.echo("Error: --no-phylo is removed; use --phylo to enable MPD.")
    raise typer.Exit(code=2)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_diversity.py::test_normalize_assignments_rejects_id_and_image -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/cli/diversity.py src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat: support otutable inputs and per-sample outputs" 
```

---

### Task 6: Add phylo warnings, MPD omission, and tree mismatch handling

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Modify: `src/otuformer/cli/diversity.py`
- Test: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests for MPD omission and tree mismatch warnings**

```python
import pandas as pd
import pytest

from otuformer.delineation.diversity import diversity_table


def test_diversity_table_no_mpd_when_tree_missing():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU_1"]})
    table = diversity_table(df, [0], tree_newick_path=None)
    assert "MPD" not in table.index


def test_mpd_tree_missing_otus_warns():
    pytest.importorskip("skbio")
    # Placeholder: construct a tiny tree with missing OTU tip and assert warning emitted.
    # Use pytest.warns to validate the warning.
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_diversity.py::test_diversity_table_no_mpd_when_tree_missing -v`
Expected: FAIL (MPD present or missing function updates).

- [ ] **Step 3: Implement MPD omission and mismatch warnings**

```python
if tree_newick_path is not None:
    mpd_value = compute_mpd_from_counts(...)
    if not np.isnan(mpd_value):
        metrics["MPD"] = mpd_value
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_diversity.py::test_diversity_table_no_mpd_when_tree_missing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py src/otuformer/cli/diversity.py tests/test_diversity.py
git commit -m "fix: mpd warnings and omission" 
```

---

### Task 7: Add per-sample output file tests

**Files:**
- Test: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests for per-sample file creation**

```python
from pathlib import Path

from otuformer.delineation.diversity import build_diversity_tables_from_otu_table


def test_per_sample_files_created(tmp_path: Path):
    otu = pd.DataFrame({"OTU_1": [2, 0], "OTU_2": [1, 3]}, index=["s1", "s2"])
    global_table, per_sample = build_diversity_tables_from_otu_table(otu, [0], None)
    per_sample_dir = tmp_path / "per-sample"
    per_sample_dir.mkdir()
    paths = build_per_sample_paths(list(per_sample.keys()), per_sample_dir)
    for sample, table in per_sample.items():
        table.reset_index().rename(columns={"index": "metric"}).to_csv(paths[sample], index=False)
    assert (per_sample_dir / "s1.csv").exists()
    assert (per_sample_dir / "s2.csv").exists()


def test_otu_table_valid_samples_gates_per_sample():
    otu = pd.DataFrame({"OTU_1": [2, 0]}, index=["s1", "s2"])
    samples = otu.index.tolist()
    assert all(s.strip() != "" for s in samples)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_diversity.py::test_per_sample_files_created -v`
Expected: FAIL or PASS depending on helpers availability.

- [ ] **Step 3: Verify implementation**

Run: `pytest tests/test_diversity.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_diversity.py
git commit -m "test: per-sample output file creation" 
```

---

### Task 8: Update help text and run full tests

**Files:**
- Modify: `src/otuformer/cli/diversity.py`

- [ ] **Step 1: Expand help text for inputs and metrics**

```python
app = typer.Typer(
    help=(
        "Compute diversity indices from OTU assignments or OTU tables.\n\n"
        "Input modes (one required, mutually exclusive):\n\n"
        "  --assignments: CSV with id/image, cluster, optional sample columns.\n"
        "  --otu-table-csv: CSV where column 1 is sample, columns 2..N are OTU IDs with abundances.\n\n"
        "Outputs:\n\n"
        "  diversity_indices.csv - global (all-samples) diversity table.\n"
        "  per-sample/ - one file per sample when sample labels are valid.\n\n"
        "Metrics:\n\n"
        "  Richness: number of unique OTUs.\n"
        "  Chao1: estimated richness (accounts for rare OTUs).\n"
        "  ACE: abundance-based coverage estimator.\n"
        "  Shannon: entropy-based diversity (higher = more diverse).\n"
        "  Simpson: probability two individuals differ (higher = more diverse).\n"
        "  Hill_q0/q1/q2: Hill numbers (richness/evenness/diversity at orders 0,1,2).\n"
        "  Pielou_J: evenness (Shannon / log(richness)).\n"
        "  MPD: mean pairwise phylogenetic distance across OTUs (--phylo + --tree required).\n\n"
        "Note: --no-phylo has been removed; omit --phylo to disable MPD."
    )
)
```

- [ ] **Step 2: Run full test suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/otuformer/cli/diversity.py
git commit -m "docs: expand diversity help text" 
```
