import pandas as pd
import pytest

from otuformer.delineation.annotate import (
    apply_corrections,
    build_annotation_summary,
    build_changed_only_table,
    build_otu_table,
    canonicalize_corrections,
    parse_bootstrap_support_from_newick,
    validate_raw_assignments,
)


def test_corrections_override_assignments():
    assignments = pd.DataFrame(
        {
            "id": ["img1", "img2", "img3"],
            "cluster": ["OTU_1", "OTU_1", "OTU_2"],
        }
    )
    corrections = pd.DataFrame(
        {
            "id": ["img2"],
            "cluster": ["OTU_3"],
        }
    )
    result = apply_corrections(assignments, corrections)
    assert result.loc[result["id"] == "img1", "cluster"].iloc[0] == "OTU_1"
    assert result.loc[result["id"] == "img2", "cluster"].iloc[0] == "OTU_3"
    assert result.loc[result["id"] == "img3", "cluster"].iloc[0] == "OTU_2"


def test_uncorrected_ids_unchanged():
    assignments = pd.DataFrame(
        {
            "id": ["a", "b"],
            "cluster": ["OTU_1", "OTU_2"],
        }
    )
    corrections = pd.DataFrame({"id": [], "cluster": []})
    result = apply_corrections(assignments, corrections)
    assert list(result["cluster"]) == ["OTU_1", "OTU_2"]


def test_annotation_summary_counts():
    assignments = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "cluster": ["OTU_1", "OTU_1", "OTU_2"],
        }
    )
    corrections = pd.DataFrame(
        {
            "id": ["b"],
            "cluster": ["OTU_3"],
        }
    )
    result = apply_corrections(assignments, corrections)
    summary = build_annotation_summary(assignments, result)
    assert summary["n_corrections"] == 1
    assert summary["n_clusters_affected"] == 2


def test_corrections_accept_image_alias():
    corrections = pd.DataFrame({"image": ["img1"], "cluster": ["OTU_2"]})
    canon = canonicalize_corrections(corrections)
    assert list(canon.columns[:2]) == ["id", "cluster"]
    assert canon.loc[0, "id"] == "img1"


def test_validate_raw_assignments_requires_columns():
    with pytest.raises(ValueError):
        validate_raw_assignments(pd.DataFrame({"id": ["x"]}))


def test_strict_missing_correction_ids_fail():
    assignments = pd.DataFrame({"id": ["a"], "cluster": ["OTU_1"]})
    corrections = pd.DataFrame({"id": ["b"], "cluster": ["OTU_2"]})
    with pytest.raises(ValueError):
        apply_corrections(assignments, corrections)


def test_conflicting_duplicate_corrections_fail():
    assignments = pd.DataFrame({"id": ["a"], "cluster": ["OTU_1"]})
    corrections = pd.DataFrame({"id": ["a", "a"], "cluster": ["OTU_2", "OTU_3"]})
    with pytest.raises(ValueError):
        apply_corrections(assignments, corrections)


def test_changed_only_rows_include_old_and_new_cluster():
    original = pd.DataFrame(
        {"id": ["a", "b"], "cluster": ["OTU_1", "OTU_1"], "sample": ["S1", "S1"]}
    )
    corrected = pd.DataFrame(
        {"id": ["a", "b"], "cluster": ["OTU_2", "OTU_1"], "sample": ["S1", "S1"]}
    )
    changed = build_changed_only_table(original, corrected)
    assert list(changed["id"]) == ["a"]
    assert changed.loc[changed["id"] == "a", "old_cluster"].iloc[0] == "OTU_1"
    assert changed.loc[changed["id"] == "a", "new_cluster"].iloc[0] == "OTU_2"


def test_otu_table_with_and_without_sample():
    with_sample = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "cluster": ["OTU_1", "OTU_1", "OTU_2"],
            "sample": ["S1", "S1", "S2"],
        }
    )
    otu = build_otu_table(with_sample)
    assert "sample" in otu.columns
    assert "OTU_1" in otu.columns
    assert "OTU_2" in otu.columns

    no_sample = pd.DataFrame({"id": ["a", "b"], "cluster": ["OTU_1", "OTU_1"]})
    otu_no_sample = build_otu_table(no_sample)
    assert otu_no_sample.loc[0, "sample"] == "all_samples"


def test_parse_bootstrap_support_from_newick(tmp_path):
    nwk = "((a:0.1,b:0.1)87.0:0.2,c:0.3)100.0;"
    path = tmp_path / "tree.nwk"
    path.write_text(nwk, encoding="utf-8")

    support = parse_bootstrap_support_from_newick(path, ["a", "b", "c"])
    assert support[frozenset({"a", "b"})] == 87.0
