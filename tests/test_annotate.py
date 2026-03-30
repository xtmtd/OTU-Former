import pandas as pd

from otuformer.delineation.annotate import apply_corrections, build_annotation_summary


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
            "corrected_cluster": ["OTU_3"],
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
    corrections = pd.DataFrame({"id": [], "corrected_cluster": []})
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
            "corrected_cluster": ["OTU_3"],
        }
    )
    result = apply_corrections(assignments, corrections)
    summary = build_annotation_summary(assignments, result)
    assert summary["n_corrections"] == 1
    assert summary["n_clusters_affected"] == 2
