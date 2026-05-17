import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from otuformer.delineation.diversity import (
    _is_number,
    build_diversity_tables_from_otu_table,
    build_mpd_inputs,
    build_per_sample_paths,
    compute_alpha_diversity,
    dedupe_sample_names,
    detect_otu_table_header,
    diversity_table,
    filter_by_min_abundance,
    has_valid_samples,
    normalize_assignments,
    parse_otu_table,
    sanitize_sample_name,
    split_assignments_by_sample,
)
from otuformer.cli.diversity import validate_input_sources


def sample_assignments():
    return pd.DataFrame(
        {
            "id": [f"img{i}" for i in range(20)],
            "cluster": ["OTU_1"] * 10 + ["OTU_2"] * 5 + ["OTU_3"] * 3 + ["OTU_4"] * 2,
        }
    )


def test_alpha_diversity_returns_all_indices():
    df = sample_assignments()
    result = compute_alpha_diversity(df)
    expected_keys = [
        "Richness",
        "Shannon",
        "Simpson",
        "InverseSimpson",
        "Pielou_J",
        "Chao1",
        "Berger_Parker",
        "Hill_q0",
        "Hill_q1",
        "Hill_q2",
    ]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"


def test_richness_correct():
    df = sample_assignments()
    result = compute_alpha_diversity(df)
    assert result["Richness"] == 4


def test_filter_by_min_abundance():
    df = sample_assignments()
    filtered = filter_by_min_abundance(df, min_abundance=3)
    assert "OTU_4" not in filtered["cluster"].values
    assert "OTU_1" in filtered["cluster"].values


def test_filter_zero_returns_all():
    df = sample_assignments()
    filtered = filter_by_min_abundance(df, min_abundance=0)
    assert len(filtered) == len(df)


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


def test_parse_otu_table_non_numeric_otu_ids():
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


def test_parse_otu_table_numeric_otu_ids_with_flag():
    df = pd.DataFrame(
        [
            ["sample", "1", "2"],
            ["s1", 2, 0],
        ]
    )
    otu = parse_otu_table(df, has_header=True)
    assert list(otu.columns) == ["1", "2"]


def test_parse_otu_table_too_few_rows_errors():
    df = pd.DataFrame(
        [
            ["", "OTU_1", "OTU_2"],
        ]
    )
    with pytest.raises(ValueError, match="at least one data row"):
        parse_otu_table(df, has_header=False)


def test_has_valid_samples_false_on_empty():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"], "sample": [" "]})
    assert has_valid_samples(df) is False


def test_has_valid_samples_true():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"], "sample": ["s1"]})
    assert has_valid_samples(df) is True


def test_has_valid_samples_missing_column():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"]})
    assert has_valid_samples(df) is False


def test_has_valid_samples_nan_invalid():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"], "sample": [None]})
    assert has_valid_samples(df) is False


def test_sanitize_sample_name_replaces_spaces():
    assert sanitize_sample_name("A/B C") == "A_B_C"


def test_sanitize_sample_name_replaces_tabs():
    assert sanitize_sample_name("A\tB") == "A_B"


def test_dedupe_sample_names():
    names = ["s1", "s1", "s2", "s1"]
    assert dedupe_sample_names(names) == ["s1", "s1_2", "s2", "s1_3"]


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


def test_build_diversity_tables_from_otu_table_thresholds():
    otu = pd.DataFrame({"OTU_1": [2, 0], "OTU_2": [1, 3]}, index=["s1", "s2"])
    global_table, per_sample = build_diversity_tables_from_otu_table(otu, [0, 2], None)
    assert set(global_table.columns) == {"min_abundance_0", "min_abundance_2"}
    assert all(
        set(t.columns) == {"min_abundance_0", "min_abundance_2"}
        for t in per_sample.values()
    )


def test_build_diversity_tables_from_otu_table_tree_path_uses_legacy_kwarg(tmp_path: Path):
    pytest.importorskip("skbio")
    otu = pd.DataFrame({"OTU_A": [2, 0], "OTU_B": [1, 3], "OTU_C": [1, 1]}, index=["s1", "s2"])
    tree_path = tmp_path / "legacy_tree.nwk"
    tree_path.write_text("((OTU_A:1.0,OTU_B:1.0):0.5,OTU_C:1.5);", encoding="utf-8")

    global_table, per_sample = build_diversity_tables_from_otu_table(
        otu, [0], tree_newick_path=tree_path
    )

    assert np.isfinite(global_table.loc["MPD", "min_abundance_0"])
    assert np.isfinite(global_table.loc["MPD_w", "min_abundance_0"])
    assert all(np.isfinite(table.loc["MPD", "min_abundance_0"]) for table in per_sample.values())


def test_build_mpd_inputs_aligns_otu_ids():
    otu_ids = ["OTU_1", "OTU_2"]
    counts = [3, 1]
    ids, arr = build_mpd_inputs(otu_ids, counts)
    assert ids == otu_ids
    assert arr.tolist() == [3, 1]


def test_diversity_table_no_mpd_when_tree_missing():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU_1"]})
    table = diversity_table(df, [0], tree_newick_path=None)
    assert "MPD" not in table.index


def test_normalize_assignments_rejects_id_and_image():
    df = pd.DataFrame({"id": ["a"], "image": ["b"], "cluster": ["OTU"]})
    with pytest.raises(ValueError):
        normalize_assignments(df)


def test_normalize_assignments_accepts_id():
    df = pd.DataFrame({"id": ["a"], "cluster": ["OTU"], "sample": ["s1"]})
    result = normalize_assignments(df)
    assert "id" in result.columns
    assert "cluster" in result.columns
    assert "sample" in result.columns


def test_normalize_assignments_accepts_image():
    df = pd.DataFrame({"image": ["a"], "cluster": ["OTU"]})
    result = normalize_assignments(df)
    assert "id" in result.columns


def test_normalize_assignments_missing_cluster_errors():
    df = pd.DataFrame({"id": ["a"]})
    with pytest.raises(ValueError):
        normalize_assignments(df)


def test_validate_input_sources_both_errors():
    with pytest.raises(ValueError):
        validate_input_sources(Path("a.csv"), Path("b.csv"))


def test_validate_input_sources_neither_errors():
    with pytest.raises(ValueError):
        validate_input_sources(None, None)


def test_validate_input_sources_ok():
    validate_input_sources(Path("a.csv"), None)
    validate_input_sources(None, Path("b.csv"))


def test_mpd_tree_missing_otus_warns(tmp_path: Path):
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_mpd_from_counts

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((OTU1:1.0,OTU2:1.0):0.5,OTU3:1.5);")
    otu_ids = ["OTU1", "OTU3", "OTU4"]
    counts = [3, 1, 2]
    with pytest.warns(UserWarning, match="missing from tree"):
        compute_mpd_from_counts(otu_ids, counts, tree_path)


def test_mpd_preserves_underscores_in_tip_names(tmp_path: Path):
    """Tip names with underscores should NOT be converted to spaces."""
    pytest.importorskip("skbio")
    import warnings
    from otuformer.delineation.diversity import compute_mpd_from_counts

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((img_01:1.0,img_02:1.0):0.5,img_03:1.5);")
    otu_ids = ["img_01", "img_02", "img_03"]
    counts = [2, 3, 1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_mpd_from_counts(otu_ids, counts, tree_path)
        mpd_warnings = [
            x
            for x in w
            if "missing from tree" in str(x.message)
            or "Extra tree tips" in str(x.message)
        ]
        assert len(mpd_warnings) == 0
    assert result["MPD"] > 0
    assert result["MPD_w"] > 0
    assert result["PD_richness_norm"] > 0


def test_mpd_no_warning_when_names_match_exactly(tmp_path: Path):
    """No warnings when all tip names match exactly."""
    pytest.importorskip("skbio")
    import warnings
    from otuformer.delineation.diversity import compute_mpd_from_counts

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((A:1.0,B:2.0):0.5,C:1.5);")
    otu_ids = ["A", "B", "C"]
    counts = [1, 2, 3]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_mpd_from_counts(otu_ids, counts, tree_path)
        mpd_warnings = [
            x
            for x in w
            if "missing from tree" in str(x.message)
            or "Extra tree tips" in str(x.message)
        ]
        assert len(mpd_warnings) == 0
    assert result["MPD"] > 0


def test_mpd_uses_id_column_when_available(tmp_path: Path):
    """compute_mpd should use 'id' column (tip names) not 'cluster' column."""
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_mpd

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((img_01:1.0,img_02:1.0):0.5,img_03:1.5);")
    assignments = pd.DataFrame(
        {
            "id": ["img_01", "img_01", "img_02", "img_03"],
            "cluster": ["OTU_A", "OTU_A", "OTU_B", "OTU_C"],
        }
    )
    result = compute_mpd(assignments, tree_path)
    assert not (result["MPD"] != result["MPD"])


def test_mpd_returns_nan_when_no_overlap(tmp_path: Path):
    """compute_mpd should return NaN when no tip names match."""
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_mpd_from_counts

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((X:1.0,Y:1.0):0.5,Z:1.5);")
    otu_ids = ["A", "B", "C"]
    counts = [1, 2, 3]
    result = compute_mpd_from_counts(otu_ids, counts, tree_path)
    assert result["MPD"] != result["MPD"]
    assert result["MPD_w"] != result["MPD_w"]
    assert result["PD_richness_norm"] != result["PD_richness_norm"]


def test_mpd_returns_dict_with_all_metrics(tmp_path: Path):
    """compute_mpd_from_counts should return dict with MPD, MPD_w, PD_richness_norm."""
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_mpd_from_counts

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((A:1.0,B:2.0):0.5,C:1.5);")
    otu_ids = ["A", "B", "C"]
    counts = [1, 2, 3]
    result = compute_mpd_from_counts(otu_ids, counts, tree_path)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"MPD", "MPD_w", "PD_richness_norm"}


def test_pd_richness_norm_equals_mpd_div_richness(tmp_path: Path):
    """PD_richness_norm should equal MPD / richness."""
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_mpd_from_counts

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((A:1.0,B:2.0):0.5,C:1.5);")
    otu_ids = ["A", "B", "C"]
    counts = [1, 2, 3]
    result = compute_mpd_from_counts(otu_ids, counts, tree_path)
    expected = result["MPD"] / 3
    assert abs(result["PD_richness_norm"] - expected) < 1e-10


def test_diversity_table_with_tree_returns_mpd(tmp_path: Path):
    """diversity_table should include MPD, MPD_w, PD_richness_norm when tree is provided."""
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import diversity_table

    tree_path = tmp_path / "test.nwk"
    tree_path.write_text("((img_01:1.0,img_02:1.0):0.5,img_03:1.5);")
    assignments = pd.DataFrame(
        {
            "id": ["img_01", "img_01", "img_02", "img_03"],
            "cluster": ["OTU_A", "OTU_A", "OTU_B", "OTU_C"],
        }
    )
    table = diversity_table(assignments, [0], tree_newick_path=tree_path)
    col = table["min_abundance_0"]
    assert "MPD" in col.index
    assert "MPD_w" in col.index
    assert "PD_richness_norm" in col.index


def test_compute_otu_centroids_shape():
    from otuformer.delineation.diversity import compute_otu_centroids
    assignments = pd.DataFrame({
        "id":      ["i1", "i2", "i3", "i4"],
        "cluster": ["OTU_1", "OTU_1", "OTU_2", "OTU_2"],
    })
    embeddings = pd.DataFrame({
        "id":    ["i1", "i2", "i3", "i4"],
        "dim_0": [1.0, 3.0, 0.0, 0.0],
        "dim_1": [0.0, 0.0, 1.0, 3.0],
    })
    centroids, otu_ids = compute_otu_centroids(assignments, embeddings)
    assert centroids.shape == (2, 2)
    assert set(otu_ids) == {"OTU_1", "OTU_2"}


def test_compute_otu_centroids_l2_normalised():
    from otuformer.delineation.diversity import compute_otu_centroids
    assignments = pd.DataFrame({
        "id":      ["i1", "i2"],
        "cluster": ["OTU_A", "OTU_B"],
    })
    embeddings = pd.DataFrame({
        "id":    ["i1", "i2"],
        "dim_0": [3.0, 0.0],
        "dim_1": [4.0, 5.0],
    })
    centroids, _ = compute_otu_centroids(assignments, embeddings)
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_compute_otu_centroids_missing_ids_ignored():
    """Individuals in assignments but absent from embeddings are silently dropped."""
    from otuformer.delineation.diversity import compute_otu_centroids
    assignments = pd.DataFrame({
        "id":      ["i1", "i2", "MISSING"],
        "cluster": ["OTU_1", "OTU_1", "OTU_1"],
    })
    embeddings = pd.DataFrame({
        "id":    ["i1", "i2"],
        "dim_0": [1.0, 1.0],
        "dim_1": [0.0, 0.0],
    })
    centroids, otu_ids = compute_otu_centroids(assignments, embeddings)
    assert centroids.shape == (1, 2)
    assert otu_ids == ["OTU_1"]


# ---------------------------------------------------------------------------
# NJ-centroid PD tests
# ---------------------------------------------------------------------------

def _make_assignments_and_embeddings(n_otus: int = 4, members_per_otu: int = 3, n_dims: int = 8, seed: int = 0):
    """Helper: synthetic assignments + embeddings with n_otus distinct clusters."""
    rng = np.random.default_rng(seed)
    rows_a, rows_e = [], []
    idx = 0
    for otu_i in range(n_otus):
        center = rng.standard_normal(n_dims)
        for _ in range(members_per_otu):
            iid = f"img_{idx:04d}"
            emb = center + rng.standard_normal(n_dims) * 0.05
            rows_a.append({"id": iid, "cluster": f"OTU_{otu_i + 1}"})
            row_e = {"id": iid}
            row_e.update({f"dim_{d}": float(emb[d]) for d in range(n_dims)})
            rows_e.append(row_e)
            idx += 1
    return pd.DataFrame(rows_a), pd.DataFrame(rows_e)


def test_compute_pd_returns_three_keys():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments, embeddings = _make_assignments_and_embeddings()
    result = compute_pd(assignments, embeddings)
    assert set(result.keys()) == {"MPD", "MPD_w", "PD_richness_norm"}


def test_compute_pd_values_are_finite():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments, embeddings = _make_assignments_and_embeddings()
    result = compute_pd(assignments, embeddings)
    for k, v in result.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_compute_pd_single_otu_returns_nan():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments = pd.DataFrame({"id": ["i1", "i2"], "cluster": ["OTU_1", "OTU_1"]})
    embeddings = pd.DataFrame({"id": ["i1", "i2"], "dim_0": [1.0, 1.0], "dim_1": [0.0, 0.0]})
    result = compute_pd(assignments, embeddings)
    for v in result.values():
        assert np.isnan(v)


def test_compute_pd_richness_norm_relation():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments, embeddings = _make_assignments_and_embeddings(n_otus=5)
    result = compute_pd(assignments, embeddings)
    n_otus = assignments["cluster"].nunique()
    expected_norm = result["MPD"] / n_otus
    assert abs(result["PD_richness_norm"] - expected_norm) < 1e-9
