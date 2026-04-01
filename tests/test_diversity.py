import pandas as pd
import pytest

from pathlib import Path

from otuformer.delineation.diversity import (
    _is_number,
    build_diversity_tables_from_otu_table,
    build_per_sample_paths,
    compute_alpha_diversity,
    dedupe_sample_names,
    detect_otu_table_header,
    filter_by_min_abundance,
    has_valid_samples,
    parse_otu_table,
    sanitize_sample_name,
    split_assignments_by_sample,
)


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
