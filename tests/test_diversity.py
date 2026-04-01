import pandas as pd
import pytest

from otuformer.delineation.diversity import (
    _is_number,
    compute_alpha_diversity,
    detect_otu_table_header,
    filter_by_min_abundance,
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
