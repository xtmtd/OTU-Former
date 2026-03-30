import pandas as pd

from otuformer.delineation.diversity import (
    compute_alpha_diversity,
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
