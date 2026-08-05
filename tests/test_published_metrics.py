from pathlib import Path

import numpy as np

from nuclei_counting.analyze_reliability import (
    alternative_group_table,
    image_average,
    load_evaluations,
    operational_group_table,
    predicted_count_groups,
    seed_level_table,
)


ROOT = Path(__file__).resolve().parents[1]


def test_seed_level_values_match_reported_rounding():
    raw = load_evaluations(ROOT / "results" / "per_seed")
    table, summary = seed_level_table(raw)
    assert table["Seed"].tolist() == [42, 123, 2024]
    assert np.allclose(table["MAE"], [13.13, 13.39, 13.20], atol=1e-12)
    assert np.allclose(table["RMSE"], [33.0492057393, 33.6028272629, 32.3428368888], atol=1e-9)
    assert round(float(summary.loc[summary["Statistic"] == "Mean", "MAE"].iloc[0]), 3) == 13.240


def test_operational_high_group_matches_reported_values():
    raw = load_evaluations(ROOT / "results" / "per_seed")
    averaged = image_average(raw)
    grouped, cutoffs = predicted_count_groups(averaged)
    table = operational_group_table(grouped).set_index("Pred_count_group")
    assert cutoffs == {"q33": 18.0, "q67": 34.0}
    assert int(table.loc["High", "n"]) == 34
    assert round(float(table.loc["High", "MAE"]), 2) == 32.38
    assert round(float(table.loc["High", "Recall"]), 3) == 0.575
    assert round(float(table.loc["High", "Missed"]), 2) == 47.13
    assert round(float(table.loc["High", "Merge"]), 2) == 15.34


def test_alternative_group_sizes():
    raw = load_evaluations(ROOT / "results" / "per_seed")
    table = alternative_group_table(raw)
    assert table["n"].tolist() == [98, 74, 103, 99]
