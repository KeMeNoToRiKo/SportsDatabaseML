"""Tests for Group 24: normal, missing, boundary, invalid and uncertain cases, plus team balance.

Run:  pytest -q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import team_balancer as tb

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def pipeline():
    raw = tb.load_players(DATA / "players_raw.csv")
    usable, issues = tb.validate_players(raw)
    fills = tb.compute_fill_values(usable)
    clean = tb.impute(usable, fills)
    scaler = tb.fit_scaler(clean)
    X = tb.scale(scaler, clean)
    model = tb.fit_kmeans(X, 4, clean)
    return {"raw": raw, "issues": issues, "fills": fills, "clean": clean,
            "scaler": scaler, "X": X, "model": model}


def player(**overrides):
    row = {"player_id": "X1", "player_name": "Test", "preferred_position": "Guard", "shooting": 7,
           "passing": 7, "defense": 5, "rebounding": 4, "fitness_level": 7, "availability_pct": 80}
    row.update(overrides)
    return pd.DataFrame([row])


# --- Normal case -------------------------------------------------------------
def test_sample_roster_cleans_to_60_players(pipeline):
    assert len(pipeline["raw"]) == 62
    assert len(pipeline["clean"]) == 60
    assert pipeline["clean"][tb.FEATURES + [tb.POSITION_COL]].notna().all().all()
    assert (pipeline["issues"]["severity"] == "rejected").sum() == 0


def test_normal_player_is_accepted_without_issues():
    usable, issues = tb.validate_players(player())
    assert len(usable) == 1 and issues.empty


def test_kmeans_finds_four_ordered_groups(pipeline):
    labels = pipeline["model"].labels_
    assert sorted(np.unique(labels)) == [0, 1, 2, 3]
    strength = tb.overall_skill(pipeline["clean"]).groupby(labels).mean()
    assert strength.is_monotonic_decreasing  # C1 strongest ... C4 weakest


def test_saved_style_predict_matches_training_labels(pipeline):
    assert (pipeline["model"].predict(pipeline["X"]) == pipeline["model"].labels_).all()


# --- Missing values ----------------------------------------------------------
def test_duplicates_removed_and_missing_values_imputed(pipeline):
    issues = pipeline["issues"]
    assert "Removed 2 exact duplicate row(s)" in issues["issue"].tolist()
    assert (issues["severity"] == "warning").sum() == 5
    clean = pipeline["clean"].set_index("player_id")
    assert clean.at["P008", "fitness_level"] == pipeline["fills"]["fitness_level"]
    assert clean.at["P053", tb.POSITION_COL] == pipeline["fills"][tb.POSITION_COL]
    assert clean["was_imputed"].sum() == 5


def test_one_missing_value_is_imputed_but_flagged():
    usable, issues = tb.validate_players(player(fitness_level=None))
    assert len(usable) == 1
    assert issues.iloc[0]["severity"] == "warning"


def test_too_many_missing_values_is_rejected():
    usable, issues = tb.validate_players(player(shooting=None, passing=None, defense=None))
    assert usable.empty
    assert "too little information" in issues.iloc[0]["issue"]


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="rebounding"):
        tb.validate_players(player().drop(columns="rebounding"))


# --- Boundary and invalid values --------------------------------------------
@pytest.mark.parametrize("column, value", [
    ("shooting", 1), ("shooting", 10), ("fitness_level", 1), ("availability_pct", 0), ("availability_pct", 100),
])
def test_boundary_values_are_accepted(column, value):
    usable, _ = tb.validate_players(player(**{column: value}))
    assert len(usable) == 1


@pytest.mark.parametrize("column, value", [
    ("shooting", 0), ("shooting", 11), ("fitness_level", 10.5), ("availability_pct", -1),
    ("availability_pct", 101), ("passing", "eight"), ("preferred_position", "Point Guard"),
])
def test_invalid_values_are_rejected(column, value):
    usable, issues = tb.validate_players(player(**{column: value}))
    assert usable.empty
    assert (issues["severity"] == "rejected").any()


def test_position_text_is_normalised():
    usable, issues = tb.validate_players(player(preferred_position="  forward "))
    assert usable.at[0, tb.POSITION_COL] == "Forward" and issues.empty


def test_edge_case_file():
    usable, issues = tb.validate_players(tb.load_players(DATA / "test_roster_edge_cases.csv"))
    rejected = set(issues.loc[issues["severity"] == "rejected", "player_id"])
    assert rejected == {"T02", "T05", "T06", "T07", "T08"}
    assert {"T01", "T03", "T04", "T09", "T10", "T11"} <= set(usable["player_id"])


# --- Uncertain cases ---------------------------------------------------------
def test_player_at_a_centroid_is_certain(pipeline):
    centre = pipeline["model"].cluster_centers_[:1]
    assert tb.assignment_margin(pipeline["model"], centre)[0] > 0.99


def test_player_between_two_centroids_is_uncertain(pipeline):
    c = pipeline["model"].cluster_centers_
    midpoint = (c[0] + c[1]) / 2
    assert tb.assignment_margin(pipeline["model"], midpoint[None, :])[0] < 0.01


def test_in_between_player_is_flagged(pipeline):
    usable, _ = tb.validate_players(player(shooting=7, passing=8, defense=5, rebounding=6,
                                           fitness_level=8, availability_pct=85))
    x = tb.scale(pipeline["scaler"], tb.impute(usable, pipeline["fills"]))
    assert tb.assignment_margin(pipeline["model"], x)[0] < tb.UNCERTAIN_MARGIN


# --- Balanced teams ----------------------------------------------------------
@pytest.mark.parametrize("n_teams", range(2, 13))
def test_teams_are_even_in_size_and_group_mix(pipeline, n_teams):
    teams = tb.build_teams(pipeline["clean"], pipeline["model"].labels_, n_teams)
    sizes = teams["team"].value_counts()
    assert sizes.max() - sizes.min() <= 1
    per_group = teams.groupby(["cluster", "team"]).size().unstack(fill_value=0)
    assert (per_group.max(axis=1) - per_group.min(axis=1)).max() <= 1


@pytest.mark.parametrize("n_teams", [2, 4, 6])
def test_teams_are_fairer_than_random(pipeline, n_teams):
    teams = tb.build_teams(pipeline["clean"], pipeline["model"].labels_, n_teams)
    ours = tb.balance_spread(teams)
    random = tb.random_team_baseline(pipeline["clean"], n_teams, trials=200)
    assert ours["skill_gap"] < random["skill_gap"]
    assert ours["fitness_gap"] < random["fitness_gap"]


def test_swaps_keep_group_and_position_mix(pipeline):
    draft = tb.assign_teams(pipeline["clean"], pipeline["model"].labels_, 4)
    refined = tb.refine_teams(draft)
    mix = lambda t: t.groupby(["team", "cluster", tb.POSITION_COL]).size().sort_index()
    pd.testing.assert_series_equal(mix(draft), mix(refined))


def test_too_many_teams_is_refused(pipeline):
    with pytest.raises(ValueError, match="at most 12 teams"):
        tb.assign_teams(pipeline["clean"], pipeline["model"].labels_, 13)


def test_align_labels_matches_renumbered_groups():
    reference = np.array([0, 0, 1, 1, 2, 2])
    assert (tb.align_labels(np.array([2, 2, 0, 0, 1, 1]), reference) == reference).all()
