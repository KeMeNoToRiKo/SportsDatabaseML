"""Run the full Group 24 process once: check data, cluster, compare, build teams, save model.

    python train.py                 # default: data/players_raw.csv, 4 teams
    python train.py --teams 5

Outputs go to data/ (clean CSV), outputs/ (tables + charts) and models/ (scaler + K-Means).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

import charts
import team_balancer as tb

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"


def main(raw_path: Path, n_teams: int) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    # 1. Load and check -----------------------------------------------------
    raw = tb.load_players(raw_path)
    quality, n_dupes = tb.data_quality_report(raw)
    quality.to_csv(OUTPUTS / "data_quality_report.csv", index=False)
    print(f"Loaded {len(raw)} rows from {raw_path.name}; exact duplicate rows: {n_dupes}")
    print(quality[["column", "missing"]].query("missing > 0").to_string(index=False), "\n")

    # 2. Validate, dedupe, impute -------------------------------------------
    usable, issues = tb.validate_players(raw)
    print(issues.to_string(index=False), "\n")
    fill_values = tb.compute_fill_values(usable)
    clean = tb.impute(usable, fill_values)
    clean.to_csv(ROOT / "data" / "players_clean.csv", index=False)
    print(f"Clean dataset: {len(clean)} players, {len(tb.FEATURES)} clustering inputs + position\n")

    # 3. Scale and choose k -------------------------------------------------
    scaler = tb.fit_scaler(clean)
    X = tb.scale(scaler, clean)
    k_table = tb.evaluate_k_range(X)
    k = tb.recommend_k(k_table)
    k_table.round(3).to_csv(OUTPUTS / "k_selection.csv", index=False)
    print(k_table.round(3).to_string(index=False))
    print(f"-> chosen k = {k} (highest K-Means silhouette; check the elbow chart agrees)\n")

    # 4. Fit both algorithms ------------------------------------------------
    kmeans = tb.fit_kmeans(X, k, clean)
    km_labels = kmeans.labels_
    hc_labels = tb.align_labels(tb.fit_hierarchical(X, k, clean), km_labels)

    # 5. Interpret ----------------------------------------------------------
    km_profile = tb.profile_clusters(clean, km_labels)
    names = km_profile["cluster_name"].tolist()
    hc_profile = tb.profile_clusters(clean, hc_labels)
    km_profile.to_csv(OUTPUTS / "cluster_profiles_kmeans.csv")
    hc_profile.to_csv(OUTPUTS / "cluster_profiles_hierarchical.csv")
    print("K-Means cluster profiles (averages in original units):")
    print(km_profile.to_string(), "\n")

    # 6. Balanced teams -----------------------------------------------------
    km_teams = tb.build_teams(clean, km_labels, n_teams)
    hc_teams = tb.build_teams(clean, hc_labels, n_teams)
    draft_only = tb.build_teams(clean, km_labels, n_teams, refine=False)
    margins = pd.Series(tb.assignment_margin(kmeans, X), index=clean["player_id"])
    km_teams["cluster_name"] = [names[c] for c in km_teams["cluster"]]
    km_teams["assignment_margin"] = km_teams["player_id"].map(margins).round(3)
    km_teams["uncertain"] = km_teams["assignment_margin"] < tb.UNCERTAIN_MARGIN
    km_teams.to_csv(OUTPUTS / "team_assignments.csv", index=False)
    summary = tb.team_summary(km_teams, names)
    summary.to_csv(OUTPUTS / "team_summary.csv")
    print(f"Team summary ({n_teams} teams, K-Means groups):")
    print(summary.to_string(), "\n")

    balance = pd.DataFrame({
        "Random split (avg of 500)": tb.random_team_baseline(clean, n_teams),
        "Group draft only": tb.balance_spread(draft_only),
        "Group draft + swaps (final)": tb.balance_spread(km_teams),
    }).T.round(2)
    balance.to_csv(OUTPUTS / "team_balance_vs_random.csv")
    print(balance.to_string(), "\n")

    # 7. Compare algorithms -------------------------------------------------
    km_m, hc_m = tb.cluster_metrics(X, km_labels), tb.cluster_metrics(X, hc_labels)
    km_gap, hc_gap = tb.balance_spread(km_teams)["skill_gap"], tb.balance_spread(hc_teams)["skill_gap"]
    comparison = pd.DataFrame([
        ["Silhouette score (higher better)", km_m["silhouette"], hc_m["silhouette"]],
        ["Davies-Bouldin index (lower better)", km_m["davies_bouldin"], hc_m["davies_bouldin"]],
        ["Calinski-Harabasz index (higher better)", km_m["calinski_harabasz"], hc_m["calinski_harabasz"]],
        ["Cluster sizes", km_m["cluster_sizes"], hc_m["cluster_sizes"]],
        ["Smallest cluster", km_m["smallest_cluster"], hc_m["smallest_cluster"]],
        ["Same result on re-run (mean ARI over 20 seeds)",
         tb.kmeans_seed_stability(X, k, km_labels), "1.00 (deterministic)"],
        [f"Team skill gap, {n_teams} teams (lower better)", km_gap, hc_gap],
        ["Can place a new player without refitting", "Yes (nearest centroid)", "No (must re-cluster)"],
    ], columns=["Measure", "K-Means", f"Hierarchical ({tb.HIER_LINKAGE} linkage)"])
    comparison = comparison.map(lambda v: round(v, 3) if isinstance(v, float) else v)
    agreement = adjusted_rand_score(km_labels, hc_labels)
    comparison.to_csv(OUTPUTS / "comparison_table.csv", index=False)
    (OUTPUTS / "comparison_table.md").write_text(
        comparison.to_markdown(index=False)
        + f"\n\nAgreement between the two methods (adjusted Rand index): {agreement:.3f}"
        + f"\nPlayers placed in a different group: {(km_labels != hc_labels).sum()} of {len(clean)}\n"
    )
    print(comparison.to_string(index=False))
    print(f"Agreement (ARI) = {agreement:.3f}; players grouped differently: {(km_labels != hc_labels).sum()}\n")

    # 8. Charts -------------------------------------------------------------
    charts.plot_k_selection(k_table, k, OUTPUTS / "k_selection.png")
    charts.plot_cluster_scatter(clean, {
        f"K-Means (silhouette {km_m['silhouette']:.3f})": (km_labels, names),
        f"Hierarchical, {tb.HIER_LINKAGE} linkage (silhouette {hc_m['silhouette']:.3f})":
            (hc_labels, hc_profile["cluster_name"].tolist()),
    }, OUTPUTS / "cluster_scatter.png")
    charts.plot_dendrogram(X, clean["player_id"].tolist(), k, tb.HIER_LINKAGE, OUTPUTS / "dendrogram.png")
    charts.plot_team_balance(balance, n_teams, OUTPUTS / "team_balance.png")

    # 9. Save scaler + model ------------------------------------------------
    tb.save_artifacts(MODELS, scaler, kmeans, {
        "trained_on": raw_path.name,
        "trained_date": date.today().isoformat(),
        "n_players": len(clean),
        "features": tb.FEATURES,
        "k": k,
        "cluster_names": names,
        "fill_values": fill_values,
        "silhouette": round(km_m["silhouette"], 4),
        "uncertain_margin": tb.UNCERTAIN_MARGIN,
    })
    print(f"Saved scaler, model and metadata to {MODELS.relative_to(ROOT)}/; tables and charts to "
          f"{OUTPUTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "players_raw.csv")
    parser.add_argument("--teams", type=int, default=4)
    args = parser.parse_args()
    main(args.data, args.teams)
