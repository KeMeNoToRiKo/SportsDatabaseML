"""Core process for Group 24 - Sports Team Balancing.

Primary algorithm:    K-Means clustering
Comparison algorithm: Hierarchical (agglomerative) clustering, average linkage

The same functions are used by train.py (batch run that saves the scaler and
model) and app.py (Streamlit interface), so both run one identical process:

    load -> validate -> clean (dedupe + impute) -> scale -> cluster
         -> evaluate -> interpret clusters -> draft balanced teams
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data definition
# ---------------------------------------------------------------------------
ID_COLS = ["player_id", "player_name"]          # reference only, never clustered
POSITION_COL = "preferred_position"              # used to balance teams, not clustered
SKILLS = ["shooting", "passing", "defense", "rebounding"]
FEATURES = SKILLS + ["fitness_level", "availability_pct"]   # clustering inputs
POSITIONS = ["Guard", "Forward", "Center"]
VALID_RANGES = {
    "shooting": (1, 10),
    "passing": (1, 10),
    "defense": (1, 10),
    "rebounding": (1, 10),
    "fitness_level": (1, 10),
    "availability_pct": (0, 100),
}
REQUIRED_COLS = ["player_id", POSITION_COL] + FEATURES

RANDOM_STATE = 42
HIER_LINKAGE = "average"      # available in KNIME's Hierarchical Clustering node (Ward is not)
MAX_MISSING_FEATURES = 2      # a row missing more than this many inputs is rejected
MIN_TEAM_SIZE = 5             # basketball needs at least five players per team
UNCERTAIN_MARGIN = 0.15       # nearest vs second-nearest centroid margin below this = uncertain


# ---------------------------------------------------------------------------
# 1. Load, check, validate, clean
# ---------------------------------------------------------------------------
def load_players(source) -> pd.DataFrame:
    """Read a CSV path or file-like object and tidy headers and text cells."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].str.strip()
    return df


def data_quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Per-column missing / out-of-range / invalid counts, plus the duplicate-row count."""
    rows = []
    for col in df.columns:
        entry = {"column": col, "dtype": str(df[col].dtype), "missing": int(df[col].isna().sum())}
        if col in VALID_RANGES:
            values = pd.to_numeric(df[col], errors="coerce")
            lo, hi = VALID_RANGES[col]
            entry["out_of_range"] = int(((values < lo) | (values > hi)).sum())
            entry["min"], entry["max"] = values.min(), values.max()
        if col == POSITION_COL:
            known = df[col].dropna().str.title().isin(POSITIONS)
            entry["invalid_category"] = int((~known).sum())
        rows.append(entry)
    return pd.DataFrame(rows), int(df.duplicated().sum())


def validate_players(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a roster into usable rows and a list of issues.

    Missing values are allowed (they are imputed later) but reported.
    Invalid values (non-numeric, out of range, unknown position) and rows with
    too many missing inputs are rejected. Exact duplicate rows are dropped.
    """
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}")

    df = df.copy()
    if "player_name" not in df.columns:
        df["player_name"] = df["player_id"]
    issues: list[dict] = []

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        issues.append({"player_id": "-", "severity": "info",
                       "issue": f"Removed {n_before - len(df)} exact duplicate row(s)"})

    blank_id = df["player_id"].isna() | (df["player_id"].astype(str).str.strip() == "")
    df.loc[blank_id, "player_id"] = [f"NEW{i + 1:03d}" for i in range(int(blank_id.sum()))]
    for pid in df.loc[blank_id, "player_id"]:
        issues.append({"player_id": pid, "severity": "warning", "issue": "Missing player_id; temporary ID assigned"})
    df["player_name"] = df["player_name"].fillna(df["player_id"])

    rejected = pd.Series(False, index=df.index)

    for col in FEATURES:
        raw = df[col]
        values = pd.to_numeric(raw, errors="coerce")
        not_numeric = values.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
        lo, hi = VALID_RANGES[col]
        out_of_range = (values < lo) | (values > hi)
        for idx in df.index[not_numeric]:
            issues.append({"player_id": df.at[idx, "player_id"], "severity": "rejected",
                           "issue": f"{col} = '{raw[idx]}' is not a number"})
        for idx in df.index[out_of_range]:
            issues.append({"player_id": df.at[idx, "player_id"], "severity": "rejected",
                           "issue": f"{col} = {values[idx]:g} is outside {lo}-{hi}"})
        rejected |= not_numeric | out_of_range
        df[col] = values

    position = df[POSITION_COL].astype("string").str.strip().str.title()
    bad_position = position.notna() & (position != "") & ~position.isin(POSITIONS)
    for idx in df.index[bad_position]:
        issues.append({"player_id": df.at[idx, "player_id"], "severity": "rejected",
                       "issue": f"preferred_position '{df.at[idx, POSITION_COL]}' is not one of {POSITIONS}"})
    rejected |= bad_position
    df[POSITION_COL] = position.where(position.isin(POSITIONS)).astype(object)

    n_missing = df[FEATURES].isna().sum(axis=1)
    too_sparse = (n_missing > MAX_MISSING_FEATURES) & ~rejected
    for idx in df.index[too_sparse]:
        issues.append({"player_id": df.at[idx, "player_id"], "severity": "rejected",
                       "issue": f"{n_missing[idx]} of {len(FEATURES)} inputs missing - too little information"})
    rejected |= too_sparse

    usable = ~rejected
    for idx in df.index[usable]:
        gaps = [c for c in FEATURES + [POSITION_COL] if pd.isna(df.at[idx, c])]
        if gaps:
            issues.append({"player_id": df.at[idx, "player_id"], "severity": "warning",
                           "issue": f"Missing {', '.join(gaps)} - will be imputed"})

    dup_ids = df.loc[usable, "player_id"].duplicated()
    for idx in dup_ids[dup_ids].index:
        issues.append({"player_id": df.at[idx, "player_id"], "severity": "rejected",
                       "issue": "Duplicate player_id with different values (first entry kept)"})
        usable[idx] = False

    issue_df = pd.DataFrame(issues, columns=["player_id", "severity", "issue"])
    return df[usable].reset_index(drop=True), issue_df


def compute_fill_values(df: pd.DataFrame) -> dict:
    """Median for each numeric input, most frequent value for position."""
    fills = {col: float(df[col].median()) for col in FEATURES}
    fills[POSITION_COL] = str(df[POSITION_COL].mode().iloc[0])
    return fills


def impute(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    df = df.copy()
    df["was_imputed"] = df[FEATURES + [POSITION_COL]].isna().any(axis=1)
    return df.fillna(fill_values)


# ---------------------------------------------------------------------------
# 2. Scale and choose k
# ---------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(df[FEATURES])


def scale(scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    return scaler.transform(df[FEATURES])


def evaluate_k_range(X: np.ndarray, k_values=range(2, 9)) -> pd.DataFrame:
    """Elbow (inertia) and silhouette for both algorithms across candidate k."""
    rows = []
    for k in k_values:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        hc = AgglomerativeClustering(n_clusters=k, linkage=HIER_LINKAGE).fit_predict(X)
        rows.append({
            "k": k,
            "kmeans_inertia": km.inertia_,
            "kmeans_silhouette": silhouette_score(X, km.labels_),
            "hierarchical_silhouette": silhouette_score(X, hc),
        })
    return pd.DataFrame(rows)


def recommend_k(k_table: pd.DataFrame) -> int:
    return int(k_table.loc[k_table["kmeans_silhouette"].idxmax(), "k"])


# ---------------------------------------------------------------------------
# 3. Fit both algorithms (cluster numbers ordered strongest -> weakest)
# ---------------------------------------------------------------------------
def overall_skill(df: pd.DataFrame) -> pd.Series:
    return df[SKILLS].mean(axis=1)


def order_labels_by_strength(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Renumber clusters so 0 has the highest average skill. Labels are otherwise arbitrary."""
    strength = overall_skill(df).groupby(labels).mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(strength.index)}
    return np.array([mapping[label] for label in labels])


def fit_kmeans(X: np.ndarray, k: int, df: pd.DataFrame) -> KMeans:
    """Fit K-Means, then refit from the sorted centroids so model.predict() returns ordered labels."""
    first = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    strength = overall_skill(df).groupby(first.labels_).mean()
    order = strength.sort_values(ascending=False).index.to_numpy()
    return KMeans(n_clusters=k, init=first.cluster_centers_[order], n_init=1,
                  random_state=RANDOM_STATE).fit(X)


def fit_hierarchical(X: np.ndarray, k: int, df: pd.DataFrame) -> np.ndarray:
    labels = AgglomerativeClustering(n_clusters=k, linkage=HIER_LINKAGE).fit_predict(X)
    return order_labels_by_strength(labels, df)


def align_labels(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Renumber `labels` to best match `reference` so the same group keeps the same number/colour."""
    k = max(labels.max(), reference.max()) + 1
    overlap = np.zeros((k, k), dtype=int)
    np.add.at(overlap, (labels, reference), 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = dict(zip(rows, cols))
    return np.array([mapping[label] for label in labels])


# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "cluster_sizes": " / ".join(str(s) for s in sizes),
        "smallest_cluster": int(sizes.min()),
    }


def kmeans_seed_stability(X: np.ndarray, k: int, reference: np.ndarray, n_seeds: int = 20) -> float:
    """Mean adjusted Rand index between the reference labels and K-Means rerun with other seeds."""
    scores = [adjusted_rand_score(reference, KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(X))
              for s in range(n_seeds)]
    return float(np.mean(scores))


def assignment_margin(model: KMeans, X: np.ndarray) -> np.ndarray:
    """(d2 - d1) / d2 where d1, d2 are distances to the nearest and second-nearest centroid.

    Near 0 = the player sits between two groups (uncertain); near 1 = clearly in one group.
    """
    d = np.sort(model.transform(X), axis=1)
    return (d[:, 1] - d[:, 0]) / d[:, 1]


# ---------------------------------------------------------------------------
# 5. Interpret clusters
# ---------------------------------------------------------------------------
def name_cluster(cluster_means: pd.Series, roster_means: pd.Series, roster_stds: pd.Series) -> str:
    """Rule-based archetype name from how a cluster differs from the whole roster."""
    z = (cluster_means[FEATURES] - roster_means[FEATURES]) / roster_stds[FEATURES]
    perimeter = (z["shooting"] + z["passing"]) / 2
    interior = (z["defense"] + z["rebounding"]) / 2
    if perimeter > 0.5 and interior > 0.5:
        name = "Core all-rounders"
    elif perimeter < -0.5 and interior < -0.5:
        name = "Developing players"
    elif perimeter - interior > 0.8:
        name = "Perimeter playmakers"
    elif interior - perimeter > 0.8:
        name = "Interior anchors"
    else:
        name = "Mixed-skill group"
    if z["availability_pct"] < -0.75:
        name += " (limited availability)"
    return name


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    """Average inputs per cluster in original units, plus size, position mix and a name."""
    data = df.assign(cluster=labels)
    profile = data.groupby("cluster")[FEATURES].mean()
    profile.insert(0, "players", data.groupby("cluster").size())
    profile["overall_skill"] = profile[SKILLS].mean(axis=1)
    mix = data.groupby("cluster")[POSITION_COL].value_counts().unstack(fill_value=0)
    profile["position_mix"] = [
        ", ".join(f"{mix.at[c, p]} {p[0]}" for p in POSITIONS if p in mix.columns and mix.at[c, p] > 0)
        for c in profile.index
    ]
    if names is None:
        means, stds = df[FEATURES].mean(), df[FEATURES].std(ddof=0)
        names = [name_cluster(profile.loc[c], means, stds) for c in profile.index]
    profile.insert(0, "cluster_name", [names[c] for c in profile.index])
    profile.index = [f"C{c + 1}" for c in profile.index]
    profile.index.name = "cluster"
    return profile.round(2)


def cluster_label(cluster: int, names: list[str]) -> str:
    return f"C{cluster + 1} {names[cluster]}"


# ---------------------------------------------------------------------------
# 6. Balanced teams: group-by-group draft, then balance-preserving swaps
# ---------------------------------------------------------------------------
def max_teams(n_players: int) -> int:
    return n_players // MIN_TEAM_SIZE


def assign_teams(df: pd.DataFrame, labels: np.ndarray, n_teams: int) -> pd.DataFrame:
    """Draft players group by group so every team gets an even share of each group.

    Players are taken one cluster at a time, best overall skill first. Each one
    joins the team that has (1) the fewest players from that cluster, then
    (2) the fewest players overall, then (3) the fewest players in the same
    position, then (4) the lowest total skill so far. Rules 1-2 guarantee that
    cluster counts and team sizes differ by at most one between teams.
    """
    if n_teams < 2:
        raise ValueError("Choose at least 2 teams.")
    if n_teams > max_teams(len(df)):
        raise ValueError(f"{len(df)} players can form at most {max_teams(len(df))} teams "
                         f"of {MIN_TEAM_SIZE}+ players.")

    data = df.assign(cluster=labels, overall_skill=overall_skill(df).round(2))
    data = data.sort_values(["cluster", "overall_skill", "player_id"],
                            ascending=[True, False, True]).reset_index(drop=True)
    k = int(data["cluster"].max()) + 1
    in_cluster = np.zeros((n_teams, k))
    in_position = {p: np.zeros(n_teams) for p in POSITIONS}
    size, skill = np.zeros(n_teams), np.zeros(n_teams)
    team = []
    for cluster, position, player_skill in data[["cluster", POSITION_COL, "overall_skill"]].itertuples(index=False):
        t = min(range(n_teams), key=lambda t: (in_cluster[t, cluster], size[t],
                                               in_position[position][t], skill[t]))
        team.append(t)
        in_cluster[t, cluster] += 1
        in_position[position][t] += 1
        size[t] += 1
        skill[t] += player_skill
    data["team"] = [f"Team {t + 1}" for t in team]
    return data


BALANCE_COLS = ["overall_skill", "fitness_level", "availability_pct"]


def refine_teams(teams: pd.DataFrame, max_passes: int = 50) -> pd.DataFrame:
    """Improve the draft by swapping players who share a cluster AND a position.

    Such swaps never change a team's cluster mix, position mix or size; they only
    move team averages of skill, fitness and availability (z-scored, equal weight)
    closer to the roster average. Repeats until no swap helps.
    """
    data = teams.reset_index(drop=True).copy()
    values = data[BALANCE_COLS].to_numpy(dtype=float)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    team_names = sorted(data["team"].unique(), key=lambda t: int(t.split()[-1]))
    team = data["team"].map({t: i for i, t in enumerate(team_names)}).to_numpy().copy()
    sizes = np.bincount(team).astype(float)
    sums = np.zeros((len(team_names), len(BALANCE_COLS)))
    np.add.at(sums, team, z)

    groups = data.groupby(["cluster", POSITION_COL]).indices.values()
    for _ in range(max_passes):
        improved = False
        for members in groups:
            for a_pos, i in enumerate(members):
                for j in members[a_pos + 1:]:
                    ta, tb = team[i], team[j]
                    if ta == tb:
                        continue
                    diff = z[j] - z[i]
                    old = (sums[ta] / sizes[ta]) ** 2 + (sums[tb] / sizes[tb]) ** 2
                    new = ((sums[ta] + diff) / sizes[ta]) ** 2 + ((sums[tb] - diff) / sizes[tb]) ** 2
                    if new.sum() < old.sum() - 1e-12:
                        sums[ta] += diff
                        sums[tb] -= diff
                        team[i], team[j] = tb, ta
                        improved = True
        if not improved:
            break
    data["team"] = [team_names[t] for t in team]
    return data


def build_teams(df: pd.DataFrame, labels: np.ndarray, n_teams: int, refine: bool = True) -> pd.DataFrame:
    """Group-by-group draft, then (optionally) balance-preserving swaps."""
    teams = assign_teams(df, labels, n_teams)
    teams = refine_teams(teams) if refine else teams
    return teams.sort_values(["team", "cluster", "overall_skill"], ascending=[True, True, False],
                             key=lambda s: s.str.extract(r"(\d+)")[0].astype(int) if s.name == "team" else s)


def suggest_team(teams: pd.DataFrame, cluster: int, position: str) -> str:
    """Team a new player should join: fewest of their group, then of their position, then smallest/weakest."""
    stats = teams.groupby("team").agg(
        same_cluster=("cluster", lambda s: int((s == cluster).sum())),
        same_position=(POSITION_COL, lambda s: int((s == position).sum())),
        players=("player_id", "size"),
        skill=("overall_skill", "mean"),
    )
    return stats.sort_values(["same_cluster", "same_position", "players", "skill"]).index[0]


def team_summary(teams: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    grouped = teams.groupby("team", sort=False)
    summary = grouped[["overall_skill"] + FEATURES].mean().round(2)
    summary.insert(0, "players", grouped.size())
    for p in POSITIONS:
        summary[p.lower() + "s"] = grouped[POSITION_COL].apply(lambda s, p=p: int((s == p).sum()))
    k = int(teams["cluster"].max()) + 1
    for c in range(k):
        col = f"C{c + 1}" if names is None else cluster_label(c, names)
        summary[col] = grouped["cluster"].apply(lambda s, c=c: int((s == c).sum()))
    return summary.sort_index(key=lambda idx: idx.str.extract(r"(\d+)")[0].astype(int))


def balance_spread(teams: pd.DataFrame) -> dict:
    """Gap between the strongest and weakest team average (smaller = more balanced)."""
    means = teams.groupby("team")[BALANCE_COLS].mean()
    spread = means.max() - means.min()
    return {
        "skill_gap": float(spread["overall_skill"]),
        "fitness_gap": float(spread["fitness_level"]),
        "availability_gap": float(spread["availability_pct"]),
    }


def random_team_baseline(df: pd.DataFrame, n_teams: int, trials: int = 500) -> dict:
    """Average balance_spread when the same roster is split into teams at random."""
    rng = np.random.default_rng(RANDOM_STATE)
    data = df.assign(overall_skill=overall_skill(df))
    base = np.arange(len(data)) % n_teams
    gaps = []
    for _ in range(trials):
        shuffled = data.assign(team=rng.permutation(base))
        gaps.append(balance_spread(shuffled))
    return pd.DataFrame(gaps).mean().to_dict()


# ---------------------------------------------------------------------------
# 7. Persist the scaler and model
# ---------------------------------------------------------------------------
def save_artifacts(model_dir: Path, scaler: StandardScaler, model: KMeans, metadata: dict) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(model, model_dir / "kmeans_model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_artifacts(model_dir: Path) -> tuple[StandardScaler, KMeans, dict]:
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir / "scaler.joblib")
    model = joblib.load(model_dir / "kmeans_model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    return scaler, model, metadata
