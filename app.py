"""Streamlit interface for Group 24 - Sports Team Balancing.

    streamlit run app.py

Runs the same process as train.py (team_balancer.py): validate -> impute ->
scale -> K-Means (saved model or refit) -> draft balanced teams, and
compares K-Means with hierarchical clustering on the same roster.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import adjusted_rand_score

import team_balancer as tb

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "players_raw.csv"
MODEL_DIR = ROOT / "models"
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]   # validated for scatter (all pairs)
CLUSTER_SHAPES = ["circle", "square", "triangle-up", "diamond"]
MUTED = "#c3c2b7"
INPUT_COLS = tb.ID_COLS + [tb.POSITION_COL] + tb.FEATURES

st.set_page_config(page_title="Team Balancer", page_icon="🏀", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained():
    if not (MODEL_DIR / "kmeans_model.joblib").exists():
        return None
    return tb.load_artifacts(MODEL_DIR)


@st.cache_data
def load_sample() -> pd.DataFrame:
    return tb.load_players(SAMPLE_PATH)


def template_csv() -> bytes:
    example = load_sample().dropna().head(3)[INPUT_COLS]
    return example.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Sidebar: inputs and settings
# ---------------------------------------------------------------------------
trained = load_trained()

with st.sidebar:
    st.header("1. Roster")
    source = st.radio("Where are the players coming from?",
                      ["Sample roster (60 players)", "Upload a CSV", "Enter players by hand"])
    uploaded = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Player CSV", type="csv")
        st.download_button("Download a template CSV", template_csv(), "player_template.csv", "text/csv")
        st.caption("Required columns: " + ", ".join(tb.REQUIRED_COLS) + ". player_name is optional.")

    st.header("2. Grouping")
    mode_options = ["Re-fit on this roster"]
    if trained is not None:
        mode_options.insert(0, f"Use trained model (k = {trained[2]['k']})")
    mode = st.radio("Clustering model", mode_options,
                    help="The trained model keeps group meanings stable between rosters. "
                         "Re-fitting lets the groups adapt to a very different roster.")
    refit = mode.startswith("Re-fit")
    k_choice = "Auto"
    if refit:
        k_choice = st.select_slider("Number of groups (k)", ["Auto", 2, 3, 4, 5, 6], value="Auto",
                                    help="Auto picks the k with the highest K-Means silhouette score.")

    st.header("3. Teams")
    n_teams_input = st.number_input("Number of teams", min_value=2, max_value=12, value=4, step=1)


# ---------------------------------------------------------------------------
# Collect the roster
# ---------------------------------------------------------------------------
st.title("Sports Team Balancer")
st.caption("K-Means finds natural player groups from skills, fitness and availability. "
           "Every team then receives an even share of each group and each position.")

if source == "Sample roster (60 players)":
    roster_raw = load_sample()
elif source == "Upload a CSV":
    if uploaded is None:
        st.info("Upload a player CSV in the sidebar, or download the template to see the format.")
        st.stop()
    try:
        roster_raw = tb.load_players(uploaded)
    except Exception as exc:  # unreadable file
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()
else:
    st.subheader("Enter players")
    st.caption("Edit cells, add rows at the bottom, or leave a cell empty to have it imputed. "
               "At least 10 valid players are needed for two teams.")
    starter = load_sample().drop_duplicates().head(12)[INPUT_COLS].reset_index(drop=True)
    roster_raw = st.data_editor(
        starter, num_rows="dynamic", width="stretch", key="manual_roster",
        column_config={
            "player_id": st.column_config.TextColumn("player_id", required=True),
            "player_name": st.column_config.TextColumn("player_name"),
            tb.POSITION_COL: st.column_config.SelectboxColumn("preferred_position", options=tb.POSITIONS),
            **{c: st.column_config.NumberColumn(c, min_value=lo, max_value=hi, step=1)
               for c, (lo, hi) in tb.VALID_RANGES.items()},
        },
    )

# ---------------------------------------------------------------------------
# Validate and clean
# ---------------------------------------------------------------------------
try:
    roster, issues = tb.validate_players(roster_raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

n_rejected = int((issues["severity"] == "rejected").sum())
n_warnings = int((issues["severity"] == "warning").sum())

with st.expander(f"Input check: {len(roster)} usable players, {n_rejected} rejected, "
                 f"{n_warnings} warning(s)", expanded=n_rejected > 0):
    if issues.empty:
        st.success("No missing, duplicate or invalid values found.")
    else:
        st.dataframe(issues, hide_index=True, width="stretch")
        st.caption("Rejected rows are left out. Missing values in usable rows are filled with the "
                   "median (numbers) or most common value (position) before grouping.")

min_players = 2 * tb.MIN_TEAM_SIZE
if len(roster) < min_players:
    st.error(f"Need at least {min_players} valid players to form two teams of {tb.MIN_TEAM_SIZE}; "
             f"only {len(roster)} are usable.")
    st.stop()

# ---------------------------------------------------------------------------
# Run the process
# ---------------------------------------------------------------------------
k_table = None
if refit:
    fill_values = tb.compute_fill_values(roster)
    clean = tb.impute(roster, fill_values)
    scaler = tb.fit_scaler(clean)
    X = tb.scale(scaler, clean)
    k_table = tb.evaluate_k_range(X, range(2, min(7, len(clean) - 1)))
    k = tb.recommend_k(k_table) if k_choice == "Auto" else int(k_choice)
    model = tb.fit_kmeans(X, k, clean)
    labels = model.labels_
    names = None
else:
    scaler, model, meta = trained
    fill_values = meta["fill_values"]
    clean = tb.impute(roster, fill_values)
    X = tb.scale(scaler, clean)
    k = meta["k"]
    labels = model.predict(X)
    names = meta["cluster_names"]

profile = tb.profile_clusters(clean, labels, names)
if names is None:
    names = profile["cluster_name"].tolist()
names = [names[c] if c < len(names) else f"Group {c + 1}" for c in range(k)]
margins = tb.assignment_margin(model, X)

n_teams = int(n_teams_input)
if n_teams > tb.max_teams(len(clean)):
    st.warning(f"{len(clean)} players can make at most {tb.max_teams(len(clean))} teams of "
               f"{tb.MIN_TEAM_SIZE}+; using {tb.max_teams(len(clean))}.")
    n_teams = tb.max_teams(len(clean))

teams = tb.build_teams(clean, labels, n_teams)
lookup = dict(zip(clean["player_id"], margins))
teams["group"] = [tb.cluster_label(c, names) for c in teams["cluster"]]
teams["certainty"] = teams["player_id"].map(lookup).round(2)
teams["uncertain"] = teams["certainty"] < tb.UNCERTAIN_MARGIN

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
spread = tb.balance_spread(teams)
baseline = tb.random_team_baseline(clean, n_teams, trials=200)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", len(clean))
c2.metric("Player groups (k)", k)
c3.metric("Teams", n_teams)
c4.metric("Team skill gap", f"{spread['skill_gap']:.2f}",
          delta=f"{spread['skill_gap'] - baseline['skill_gap']:.2f} vs random teams", delta_color="inverse",
          help="Strongest minus weakest team's average skill rating (1-10). Lower is fairer.")

tab_teams, tab_groups, tab_compare, tab_check = st.tabs(
    ["Balanced teams", "Player groups", "K-Means vs hierarchical", "Check one player"])

# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
with tab_teams:
    group_counts = teams.groupby("cluster").size()
    st.markdown(
        f"**How the teams were built.** K-Means sorted the {len(clean)} players into {k} groups of similar "
        f"players. Each group was then drafted across the {n_teams} teams, best player first, so every "
        "team gets an even share of every group, an equal roster size and a spread of positions. "
        "Finally, players with the same "
        "group and position were swapped between teams wherever that brought the teams' average skill, "
        "fitness and availability closer together. "
        f"Strongest and weakest team differ by **{spread['skill_gap']:.2f}** skill points "
        f"(random teams: {baseline['skill_gap']:.2f}), **{spread['fitness_gap']:.2f}** fitness points and "
        f"**{spread['availability_gap']:.1f}** availability points."
    )
    if teams["uncertain"].any() or teams["was_imputed"].any():
        st.caption(f"⚠ {int(teams['uncertain'].sum())} player(s) sit between two groups (certainty < "
                   f"{tb.UNCERTAIN_MARGIN}) and {int(teams['was_imputed'].sum())} had missing values "
                   "filled in. Their group placement is less reliable, so check them by eye.")

    summary = tb.team_summary(teams, names)
    st.subheader("Team summary")
    st.dataframe(summary, width="stretch")

    show_cols = ["player_id", "player_name", tb.POSITION_COL, "group", "overall_skill",
                 "fitness_level", "availability_pct", "certainty", "uncertain", "was_imputed"]
    cols = st.columns(min(n_teams, 3))
    for i, (team_name, members) in enumerate(teams.groupby("team", sort=False)):
        with cols[i % len(cols)]:
            st.markdown(f"**{team_name}** · {len(members)} players · avg skill "
                        f"{members['overall_skill'].mean():.2f}")
            st.dataframe(members[show_cols].rename(columns={tb.POSITION_COL: "position"}),
                         hide_index=True, width="stretch", height=min(38 + 35 * len(members), 420))

    st.download_button("Download team assignments (CSV)", teams.to_csv(index=False).encode(),
                       "team_assignments.csv", "text/csv")

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def scatter(data: pd.DataFrame, label_col: str, title: str) -> alt.Chart:
    rng = np.random.default_rng(1)
    plot = data.assign(
        perimeter=data[["shooting", "passing"]].mean(axis=1) + rng.uniform(-0.09, 0.09, len(data)),
        interior=data[["defense", "rebounding"]].mean(axis=1) + rng.uniform(-0.09, 0.09, len(data)),
    )
    x = alt.X("perimeter:Q", title="Perimeter skill (avg shooting, passing)", scale=alt.Scale(domain=[1, 10]))
    y = alt.Y("interior:Q", title="Interior skill (avg defense, rebounding)", scale=alt.Scale(domain=[1, 10]))
    tooltip = ["player_name", "player_id", tb.POSITION_COL, label_col, "shooting", "passing", "defense",
               "rebounding", "fitness_level", "availability_pct"]
    domain = [tb.cluster_label(c, names) for c in range(k)]
    if k <= len(CLUSTER_COLORS):
        return alt.Chart(plot, title=title).mark_point(filled=True, size=90, opacity=1, stroke="#fcfcfb",
                                                       strokeWidth=1.5).encode(
            x=x, y=y, tooltip=tooltip,
            color=alt.Color(f"{label_col}:N", scale=alt.Scale(domain=domain, range=CLUSTER_COLORS[:k]),
                            legend=alt.Legend(orient="bottom", columns=1, title=None)),
            shape=alt.Shape(f"{label_col}:N", scale=alt.Scale(domain=domain, range=CLUSTER_SHAPES[:k])),
        ).properties(height=380)
    # More than four groups: small multiples, one highlighted group per panel
    panels = pd.concat([plot.assign(panel=g, highlight=plot[label_col] == g) for g in domain])
    return alt.Chart(panels, title=title).mark_circle(size=55, opacity=1).encode(
        x=x, y=y, tooltip=tooltip,
        color=alt.condition("datum.highlight", alt.value(CLUSTER_COLORS[0]), alt.value(MUTED)),
        order=alt.Order("highlight:N"),
    ).properties(width=220, height=200).facet(facet=alt.Facet("panel:N", title=None), columns=3)


with tab_groups:
    st.markdown("Each group is a **type of player**, not a team. A group with only strong players would make "
                "an unfair team, so every team takes a share of each group instead.")
    plot_data = clean.assign(group=[tb.cluster_label(c, names) for c in labels])
    st.altair_chart(scatter(plot_data, "group", "Players by K-Means group"), width="stretch")
    st.dataframe(profile, width="stretch")
    st.caption("Averages are in the original units (skills and fitness 1-10, availability %). "
               "Names come from how each group's averages differ from the whole roster.")

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
with tab_compare:
    hc_labels = tb.align_labels(tb.fit_hierarchical(X, k, clean), labels)
    if len(np.unique(labels)) < 2 or len(np.unique(hc_labels)) < 2:
        st.info("This roster falls into a single group, so cluster quality can't be compared.")
    else:
        km_m, hc_m = tb.cluster_metrics(X, labels), tb.cluster_metrics(X, hc_labels)
        hc_teams = tb.build_teams(clean, hc_labels, n_teams)
        comparison = pd.DataFrame({
            "K-Means": [km_m["silhouette"], km_m["davies_bouldin"], km_m["calinski_harabasz"],
                        km_m["cluster_sizes"], spread["skill_gap"]],
            f"Hierarchical ({tb.HIER_LINKAGE})": [hc_m["silhouette"], hc_m["davies_bouldin"],
                                                 hc_m["calinski_harabasz"], hc_m["cluster_sizes"],
                                                 tb.balance_spread(hc_teams)["skill_gap"]],
        }, index=["Silhouette (higher better)", "Davies-Bouldin (lower better)",
                  "Calinski-Harabasz (higher better)", "Group sizes", "Team skill gap (lower better)"])
        comparison = comparison.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v)
        st.dataframe(comparison, width="stretch")
        ari = adjusted_rand_score(labels, hc_labels)
        moved = int((labels != hc_labels).sum())
        st.markdown(f"The two methods agree closely (adjusted Rand index **{ari:.2f}**); "
                    f"**{moved}** of {len(clean)} players land in a different group. K-Means is kept as the "
                    "primary method because it can place a new player with the saved model, without "
                    "re-clustering the whole roster.")
    if k_table is not None:
        long = k_table.melt("k", ["kmeans_silhouette", "hierarchical_silhouette"],
                            var_name="method", value_name="silhouette")
        long["method"] = long["method"].map({"kmeans_silhouette": "K-Means",
                                             "hierarchical_silhouette": "Hierarchical"})
        line = alt.Chart(long, title="Choosing k: silhouette score by number of groups").mark_line(
            strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
            x=alt.X("k:O", title="Number of groups (k)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("silhouette:Q", title="Silhouette (higher = better separated)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("method:N", scale=alt.Scale(domain=["K-Means", "Hierarchical"],
                                                        range=CLUSTER_COLORS[:2]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["method", "k", alt.Tooltip("silhouette:Q", format=".3f")],
        ).properties(height=300)
        st.altair_chart(line, width="stretch")
    else:
        st.caption(f"The trained model uses k = {k}, chosen in train.py by the highest silhouette score "
                   "and confirmed by the elbow chart. Switch to “Re-fit on this roster” to explore other k.")

# ---------------------------------------------------------------------------
# Check one player
# ---------------------------------------------------------------------------
with tab_check:
    st.markdown("Enter one player to see which group they belong to and which team needs them most. "
                "Leave a box empty to test how missing values are handled.")
    with st.form("one_player"):
        a, b = st.columns(2)
        pid = a.text_input("Player ID", "NEW001")
        pname = b.text_input("Name (optional)", "")
        position = a.selectbox("Preferred position", tb.POSITIONS + ["(unknown)"])
        shooting = b.number_input("Shooting (1-10)", value=None, step=1, placeholder="empty = missing")
        passing = a.number_input("Passing (1-10)", value=None, step=1, placeholder="empty = missing")
        defense = b.number_input("Defense (1-10)", value=None, step=1, placeholder="empty = missing")
        rebounding = a.number_input("Rebounding (1-10)", value=None, step=1, placeholder="empty = missing")
        fitness = b.number_input("Fitness level (1-10)", value=None, step=1, placeholder="empty = missing")
        availability = a.number_input("Availability (% of sessions, 0-100)", value=None, step=5,
                                      placeholder="empty = missing")
        submitted = st.form_submit_button("Check player")

    if submitted:
        one = pd.DataFrame([{
            "player_id": pid, "player_name": pname or pid,
            tb.POSITION_COL: None if position == "(unknown)" else position,
            "shooting": shooting, "passing": passing, "defense": defense, "rebounding": rebounding,
            "fitness_level": fitness, "availability_pct": availability,
        }])
        valid, one_issues = tb.validate_players(one)
        for _, issue in one_issues.iterrows():
            (st.error if issue["severity"] == "rejected" else st.warning)(issue["issue"])
        if not valid.empty:
            filled = tb.impute(valid, fill_values)
            x_one = tb.scale(scaler, filled)
            cluster = int(model.predict(x_one)[0])
            margin = float(tb.assignment_margin(model, x_one)[0])
            team = tb.suggest_team(teams, cluster, filled.at[0, tb.POSITION_COL])
            st.success(f"Group: **{tb.cluster_label(cluster, names)}** · suggested team: **{team}** "
                       f"(fewest players of this group and position).")
            if margin < tb.UNCERTAIN_MARGIN:
                d = model.transform(x_one)[0]
                second = int(np.argsort(d)[1])
                st.warning(f"Uncertain: certainty {margin:.2f} is below {tb.UNCERTAIN_MARGIN}. This player is "
                           f"almost as close to **{tb.cluster_label(second, names)}**; a coach should decide.")
            else:
                st.caption(f"Certainty {margin:.2f} (0 = exactly between two groups, 1 = at a group's centre).")
            if filled.at[0, "was_imputed"]:
                st.caption("Some inputs were missing and were filled with the roster median / most common "
                           "position, so this placement is provisional.")
