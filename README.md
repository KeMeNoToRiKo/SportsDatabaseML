# Group 24: Sports Team Balancing (Unsupervised Clustering)

K-Means finds natural player groups (skill profile, fitness, availability) in a 60-player basketball roster. Hierarchical clustering is the comparison method. The groups are then used to draft **balanced teams**, so each team gets an even share of every player type and position.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py            # checks data, clusters, compares, builds teams, saves scaler + model
streamlit run app.py       # interface
pytest -q                  # 42 tests: normal, missing, boundary, invalid, uncertain, team balance
```

`python data/generate_dataset.py` recreates the synthetic roster (seed 24). `python train.py --teams 5` changes the number of teams.

## Deliverables

| Required output | File |
|---|---|
| 30–100 record dataset | `data/players_raw.csv` (62 rows → 60 players). Cleaned version: `data/players_clean.csv` |
| Data dictionary | `docs/data_dictionary.md` |
| KNIME workflow | `knime/KNIME_workflow_guide.md` (node-by-node build and expected results) |
| Python implementation | `team_balancer.py` (process), `train.py` (run + save), `charts.py` |
| Streamlit interface | `app.py` |
| Comparison table / chart | `outputs/comparison_table.md`, `outputs/k_selection.png`, `outputs/cluster_scatter.png` |
| Interpretation and limitations | `docs/report.md` (also covers Parts 1–6 and the critical-thinking questions) |

## Layout

```
data/       players_raw.csv, players_clean.csv, test_roster_edge_cases.csv, generate_dataset.py
models/     scaler.joblib, kmeans_model.joblib, metadata.json      (written by train.py)
outputs/    comparison, cluster profiles, team assignments, charts  (written by train.py)
docs/       data_dictionary.md, report.md
knime/      KNIME_workflow_guide.md
tests/      test_team_balancer.py
```

## Results at a glance

- **k = 4** (silhouette peak and elbow): Core all-rounders (12), Perimeter playmakers (16), Interior anchors (14), Developing players with limited availability (18).
- **K-Means vs hierarchical (average linkage):** silhouette 0.385 vs 0.382. They agree on 59 of 60 players (ARI 0.958).
- **4 teams of 15:** the gap between the strongest and weakest team's average skill is 0.02 points, compared with 0.67 for random teams.
