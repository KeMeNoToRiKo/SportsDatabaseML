# KNIME Workflow Guide: Group 24 Sports Team Balancing

Build this in **KNIME Analytics Platform 5.x**. The guide follows the suggested workflow:

`CSV Reader → Column Filter → Missing Value → Normalizer → k-Means → Cluster Assigner → Scatter Plot`

It adds three things to that chain: a Duplicate Row Filter, the hierarchical clustering branch for the comparison, and evaluation nodes. Save the finished workflow as `knime/Group24_TeamBalancing.knwf` (File → Export KNIME Workflow).

## Layout

```
                                                       ┌─> k-Means (k=4) ──┬─> Cluster Assigner ─> Color Manager ─> Scatter Plot
                                                       │                   ├─> Silhouette Coefficient
CSV Reader ─> Duplicate Row ─> Missing ─> Column ─> Normalizer             └─> Joiner ─> GroupBy (cluster profiles)
              Filter           Value      Filter    (z-score)  ──┤
                                                       │
                                                       └─> Hierarchical Clustering (average, k=4) ─┬─> Silhouette Coefficient
                                                                                                   └─> Joiner ─> Crosstab (K-Means vs Hierarchical)
Optional: Math Formula (overall skill) ─> Sorter ─> Math Formula (team) ─> GroupBy (team summary) ─> CSV Writer
```

## Node settings

| # | Node (search name) | Settings | Why |
|---|---|---|---|
| 1 | **CSV Reader** | File: `data/players_raw.csv`. Has column header ✔. In the **Transformation** tab, set `shooting`, `passing`, `defense`, `rebounding`, `fitness_level` and `availability_pct` to **Number (double)**. | Empty cells are read as missing. Using the double type lets the median be 6.5 (as in Python) instead of a rounded integer. |
| 2 | **Duplicate Row Filter** | Columns: all. Action: *Remove duplicate rows*, keep the first occurrence. | Removes the 2 duplicate registrations (62 → 60 rows). |
| 3 | **Missing Value** | Default tab: *Number (double)* → **Median**. *String* → **Most Frequent Value**. | Same imputation as Python. Check the output: P008 fitness = 6.5, P053 position = Guard. |
| 4 | **Column Filter** | Exclude `player_name`. Keep `player_id`, `preferred_position` and the six numeric inputs. | Names are reference-only, personal data and not needed for analysis. `player_id` is kept for joining and display. |
| 5 | **Normalizer** | Columns: the six numeric inputs only. Method: **Z-Score Normalization (Gaussian)**. | Puts 0–100 availability and the 1–10 ratings on the same scale (= `StandardScaler`). See the note on standard deviation below. |
| 6 | **k-Means** | Number of clusters: **4**. Centroid initialization: *Random initialization*, **Use static random seed** ✔ (e.g. 42). Max iterations: 99. Include: the six normalized columns only. | Primary algorithm. Output port 1 holds the data plus a `Cluster` column, and output port 2 holds the cluster model. |
| 7 | **Cluster Assigner** | Model input ← k-Means port 2. Data input ← Normalizer output (or new players passed through **Normalizer (Apply)** using the Normalizer's model port). | Shows how the saved model places players without re-training, like `model.predict()` in Python. |
| 8 | **Color Manager** | Column: `Cluster`. Set colors to `#2a78d6`, `#eb6834`, `#1baf7a`, `#4a3aa7` in the order that matches C1–C4 (check with the GroupBy profile). | Same colorblind-checked palette as the Python charts. |
| 9 | **Scatter Plot** | X: `shooting`, Y: `rebounding` (or perimeter/interior averages from a Math Formula, as in the Python chart). Color by `Cluster`. | Shooting vs rebounding separates the four groups well. Availability is the main thing separating *Developing players*. |
| 10 | **Silhouette Coefficient** | Cluster column: `Cluster`. Distance: Euclidean on the six normalized columns. Add one node after k-Means and one after hierarchical clustering. | Evaluation metric for both methods. If your installation lacks this node, compute it in a **Python Script** node with `sklearn.metrics.silhouette_score`. |
| 11 | **Hierarchical Clustering** | Distance function: **Euclidean**. Linkage type: **Average**. Number output clusters: **4**. Columns: the six normalized inputs. | Comparison algorithm. The view shows the dendrogram. In versions where this node is deprecated, use **Distance Matrix Calculate → Hierarchical Clustering (DistMatrix) → Hierarchical Cluster Assigner** with average linkage. |
| 12 | **Joiner** + **GroupBy** (profiles) | Join k-Means output to the *un-normalized* table (output of node 4) on `player_id`. GroupBy `Cluster`: Mean of the six inputs, Count of `player_id`. | Cluster profiles in original units for interpretation. |
| 13 | **Joiner** + **Crosstab** | Join both clustering outputs on `player_id`. Crosstab: rows = K-Means `Cluster`, columns = hierarchical `Cluster`. | Shows where the two methods agree (expect 59 of 60 on the diagonal after matching labels). |
| 14 *(optional)* | **Math Formula** → **Sorter** → **Math Formula** → **GroupBy** → **CSV Writer** | ① `overall_skill = ($shooting$ + $passing$ + $defense$ + $rebounding$) / 4` on the un-normalized columns. ② Sort by `Cluster` ascending, then `overall_skill` descending. ③ `team = mod($$ROWINDEX$$, 4) + 1` (deal players round-robin, like cards). ④ GroupBy `team`: mean `overall_skill`, mean `fitness_level`, mean `availability_pct`, count. | Simple team dealing that gives every team an even share of each group. Python adds tie-breaking by position and skill-balancing swaps, so its teams are more even (see the report). |

## Matching the Python results

After running, compare with `outputs/` from `python train.py`:

| Check | Expected |
|---|---|
| Rows after Duplicate Row Filter | 60 |
| Medians used for imputation | shooting 5.0, passing 5.5, defense 5.0, rebounding 5.0, fitness 6.5, availability 80.0. Position: Guard |
| K-Means cluster sizes | 12 / 16 / 14 / 18 (order may differ) |
| Hierarchical cluster sizes (average linkage) | 13 / 15 / 14 / 18 |
| Overall silhouette | K-Means ≈ 0.385, Hierarchical ≈ 0.382 |
| Cluster profile (K-Means) | Core all-rounders: high on everything (skill ≈ 7.5, availability ≈ 88%). Perimeter playmakers: shooting/passing ≈ 7.2–7.6, rebounding ≈ 3.6. Interior anchors: defense/rebounding ≈ 7.2–7.9, shooting ≈ 3.7. Developing players: all skills ≈ 4, availability ≈ 50%. |

**Expected small differences:**
- **Cluster numbers.** KNIME names clusters `cluster_0` … `cluster_3` in arbitrary order, while Python numbers them from highest to lowest skill. Match them using the profile table, not the number.
- **Standard deviation.** KNIME's z-score uses the sample standard deviation (n − 1), and `StandardScaler` uses the population one (n). Every value differs by the same factor (√(59/60) ≈ 0.992), so distances shrink uniformly. Cluster assignments and silhouette scores do not change.
- **Initialization.** KNIME's k-Means runs once from one random start, while Python keeps the best of 20 starts (`n_init=20`). If KNIME's sizes differ from 12/16/14/18, try another seed. That variation is itself a stability finding worth reporting.
- **Linkage.** Ward linkage is not offered by KNIME's built-in hierarchical nodes, so both tools use **average** linkage.

## Optional: choosing k inside KNIME

Wrap nodes 6 and 10 in a **Counting Loop Start** (k = 2…8, passed to k-Means as a flow variable for *number of clusters*) and a **Loop End**. Plot the overall silhouette against k with a **Line Plot**. The peak should be at k = 4, matching `outputs/k_selection.png`.
