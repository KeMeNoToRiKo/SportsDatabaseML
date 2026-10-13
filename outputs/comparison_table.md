| Measure                                        | K-Means                | Hierarchical (average linkage)   |
|:-----------------------------------------------|:-----------------------|:---------------------------------|
| Silhouette score (higher better)               | 0.385                  | 0.382                            |
| Davies-Bouldin index (lower better)            | 0.911                  | 0.917                            |
| Calinski-Harabasz index (higher better)        | 43.701                 | 43.53                            |
| Cluster sizes                                  | 12 / 16 / 14 / 18      | 13 / 15 / 14 / 18                |
| Smallest cluster                               | 12                     | 13                               |
| Same result on re-run (mean ARI over 20 seeds) | 0.964                  | 1.00 (deterministic)             |
| Team skill gap, 4 teams (lower better)         | 0.017                  | 0.033                            |
| Can place a new player without refitting       | Yes (nearest centroid) | No (must re-cluster)             |

Agreement between the two methods (adjusted Rand index): 0.958
Players placed in a different group: 1 of 60
