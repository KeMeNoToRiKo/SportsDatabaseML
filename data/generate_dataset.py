"""Generate the synthetic player roster used by Group 24 (Sports Team Balancing).

The data is SYNTHETIC. No real people are described. Four player profiles are
planted so the clustering has natural structure to discover, and a few data
quality problems (missing values, duplicate rows) are injected on purpose so the
cleaning steps in KNIME and Python have something to catch.

Run:  python data/generate_dataset.py
Output: data/players_raw.csv  (62 rows = 60 unique players + 2 duplicates)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 24
OUT_PATH = Path(__file__).with_name("players_raw.csv")

# Hidden profiles: (count, means, standard deviations, position probabilities)
# means/sds order: shooting, passing, defense, rebounding, fitness_level, availability_pct
# position probabilities order: Guard, Forward, Center
PROFILES = {
    "core_all_rounder": (12, [8.0, 7.5, 7.5, 7.0, 8.5, 90], [0.9, 0.9, 0.9, 0.9, 0.8, 6], [0.40, 0.40, 0.20]),
    "perimeter_playmaker": (16, [7.5, 7.5, 4.5, 3.5, 7.0, 82], [0.9, 0.9, 1.0, 0.9, 1.0, 8], [0.80, 0.20, 0.00]),
    "interior_anchor": (14, [3.8, 4.0, 7.5, 8.0, 6.5, 80], [0.9, 0.9, 0.9, 0.9, 1.0, 8], [0.00, 0.40, 0.60]),
    "developing_casual": (18, [4.0, 4.0, 4.0, 4.0, 5.0, 50], [1.0, 1.0, 1.0, 1.0, 1.2, 12], [0.35, 0.40, 0.25]),
}

FIRST_NAMES = [
    "Aaron", "Bea", "Carlo", "Dana", "Elias", "Faye", "Gab", "Hana", "Ivan", "Jules",
    "Kai", "Lara", "Migs", "Nina", "Oscar", "Pia", "Quinn", "Rafa", "Sam", "Tess",
    "Uri", "Vince", "Wren", "Xander", "Yna", "Zed", "Andre", "Bianca", "Cedric", "Dani",
    "Enzo", "Fritz", "Gelo", "Hiro", "Isa", "Jomar", "Kiko", "Lance", "Mara", "Noel",
    "Owen", "Paolo", "Rhea", "Sol", "Tomas", "Uma", "Vito", "Wes", "Yuri", "Zia",
    "Abel", "Bryce", "Cyd", "Drew", "Ely", "Franco", "Gino", "Harley", "Ian", "Jessa",
]
LAST_INITIALS = list("ABCDEFGHIJKLMNOPRSTUVWY")

SKILL_COLS = ["shooting", "passing", "defense", "rebounding", "fitness_level"]


def build_roster(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for profile, (n, means, sds, pos_p) in PROFILES.items():
        values = rng.normal(means, sds, size=(n, len(means)))
        positions = rng.choice(["Guard", "Forward", "Center"], size=n, p=pos_p)
        for vals, pos in zip(values, positions):
            rows.append({
                "preferred_position": pos,
                **{col: int(np.clip(round(v), 1, 10)) for col, v in zip(SKILL_COLS, vals[:5])},
                "availability_pct": int(np.clip(round(vals[5]), 0, 100)),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    names = rng.permutation(FIRST_NAMES)[: len(df)]
    initials = rng.choice(LAST_INITIALS, size=len(df))
    df.insert(0, "player_name", [f"{f} {i}." for f, i in zip(names, initials)])
    df.insert(0, "player_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
    return df


def inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    numeric = ["shooting", "passing", "defense", "rebounding", "fitness_level", "availability_pct"]
    df[numeric] = df[numeric].astype("Int64")  # nullable ints keep the CSV free of "7.0"

    # Missing values the cleaning step must handle
    df.loc[[7, 41], "fitness_level"] = pd.NA
    df.loc[[19], "availability_pct"] = pd.NA
    df.loc[[33], "shooting"] = pd.NA
    df.loc[[52], "preferred_position"] = None

    # Two exact duplicate rows (e.g., a player registered twice)
    dupes = df.iloc[[12, 45]]
    df = pd.concat([df, dupes]).sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    roster = build_roster(rng)
    raw = inject_quality_issues(roster, rng)
    raw.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(raw)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
