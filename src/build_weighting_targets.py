#!/usr/bin/env python3
"""
build_weighting_targets.py - turn the Destatis population table into the
weighting targets used by R/04_weighting.R.

Source:  data/external/destatis_12411-0013_2025-12-31_age18-69.txt
         Destatis GENESIS-Online table 12411-0013, population on 31.12.2025
         by Land, sex and single year of age (Zensus 2022 basis).
Output:  config/weighting_targets.csv
         one row per variable x level: population aged 18-69 and its share,
         for age_band, gender_w and region (the three raking margins), plus
         the joint age_band x gender_w cells for reference.

Checks:  male + female = total for every Land and age (catches transcription
         errors), 52 ages x 16 Laender present, shares sum to 1 per variable.

Usage:   python src/build_weighting_targets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "destatis_12411-0013_2025-12-31_age18-69.txt"
OUT = ROOT / "config" / "weighting_targets.csv"
CODEBOOK = ROOT / "config" / "codebook.yaml"

# Column order of the GENESIS table -> codebook state codes (official order 1-16)
LAENDER = [("Baden-Wuerttemberg", 8), ("Bayern", 9), ("Berlin", 11), ("Brandenburg", 12),
           ("Bremen", 4), ("Hamburg", 2), ("Hessen", 6), ("Mecklenburg-Vorpommern", 13),
           ("Niedersachsen", 3), ("Nordrhein-Westfalen", 5), ("Rheinland-Pfalz", 7),
           ("Saarland", 10), ("Sachsen", 14), ("Sachsen-Anhalt", 15),
           ("Schleswig-Holstein", 1), ("Thueringen", 16)]

META = {
    "reference_date": "2025-12-31",
    "source": "Destatis GENESIS-Online 12411-0013 (Bevoelkerungsfortschreibung, Basis Zensus 2022)",
    "source_status": "2026-09-21",
    "retrieved": "2026-11-15",
    "universe": "population aged 18-69, Germany",
}


def to_int(token: str) -> int:
    return int(token.replace(".", ""))


def parse(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        label, *nums = line.split()
        if len(nums) != 48:
            raise ValueError(f"{label}: expected 48 numbers, found {len(nums)}")
        age = None if label == "Insgesamt" else int(label.split("-")[0])
        for i, (land, code) in enumerate(LAENDER):
            m, w, t = (to_int(x) for x in nums[3 * i: 3 * i + 3])
            if m + w != t:
                raise ValueError(f"{label} {land}: male {m} + female {w} != total {t}")
            rows.append({"age": age, "land": land, "state": code, "male": m, "female": w, "total": t})
    df = pd.DataFrame(rows)
    ages = sorted(df["age"].dropna().unique())
    if ages != list(range(18, 70)):
        raise ValueError(f"expected ages 18-69, found {ages[0]}-{ages[-1]} ({len(ages)})")
    return df


def build() -> pd.DataFrame:
    cb = yaml.safe_load(open(CODEBOOK, encoding="utf-8"))
    sets, der = cb["value_sets"], cb["derivations"]
    df = parse(SOURCE)
    pop = df[df["age"].notna()].copy()
    breaks = der["age_band_breaks"]
    pop["age_band"] = pd.cut(pop["age"], bins=breaks, right=False, labels=range(1, len(breaks))).astype(int)
    pop["region"] = pop["state"].map({int(k): int(v) for k, v in der["state_to_region"].items()})
    long = pop.melt(id_vars=["age_band", "region", "state"], value_vars=["male", "female"],
                    var_name="sex", value_name="n")
    long["gender_w"] = long["sex"].map({"male": 1, "female": 2})

    out = []
    total = long["n"].sum()
    for var in ("age_band", "gender_w", "region"):
        for level, n in long.groupby(var)["n"].sum().items():
            out.append({"variable": var, "level": int(level), "label": sets[var][int(level)],
                        "population": int(n), "share": n / total})
    for (b, g), n in long.groupby(["age_band", "gender_w"])["n"].sum().items():
        out.append({"variable": "age_band_x_gender_w", "level": int(b) * 10 + int(g),
                    "label": f"{sets['age_band'][int(b)]} {sets['gender_w'][int(g)]}",
                    "population": int(n), "share": n / total})
    t = pd.DataFrame(out)
    for k, v in META.items():
        t[k] = v
    t["share"] = t["share"].round(6)
    for var, g in t.groupby("variable"):
        if abs(g["share"].sum() - 1) > 1e-5:
            raise ValueError(f"{var} shares sum to {g['share'].sum()}")
    return t


def main() -> int:
    t = build()
    t.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(t)} rows), population 18-69: "
          f"{t[t.variable == 'age_band'].population.sum():,}")
    for var in ("age_band", "gender_w", "region"):
        g = t[t["variable"] == var]
        print(f"  {var:9s} " + " | ".join(f"{l} {s:.1%}" for l, s in zip(g["label"], g["share"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
