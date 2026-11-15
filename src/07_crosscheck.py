#!/usr/bin/env python3
"""
07_crosscheck.py - Mosaik Bank CX Tracker 2026: recompute R's numbers in pandas.

R produced the weights (step 04) and the tables (step 05). This script rebuilds
the same numbers independently, with its own table definitions and its own
significance tests, and fails if anything disagrees:

  weights      mean 1 per wave, inside [0.3, 3.0], weighted margins within
               0.5 points of config/weighting_targets.csv
  banners      every cell of outputs/tables/banners_long.csv:
               unweighted base (exact), effective base (+-0.1),
               estimate (+-0.1 percentage points; NPS +-0.1 points),
               small-base / suppression flags and significance letters (exact)
  NPS trend    outputs/tables/nps_by_wave.csv (and _group.csv if present):
               NPS, SE, CI, change vs previous wave (+-0.1 points), bases, flags

Usage
-----
    python src/07_crosscheck.py
    python src/07_crosscheck.py --tolerance 0.05

Outputs
-------
    outputs/crosscheck_report.md     what was compared and any disagreement
Exit code 1 if any metric disagrees beyond the tolerance.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger("crosscheck")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean" / "tracker_weighted.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
TARGETS = ROOT / "config" / "weighting_targets.csv"
REPORT = ROOT / "outputs" / "crosscheck_report.md"

TOL = 0.1               # points (percentage points for shares, NPS points for NPS)
TOL_NEFF = 0.1          # R writes n_eff rounded to 1 decimal: max rounding error 0.05
MARGIN_TOL_PP = 0.5
TRIM = (0.3, 3.0)
FLAG_BASE, MIN_BASE, ALPHA = 50, 30, 0.05
Z95 = norm.ppf(0.975)


# =============================================================================
# Independent definitions (written from the questionnaire, not copied from R)
# =============================================================================

def flag(cond: pd.Series) -> pd.Series:
    return cond.astype(float).where(cond.notna())


def t2b(x: pd.Series) -> pd.Series:
    return (x >= 4).astype(float).where(x.notna())


def cust(d):
    return d["mosaik_cust"] == 1


def everyone(d):
    return pd.Series(True, index=d.index)


TABLES = {
    "T01": (cust, [("mean", lambda d: d["nps_score"]),
                   ("prop", lambda d: flag(d["nps_group"] == 3)),
                   ("prop", lambda d: flag(d["nps_group"] == 2)),
                   ("prop", lambda d: flag(d["nps_group"] == 1))]),
    "T02": (cust, [("prop", lambda d, i=i: d[f"c3_t2b_r{i}"]) for i in range(1, 9)]),
    "T03": (cust, [("prop", lambda d: t2b(d["c4_overall"])), ("prop", lambda d: t2b(d["c5_switch"]))]),
    "T04": (cust, [("prop", lambda d, i=i: d[f"c2_code_{i}"]) for i in range(1, 9)]),
    "T05": (cust, [("prop", lambda d, k=k: flag(d["d1_app_freq"] == k)) for k in range(1, 5)]),
    "T06": (cust, [("prop", lambda d: d["d2_contact"]),
                   ("prop", lambda d: flag(d["d3_resolved"] == 1).where(d["d2_contact"] == 1)),
                   ("prop", lambda d: flag(d["d3_resolved"] == 3).where(d["d2_contact"] == 1))]),
    "T07": (everyone, [("prop", lambda d: d["b1_r1"]), ("prop", lambda d: (d["b2_r1"] == 1).astype(float)),
                       ("prop", lambda d: d["a1_r1"]), ("prop", lambda d: (d["main_bank"] == 1).astype(float))]),
    "T08": (everyone, [("prop", lambda d, i=i: d[f"b1_r{i}"]) for i in range(1, 6)]),
    "T09": (everyone, [("prop", lambda d, i=i: (d[f"b2_r{i}"] == 1).astype(float)) for i in range(1, 6)]),
    "T10": (everyone, [("prop", lambda d, i=i: (d["main_bank"] == i).astype(float)) for i in range(1, 7)]),
    "T11": (everyone, [("prop", lambda d, i=i: t2b(d[f"e1_r{i}"])) for i in range(1, 7)]),
}


def banner_columns(d: pd.DataFrame) -> dict[str, tuple]:
    """banner -> (row filter, [(group, column label, mask function), ...])"""
    labels = {1: "W1 Mar 26", 2: "W2 Jun 26", 3: "W3 Sep 26", 4: "W4 Dec 26"}
    latest = int(d["wave"].max())
    trend = [("Total", "Total W1-W4", lambda x: pd.Series(True, index=x.index))] + \
            [("Wave", labels[w], lambda x, w=w: x["wave"] == w) for w in sorted(d["wave"].unique())]
    prof = [("Total", f"Total {labels[latest]}", lambda x: pd.Series(True, index=x.index))]
    prof += [("Age", lab, lambda x, b=b: x["age_band"] == b)
             for b, lab in enumerate(["18-29", "30-39", "40-49", "50-59", "60-69"], start=1)]
    prof += [("Gender", "Male", lambda x: x["gender"] == 1), ("Gender", "Female", lambda x: x["gender"] == 2)]
    prof += [("Mosaik relationship", "Main bank", lambda x: (x["mosaik_cust"] == 1) & (x["mosaik_main"] == 1)),
             ("Mosaik relationship", "Secondary", lambda x: (x["mosaik_cust"] == 1) & (x["mosaik_main"] == 0)),
             ("Mosaik relationship", "Not a customer", lambda x: x["mosaik_cust"] == 0)]
    prof += [("Region", lab, lambda x, r=r: x["region"] == r)
             for r, lab in enumerate(["North", "East", "South", "West"], start=1)]
    return {"trend": (lambda x: pd.Series(True, index=x.index), trend),
            "latest": (lambda x: x["wave"] == latest, prof)}


# =============================================================================
# Statistics, implemented here from the formulas
# =============================================================================

def n_eff(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def letters_for(est, ne, sd, kind, col_letters) -> list[str]:
    out = []
    for i in range(len(est)):
        if np.isnan(est[i]):
            out.append("")
            continue
        hits = []
        for j in range(len(est)):
            if i == j or np.isnan(est[j]):
                continue
            if kind == "prop":
                pool = (est[i] * ne[i] + est[j] * ne[j]) / (ne[i] + ne[j])
                se = np.sqrt(pool * (1 - pool) * (1 / ne[i] + 1 / ne[j]))
            else:
                se = np.sqrt(sd[i] ** 2 / ne[i] + sd[j] ** 2 / ne[j])
            if se > 0 and 2 * norm.sf(abs((est[i] - est[j]) / se)) < ALPHA and est[i] > est[j]:
                hits.append(col_letters[j])
        out.append("".join(sorted(hits)))
    return out


def recompute_banners(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bname, (bfilter, cols) in banner_columns(d).items():
        dd = d[bfilter(d)]
        for tid, (base, trows) in TABLES.items():
            in_base = base(dd).fillna(False)
            for r, (kind, f) in enumerate(trows, start=1):
                x = f(dd)
                cells = []
                for c, (group, label, mf) in enumerate(cols, start=1):
                    m = in_base & mf(dd).fillna(False) & x.notna()
                    w, v = dd.loc[m, "weight"].to_numpy(), x[m].to_numpy(float)
                    n = int(m.sum())
                    est = float(np.average(v, weights=w)) if n else np.nan
                    sd = float(np.sqrt(np.average((v - est) ** 2, weights=w))) if n and kind == "mean" else np.nan
                    sup = n < MIN_BASE
                    cells.append({"banner": bname, "table_id": tid, "row_order": r, "col_order": c, "type": kind,
                                  "group": group, "column": label, "n_unweighted": n, "n_eff": n_eff(w),
                                  "estimate": np.nan if sup else est, "sd": np.nan if sup else sd,
                                  "small_base": (n < FLAG_BASE) and not sup, "suppressed": sup})
                cells = pd.DataFrame(cells)
                letters = [chr(64 + i) for i in range(1, len(cols) + 1)]
                cells["letter"] = letters
                cells["sig"] = ""
                for g, idx in cells.groupby("group").groups.items():
                    if len(idx) < 2:
                        continue
                    sub = cells.loc[idx]
                    cells.loc[idx, "sig"] = letters_for(sub["estimate"].to_numpy(), sub["n_eff"].to_numpy(),
                                                        sub["sd"].to_numpy(), kind, sub["letter"].tolist())
                rows.append(cells)
    return pd.concat(rows, ignore_index=True)


def recompute_nps(d: pd.DataFrame, group_var=None, levels=None, labels=None) -> pd.DataFrame:
    """NPS by wave with a linearisation SE (as survey::svymean on a domain of the full design)."""
    N = len(d)
    specs = [("Total", "Total", pd.Series(True, index=d.index))] if group_var is None else \
            [(group_var, lab, d[group_var] == lev) for lev, lab in zip(levels, labels)]
    out = []
    for grp, lab, keep in specs:
        prev = None
        first = None
        for w in sorted(d["wave"].unique()):
            dom = (d["mosaik_cust"] == 1) & (d["wave"] == w) & keep
            g = d[dom]
            n = len(g)
            row = {"group": grp, "subgroup": lab, "wave": w, "n_unweighted": n,
                   "n_eff": n_eff(g["weight"].to_numpy()), "suppressed": n < MIN_BASE,
                   "small_base": MIN_BASE <= n < FLAG_BASE}
            if n >= MIN_BASE:
                wt, y = g["weight"].to_numpy(), g["nps_score"].to_numpy(float)
                m = np.average(y, weights=wt)
                se = np.sqrt(N / (N - 1) * np.sum((wt * (y - m)) ** 2)) / wt.sum()
                row.update(nps=m, se=se, ci_low=m - Z95 * se, ci_high=m + Z95 * se)
            else:
                row.update(nps=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan)
            for ref, key in ((prev, "prev"), (first, "w1")):
                if ref is None or np.isnan(row["nps"]) or np.isnan(ref["nps"]):
                    row[f"change_vs_{key}"], row[f"sig_vs_{key}"] = np.nan, False
                else:
                    diff = row["nps"] - ref["nps"]
                    p = 2 * norm.sf(abs(diff / np.sqrt(row["se"] ** 2 + ref["se"] ** 2)))
                    row[f"change_vs_{key}"], row[f"sig_vs_{key}"] = diff, p < ALPHA
            out.append(row)
            prev = row
            first = first or row
    return pd.DataFrame(out)


# =============================================================================
# Comparisons
# =============================================================================

class Findings:
    def __init__(self):
        self.items: list[dict] = []
        self.counts: dict[str, int] = {}

    def compared(self, what: str, n: int):
        self.counts[what] = self.counts.get(what, 0) + n

    def add(self, area, key, metric, r_value, py_value, diff=None):
        self.items.append({"area": area, "key": key, "metric": metric, "R": r_value, "pandas": py_value,
                           "diff": diff})


def check_weights(d: pd.DataFrame, targets: pd.DataFrame, f: Findings):
    for w, g in d.groupby("wave"):
        f.compared("weights: wave checks", 3)
        if abs(g["weight"].mean() - 1) > 1e-6:
            f.add("weights", f"W{w}", "mean weight", round(g["weight"].mean(), 6), 1.0)
        lo, hi = g["weight"].min(), g["weight"].max()
        if lo < TRIM[0] - 1e-6 or hi > TRIM[1] + 1e-6:
            f.add("weights", f"W{w}", "weight range", f"{lo:.3f}-{hi:.3f}", f"{TRIM[0]}-{TRIM[1]}")
        for var in ("age_band", "gender_w", "region"):
            t = targets[targets["variable"] == var].set_index("level")["share"]
            got = g.groupby(var)["weight"].sum() / g["weight"].sum()
            for lev, share in t.items():
                f.compared("weights: margin cells", 1)
                diff = 100 * (got.get(lev, 0) - share)
                if abs(diff) > MARGIN_TOL_PP:
                    f.add("weights", f"W{w} {var}={lev}", "weighted % vs target", round(100 * got.get(lev, 0), 2),
                          round(100 * share, 2), round(diff, 2))


def check_banners(r: pd.DataFrame, py: pd.DataFrame, tol: float, f: Findings):
    key = ["banner", "table_id", "row_order", "col_order"]
    r = r.copy()
    r["sig"] = r["sig"].fillna("").astype(str)
    m = r.merge(py, on=key, how="outer", suffixes=("_r", "_py"), indicator=True)
    for side, label in (("left_only", "only in R"), ("right_only", "only in pandas")):
        for _, row in m[m["_merge"] == side].head(20).iterrows():
            f.add("banners", "/".join(str(row[k]) for k in key), "cell exists", label, "")
    both = m[m["_merge"] == "both"]
    f.compared("banners: cells", len(both))
    for _, row in both.iterrows():
        k = f"{row['banner']}/{row['table_id']}/row {row['row_order']}/{row['column_r']}"
        if row["column_r"] != row["column_py"]:
            f.add("banners", k, "column label", row["column_r"], row["column_py"])
        if int(row["n_unweighted_r"]) != int(row["n_unweighted_py"]):
            f.add("banners", k, "unweighted base", row["n_unweighted_r"], row["n_unweighted_py"])
        if abs(row["n_eff_r"] - row["n_eff_py"]) > TOL_NEFF:
            f.add("banners", k, "effective base", row["n_eff_r"], round(row["n_eff_py"], 2),
                  round(row["n_eff_r"] - row["n_eff_py"], 3))
        if bool(row["suppressed_r"]) != bool(row["suppressed_py"]) or bool(row["small_base_r"]) != bool(row["small_base_py"]):
            f.add("banners", k, "base flags", f"{row['small_base_r']}/{row['suppressed_r']}",
                  f"{row['small_base_py']}/{row['suppressed_py']}")
        er, ep = row["estimate_r"], row["estimate_py"]
        if pd.isna(er) != pd.isna(ep):
            f.add("banners", k, "estimate present", er, ep)
        elif not pd.isna(er):
            scale = 100 if row["type_r"] == "prop" else 1
            diff = scale * (er - ep)
            if abs(diff) > tol:
                f.add("banners", k, "estimate (points)", round(scale * er, 3), round(scale * ep, 3), round(diff, 3))
        if row["sig_r"] != row["sig_py"]:
            f.add("banners", k, "significance letters", row["sig_r"] or "(none)", row["sig_py"] or "(none)")


def check_nps(r: pd.DataFrame, py: pd.DataFrame, tol: float, f: Findings, area: str):
    key = ["subgroup", "wave"]
    m = r.merge(py, on=key, suffixes=("_r", "_py"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        f.add(area, "rows", "row sets differ", int((m["_merge"] == "left_only").sum()),
              int((m["_merge"] == "right_only").sum()))
    m = m[m["_merge"] == "both"]
    f.compared(f"{area}: rows", len(m))
    for _, row in m.iterrows():
        k = f"{row['subgroup']} W{row['wave']}"
        if int(row["n_unweighted_r"]) != int(row["n_unweighted_py"]):
            f.add(area, k, "unweighted base", row["n_unweighted_r"], row["n_unweighted_py"])
        if abs(row["n_eff_r"] - row["n_eff_py"]) > TOL_NEFF:
            f.add(area, k, "effective base", row["n_eff_r"], round(row["n_eff_py"], 2))
        for metric in ("nps", "se", "ci_low", "ci_high", "change_vs_prev", "change_vs_w1"):
            a, b = row[f"{metric}_r"], row[f"{metric}_py"]
            if pd.isna(a) != pd.isna(b):
                f.add(area, k, metric, a, b)
            elif not pd.isna(a) and abs(a - b) > tol:
                f.add(area, k, metric, round(a, 3), round(b, 3), round(a - b, 3))
        for metric in ("sig_vs_prev", "sig_vs_w1", "suppressed", "small_base"):
            if bool(row[f"{metric}_r"]) != bool(row[f"{metric}_py"]):
                f.add(area, k, metric, row[f"{metric}_r"], row[f"{metric}_py"])


# =============================================================================
# Main
# =============================================================================

def write_report(f: Findings, tol: float, path: Path):
    lines = ["# Cross-check: R outputs vs independent pandas recomputation", "",
             f"_Generated {date.today().isoformat()} by `src/07_crosscheck.py`. Tolerance {tol} points "
             f"(effective bases {TOL_NEFF}, weighting margins {MARGIN_TOL_PP} points)._", "",
             "| Compared | Count |", "| --- | --- |"]
    lines += [f"| {k} | {v:,} |" for k, v in f.counts.items()]
    if not f.items:
        lines += ["", "**All values agree.**"]
    else:
        lines += ["", f"**{len(f.items)} disagreement(s)**", "", "| Area | Key | Metric | R | pandas | Diff |",
                  "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {i['area']} | {i['key']} | {i['metric']} | {i['R']} | {i['pandas']} | "
                  f"{'' if i['diff'] is None else i['diff']} |" for i in f.items[:200]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute R's tables in pandas and compare.")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    ap.add_argument("--targets", type=Path, default=TARGETS)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--tolerance", type=float, default=TOL)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    needed = [args.data, args.tables_dir / "banners_long.csv", args.tables_dir / "nps_by_wave.csv"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        log.error("Missing R outputs: %s. Run R/04_weighting.R and R/05_tables_sigtests.R first.", missing)
        return 1

    d = pd.read_csv(args.data, low_memory=False)
    f = Findings()
    check_weights(d, pd.read_csv(args.targets), f)
    check_banners(pd.read_csv(args.tables_dir / "banners_long.csv"), recompute_banners(d), args.tolerance, f)
    check_nps(pd.read_csv(args.tables_dir / "nps_by_wave.csv"), recompute_nps(d), args.tolerance, f, "nps_by_wave")
    grp_path = args.tables_dir / "nps_by_wave_group.csv"
    if grp_path.exists():
        py_grp = pd.concat([recompute_nps(d, "mosaik_main", [1, 0], ["Main bank", "Secondary"]),
                            recompute_nps(d, "age_band", [1, 2, 3, 4, 5],
                                          ["18-29", "30-39", "40-49", "50-59", "60-69"])])
        check_nps(pd.read_csv(grp_path), py_grp, args.tolerance, f, "nps_by_wave_group")

    write_report(f, args.tolerance, args.report)
    for what, n in f.counts.items():
        log.info("compared %-26s %6d", what, n)
    if f.items:
        log.error("%d disagreement(s) beyond tolerance; first ones:", len(f.items))
        for i in f.items[:15]:
            log.error("  %s | %s | %s: R=%s pandas=%s", i["area"], i["key"], i["metric"], i["R"], i["pandas"])
        log.error("Full list: %s", args.report)
        return 1
    log.info("All R outputs agree with pandas within tolerance. Report: %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
