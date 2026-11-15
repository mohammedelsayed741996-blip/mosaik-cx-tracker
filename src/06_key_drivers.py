#!/usr/bin/env python3
"""
06_key_drivers.py - Mosaik Bank CX Tracker 2026: what drives NPS?

Method
------
* Mosaik customers, all waves pooled, weighted (weights from R/04).
* Outcome: likelihood to recommend (C1, 0-10). Predictors: the 7 attributes
  rated in every wave (C3 r1-r7, 1-5). Investing (r8) is left out because it
  only exists from W3. "Don't know" answers are excluded (complete cases).
* Shapley value regression: the R2 of the full weighted regression is split
  fairly between the attributes by averaging each attribute's added R2 over
  all 2^7 = 128 subsets of the others. Unlike regression coefficients, the
  shares stay stable when attributes are correlated, and they add up to R2.
* A bootstrap (resampling respondents within wave) gives 95% intervals for
  each share and shows how stable the ranking is.
* Priority matrix: importance (share of R2) against performance (weighted
  top-2-box in the latest wave), split at the averages:
      Fix first = above-average importance, below-average performance
      Maintain  = above-average importance, above-average performance
      Monitor   = below-average importance, below-average performance
      Secondary = below-average importance, above-average performance

Afterwards, and only because the data is simulated, the ranking is checked
against the planted driver weights in data/_truth/ (--no-evaluate to skip).

Usage
-----
    python src/06_key_drivers.py
    python src/06_key_drivers.py --bootstrap 500 --data data/clean/tracker_weighted.csv

Outputs
-------
    outputs/tables/drivers.csv          one row per attribute
    outputs/tables/drivers_summary.json  R2, n, bootstrap stability, truth check
Exit code 1 if the top 3 drivers do not match the planted weights in rank order.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("drivers")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean" / "tracker_weighted.csv"
TRUTH = ROOT / "data" / "_truth" / "driver_weights.csv"
OUT_DIR = ROOT / "outputs" / "tables"

ATTRIBUTES = {              # item -> (attribute key used in the truth file, label)
    "c3_r1": ("app", "App ease"),
    "c3_r2": ("speed", "Transfer speed"),
    "c3_r3": ("fees", "Fees / value for money"),
    "c3_r4": ("service", "Customer service"),
    "c3_r5": ("security", "Security & trust"),
    "c3_r6": ("card", "Card features"),
    "c3_r7": ("interest", "Savings interest"),
}
ITEMS = list(ATTRIBUTES)
OUTCOME = "c1_nps"
SEED = 42


# =============================================================================
# Weighted regression and Shapley decomposition
# =============================================================================

def weighted_r2_all_subsets(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[frozenset, float]:
    """Weighted R2 for every subset of predictors (the empty set has R2 = 0)."""
    sw = np.sqrt(w)
    y_mean = np.average(y, weights=w)
    sst = np.sum(w * (y - y_mean) ** 2)
    p = X.shape[1]
    r2 = {frozenset(): 0.0}
    for k in range(1, p + 1):
        for subset in combinations(range(p), k):
            A = np.column_stack([np.ones(len(y)), X[:, subset]]) * sw[:, None]
            beta, *_ = np.linalg.lstsq(A, y * sw, rcond=None)
            resid = y - np.column_stack([np.ones(len(y)), X[:, subset]]) @ beta
            r2[frozenset(subset)] = 1 - np.sum(w * resid ** 2) / sst
    return r2


def shapley_values(r2: dict[frozenset, float], p: int) -> np.ndarray:
    """Exact Shapley value of each predictor's contribution to R2."""
    phi = np.zeros(p)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            weight = factorial(k) * factorial(p - k - 1) / factorial(p)
            for s in combinations(others, k):
                s = frozenset(s)
                phi[j] += weight * (r2[s | {j}] - r2[s])
    return phi


def standardised_betas(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    def z(a):
        m = np.average(a, weights=w, axis=0)
        s = np.sqrt(np.average((a - m) ** 2, weights=w, axis=0))
        return (a - m) / s
    Xz, yz = z(X), z(y)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xz * sw[:, None], yz * sw, rcond=None)
    return beta


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


# =============================================================================
# Analysis
# =============================================================================

def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run R/04_weighting.R first")
    d = pd.read_csv(path, low_memory=False)
    if "weight" not in d or d["weight"].isna().any():
        raise ValueError("weight column missing or incomplete")
    return d


def analysis_frame(d: pd.DataFrame) -> pd.DataFrame:
    cust = d[d["mosaik_cust"] == 1]
    frame = cust[["wave", "weight", OUTCOME] + ITEMS].dropna()
    log.info("Mosaik customers %d, complete cases for the driver model %d (%.0f%%)",
             len(cust), len(frame), 100 * len(frame) / len(cust))
    return frame.reset_index(drop=True)


def run_shapley(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    X, y, w = frame[ITEMS].to_numpy(float), frame[OUTCOME].to_numpy(float), frame["weight"].to_numpy(float)
    r2 = weighted_r2_all_subsets(X, y, w)
    phi = shapley_values(r2, len(ITEMS))
    total = r2[frozenset(range(len(ITEMS)))]
    assert abs(phi.sum() - total) < 1e-9, "Shapley values must add up to the full-model R2"
    return phi, total


def bootstrap(frame: pd.DataFrame, reps: int, seed: int = SEED) -> np.ndarray:
    """Shapley shares for `reps` resamples (respondents resampled within wave)."""
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in frame.groupby("wave")]
    out = np.zeros((reps, len(ITEMS)))
    for b in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        phi, total = run_shapley(frame.loc[idx].reset_index(drop=True))
        out[b] = phi / total
    return out


def performance(d: pd.DataFrame) -> pd.DataFrame:
    """Weighted top-2-box per attribute: latest wave, previous wave and all waves."""
    cust = d[d["mosaik_cust"] == 1]
    latest = int(d["wave"].max())
    rows = []
    for item in ITEMS:
        t2b = f"c3_t2b_{item.split('_')[1]}"

        def wt(g):
            g = g[g[t2b].notna()]
            return 100 * np.average(g[t2b], weights=g["weight"]) if len(g) else np.nan
        rows.append({"item": item,
                     "perf_t2b_latest": wt(cust[cust["wave"] == latest]),
                     "perf_t2b_previous": wt(cust[cust["wave"] == latest - 1]),
                     "perf_t2b_all_waves": wt(cust)})
    perf = pd.DataFrame(rows)
    perf["perf_change_pp"] = perf["perf_t2b_latest"] - perf["perf_t2b_previous"]
    return perf


def quadrant(imp: float, perf: float, imp_cut: float, perf_cut: float) -> str:
    if imp >= imp_cut:
        return "Fix first" if perf < perf_cut else "Maintain"
    return "Monitor" if perf < perf_cut else "Secondary"


def evaluate_truth(drivers: pd.DataFrame, truth_path: Path) -> dict | None:
    if not truth_path.exists():
        log.warning("No truth file at %s: skipping the check", truth_path)
        return None
    truth = pd.read_csv(truth_path)
    truth = truth[truth["attribute"].isin([a for a, _ in ATTRIBUTES.values()])]
    truth_order = truth.sort_values("true_weight", ascending=False)["attribute"].tolist()
    found_order = drivers.sort_values("rank")["attribute"].tolist()
    top3_match = truth_order[:3] == found_order[:3]
    from scipy.stats import spearmanr          # only needed for the check
    merged = drivers.merge(truth[["attribute", "true_weight"]], on="attribute")
    rho = spearmanr(merged["share_of_r2"], merged["true_weight"]).statistic
    return {"truth_top3": truth_order[:3], "found_top3": found_order[:3], "top3_match": bool(top3_match),
            "truth_order": truth_order, "found_order": found_order, "spearman_rank_corr": round(float(rho), 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Weighted Shapley driver analysis of NPS.")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--truth", type=Path, default=TRUTH)
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples (0 = skip)")
    ap.add_argument("--no-evaluate", action="store_true", help="skip the check against data/_truth/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")

    d = load(args.data)
    frame = analysis_frame(d)
    phi, total_r2 = run_shapley(frame)
    X, y, w = frame[ITEMS].to_numpy(float), frame[OUTCOME].to_numpy(float), frame["weight"].to_numpy(float)

    drivers = pd.DataFrame({
        "item": ITEMS,
        "attribute": [ATTRIBUTES[i][0] for i in ITEMS],
        "label": [ATTRIBUTES[i][1] for i in ITEMS],
        "shapley_r2": phi,
        "share_of_r2": phi / total_r2,
        "std_beta": standardised_betas(X, y, w),
        "corr_with_nps": [weighted_corr(X[:, j], y, w) for j in range(len(ITEMS))],
    })
    drivers["rank"] = drivers["share_of_r2"].rank(ascending=False, method="first").astype(int)

    stability = None
    if args.bootstrap > 0:
        log.info("Bootstrapping %d resamples ...", args.bootstrap)
        boot = bootstrap(frame, args.bootstrap)
        drivers["share_ci_low"] = np.percentile(boot, 2.5, axis=0)
        drivers["share_ci_high"] = np.percentile(boot, 97.5, axis=0)
        top3 = drivers.sort_values("rank")["item"].tolist()[:3]
        top3_idx = [ITEMS.index(i) for i in top3]
        same_top3 = np.mean([list(np.argsort(-row)[:3]) == top3_idx for row in boot])
        drivers["boot_rank_mode"] = [int(pd.Series(np.argsort(np.argsort(-boot, axis=1), axis=1)[:, j] + 1).mode()[0])
                                     for j in range(len(ITEMS))]
        stability = {"reps": args.bootstrap, "share_same_top3_order": round(float(same_top3), 3)}

    perf = performance(d)
    drivers = drivers.merge(perf, on="item")
    imp_cut = 1 / len(ITEMS)                                  # average share of R2
    perf_cut = drivers["perf_t2b_latest"].mean()
    drivers["quadrant"] = [quadrant(i, p, imp_cut, perf_cut)
                           for i, p in zip(drivers["share_of_r2"], drivers["perf_t2b_latest"])]
    drivers = drivers.sort_values("rank").reset_index(drop=True)

    ev = None if args.no_evaluate else evaluate_truth(drivers, args.truth)
    if ev:
        truth = pd.read_csv(args.truth).set_index("attribute")
        drivers["true_weight"] = drivers["attribute"].map(truth["true_weight"])
        drivers["true_rank"] = drivers["true_weight"].rank(ascending=False, method="first").astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = drivers.copy()
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(4)
    out.to_csv(args.out_dir / "drivers.csv", index=False)
    summary = {"model": "weighted OLS of C1 (0-10) on C3 r1-r7 (1-5), Shapley decomposition of R2",
               "base": "Mosaik customers, all waves pooled, complete cases",
               "n": int(len(frame)), "n_eff": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
               "r2": round(float(total_r2), 4), "importance_cut": imp_cut,
               "performance_cut_t2b": round(float(perf_cut), 2),
               "latest_wave": int(d["wave"].max()), "bootstrap": stability, "truth_check": ev}
    with open(args.out_dir / "drivers_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nKey drivers of NPS (weighted Shapley regression, n = {len(frame):,}, R2 = {total_r2:.3f})")
    cols = ["rank", "label", "share_of_r2", "share_ci_low", "share_ci_high", "perf_t2b_latest",
            "perf_change_pp", "quadrant"]
    show = drivers[[c for c in cols if c in drivers]].copy()
    for c in ("share_of_r2", "share_ci_low", "share_ci_high"):
        if c in show:
            show[c] = (100 * show[c]).round(1)
    for c in ("perf_t2b_latest", "perf_change_pp"):
        show[c] = show[c].round(1)
    print(show.to_string(index=False))
    if stability:
        print(f"Bootstrap: same top-3 order in {100 * stability['share_same_top3_order']:.0f}% of {stability['reps']} resamples")
    if ev:
        print(f"\nTruth check (simulated data only): planted top 3 {ev['truth_top3']}, found {ev['found_top3']} "
              f"-> {'MATCH' if ev['top3_match'] else 'MISMATCH'}; rank correlation {ev['spearman_rank_corr']}")
        if not ev["top3_match"]:
            log.error("Top 3 drivers do not match the planted weights in rank order")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
