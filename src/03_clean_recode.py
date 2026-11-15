#!/usr/bin/env python3
"""
03_clean_recode.py - Mosaik Bank CX Tracker 2026: build the clean analysis file.

Input:  data/interim/flagged.parquet (step 02)
Steps:
  1. drop respondents flagged qc_remove (kept in an audit file with reasons)
  2. turn special codes (-99 refused, -98 not asked, -97 don't know) into NA,
     keeping the reason in <var>_miss_reason for gender, C3 and F1
  3. derive: age_band, region, gender_w, mosaik_cust, mosaik_main, nps_group,
     nps_score, c3_t2b_r*, and the C2 open-end codes (8-code frame)
  4. check routing and codes against the codebook (any failure stops the run)
  5. write the clean file as parquet, CSV (for R) and SPSS .sav (labelled)

Usage
-----
    python src/03_clean_recode.py

Outputs
-------
    data/clean/tracker_clean.parquet     typed, for Python steps (needs pyarrow)
    data/clean/tracker_clean.csv         UTF-8, empty cell = missing, for R
    data/clean/tracker_clean.sav         labelled SPSS file (needs pyreadstat)
    data/clean/removed_respondents.csv   who was dropped and why
    outputs/cleaning_report.json         counts for every step
    outputs/c2_coding_review.csv         every open end with its codes, for checking
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("clean")

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
CONFIG = ROOT / "config"

SEED = 42                               # gender_w imputation is reproducible
MISS_REASON_VARS = ["gender", "f1_income"] + [f"c3_r{i}" for i in range(1, 9)]
C3_ITEMS = [f"c3_r{i}" for i in range(1, 9)]
C_SECTION = (["c1_nps", "c2_reason", "c3x_attn", "c4_overall", "c5_switch", "d1_app_freq", "d2_contact"]
             + C3_ITEMS + [f"c6_r{i}" for i in range(1, 6)])
QC_KEEP = ["qc_speeder", "qc_straightliner", "qc_gibberish"]   # the only flags left on kept rows


class CleaningError(RuntimeError):
    """A check that must stop the pipeline."""


# =============================================================================
# Open-end coding (C2)
# =============================================================================
# Keyword stems per code, matched on normalised text (lower case, umlauts
# spelled out). Several stems per code, so one typo rarely loses the code; a
# fuzzy pass catches swapped letters in longer words ("trnasfers").
CODE_FRAME_STEMS: dict[int, list[str]] = {
    1: ["gebuehr", "gebuhr", r"\bfees?\b", "teuer", "expensive", "kosten", "preis", "price",
        "4,90", "4.90", "im monat", "a month", "pricing"],
    2: [r"\bapp\b", r"\bapps\b", "abstuerz", "crash", "unuebersichtlich", "uebersichtlich"],
    3: ["service", "support", "chat", "erreichbar", "wartezeit", "waited", "hotline", "berater"],
    4: ["sicher", "secur", "safe", "phishing", "betrug", "fraud"],
    5: ["ueberweis", "uberweis", "transfer"],
    6: ["karte", "card"],
    7: ["zins", "zinsen", "interest", "tagesgeld", "savings", "etf", "invest", "anlage", "sparplan", "depot"],
}
OTHER_CODE = 8
FUZZY_MIN_LEN = 5
FUZZY_RATIO = 0.80


def normalise(text: str) -> str:
    t = text.lower().strip()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t)


def code_open_end(text) -> tuple[set[int], str]:
    """Return (codes, method) for one C2 answer. method: keyword | fuzzy | other."""
    if not isinstance(text, str) or not text.strip():
        return set(), "none"
    t = normalise(text)
    codes = {code for code, stems in CODE_FRAME_STEMS.items()
             if any(re.search(s, t) if s.startswith(r"\b") else s in t for s in stems)}
    method = "keyword"
    if not codes:
        tokens = [w for w in re.findall(r"[a-z]+", t) if len(w) >= FUZZY_MIN_LEN]
        for code, stems in CODE_FRAME_STEMS.items():
            plain = [s for s in stems if len(s) >= FUZZY_MIN_LEN and s.isalpha()]
            for tok in tokens:
                if any(difflib.SequenceMatcher(None, tok[:len(s) + 2], s).ratio() >= FUZZY_RATIO
                       for s in plain):
                    codes.add(code)
        method = "fuzzy" if codes else "other"
    # coding rule: a complaint about the card FEE is a fees answer, not a card answer
    if 1 in codes and 6 in codes:
        codes.discard(6)
    return (codes or {OTHER_CODE}), method


# =============================================================================
# I/O
# =============================================================================

def read_frame(stem: Path) -> pd.DataFrame:
    pq, gz = stem.with_suffix(".parquet"), stem.with_suffix(".csv.gz")
    if pq.exists():
        return pd.read_parquet(pq)
    if gz.exists():
        df = pd.read_csv(gz, low_memory=False, parse_dates=["start_ts", "end_ts"])
        for c in ("resp_id", "panel_id", "source_file", "c2_reason", "qc_reasons"):
            if c in df:
                df[c] = df[c].astype("string")
        return df
    raise FileNotFoundError(f"neither {pq.name} nor {gz.name} in {stem.parent}; run step 02 first")


def load_codebook(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# =============================================================================
# Steps
# =============================================================================

def drop_removed(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed = df[df["qc_remove"] == 1]
    report["rows_in"] = int(len(df))
    report["removed_by_wave"] = {f"W{w}": int(n) for w, n in removed["wave"].value_counts().sort_index().items()}
    report["rows_out"] = int(len(df) - len(removed))
    audit = removed[["resp_id", "wave", "record", "panel_id", "qc_reasons"]].sort_values(["wave", "record"])
    return df[df["qc_remove"] == 0].copy(), audit


def special_codes_to_na(df: pd.DataFrame, codebook: dict, report: dict) -> pd.DataFrame:
    """-99 / -98 / -97 -> NA everywhere; <var>_miss_reason keeps why, where it matters."""
    special = [int(k) for k in codebook["missing_codes"]]
    int_vars = [v["name"] for v in codebook["variables"] if v["type"] == "int" and v["name"] in df]
    cust = df["a1_r1"] == 1
    counts = {}
    for var in MISS_REASON_VARS:
        if var not in df:
            continue
        reason = df[var].where(df[var].isin(special))
        if var.startswith("c3_"):
            asked = cust & (df["wave"] >= 3) if var == "c3_r8" else cust
            reason = reason.mask(reason.isna() & ~asked, -98)       # blank because not asked
        df[f"{var}_miss_reason"] = reason.astype("Int64")
    for var in int_vars:
        hit = df[var].isin(special)
        if hit.any():
            counts[var] = {str(k): int(v) for k, v in df.loc[hit, var].value_counts().items()}
            df[var] = df[var].mask(hit)
    report["special_codes_to_na"] = counts
    return df


def impute_gender_w(df: pd.DataFrame, report: dict) -> pd.Series:
    """Weighting needs male/female (census has two categories). 'Diverse' and
    'prefer not to say' get one by a seeded draw, in proportion to the male /
    female split in the same wave x age band. Tables keep the full `gender`."""
    rng = np.random.default_rng(SEED)
    gw = df["gender"].where(df["gender"].isin([1, 2])).astype("float64")
    todo = df.index[gw.isna()].sort_values()
    for (w, b), g in df.loc[todo].groupby(["wave", "age_band"]):
        ref = df[(df["wave"] == w) & (df["age_band"] == b) & df["gender"].isin([1, 2])]["gender"]
        p_male = (ref == 1).mean() if len(ref) else 0.5
        gw.loc[g.index] = np.where(rng.random(len(g)) < p_male, 1, 2)
    report["gender_w_imputed"] = int(len(todo))
    return gw.astype("Int64")


def derive(df: pd.DataFrame, codebook: dict, report: dict) -> pd.DataFrame:
    d = codebook["derivations"]
    breaks = d["age_band_breaks"]
    df["age_band"] = pd.cut(df["age"], bins=breaks, right=False, labels=range(1, len(breaks))).astype("Int64")
    df["region"] = df["state"].map({int(k): int(v) for k, v in d["state_to_region"].items()}).astype("Int64")
    df["gender_w"] = impute_gender_w(df, report)

    df["mosaik_cust"] = (df["a1_r1"] == 1).astype("Int64")
    df["mosaik_main"] = (df["main_bank"] == 1).astype("Int64")

    nps = pd.to_numeric(df["c1_nps"]).astype("float64")       # works for Int64 and float input
    df["nps_group"] = pd.Series(np.select([nps <= 6, nps <= 8, nps <= 10], [1, 2, 3], default=0),
                                index=df.index).where(nps.notna()).astype("Int64")
    df["nps_score"] = df["nps_group"].map({1: -100, 2: 0, 3: 100}).astype("Int64")

    t2b = set(d["top2box"])
    for item in C3_ITEMS:
        r = item.split("_")[1]
        df[f"c3_t2b_{r}"] = df[item].isin(t2b).astype("Int64").where(df[item].notna())

    # open-end coding
    coded = df["c2_reason"].map(code_open_end)
    codes = coded.map(lambda x: x[0])
    for c in range(1, 9):
        df[f"c2_code_{c}"] = codes.map(lambda s, c=c: int(c in s)).astype("Int64").where(df["mosaik_cust"] == 1)
    methods = coded.map(lambda x: x[1])
    report["c2_coding"] = {
        "answers": int(df["c2_reason"].notna().sum()),
        "method": {k: int(v) for k, v in methods[df["c2_reason"].notna()].value_counts().items()},
        "multi_coded": int((codes.map(len) > 1).sum()),
        "code_counts": {f"c2_code_{c}": int(df[f"c2_code_{c}"].sum()) for c in range(1, 9)},
    }
    df.attrs["c2_method"] = methods
    return df


# =============================================================================
# Checks (any failure stops the run)
# =============================================================================

def check(df: pd.DataFrame, codebook: dict, report: dict) -> None:
    problems = []
    cust = df["mosaik_cust"] == 1

    def expect(name: str, bad: pd.Series):
        n = int(bad.fillna(False).sum())
        report["checks"][name] = n
        if n:
            problems.append(f"{name}: {n} rows")

    report["checks"] = {}
    expect("C/D answered by non-customer", ~cust & df[[c for c in C_SECTION if c != "c2_reason"]].notna().any(axis=1))
    expect("customer without NPS", cust & df["c1_nps"].isna())
    expect("B2 answered while unaware", pd.concat([df[f"b2_r{i}"].notna() & (df[f"b1_r{i}"] == 0)
                                                    for i in range(1, 6)], axis=1).any(axis=1))
    expect("D3 answered without contact", df["d3_resolved"].notna() & (df["d2_contact"] != 1))
    expect("C3r8 present before W3", df["c3_r8"].notna() & (df["wave"] < 3))
    expect("main bank not ticked at A1", pd.concat([(df["main_bank"] == b) & (df[f"a1_r{b}"] != 1)
                                                     for b in range(1, 7)], axis=1).any(axis=1))
    expect("age outside 18-69", ~df["age"].between(18, 69))
    expect("duplicate resp_id", df["resp_id"].duplicated())
    expect("removed respondent still present", df.get("qc_remove", pd.Series(0, index=df.index)) == 1)

    # every coded variable holds only codebook codes (no special codes left)
    sets = codebook["value_sets"]
    bad_codes = {}
    for v in codebook["variables"]:
        if v["name"] in df and v.get("values") and v["type"] == "int":
            allowed = {int(k) for k in sets[v["values"]]} - (set() if v["name"].endswith("_miss_reason")
                                                             else {int(k) for k in codebook["missing_codes"]})
            vals = df[v["name"]].dropna()
            if v["name"] == "c1_nps":
                allowed = set(range(11))
            n = int((~vals.isin(allowed)).sum())
            if n:
                bad_codes[v["name"]] = n
    report["checks"]["values outside codebook"] = bad_codes
    if bad_codes:
        problems.append(f"values outside codebook: {bad_codes}")
    if problems:
        raise CleaningError("clean file failed checks:\n  " + "\n  ".join(problems))


# =============================================================================
# Output
# =============================================================================

def final_columns(df: pd.DataFrame, codebook: dict) -> list[str]:
    """Codebook order; drops raw-only variables (keep: false) and the weight (made in R)."""
    cols = []
    for v in codebook["variables"]:
        name = v["name"]
        if v.get("keep") is False or name in ("weight", "qc_remove", "qc_attention", "qc_duplicate",
                                               "qc_inconsistent"):
            continue
        if name in df:
            cols.append(name)
    cols = ["wave", "resp_id"] + [c for c in cols if c not in ("wave", "resp_id")]
    extra = [c for c in df.columns if c not in cols and c not in
             ("status", "s4_account", "s5_r1", "s5_r2", "s5_r3", "s5_r4", "source_file",
              "qc_remove", "qc_attention", "qc_duplicate", "qc_inconsistent", "qc_soft_count", "qc_reasons")]
    if extra:
        raise CleaningError(f"columns not defined in the codebook: {extra}")
    return cols


def type_columns(df: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    types = {v["name"]: v["type"] for v in codebook["variables"]}
    for c in df.columns:
        t = types.get(c)
        if t == "int" or c == "wave":
            df[c] = pd.to_numeric(df[c]).astype("Int64")
        elif t in ("string", "text"):
            df[c] = df[c].astype("string")
        elif t == "float":
            df[c] = df[c].astype("float64")
    return df


def write_sav(df: pd.DataFrame, path: Path, codebook: dict) -> bool:
    try:
        import pyreadstat
    except ImportError:
        log.warning("pyreadstat not installed: skipping %s (pip install pyreadstat)", path.name)
        return False
    sets = codebook["value_sets"]
    var_labels, value_labels = {}, {}
    for v in codebook["variables"]:
        if v["name"] in df:
            var_labels[v["name"]] = v["label"][:255]
            if v.get("values") and v["type"] == "int":
                special = {int(k) for k in codebook["missing_codes"]}
                labels = {int(k): str(l) for k, l in sets[v["values"]].items()
                          if v["name"].endswith("_miss_reason") or int(k) not in special}
                value_labels[v["name"]] = labels
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) == "Int64":
            out[c] = out[c].astype("float64")
        elif str(out[c].dtype).startswith("string"):
            out[c] = out[c].astype(object).where(out[c].notna(), "")
    pyreadstat.write_sav(out, str(path), column_labels=var_labels, variable_value_labels=value_labels)
    return True


def write_outputs(df: pd.DataFrame, audit: pd.DataFrame, clean_dir: Path, codebook: dict,
                  report: dict) -> list[Path]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        df.to_parquet(clean_dir / "tracker_clean.parquet", index=False)
        written.append(clean_dir / "tracker_clean.parquet")
    except ImportError:
        log.warning("No parquet engine (pip install pyarrow): skipping tracker_clean.parquet")
    df.to_csv(clean_dir / "tracker_clean.csv", index=False, encoding="utf-8", na_rep="",
              date_format="%Y-%m-%d %H:%M:%S")
    written.append(clean_dir / "tracker_clean.csv")
    if write_sav(df, clean_dir / "tracker_clean.sav", codebook):
        written.append(clean_dir / "tracker_clean.sav")
    audit.to_csv(clean_dir / "removed_respondents.csv", index=False)
    written.append(clean_dir / "removed_respondents.csv")
    report["files"] = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in written]
    return written


# =============================================================================
# Main
# =============================================================================

def clean(flagged: pd.DataFrame, codebook: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report: dict = {}
    df, audit = drop_removed(flagged, report)
    df = special_codes_to_na(df, codebook, report)
    df = derive(df, codebook, report)
    methods = df.attrs.pop("c2_method")
    check(df, codebook, report)
    review = df.loc[df["c2_reason"].notna(), ["resp_id", "nps_group", "c2_reason"]].copy()
    review["codes"] = [",".join(str(c) for c in range(1, 9) if df.at[i, f"c2_code_{c}"] == 1)
                       for i in review.index]
    review["method"] = methods[review.index]
    df = type_columns(df[final_columns(df, codebook)].reset_index(drop=True), codebook)
    report["rows_by_wave"] = {f"W{w}": int(n) for w, n in df["wave"].value_counts().sort_index().items()}
    report["customers_by_wave"] = {f"W{w}": int(n) for w, n in
                                   df[df["mosaik_cust"] == 1]["wave"].value_counts().sort_index().items()}
    report["columns"] = len(df.columns)
    return df, audit, report | {"_review": review}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drop removed respondents, recode and derive variables.")
    ap.add_argument("--interim-dir", type=Path, default=INTERIM)
    ap.add_argument("--clean-dir", type=Path, default=CLEAN)
    ap.add_argument("--outputs-dir", type=Path, default=OUTPUTS)
    ap.add_argument("--codebook", type=Path, default=CONFIG / "codebook.yaml")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")
    codebook = load_codebook(args.codebook)
    try:
        df, audit, report = clean(read_frame(args.interim_dir / "flagged"), codebook)
    except CleaningError as exc:
        log.error("%s", exc)
        return 1

    review = report.pop("_review")
    write_outputs(df, audit, args.clean_dir, codebook, report)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.outputs_dir / "c2_coding_review.csv", index=False)
    with open(args.outputs_dir / "cleaning_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    log.info("Kept %d of %d (removed %s)", report["rows_out"], report["rows_in"], report["removed_by_wave"])
    log.info("Rows by wave %s | Mosaik customers %s", report["rows_by_wave"], report["customers_by_wave"])
    log.info("Special codes -> NA: %s", {k: sum(v.values()) for k, v in report["special_codes_to_na"].items()})
    log.info("gender_w imputed for %d | C2 coded %d answers %s, multi-coded %d",
             report["gender_w_imputed"], report["c2_coding"]["answers"], report["c2_coding"]["method"],
             report["c2_coding"]["multi_coded"])
    log.info("All checks passed. Files: %s", ", ".join(report["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
