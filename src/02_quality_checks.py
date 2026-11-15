#!/usr/bin/env python3
"""
02_quality_checks.py - Mosaik Bank CX Tracker 2026: flag bad respondents.

Six checks, each a 0/1 column, computed from the survey data alone:

  qc_speeder        interview shorter than 40% of the median for the same
                    wave x device x questionnaire path (customers see ~2x more
                    questions, so they get their own median)            soft
  qc_straightliner  identical answers across a whole grid (C3 or E1) with
                    6+ valid answers ("don't know" excluded)             soft
  qc_gibberish      open end C2 that is not an answer: too few letters,
                    one token repeated, no vowels, a repeated pattern, or a
                    keyboard run such as "asdf" / "qwertz"               soft
  qc_attention      failed the in-grid check (C3x must be 4)             hard
  qc_duplicate      same panel ID twice in a wave; later entry flagged   hard
  qc_inconsistent   breaks a rule in config/consistency_rules.yaml       hard

Removal rule: any hard flag, or 2+ soft flags. A single soft flag is kept and
logged: fast readers and honest "all 4s" respondents exist.

Then, and only because this data is simulated, the flags are scored against
data/_truth/ (recall, precision, F1). The truth is read AFTER every flag is
final and never changes a flag. Turn it off with --no-evaluate.

Usage
-----
    python src/02_quality_checks.py
    python src/02_quality_checks.py --no-evaluate
    python src/02_quality_checks.py --fail-under 0.95    # exit 1 if removal recall/precision < 0.95

Outputs
-------
    data/interim/flagged.parquet    stacked file + qc_* columns + qc_reasons
                                    (flagged.csv.gz if no parquet engine)
    outputs/quality_report.md       the report a project lead reads
    outputs/quality_metrics.json    the same numbers, machine-readable
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("quality")

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TRUTH = ROOT / "data" / "_truth"
OUTPUTS = ROOT / "outputs"
CONFIG = ROOT / "config"

# --- thresholds (the decisions a reviewer will ask about) ----------------------
SPEED_FRACTION = 0.40          # flag if duration < 40% of the comparable median
STRAIGHTLINE_MIN_ITEMS = 6     # a grid needs 6+ valid answers to judge
ATTENTION_CORRECT = 4          # C3x instruction: "please select 'satisfied' (4)"
MIN_LETTERS = 3                # open ends with fewer letters are not answers
KEYBOARD_MIN_RUN = 4           # "asdf" (4) is flagged, "wer" (a German word) is not
SOFT_FLAGS = ["qc_speeder", "qc_straightliner", "qc_gibberish"]
HARD_FLAGS = ["qc_attention", "qc_duplicate", "qc_inconsistent"]
ALL_FLAGS = SOFT_FLAGS + HARD_FLAGS
SOFT_TO_REMOVE = 2

C3_ITEMS = [f"c3_r{i}" for i in range(1, 9)]
E1_ITEMS = [f"e1_r{i}" for i in range(1, 7)]
GRIDS = {"C3": C3_ITEMS, "E1": E1_ITEMS}
VALID_SCALE = (1, 5)

KEYBOARD_ROWS = ["1234567890", "qwertzuiop", "qwertyuiop", "asdfghjkl", "yxcvbnm", "zxcvbnm"]
VOWELS = set("aeiouyäöü")

TRUTH_MAP = {
    "qc_speeder": "true_speeder",
    "qc_straightliner": "true_straightliner",
    "qc_attention": "true_attention",
    "qc_duplicate": "true_duplicate",
    "qc_gibberish": "true_gibberish",
    "qc_inconsistent": "true_inconsistent",
    "qc_remove": "should_remove",
}
LABELS = {
    "qc_speeder": "Speeder", "qc_straightliner": "Straightliner", "qc_gibberish": "Gibberish open end",
    "qc_attention": "Attention check failed", "qc_duplicate": "Duplicate panel ID",
    "qc_inconsistent": "Inconsistent answers", "qc_remove": "Removed",
}


# =============================================================================
# I/O
# =============================================================================

def read_frame(stem: Path) -> pd.DataFrame:
    """Read <stem>.parquet, or <stem>.csv.gz written when no parquet engine exists."""
    pq, gz = stem.with_suffix(".parquet"), stem.with_suffix(".csv.gz")
    if pq.exists():
        return pd.read_parquet(pq)
    if gz.exists():
        df = pd.read_csv(gz, low_memory=False, parse_dates=["start_ts", "end_ts"])
        for c in ("resp_id", "panel_id", "source_file", "c2_reason"):
            if c in df:
                df[c] = df[c].astype("string")
        return df
    raise FileNotFoundError(f"neither {pq.name} nor {gz.name} found in {stem.parent}; run step 01 first")


def write_frame(df: pd.DataFrame, stem: Path) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
        return stem.with_suffix(".parquet")
    except ImportError:
        path = stem.with_suffix(".csv.gz")
        log.warning("No parquet engine (pip install pyarrow): writing %s instead", path.name)
        df.to_csv(path, index=False, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
        return path


# =============================================================================
# Checks
# =============================================================================

def flag_speeders(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Duration below SPEED_FRACTION of the median for the same wave x device x path."""
    path = np.where(df["c1_nps"].notna(), "customer", "non-customer")
    keys = [df["wave"], df["device"], pd.Series(path, index=df.index, name="path")]
    median = df.groupby(keys)["duration_s"].transform("median")
    flag = df["duration_s"] < SPEED_FRACTION * median
    table = (df.assign(path=path).groupby(["wave", "device", "path"])["duration_s"]
               .agg(n="size", median="median").reset_index())
    table["threshold_s"] = (SPEED_FRACTION * table["median"]).round(0)
    table["flagged"] = df.assign(path=path, f=flag).groupby(["wave", "device", "path"])["f"].sum().to_numpy()
    return flag, table


def flag_straightliners(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """True if any grid has STRAIGHTLINE_MIN_ITEMS+ valid answers, all identical."""
    per_grid = {}
    for name, items in GRIDS.items():
        cols = [c for c in items if c in df]
        grid = df[cols].where(df[cols].ge(VALID_SCALE[0]) & df[cols].le(VALID_SCALE[1]))
        answered = grid.notna().sum(axis=1)
        per_grid[name] = (grid.nunique(axis=1) == 1) & (answered >= STRAIGHTLINE_MIN_ITEMS)
    grids = pd.DataFrame(per_grid)
    return grids.any(axis=1), grids


def flag_attention(df: pd.DataFrame) -> pd.Series:
    shown = df["c3x_attn"].notna()
    return shown & (df["c3x_attn"] != ATTENTION_CORRECT)


def flag_duplicates(df: pd.DataFrame) -> pd.Series:
    """Same panel ID twice within a wave: keep the earliest interview, flag the rest."""
    order = df.sort_values(["wave", "start_ts", "record"]).index
    dup = df.loc[order].duplicated(["wave", "panel_id"], keep="first")
    return dup.reindex(df.index)


def is_keyboard_run(letters: str) -> bool:
    """Whole answer is a run along one keyboard row, forwards or backwards (asdf, lkjh)."""
    if len(letters) < KEYBOARD_MIN_RUN:
        return False
    return any(letters in row or letters in row[::-1] for row in KEYBOARD_ROWS)


def gibberish_reason(text) -> str | None:
    """Why an open end is not an answer, or None if it looks like one."""
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    letters = re.sub(r"[^a-zäöüß]", "", t)
    compact = re.sub(r"\s+", "", t)
    tokens = re.findall(r"[a-zäöüß0-9]+", t)
    if len(letters) < MIN_LETTERS:
        return "too few letters"
    if len(tokens) >= 2 and len(set(tokens)) == 1:
        return "same word repeated"
    if re.fullmatch(r"(.)\1{3,}", compact) or re.fullmatch(r"(.{2,3})\1{2,}", compact):
        return "repeated pattern"
    if is_keyboard_run(compact):
        return "keyboard run"
    if not set(letters) & VOWELS:
        return "no vowels"
    return None


def flag_gibberish(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    reasons = df["c2_reason"].map(gibberish_reason)
    return reasons.notna(), reasons


def load_rules(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["rules"]


def flag_inconsistent(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.Series, pd.DataFrame]:
    hits = {}
    num = df.drop(columns=["c2_reason", "resp_id", "panel_id", "source_file"], errors="ignore")
    num = num.apply(lambda s: s.astype("float64") if str(s.dtype) == "Int64" else s)
    for rule in rules:
        try:
            res = num.eval(" ".join(rule["when"].split()), engine="python")
        except Exception as exc:                     # a broken rule must stop the run
            raise RuntimeError(f"consistency rule {rule['id']!r} failed: {exc}") from exc
        hits[rule["id"]] = pd.Series(res, index=df.index).fillna(False).astype(bool)
    table = pd.DataFrame(hits)
    return table.any(axis=1), table


def run_checks(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    detail = {}
    out["qc_speeder"], detail["speed_table"] = flag_speeders(out)
    out["qc_straightliner"], detail["grids"] = flag_straightliners(out)
    out["qc_attention"] = flag_attention(out)
    out["qc_duplicate"] = flag_duplicates(out)
    out["qc_gibberish"], detail["gibberish_reasons"] = flag_gibberish(out)
    out["qc_inconsistent"], detail["rules"] = flag_inconsistent(out, rules)

    soft = out[SOFT_FLAGS].sum(axis=1)
    out["qc_soft_count"] = soft
    out["qc_remove"] = out[HARD_FLAGS].any(axis=1) | (soft >= SOFT_TO_REMOVE)
    for f in ALL_FLAGS + ["qc_remove"]:
        out[f] = out[f].fillna(False).astype(int)

    # human-readable reasons, e.g. "speeder; straightliner(E1)"
    reasons = []
    for idx in out.index:
        r = []
        if out.at[idx, "qc_speeder"]:
            r.append("speeder")
        if out.at[idx, "qc_straightliner"]:
            g = [k for k in GRIDS if detail["grids"].at[idx, k]]
            r.append(f"straightliner({','.join(g)})")
        if out.at[idx, "qc_gibberish"]:
            r.append(f"gibberish({detail['gibberish_reasons'].at[idx]})")
        if out.at[idx, "qc_attention"]:
            r.append("attention")
        if out.at[idx, "qc_duplicate"]:
            r.append("duplicate")
        if out.at[idx, "qc_inconsistent"]:
            rr = [k for k in detail["rules"].columns if detail["rules"].at[idx, k]]
            r.append(f"inconsistent({','.join(rr)})")
        reasons.append("; ".join(r))
    out["qc_reasons"] = pd.array(reasons, dtype="string")
    return out, detail


# =============================================================================
# Evaluation against the simulated truth (never feeds back into the flags)
# =============================================================================

def score(pred: pd.Series, true: pd.Series) -> dict:
    pred, true = pred.astype(bool), true.astype(bool)
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else float("nan")
    return {"planted": tp + fn, "flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4)}


def evaluate(flagged: pd.DataFrame, truth_dir: Path) -> dict | None:
    path = truth_dir / "respondents_truth.csv"
    if not path.exists():
        log.warning("No truth file at %s: skipping evaluation", path)
        return None
    truth = pd.read_csv(path, low_memory=False)
    cols = ["wave", "record"] + list(TRUTH_MAP.values()) + ["bad_profile", "inconsistency_type"]
    merged = flagged[["wave", "record", "resp_id", "qc_reasons", "c2_reason", "duration_s"] +
                     list(TRUTH_MAP)].merge(truth[cols], on=["wave", "record"], how="left",
                                            validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise RuntimeError(f"{int((merged['_merge'] != 'both').sum())} rows not found in the truth file")
    overall = {flag: score(merged[flag], merged[t]) for flag, t in TRUTH_MAP.items()}
    by_wave = {int(w): score(g["qc_remove"], g["should_remove"]) for w, g in merged.groupby("wave")}
    errors = {}
    for flag, t in TRUTH_MAP.items():
        fn = merged[(merged[flag] == 0) & (merged[t] == 1)]
        fp = merged[(merged[flag] == 1) & (merged[t] == 0)]
        keep = ["resp_id", "qc_reasons", "bad_profile", "c2_reason"]
        errors[flag] = {"missed": fn[keep].head(10).to_dict("records"),
                        "false_alarms": fp[keep].head(10).to_dict("records")}
    # Outcome by planted profile. "should_remove" applies the same documented rule,
    # so the scores above test the implementation; this table tests the rule itself.
    prof = merged.assign(profile=merged["bad_profile"].fillna("").replace("", "genuine"),
                         customer=flagged["c1_nps"].notna().to_numpy())
    profiles = (prof.groupby(["profile", "customer"])
                    .agg(n=("qc_remove", "size"), removed=("qc_remove", "sum"))
                    .reset_index())
    profiles["kept"] = profiles["n"] - profiles["removed"]
    return {"overall": overall, "removal_by_wave": by_wave, "errors": errors,
            "profiles": profiles.to_dict("records")}


# =============================================================================
# Report
# =============================================================================

def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def pct(n, d) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def build_report(flagged: pd.DataFrame, detail: dict, rules: list[dict], ev: dict | None) -> str:
    n = len(flagged)
    removed = int(flagged["qc_remove"].sum())
    L = []
    L.append("# Data quality report: Mosaik Bank CX Tracker 2026")
    L.append(f"\n_Generated {date.today().isoformat()} by `src/02_quality_checks.py`. "
             "All data is simulated._\n")
    L.append(f"**{removed:,} of {n:,} completes ({pct(removed, n)}) are removed**, leaving "
             f"{n - removed:,} for analysis. A respondent is removed for any hard flag "
             f"(attention check, duplicate, inconsistent answers) or for {SOFT_TO_REMOVE}+ soft flags "
             "(speeder, straightliner, gibberish). One soft flag alone is kept and logged.\n")

    # --- by wave -------------------------------------------------------------------
    L.append("## Removals by wave\n")
    rows = []
    for w, g in flagged.groupby("wave"):
        cust = g["c1_nps"].notna()
        rows.append({"Wave": f"W{w}", "Completes": len(g), "Removed": int(g["qc_remove"].sum()),
                     "Removed %": pct(g["qc_remove"].sum(), len(g)), "Kept": int(len(g) - g["qc_remove"].sum()),
                     "Mosaik customers kept": int((cust & (g["qc_remove"] == 0)).sum())})
    L.append(md_table(pd.DataFrame(rows)))

    # --- flags by wave ---------------------------------------------------------------
    L.append("\n## Flags by wave\n")
    L.append("A respondent can carry several flags, so the columns do not add up to the removals.\n")
    t = flagged.groupby("wave")[ALL_FLAGS + ["qc_remove"]].sum().T
    t.columns = [f"W{c}" for c in t.columns]
    t["Total"] = t.sum(axis=1)
    t["Total %"] = [pct(v, n) for v in t["Total"]]
    t.insert(0, "Flag", [LABELS[i] + (" (soft)" if i in SOFT_FLAGS else " (hard)" if i in HARD_FLAGS else "")
                         for i in t.index])
    L.append(md_table(t.reset_index(drop=True)))

    # --- soft-flag combinations ------------------------------------------------------
    L.append("\n## How soft flags combine\n")
    soft_rows = flagged[flagged["qc_soft_count"] > 0]
    combo = soft_rows[SOFT_FLAGS].apply(lambda r: " + ".join(LABELS[c] for c in SOFT_FLAGS if r[c]), axis=1)
    hard_any = soft_rows[HARD_FLAGS].any(axis=1)
    ct = pd.DataFrame({"combo": combo, "hard": hard_any})
    ct = ct.groupby("combo").agg(Respondents=("hard", "size"), **{"Also a hard flag": ("hard", "sum")})
    ct["Decision"] = ["removed" if " + " in c else "kept unless a hard flag" for c in ct.index]
    L.append(md_table(ct.reset_index().rename(columns={"combo": "Soft flags"})))
    single = flagged[(flagged["qc_soft_count"] == 1) & (flagged["qc_remove"] == 0)]
    L.append(f"\n{len(single):,} respondents with exactly one soft flag are **kept**. "
             "They are worth a manual look in a real study.\n")

    # --- method details ---------------------------------------------------------------
    L.append("## Check definitions\n")
    L.append(md_table(pd.DataFrame([
        {"Check": "Speeder", "Rule": f"duration < {SPEED_FRACTION:.0%} of the median for the same wave x device x path (customer / non-customer)", "Type": "soft"},
        {"Check": "Straightliner", "Rule": f"all valid answers identical in C3 or E1, with {STRAIGHTLINE_MIN_ITEMS}+ answered (don't know excluded)", "Type": "soft"},
        {"Check": "Gibberish", "Rule": f"C2 has < {MIN_LETTERS} letters, one word repeated, a repeated pattern, a keyboard run of {KEYBOARD_MIN_RUN}+ keys, or no vowels", "Type": "soft"},
        {"Check": "Attention", "Rule": f"C3x answered and not {ATTENTION_CORRECT}", "Type": "hard"},
        {"Check": "Duplicate", "Rule": "same panel ID twice in a wave; the later interview is flagged", "Type": "hard"},
        {"Check": "Inconsistent", "Rule": "breaks any rule below", "Type": "hard"},
    ])))

    L.append("\n### Speed thresholds\n")
    st = detail["speed_table"].copy()
    st["device"] = st["device"].map({1: "Desktop", 2: "Mobile", 3: "Tablet"})
    st["wave"] = "W" + st["wave"].astype(str)
    st["median"] = st["median"].round(0).astype(int)
    st["threshold_s"] = st["threshold_s"].astype(int)
    L.append(md_table(st.rename(columns={"wave": "Wave", "device": "Device", "path": "Path", "n": "n",
                                         "median": "Median (s)", "threshold_s": "Threshold (s)",
                                         "flagged": "Flagged"})))

    L.append("\n### Consistency rules\n")
    r_counts = detail["rules"].sum()
    L.append(md_table(pd.DataFrame([{"Rule": r["id"], "Description": r["description"],
                                     "Respondents": int(r_counts[r["id"]])} for r in rules])))

    L.append("\n### Gibberish open ends\n")
    gr = detail["gibberish_reasons"].dropna()
    ex = (pd.DataFrame({"Reason": gr, "Text": flagged.loc[gr.index, "c2_reason"]})
          .groupby("Reason").agg(Respondents=("Text", "size"),
                                 Examples=("Text", lambda s: ", ".join(f"`{x}`" for x in s.unique()[:5]))))
    L.append(md_table(ex.reset_index()))

    # --- validation ---------------------------------------------------------------------
    if ev:
        L.append("\n## Validation against the simulated truth\n")
        L.append("_Only possible because the data is simulated: `data/_truth/` records which respondents "
                 "were planted as bad. It was read after all flags were final and changed none of them._\n")
        rows = []
        for flag, s in ev["overall"].items():
            rows.append({"Flag": LABELS[flag], "Planted": s["planted"], "Flagged": s["flagged"],
                         "Caught": s["tp"], "Missed": s["fn"], "False alarms": s["fp"],
                         "Recall": f"{s['recall']:.3f}", "Precision": f"{s['precision']:.3f}",
                         "F1": f"{s['f1']:.3f}"})
        L.append(md_table(pd.DataFrame(rows)))
        L.append("\n**Removal decision by wave**\n")
        L.append(md_table(pd.DataFrame([{"Wave": f"W{w}", "Should remove": s["planted"], "Removed": s["flagged"],
                                         "Recall": f"{s['recall']:.3f}", "Precision": f"{s['precision']:.3f}"}
                                        for w, s in ev["removal_by_wave"].items()])))
        L.append("\n### Does the rule remove the right people?\n")
        L.append("The scores above compare the flags with a truth that applies the same removal rule, "
                 "so they show the rule is implemented correctly, not that it is a good rule. "
                 "This table shows what happened to each type of planted respondent.\n")
        pt = pd.DataFrame(ev["profiles"])
        pt["customer"] = pt["customer"].map({True: "customer", False: "non-customer"})
        pt["Removed %"] = [pct(r, n_) for r, n_ in zip(pt["removed"], pt["n"])]
        L.append(md_table(pt.rename(columns={"profile": "Planted profile", "customer": "Path", "n": "n",
                                             "removed": "Removed", "kept": "Kept"})))
        rnd = pt[(pt["profile"] == "speeder_random") & (pt["kept"] > 0)]
        if len(rnd):
            k = int(rnd["kept"].sum())
            L.append(f"\n**Known gap:** {k} random responders who rushed through the survey are kept. "
                     "They are non-customers, so they never saw the open end or the attention check, and "
                     "speed is their only flag. Speed alone cannot separate them from fast honest "
                     "readers (both finish in 15-33% of the median). A next check could test answer "
                     "coherence, for example E1 answers that contradict each other.\n")
        misses = [(f, e) for f, e in ev["errors"].items() if e["missed"] or e["false_alarms"]]
        if misses:
            L.append("\n### Errors to inspect (up to 10 each)\n")
            for flag, e in misses:
                for kind in ("missed", "false_alarms"):
                    if e[kind]:
                        L.append(f"**{LABELS[flag]}: {kind.replace('_', ' ')}**\n")
                        L.append(md_table(pd.DataFrame(e[kind]).fillna("")))
                        L.append("")
        else:
            L.append("\nNo missed or false flags.\n")
    return "\n".join(L) + "\n"


# =============================================================================
# Main
# =============================================================================

def print_metrics(ev: dict) -> None:
    print("\nValidation against simulated truth")
    print(f"{'Flag':26s} {'Planted':>8s} {'Flagged':>8s} {'Recall':>8s} {'Precision':>10s} {'F1':>7s}")
    for flag, s in ev["overall"].items():
        print(f"{LABELS[flag]:26s} {s['planted']:8d} {s['flagged']:8d} {s['recall']:8.3f} "
              f"{s['precision']:10.3f} {s['f1']:7.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flag bad respondents and write the quality report.")
    ap.add_argument("--interim-dir", type=Path, default=INTERIM)
    ap.add_argument("--truth-dir", type=Path, default=TRUTH)
    ap.add_argument("--outputs-dir", type=Path, default=OUTPUTS)
    ap.add_argument("--rules", type=Path, default=CONFIG / "consistency_rules.yaml")
    ap.add_argument("--no-evaluate", action="store_true", help="skip scoring against data/_truth/")
    ap.add_argument("--fail-under", type=float, default=0.90,
                    help="exit 1 if removal recall or precision is below this (when evaluating)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")

    stacked = read_frame(args.interim_dir / "stacked")
    rules = load_rules(args.rules)
    flagged, detail = run_checks(stacked, rules)            # flags are final here
    path = write_frame(flagged, args.interim_dir / "flagged")
    log.info("Flagged %d of %d completes for removal -> %s",
             int(flagged["qc_remove"].sum()), len(flagged), path.name)
    for w, g in flagged.groupby("wave"):
        log.info("  W%d: %s | removed %d", w,
                 ", ".join(f"{f[3:]} {int(g[f].sum())}" for f in ALL_FLAGS), int(g["qc_remove"].sum()))

    ev = None if args.no_evaluate else evaluate(flagged, args.truth_dir)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(flagged, detail, rules, ev)
    (args.outputs_dir / "quality_report.md").write_text(report, encoding="utf-8")
    metrics = {
        "completes": int(len(flagged)), "removed": int(flagged["qc_remove"].sum()),
        "flags_by_wave": {f"W{w}": {f: int(g[f].sum()) for f in ALL_FLAGS + ["qc_remove"]}
                          for w, g in flagged.groupby("wave")},
        "rules": {k: int(v) for k, v in detail["rules"].sum().items()},
        "evaluation": None if ev is None else {"overall": ev["overall"], "removal_by_wave": ev["removal_by_wave"],
                                               "profiles": ev["profiles"]},
    }
    with open(args.outputs_dir / "quality_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Report -> %s", args.outputs_dir / "quality_report.md")

    if ev:
        print_metrics(ev)
        rem = ev["overall"]["qc_remove"]
        if rem["recall"] < args.fail_under or rem["precision"] < args.fail_under:
            log.error("Removal recall %.3f / precision %.3f below %.2f", rem["recall"], rem["precision"],
                      args.fail_under)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
