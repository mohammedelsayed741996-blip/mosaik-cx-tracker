#!/usr/bin/env python3
"""
01_ingest_harmonise.py - Mosaik Bank CX Tracker 2026: read the 4 raw exports
and stack them into one harmonised file.

Every wave arrives in a slightly different shape (see config/wave_mappings.yaml).
This step makes them all look like the standard export, nothing more:

* reads each file with its own encoding / format (CSV or SPSS .sav)
* renames wave-specific columns and checks that every column is either mapped
  or explicitly dropped (an unknown or missing column stops the run)
* keeps completes only (status 1) and counts the rest
* parses dates (ISO, DD.MM.YYYY), durations (seconds or h:mm:ss), typed ages
  ("34 Jahre") and typed federal states ("bavaria ", "NRW")
* maps refusal aliases (99) to -99 and restores 0 for unticked multi-select
  items exported as blanks, using the routing so "not asked" stays blank
* tidies open ends (whitespace, euro signs lost to Latin-1, garbled UTF-8)
* checks codes against the codebook and SPSS value labels against the codebook
* adds wave, resp_id and source_file, then writes data/interim/stacked.parquet

Special codes (-99 refused, -97 don't know) are kept as codes here; turning them
into NA is a cleaning decision for step 03. Quality flags are step 02.

Usage
-----
    python src/01_ingest_harmonise.py
    python src/01_ingest_harmonise.py --raw-dir data/raw --out-dir data/interim

Outputs
-------
    data/interim/stacked.parquet        one row per complete, all 4 waves
                                        (stacked.csv.gz if no parquet engine)
    data/interim/ingest_report.json     counts of every row dropped and fix applied

Exit code 1 if a hard check fails (missing file, unknown or missing column,
duplicate record, unparseable values in a key column).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("ingest")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = ROOT / "config"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "data" / "interim"

ID_COLS = ["wave", "resp_id", "record", "panel_id", "source_file"]
TIME_COLS = ["start_ts", "end_ts"]
# Columns that must parse for every complete; a failure here is a hard error
KEY_COLS = ["record", "panel_id", "start_ts", "duration_s", "age", "gender", "state"]
DURATION_TOLERANCE_S = 60   # duration vs (end - start) disagreement to report


class IngestError(RuntimeError):
    """A problem that must stop the pipeline."""


# =============================================================================
# Config
# =============================================================================

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def codebook_index(codebook: dict) -> dict[str, dict]:
    return {v["name"]: v for v in codebook["variables"]}


def check_config(mappings: dict, codebook: dict) -> None:
    """standard_columns must agree with the codebook's raw names, both ways."""
    std = mappings["standard_columns"]
    cb_raw = {v["raw"]: v["name"] for v in codebook["variables"] if v.get("raw")}
    problems = []
    for raw, clean in std.items():
        if raw not in cb_raw:
            problems.append(f"{raw} is in wave_mappings but has no `raw:` entry in the codebook")
        elif cb_raw[raw] != clean:
            problems.append(f"{raw} maps to {clean} in wave_mappings but to {cb_raw[raw]} in the codebook")
    for raw, clean in cb_raw.items():
        if raw not in std:
            problems.append(f"codebook variable {clean} (raw {raw}) is missing from standard_columns")
    for clean in mappings.get("multi_select_routing", {}):
        if clean not in std.values():
            problems.append(f"multi_select_routing names unknown variable {clean}")
    if problems:
        raise IngestError("Config mismatch:\n  " + "\n  ".join(problems))


def expected_raw_columns(wave: int, mappings: dict, cb: dict[str, dict]) -> set[str]:
    """Standard raw columns that should exist in this wave's export."""
    cols = set()
    for raw, clean in mappings["standard_columns"].items():
        waves = cb[clean].get("waves")
        if waves is None or wave in waves:
            cols.add(raw)
    return cols


# =============================================================================
# Reading
# =============================================================================

def check_encoding(path: Path, encoding: str, report: dict) -> None:
    """Fail early and clearly if a CSV is not in the encoding the config says."""
    data = path.read_bytes()
    try:
        data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise IngestError(f"{path.name} is not valid {encoding} ({exc.reason} at byte {exc.start}); "
                          f"check `encoding` in wave_mappings.yaml") from exc
    if encoding.lower().replace("-", "") in ("latin1", "iso88591", "cp1252"):
        try:
            data.decode("utf-8")
            if not data.isascii():
                report["warnings"].append(f"{path.name} also decodes as UTF-8; is {encoding} right?")
        except UnicodeDecodeError:
            pass


def read_csv_as_text(path: Path, encoding: str) -> pd.DataFrame:
    # Everything as text: parsing is explicit and counted, never guessed by pandas
    return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False,
                       na_values=[""], skipinitialspace=False)


def read_wave(wave: int, cfg: dict, raw_dir: Path, report: dict) -> tuple[pd.DataFrame, dict | None, str]:
    """Return (frame, embedded value labels or None, source file name)."""
    fmt = cfg.get("format", "csv")
    path = raw_dir / cfg["file"]

    if fmt == "sav":
        if path.exists():
            try:
                import pyreadstat
            except ImportError as exc:
                raise IngestError(f"{path.name} needs pyreadstat: pip install pyreadstat") from exc
            df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False,
                                           user_missing=False, disable_datetime_conversion=False)
            labels = {k: {int(float(c)): str(l) for c, l in v.items()}
                      for k, v in meta.variable_value_labels.items()}
            return df, labels, path.name
        fb, fb_labels = cfg.get("fallback_file"), cfg.get("fallback_labels")
        if fb and (raw_dir / fb).exists():
            report["warnings"].append(f"{path.name} not found; read fallback {fb} instead")
            check_encoding(raw_dir / fb, cfg.get("encoding", "utf-8"), report)
            df = read_csv_as_text(raw_dir / fb, cfg.get("encoding", "utf-8"))
            labels = None
            if fb_labels and (raw_dir / fb_labels).exists():
                with open(raw_dir / fb_labels, encoding="utf-8") as fh:
                    raw_labels = json.load(fh)["value_labels"]
                labels = {k: {int(c): str(l) for c, l in v.items()} for k, v in raw_labels.items()}
            return df, labels, fb
        raise IngestError(f"W{wave}: neither {path.name} nor a fallback file found in {raw_dir}")

    if not path.exists():
        raise IngestError(f"W{wave}: {path} not found")
    check_encoding(path, cfg.get("encoding", "utf-8"), report)
    return read_csv_as_text(path, cfg.get("encoding", "utf-8")), None, path.name


# =============================================================================
# Column mapping
# =============================================================================

def map_columns(df: pd.DataFrame, wave: int, cfg: dict, mappings: dict,
                cb: dict[str, dict], report: dict) -> pd.DataFrame:
    rename = cfg.get("rename") or {}
    df = df.rename(columns=rename)
    if rename:
        report["fixes"]["columns_renamed"] = rename

    df.columns = [c.strip() for c in df.columns]
    drop = set(cfg.get("drop") or [])
    expected = expected_raw_columns(wave, mappings, cb)
    present = set(df.columns)

    unknown = sorted(present - expected - drop)
    missing = sorted(expected - present)
    if unknown or missing:
        raise IngestError(f"W{wave}: columns do not match the mapping. "
                          f"Unknown: {unknown or 'none'}; missing: {missing or 'none'}. "
                          f"Add renames/drops to wave_mappings.yaml.")
    if drop & present:
        report["fixes"]["columns_dropped"] = sorted(drop & present)
        df = df.drop(columns=sorted(drop & present))
    return df.rename(columns=mappings["standard_columns"])


# =============================================================================
# Parsers (each returns parsed values and counts what it did)
# =============================================================================

def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    return pd.to_numeric(s.astype("string").str.strip(), errors="coerce").astype("float64")


def parse_datetimes(s: pd.Series, formats: list[str]) -> tuple[pd.Series, int]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, 0
    text = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = out.isna() & text.notna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce")
    failed = int((out.isna() & text.notna()).sum())
    return out, failed


def parse_duration(s: pd.Series, fmt: str) -> tuple[pd.Series, int]:
    if fmt == "seconds":
        out = to_number(s)
    elif fmt == "hms":
        text = s.astype("string").str.strip()
        parts = text.str.extract(r"^(\d+):(\d{1,2}):(\d{1,2})$").astype("float64")
        out = parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise IngestError(f"unknown duration_format {fmt!r}")
    failed = int((out.isna() & s.notna()).sum())
    return out, failed


def parse_age_text(s: pd.Series) -> tuple[pd.Series, int, int]:
    """'34 Jahre' / '24 years' / ' 34' -> 34. Returns (ages, n_fixed, n_failed)."""
    numeric = to_number(s)
    text = s.astype("string")
    needs = numeric.isna() & text.notna()
    extracted = text[needs].str.extract(r"^\s*(\d{1,3})\s*(?:jahre?|years?|yrs?|j\.?)?\s*$",
                                        flags=re.IGNORECASE)[0]
    numeric[needs] = pd.to_numeric(extracted, errors="coerce")
    fixed = int(numeric[needs].notna().sum())
    failed = int(numeric[needs].isna().sum())
    return numeric, fixed, failed


def normalise_key(text: str) -> str:
    text = text.strip().lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text)


def parse_state_text(s: pd.Series, lookup: dict[str, int]) -> tuple[pd.Series, Counter, Counter]:
    """Codes stay codes; typed names are looked up. Returns (codes, fixed, unknown)."""
    numeric = to_number(s)
    text = s.astype("string")
    needs = numeric.isna() & text.notna()
    fixed, unknown = Counter(), Counter()
    for idx, val in text[needs].items():
        code = lookup.get(normalise_key(val))
        if code is None:
            unknown[val] += 1
        else:
            numeric.at[idx] = code
            fixed[val] += 1
    return numeric, fixed, unknown


EURO_LOST = re.compile(r"(\d)\s\?(?=\s)")            # "4,90 ? im Monat" -> "4,90 € im Monat"
MOJIBAKE_HINT = re.compile(r"[ÃÂâ][\u0080-¿‘-›€ŒœŠšŽžŸ]")


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 bytes that were decoded as Latin-1 / cp1252 ('MÃ¼nchen' -> 'München')."""
    if not MOJIBAKE_HINT.search(text):
        return text
    for enc in ("latin-1", "cp1252"):
        try:
            repaired = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not MOJIBAKE_HINT.search(repaired):
            return repaired
    return text


def clean_text(s: pd.Series, ops: list[str]) -> tuple[pd.Series, dict[str, int]]:
    out = s.astype("string")
    counts: dict[str, int] = {}
    for op in ops:
        before = out.copy()
        if op == "strip":
            out = out.str.strip()
            out = out.mask(out == "")
        elif op == "restore_euro":
            out = out.str.replace(EURO_LOST, r"\1 €", regex=True)
        elif op == "fix_mojibake":
            out = out.map(lambda t: fix_mojibake(t) if isinstance(t, str) else t).astype("string")
        else:
            raise IngestError(f"unknown text operation {op!r}")
        counts[op] = int(((before != out) & before.notna()).fillna(False).sum())
    return out, counts


# =============================================================================
# Harmonisation of one wave
# =============================================================================

def harmonise_wave(df: pd.DataFrame, wave: int, cfg: dict, mappings: dict,
                   cb: dict[str, dict], state_lookup: dict[str, int], report: dict) -> pd.DataFrame:
    fixes, warn = report["fixes"], report["warnings"]

    # --- dispositions: keep completes only -------------------------------------
    status = to_number(df["status"])
    report["rows_by_status"] = {int(k): int(v) for k, v in status.value_counts(dropna=False).items()
                                if pd.notna(k)}
    if status.isna().any():
        raise IngestError(f"W{wave}: {int(status.isna().sum())} rows without a status code")
    keep = status.isin(mappings.get("keep_status", [1]))
    df = df.loc[keep].copy()
    df["status"] = status[keep]
    report["rows_dropped_noncomplete"] = int((~keep).sum())

    # --- dates and duration --------------------------------------------------------
    for col in TIME_COLS:
        df[col], failed = parse_datetimes(df[col], cfg["date_formats"])
        if failed:
            report["parse_failures"][col] = failed
    df["duration_s"], failed = parse_duration(df["duration_s"], cfg.get("duration_format", "seconds"))
    if failed:
        report["parse_failures"]["duration_s"] = failed
    if cfg.get("duration_format") == "hms":
        fixes["duration_hms_to_seconds"] = int(df["duration_s"].notna().sum())
    gap = ((df["end_ts"] - df["start_ts"]).dt.total_seconds() - df["duration_s"]).abs()
    n_gap = int((gap > DURATION_TOLERANCE_S).sum())
    if n_gap:
        warn.append(f"{n_gap} rows where duration differs from end - start by more than "
                    f"{DURATION_TOLERANCE_S}s")
    if cfg["date_formats"] != ["%Y-%m-%d %H:%M:%S"]:
        fixes["dates_reformatted"] = int(df["start_ts"].notna().sum())

    # --- typed ages and states ------------------------------------------------------
    if cfg.get("age_free_text"):
        df["age"], fixed, failed = parse_age_text(df["age"])
        fixes["age_text_parsed"] = fixed
        if failed:
            report["parse_failures"]["age"] = failed
    if cfg.get("state_free_text"):
        df["state"], fixed, unknown = parse_state_text(df["state"], state_lookup)
        fixes["state_text_parsed"] = int(sum(fixed.values()))
        fixes["state_text_variants"] = dict(fixed.most_common())
        if unknown:
            report["parse_failures"]["state"] = int(sum(unknown.values()))
            warn.append(f"unrecognised state text: {dict(unknown)} (add to state_lookup)")

    # --- open ends ---------------------------------------------------------------------
    text_cols = {c for c, v in cb.items() if v["type"] == "text"}
    for raw_col, ops in (cfg.get("text_fields") or {}).items():
        col = mappings["standard_columns"][raw_col]
        df[col], counts = clean_text(df[col], ops)
        for op, n in counts.items():
            fixes[f"{col}_{op}"] = n
    for col in text_cols:
        df[col] = df[col].astype("string")
        lost = int(df[col].str.contains("�", regex=False).fillna(False).sum())
        if lost:
            warn.append(f"{lost} {col} texts contain U+FFFD (characters lost at export)")

    # --- everything else is numeric ------------------------------------------------
    numeric_cols = [c for c in df.columns
                    if c not in TIME_COLS and c not in text_cols and c != "panel_id"]
    for col in numeric_cols:
        before = df[col]
        df[col] = to_number(before)
        bad = int((df[col].isna() & before.notna()).sum())
        if bad:
            report["parse_failures"][col] = report["parse_failures"].get(col, 0) + bad
            examples = before[df[col].isna() & before.notna()].astype(str).unique()[:5].tolist()
            warn.append(f"{bad} unparseable values in {col}, e.g. {examples}")

    # --- refusal aliases -------------------------------------------------------------
    aliases = cfg.get("refused_aliases") or []
    if aliases:
        n = 0
        for raw_col in mappings.get("refusal_columns", []):
            col = mappings["standard_columns"][raw_col]
            hit = df[col].isin(aliases)
            n += int(hit.sum())
            df.loc[hit, col] = -99
        fixes["refusal_alias_to_-99"] = n

    # --- unticked multi-select items exported as blank -----------------------------
    if cfg.get("multi_blank_is_zero"):
        n = 0
        for col, cond in mappings["multi_select_routing"].items():
            asked = pd.Series(True, index=df.index) if cond == "all" \
                else df.eval(cond, engine="python").fillna(False).astype(bool)
            fill = asked & df[col].isna()
            n += int(fill.sum())
            df.loc[fill, col] = 0
        fixes["multi_blank_to_0"] = n

    # --- panel id ----------------------------------------------------------------------
    df["panel_id"] = df["panel_id"].astype("string").str.strip()
    return df


# =============================================================================
# Checks against the codebook
# =============================================================================

def allowed_codes(var: dict, value_sets: dict) -> set[int] | None:
    vs = var.get("values")
    if not vs:
        return None
    return {int(k) for k in value_sets[vs]}


def validate_codes(df: pd.DataFrame, codebook: dict, cb: dict[str, dict], report: dict) -> None:
    """Count values outside the codebook. Out-of-range answers are reported, not fixed."""
    invalid = {}
    for col in df.columns:
        var = cb.get(col)
        if not var or var["type"] != "int":
            continue
        codes = allowed_codes(var, codebook["value_sets"])
        vals = df[col].dropna()
        if col == "c1_nps":
            bad = ~vals.isin(range(0, 11))
        elif codes is not None:
            bad = ~vals.isin(codes)
        elif col == "age":
            bad = ~vals.between(18, 69)
        else:
            bad = vals != np.floor(vals)
        if bad.any():
            invalid[col] = {"n": int(bad.sum()),
                            "values": sorted(map(float, vals[bad].unique()))[:10]}
    if invalid:
        report["invalid_codes"] = invalid
        report["warnings"].append(f"values outside the codebook in: {sorted(invalid)}")


def check_value_labels(labels: dict, mappings: dict, codebook: dict, cb: dict[str, dict],
                       report: dict) -> None:
    """Embedded SPSS labels vs codebook labels. Codes win; differences are reported."""
    diffs = []
    for raw_col, raw_labels in labels.items():
        clean = mappings["standard_columns"].get(raw_col)
        var = cb.get(clean or "")
        if not var or not var.get("values"):
            continue
        expected = {int(k): str(v) for k, v in codebook["value_sets"][var["values"]].items()}
        for code, label in raw_labels.items():
            if code not in expected:
                diffs.append({"column": raw_col, "code": code, "export_label": label,
                              "codebook_label": None})
            elif label.strip() != expected[code]:
                diffs.append({"column": raw_col, "code": code, "export_label": label,
                              "codebook_label": expected[code]})
    report["label_differences"] = diffs
    for d in diffs:
        report["warnings"].append(f"label differs for {d['column']}={d['code']}: export "
                                  f"{d['export_label']!r} vs codebook {d['codebook_label']!r} "
                                  f"(codes are used, labels ignored)")


# =============================================================================
# Output
# =============================================================================

def finalise_types(df: pd.DataFrame, cb: dict[str, dict]) -> pd.DataFrame:
    for col in df.columns:
        var = cb.get(col)
        if col in ("wave",) or (var and var["type"] == "int"):
            vals = df[col]
            if vals.notna().any() and (vals.dropna() != np.floor(vals.dropna())).any():
                raise IngestError(f"{col} has non-integer values")
            df[col] = vals.astype("Int64")
    for col in ("resp_id", "panel_id", "source_file", "c2_reason"):
        df[col] = df[col].astype("string")
    return df


def column_order(df: pd.DataFrame, codebook: dict) -> list[str]:
    order = list(ID_COLS)
    for v in codebook["variables"]:
        if v["name"] in df.columns and v["name"] not in order:
            order.append(v["name"])
    return order + [c for c in df.columns if c not in order]


def write_output(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stacked.parquet"
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = out_dir / "stacked.csv.gz"
        log.warning("No parquet engine (pip install pyarrow): writing %s instead", fallback.name)
        df.to_csv(fallback, index=False, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
        return fallback


# =============================================================================
# Main
# =============================================================================

def new_report() -> dict[str, Any]:
    return {"source_file": None, "rows_read": 0, "rows_by_status": {}, "rows_dropped_noncomplete": 0,
            "rows_kept": 0, "fixes": {}, "parse_failures": {}, "warnings": []}


def ingest(raw_dir: Path, out_dir: Path, config_dir: Path) -> tuple[pd.DataFrame, dict]:
    codebook = load_yaml(config_dir / "codebook.yaml")
    mappings = load_yaml(config_dir / "wave_mappings.yaml")
    check_config(mappings, codebook)
    cb = codebook_index(codebook)
    state_lookup = {key: int(code) for code, keys in mappings["state_lookup"].items() for key in keys}

    frames, reports, errors = [], {}, []
    for wave in sorted(int(w) for w in mappings["waves"]):
        cfg = mappings["waves"][wave]
        report = new_report()
        reports[f"W{wave}"] = report
        try:
            raw, labels, source = read_wave(wave, cfg, raw_dir, report)
            report["source_file"] = source
            report["rows_read"] = int(len(raw))
            df = map_columns(raw, wave, cfg, mappings, cb, report)
            df = harmonise_wave(df, wave, cfg, mappings, cb, state_lookup, report)
            if cfg.get("check_value_labels"):
                if labels:
                    check_value_labels(labels, mappings, codebook, cb, report)
                else:
                    report["warnings"].append("no embedded value labels found to check")
            df["wave"] = wave
            df["source_file"] = source
            df["resp_id"] = [f"W{wave}_{int(r):06d}" for r in df["record"]]
            validate_codes(df, codebook, cb, report)
            report["rows_kept"] = int(len(df))
            frames.append(df)
        except IngestError as exc:
            errors.append(str(exc))
            report["error"] = str(exc)
            log.error("%s", exc)
            continue

        hard = {c: n for c, n in report["parse_failures"].items() if c in KEY_COLS}
        if hard:
            errors.append(f"W{wave}: unparseable values in key columns {hard}")
        log.info("W%d %-18s read %5d | non-completes dropped %4d | kept %5d | fixes %s",
                 wave, report["source_file"], report["rows_read"], report["rows_dropped_noncomplete"],
                 report["rows_kept"], {k: v for k, v in report["fixes"].items()
                                       if isinstance(v, int) and v})
        for w in report["warnings"]:
            log.warning("W%d: %s", wave, w)

    if not frames:
        raise IngestError("no wave could be read:\n  " + "\n  ".join(errors))

    stacked = pd.concat(frames, ignore_index=True)
    dup = stacked.duplicated(["wave", "record"])
    if dup.any():
        errors.append(f"{int(dup.sum())} duplicate (wave, record) keys")
    stacked = finalise_types(stacked, cb)
    stacked = stacked[column_order(stacked, codebook)]

    summary = {
        "waves": reports,
        "total_rows_kept": int(len(stacked)),
        "rows_kept_by_wave": {f"W{w}": int(n) for w, n in stacked["wave"].value_counts().sort_index().items()},
        "columns": list(stacked.columns),
        "errors": errors,
    }
    if errors:
        raise IngestError("ingest failed:\n  " + "\n  ".join(errors))
    return stacked, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest and harmonise the raw tracker exports.")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        stacked, summary = ingest(args.raw_dir, args.out_dir, args.config_dir)
    except IngestError as exc:
        log.error("%s", exc)
        return 1

    path = write_output(stacked, args.out_dir)
    with open(args.out_dir / "ingest_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Stacked %d completes from %d waves (%s) -> %s",
             summary["total_rows_kept"], len(summary["rows_kept_by_wave"]),
             ", ".join(f"{w}: {n}" for w, n in summary["rows_kept_by_wave"].items()), path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
