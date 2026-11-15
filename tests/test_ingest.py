"""Tests for step 01: the stacked file must reproduce the undamaged answers.

The simulator writes the true answers to data/_truth/; the ingest step never
reads them. These tests simulate a small dataset into a temp folder, run the
ingest on its raw exports and compare the result cell by cell.

Run: pytest tests/test_ingest.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sim = load("sim", "00_simulate.py")
ing = load("ing", "01_ingest_harmonise.py")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    sim.simulate(seed=42, n_per_wave=500, out_dir=data)
    stacked, report = ing.ingest(data / "raw", data / "interim", ROOT / "config")
    truth = pd.read_csv(data / "_truth" / "true_answers.csv", low_memory=False,
                        parse_dates=["start_ts", "end_ts"])
    key = ["wave", "record"]
    return (stacked.sort_values(key).reset_index(drop=True),
            truth.sort_values(key).reset_index(drop=True), report, data)


def test_row_counts(run):
    stacked, truth, report, _ = run
    assert len(stacked) == len(truth) == 2000
    assert report["waves"]["W1"]["rows_dropped_noncomplete"] == 350   # partials, screen-outs, over quota
    assert not stacked.duplicated(["wave", "record"]).any()


def test_numeric_answers_match_truth(run):
    stacked, truth, _, _ = run
    skip = {"wave", "record", "panel_id", "c2_reason", "start_ts", "end_ts"}
    for col in [c for c in truth.columns if c not in skip]:
        a = pd.to_numeric(truth[col])
        b = pd.to_numeric(stacked[col])
        same = (a == b) | (a.isna() & b.isna())
        assert same.all(), f"{col}: {int((~same).sum())} cells differ"


def test_dates_match_to_the_minute(run):
    stacked, truth, _, _ = run          # W2 exports dates without seconds
    for col in ("start_ts", "end_ts"):
        assert (stacked[col].dt.floor("min") == truth[col].dt.floor("min")).all()


def test_open_ends_repaired(run):
    stacked, truth, _, _ = run
    a = truth["c2_reason"].fillna("").str.strip()
    b = stacked["c2_reason"].fillna("")
    assert (a == b).all(), b[a != b].head().tolist()


def test_typed_states_and_ages_parsed(run):
    _, _, report, _ = run
    w3 = report["waves"]["W3"]
    assert w3["fixes"]["state_text_parsed"] > 0 and w3["fixes"]["age_text_parsed"] > 0
    assert "state" not in w3["parse_failures"] and "age" not in w3["parse_failures"]


def test_label_difference_reported(run):
    _, _, report, _ = run
    diffs = report["waves"]["W4"]["label_differences"]
    assert any(d["column"] == "F2" and d["code"] == 6 for d in diffs)


def test_unknown_column_stops_the_run(run, tmp_path):
    _, _, _, data = run
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in (data / "raw").iterdir():
        (raw / f.name).write_bytes(f.read_bytes())
    w3 = pd.read_csv(raw / "W3_export.csv", dtype=str)
    w3["Q99_new"] = "1"
    w3.to_csv(raw / "W3_export.csv", index=False)
    with pytest.raises(ing.IngestError, match="Q99_new"):
        ing.ingest(raw, tmp_path / "out", ROOT / "config")
