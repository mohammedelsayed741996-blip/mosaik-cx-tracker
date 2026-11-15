"""Tests for step 03: clean file, derived variables and open-end coding.

Run: pytest tests/test_clean.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sim = load("sim", "00_simulate.py")
ing = load("ing", "01_ingest_harmonise.py")
qc = load("qc", "02_quality_checks.py")
cl = load("cl", "03_clean_recode.py")
CODEBOOK = yaml.safe_load(open(ROOT / "config" / "codebook.yaml", encoding="utf-8"))


@pytest.mark.parametrize("text,code", [
    ("The new card fee is a joke", 1), ("4,90 € im Monat für die Karte, ernsthaft?", 1),
    ("App ist super einfach", 2), ("waited ages for support", 3), ("I feel my money is safe there", 4),
    ("Überweisungen sind sofort da", 5), ("card got blocked abroad twice", 6),
    ("Gute Zinsen aufs Tagesgeld", 7), ("Alles gut", 8), ("trnasfers take too long", 5),
])
def test_code_frame(text, code):
    codes, _ = cl.code_open_end(text)
    assert codes == {code}


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    sim.simulate(seed=5, n_per_wave=600, out_dir=data)
    stacked, _ = ing.ingest(data / "raw", data / "interim", ROOT / "config")
    flagged, _ = qc.run_checks(stacked, qc.load_rules(ROOT / "config" / "consistency_rules.yaml"))
    df, audit, report = cl.clean(flagged, CODEBOOK)
    truth = pd.read_csv(data / "_truth" / "respondents_truth.csv")
    answers = pd.read_csv(data / "_truth" / "true_answers.csv", low_memory=False)
    return df, audit, report, flagged, truth, answers


def test_removed_rows_gone(cleaned):
    df, audit, _, flagged, _, _ = cleaned
    assert len(df) + len(audit) == len(flagged)
    assert not set(df["resp_id"]) & set(audit["resp_id"])


def test_no_special_codes_left(cleaned):
    df = cleaned[0]
    reasons = [c for c in df.columns if c.endswith("_miss_reason")]
    values = df.drop(columns=reasons).select_dtypes("number")
    assert not values.isin([-99, -98, -97]).any().any()


def test_miss_reasons_match_raw_codes(cleaned):
    df, _, _, _, _, answers = cleaned
    m = df.merge(answers[["wave", "record", "f1_income", "c3_r7"]], on=["wave", "record"], suffixes=("", "_raw"))
    assert ((m["f1_income_raw"] == -99) == (m["f1_income_miss_reason"] == -99)).all()
    assert ((m["c3_r7_raw"] == -97) == (m["c3_r7_miss_reason"] == -97)).all()


def test_nps_derivations(cleaned):
    df = cleaned[0]
    cust = df[df["mosaik_cust"] == 1]
    expected = 100 * ((cust["c1_nps"] >= 9).mean() - (cust["c1_nps"] <= 6).mean())
    assert cust["nps_score"].mean() == pytest.approx(expected)
    assert df.loc[df["mosaik_cust"] == 0, ["nps_group", "nps_score"]].isna().all().all()


def test_gender_w_is_binary(cleaned):
    assert set(cleaned[0]["gender_w"].dropna().unique()) == {1, 2}


def test_open_end_coding_accuracy(cleaned):
    df, _, _, _, truth, _ = cleaned
    cu = df[df["mosaik_cust"] == 1].merge(truth[["wave", "record", "c2_true_code"]], on=["wave", "record"])
    true_code = cu["c2_true_code"].fillna(8).astype(int)
    hit = [cu.iloc[i][f"c2_code_{c}"] == 1 for i, c in enumerate(true_code)]
    assert sum(hit) / len(hit) >= 0.98


def test_every_column_in_codebook(cleaned):
    names = {v["name"] for v in CODEBOOK["variables"]}
    assert set(cleaned[0].columns) <= names
