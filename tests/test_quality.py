"""Tests for step 02: quality flags.

Run: pytest tests/test_quality.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
qc = load("qc", "02_quality_checks.py")


@pytest.mark.parametrize("text", ["asdf", "qwertz", "lkjh", "good good good", "ok ok ok ok", "jjjjjjj",
                                  "fgfgfg", "nsdkjfh", "1234", "...", "-", "x"])
def test_gibberish_is_caught(text):
    assert qc.gibberish_reason(text) is not None


@pytest.mark.parametrize("text", ["Alles gut", "its ok", "App ist super einfach", "Zinsen sind zu niedrig",
                                  "4,90 € im Monat für die Karte, ernsthaft?", "fees keep going up",
                                  "Gute Karte für Reisen", "wer"])
def test_real_answers_pass(text):
    assert qc.gibberish_reason(text) is None


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    sim.simulate(seed=11, n_per_wave=600, out_dir=data)
    stacked, _ = ing.ingest(data / "raw", data / "interim", ROOT / "config")
    flagged, _ = qc.run_checks(stacked, qc.load_rules(ROOT / "config" / "consistency_rules.yaml"))
    return flagged, qc.evaluate(flagged, data / "_truth")


@pytest.mark.parametrize("flag", list(qc.TRUTH_MAP))
def test_flags_match_truth(evaluated, flag):
    _, ev = evaluated
    s = ev["overall"][flag]
    assert s["recall"] >= 0.95 and s["precision"] >= 0.95, s


def test_genuine_respondents_are_kept(evaluated):
    _, ev = evaluated
    genuine = [p for p in ev["profiles"] if p["profile"] == "genuine"]
    assert sum(p["removed"] for p in genuine) == 0
