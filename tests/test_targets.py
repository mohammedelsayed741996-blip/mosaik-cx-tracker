"""Tests for the Destatis weighting targets.

Run: pytest tests/test_targets.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bt", ROOT / "src" / "build_weighting_targets.py")
bt = importlib.util.module_from_spec(spec)
sys.modules["bt"] = bt
spec.loader.exec_module(bt)

# Destatis totals by Land, 31.12.2025, as published separately on statistikportal.de
PORTAL_TOTALS = {"Baden-Wuerttemberg": 11238761, "Bayern": 13245503, "Berlin": 3700577,
                 "Brandenburg": 2551889, "Bremen": 706915, "Hamburg": 1869473, "Hessen": 6278594,
                 "Mecklenburg-Vorpommern": 1573685, "Niedersachsen": 8000826,
                 "Nordrhein-Westfalen": 17986214, "Rheinland-Pfalz": 4123022, "Saarland": 1007342,
                 "Sachsen": 4026599, "Sachsen-Anhalt": 2120252, "Schleswig-Holstein": 2958519,
                 "Thueringen": 2078946}


def test_source_parses_and_adds_up():
    df = bt.parse(bt.SOURCE)              # raises if male + female != total anywhere
    assert sorted(df["age"].dropna().unique()) == list(range(18, 70))


def test_totals_match_second_source():
    df = bt.parse(bt.SOURCE)
    totals = df[df["age"].isna()].set_index("land")["total"].to_dict()
    assert totals == PORTAL_TOTALS


def test_targets_file_is_current():
    built = bt.build()
    on_disk = pd.read_csv(ROOT / "config" / "weighting_targets.csv")
    pd.testing.assert_series_equal(built["share"], on_disk["share"], check_names=False)


@pytest.mark.parametrize("var,levels", [("age_band", 5), ("gender_w", 2), ("region", 4)])
def test_margins_complete(var, levels):
    t = bt.build()
    g = t[t["variable"] == var]
    assert len(g) == levels and abs(g["share"].sum() - 1) < 1e-5
