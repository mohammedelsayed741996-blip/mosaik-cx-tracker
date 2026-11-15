"""Tests for step 06 (Shapley drivers) and step 07 (cross-check statistics).

Run: pytest tests/test_drivers_crosscheck.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kd = load("kd", "06_key_drivers.py")
cc = load("cc", "07_crosscheck.py")


def test_shapley_independent_predictors():
    """With independent predictors, each Shapley value equals beta^2 * var(x) / var(y)."""
    rng = np.random.default_rng(0)
    n = 200_000
    X = rng.normal(size=(n, 3))
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + rng.normal(size=n)
    r2 = kd.weighted_r2_all_subsets(X, y, np.ones(n))
    phi = kd.shapley_values(r2, 3)
    expected = np.array([9, 4, 1]) / 15.0            # var(y) = 9 + 4 + 1 + 1
    assert phi == pytest.approx(expected, abs=0.005)
    assert phi.sum() == pytest.approx(r2[frozenset({0, 1, 2})])


def test_shapley_splits_a_duplicated_predictor_evenly():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50_000)
    X = np.column_stack([x, x + rng.normal(scale=1e-3, size=x.size)])
    y = x + rng.normal(size=x.size)
    phi = kd.shapley_values(kd.weighted_r2_all_subsets(X, y, np.ones(x.size)), 2)
    assert phi[0] == pytest.approx(phi[1], rel=0.01)


def test_weights_change_the_fit():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 1))
    y = X[:, 0] + rng.normal(size=1000)
    w = np.where(X[:, 0] > 0, 5.0, 1.0)
    r2_w = kd.weighted_r2_all_subsets(X, y, w)[frozenset({0})]
    r2_u = kd.weighted_r2_all_subsets(X, y, np.ones(1000))[frozenset({0})]
    assert r2_w != pytest.approx(r2_u)


def test_quadrants():
    assert kd.quadrant(0.30, 20, 1 / 7, 50) == "Fix first"
    assert kd.quadrant(0.30, 70, 1 / 7, 50) == "Maintain"
    assert kd.quadrant(0.05, 20, 1 / 7, 50) == "Monitor"
    assert kd.quadrant(0.05, 70, 1 / 7, 50) == "Secondary"


def test_crosscheck_letters_match_the_r_hand_example():
    """Same worked example as R/tests/test_utils_sig_letters.R (A = 'B', B = '', C = '')."""
    est = np.array([0.60, 0.48, 0.55])
    ne = np.array([200.0, 150.0, 100.0])
    assert cc.letters_for(est, ne, np.full(3, np.nan), "prop", ["A", "B", "C"]) == ["B", "", ""]
    assert cc.letters_for(np.array([30.0, 10.0]), np.array([400.0, 300.0]), np.array([60.0, 65.0]),
                          "mean", ["A", "B"]) == ["B", ""]


def test_crosscheck_effective_base():
    assert cc.n_eff(np.array([1, 1, 2, 2.0])) == pytest.approx(3.6)
