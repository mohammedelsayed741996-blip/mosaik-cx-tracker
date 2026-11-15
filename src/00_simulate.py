#!/usr/bin/env python3
"""
00_simulate.py - Mosaik Bank CX Tracker 2026: simulate four messy survey waves.

What it does
------------
1. Builds respondents for 4 quarterly waves from hidden traits
   (digital affinity, price sensitivity, trust) and derives every answer from
   those traits, so the data has realistic structure.
2. Plants the story the analysis must find: a card fee in W3 hurts
   satisfaction with fees and NPS; an app relaunch in W4 lifts app ease
   (most among under-30s) and wins back about half of the NPS loss
   (expected NPS on a large sample: about 24 / 24 / 4 / 16; with seed 42
   the sample gives about 25 / 21 / 7 / 17 before weighting).
3. Plants bad respondents (speeders, straightliners, attention-check
   failures, duplicates, gibberish open ends, inconsistent answers).
4. Damages each wave's export differently, the way real fieldwork does
   (renamed columns, encodings, date and duration formats, free-text codes,
   an SPSS file in W4).
5. Writes the planted truth to data/_truth/. The pipeline must never read
   it; only the tests do, to score whether the pipeline found what was planted.

All brands and all data are fictional.

Usage
-----
    python src/00_simulate.py                 # seed 42, 3,000 raw completes per wave
    python src/00_simulate.py --seed 7 --n-per-wave 1000 --out-dir /tmp/sim

Outputs
-------
    data/raw/W1_export.csv      UTF-8, includes partials / screen-outs
    data/raw/W2_export.csv      Latin-1, QNPS column, hh:mm:ss durations
    data/raw/W3_export.csv      UTF-8, new C3r8 column, free-text states
    data/raw/W4_export.sav      SPSS with value labels (needs pyreadstat;
                                falls back to CSV + labels JSON without it)
    data/_truth/respondents_truth.csv   hidden traits + planted quality flags
    data/_truth/true_answers.csv        undamaged answers for every raw complete
    data/_truth/driver_weights.csv      true NPS driver weights
    data/_truth/wave_effects.csv        true wave shifts per attribute
    data/_truth/simulation_meta.json    seed, calibration, counts, file hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("simulate")

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK_PATH = ROOT / "config" / "codebook.yaml"

# =============================================================================
# Parameters (every number that shapes the data lives here)
# =============================================================================

SEED = 42
N_PER_WAVE = 3000                  # raw completes per wave, bad ones included (~720 Mosaik customers)
CALIBRATION_N = 80_000             # synthetic respondents used to calibrate NPS

WAVES = {  # wave: (first fieldwork day, number of days)
    1: ("2026-03-02", 14),
    2: ("2026-06-01", 14),
    3: ("2026-09-01", 14),
    4: ("2026-12-01", 14),
}

# --- Population frame (adults 18-69) ----------------------------------------
# APPROXIMATE shares for simulation only. Replace with Destatis figures
# (GENESIS 12411-0005, population by age, sex and state) in
# config/weighting_targets.csv before weighting; the simulator only needs
# a plausible frame, the weighting step needs the official one.
AGE_BANDS = {1: (18, 29), 2: (30, 39), 3: (40, 49), 4: (50, 59), 5: (60, 69)}
POP_AGE_BAND = {1: 0.19, 2: 0.20, 3: 0.18, 4: 0.23, 5: 0.20}
POP_GENDER = {1: 0.490, 2: 0.490, 3: 0.008, -99: 0.012}
POP_STATE = {  # millions, approximate
    1: 2.95, 2: 1.89, 3: 8.14, 4: 0.69, 5: 18.14, 6: 6.39, 7: 4.16, 8: 11.28,
    9: 13.37, 10: 0.99, 11: 3.88, 12: 2.57, 13: 1.63, 14: 4.09, 15: 2.19, 16: 2.13,
}
STATE_TO_REGION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 4, 7: 4, 8: 3, 9: 3, 10: 4,
                   11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2}
EAST_STATES = [s for s, r in STATE_TO_REGION.items() if r == 2]

# Sample skew relative to the population (what weighting has to correct)
SKEW_AGE_BAND = {1: 1.4, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.6}
SKEW_EAST = 0.8

# --- Banking --------------------------------------------------------------------
BRANDS = {1: "mosaik", 2: "volta", 3: "atlas", 4: "sparhaus", 5: "pebble"}
# P(uses brand) = logistic(intercept + slope * digital)
USAGE_COEF = {1: (-1.45, 0.9), 2: (-1.7, 0.8), 3: (-1.3, 0.2), 4: (0.4, -0.7), 5: (-2.4, 0.7)}
P_OTHER_BANK = 0.55
# Relative pull to be the main bank among the banks a respondent uses
MAIN_PULL = {1: 0.80, 2: 0.45, 3: 1.00, 4: 2.30, 5: 0.25, 6: 1.70}
MAIN_DIGITAL_TILT = {1: 0.4, 2: 0.4, 3: 0.1, 4: -0.3, 5: 0.4, 6: 0.0}
TENURE_P = [0.18, 0.30, 0.32, 0.20]           # <1, 1-2, 3-5, 5+ years
# Aware / consider (non-users): logistic(intercept + slope * digital)
AWARE_COEF = {1: (0.6, 0.5), 2: (1.2, 0.5), 3: (0.8, 0.3), 4: (2.5, 0.0), 5: (-0.3, 0.5)}
CONSIDER_COEF = {1: (-0.6, 0.6), 2: (-0.5, 0.6), 3: (-0.7, 0.2), 4: (-0.4, -0.5), 5: (-1.4, 0.5)}
CONSIDER_USER_BOOST = 1.8

# --- Experience model (Mosaik customers) --------------------------------------
ATTRS = ["app", "speed", "fees", "service", "security", "card", "interest", "investing"]
ATTR_ITEM = {a: f"c3_r{i}" for i, a in enumerate(ATTRS, start=1)}
MU = {"app": 0.60, "speed": 0.70, "fees": 0.30, "service": 0.00,
      "security": 0.40, "card": 0.30, "interest": -0.10, "investing": 0.10}
BETA = {"app": 0.30, "speed": 0.10, "fees": 0.05, "service": 0.00,
        "security": 0.05, "card": 0.10, "interest": 0.00, "investing": 0.15}
EPS_SD = 0.8
SAT_THRESHOLDS = [-1.5, -0.5, 0.4, 1.3]       # latent -> 1..5

# Planted wave shifts (W1..W4). The analysis must recover these.
WAVE_DELTA = {
    "fees":    [0.0, 0.00, -0.40, -0.40],   # the fee stays, so the hit persists
    "app":     [0.0, 0.05,  0.00,  0.20],
    "service": [0.0, 0.00, -0.15, -0.05],
}
FEES_MAIN_EXTRA_W3 = -0.20      # main-bank customers pay the fee; extra hit in W3
APP_UNDER30_W4 = 0.35           # W4 app shift for under-30s (instead of 0.20)

# Planted truth for the key driver analysis
DRIVER_WEIGHTS = {"fees": 0.30, "app": 0.25, "security": 0.20, "service": 0.12,
                  "speed": 0.05, "card": 0.05, "interest": 0.05}
TRUST_WEIGHT = 0.30
NPS_NOISE_SD = 0.35             # gives a realistic driver-model R2 of about 0.3
NPS_SCALE = 3.0                 # 0-10 score = intercept + NPS_SCALE * NPS*
NPS_TARGET_W1 = 24.0            # calibrate the intercept to this unweighted W1 NPS

# --- Bad respondents (share of raw completes per wave) --------------------------
BAD_RATES = {
    "speeder": 0.030,
    "straightliner": 0.025,
    "attention": 0.020,      # Mosaik customers only (only they see the grid)
    "duplicate": 0.010,
    "gibberish": 0.015,      # Mosaik customers only (only they see C2)
    "inconsistent": 0.010,
}
SPEEDER_RANDOM_SHARE = 0.60  # of speeders, share answering at random (rest: fast readers)
NONCOMPLETES_W1 = {2: 130, 3: 160, 4: 60}   # partials, screen-outs, over quota

# --- Raw export layout -----------------------------------------------------------
# clean name -> standard raw name. Wave-specific deviations are export damage.
RAW_NAMES: dict[str, str] = {
    "record": "record", "panel_id": "panel_id", "status": "status",
    "start_ts": "start_date", "end_ts": "end_date", "duration_s": "qtime",
    "device": "device", "age": "S1", "gender": "S2", "state": "S3",
    "s4_account": "S4",
    **{f"s5_r{i}": f"S5r{i}" for i in range(1, 5)},
    **{f"a1_r{i}": f"A1r{i}" for i in range(1, 8)},
    "main_bank": "A2", "a3_tenure": "A3",
    **{f"b1_r{i}": f"B1r{i}" for i in range(1, 6)},
    **{f"b2_r{i}": f"B2r{i}" for i in range(1, 6)},
    "c1_nps": "C1", "c2_reason": "C2",
    **{f"c3_r{i}": f"C3r{i}" for i in range(1, 9)},
    "c3x_attn": "C3x", "c4_overall": "C4", "c5_switch": "C5",
    **{f"c6_r{i}": f"C6r{i}" for i in range(1, 6)},
    "d1_app_freq": "D1", "d2_contact": "D2", "d3_resolved": "D3",
    **{f"e1_r{i}": f"E1r{i}" for i in range(1, 7)},
    "f1_income": "F1", "f2_employment": "F2", "f3_education": "F3",
}
C3_ITEMS = [f"c3_r{i}" for i in range(1, 9)]
E1_ITEMS = [f"e1_r{i}" for i in range(1, 7)]
MULTI_PREFIXES = ("a1_r", "b1_r", "b2_r", "c6_r", "s5_r")

# --- Open-end templates (C2) ------------------------------------------------------
# attribute -> code in the C2 code frame (see codebook value set c2_codes)
OE_CODE = {"fees": 1, "app": 2, "service": 3, "security": 4, "speed": 5,
           "card": 6, "interest": 7, "investing": 7, "general": 8}
OE_TEMPLATES = {
    ("fees", "neg", "en"): ["The new card fee is a joke", "too expensive since they added the monthly fee",
                            "4.90 a month for a card? no thanks", "fees keep going up"],
    ("fees", "neg", "de"): ["Die neue Kartengebühr ist eine Frechheit", "Zu teuer seit der monatlichen Gebühr",
                            "4,90 € im Monat für die Karte, ernsthaft?", "Gebühren steigen ständig"],
    ("fees", "pos", "en"): ["No hidden fees", "Free account and fair prices"],
    ("fees", "pos", "de"): ["Keine versteckten Gebühren", "Kostenloses Konto, faire Preise"],
    ("app", "neg", "en"): ["app crashes a lot", "hard to find anything in the app"],
    ("app", "neg", "de"): ["App stürzt ständig ab", "Die App ist unübersichtlich"],
    ("app", "pos", "en"): ["App is super easy to use", "love how simple the app is"],
    ("app", "pos", "de"): ["App ist super einfach", "Die App ist übersichtlich und schnell"],
    ("service", "neg", "en"): ["waited ages for support", "chat support never solves anything"],
    ("service", "neg", "de"): ["Kundenservice ist kaum erreichbar", "Lange Wartezeiten im Chat"],
    ("service", "pos", "en"): ["quick and friendly support"],
    ("service", "pos", "de"): ["Schneller, freundlicher Kundenservice"],
    ("security", "neg", "en"): ["worried about security after the phishing mails"],
    ("security", "neg", "de"): ["Mache mir Sorgen um die Sicherheit"],
    ("security", "pos", "en"): ["I feel my money is safe there"],
    ("security", "pos", "de"): ["Ich fühle mich sicher, gute Sicherheitsfunktionen"],
    ("speed", "neg", "en"): ["transfers take too long"],
    ("speed", "neg", "de"): ["Überweisungen dauern zu lange"],
    ("speed", "pos", "en"): ["instant transfers, love it"],
    ("speed", "pos", "de"): ["Überweisungen sind sofort da"],
    ("card", "neg", "en"): ["card got blocked abroad twice"],
    ("card", "neg", "de"): ["Karte wurde im Ausland gesperrt"],
    ("card", "pos", "en"): ["great card for travelling"],
    ("card", "pos", "de"): ["Gute Karte für Reisen"],
    ("interest", "neg", "en"): ["interest rates are too low"],
    ("interest", "neg", "de"): ["Zinsen sind zu niedrig"],
    ("interest", "pos", "en"): ["good interest on savings"],
    ("interest", "pos", "de"): ["Gute Zinsen aufs Tagesgeld"],
    ("investing", "neg", "en"): ["investing options are limited"],
    ("investing", "neg", "de"): ["Zu wenig Anlagemöglichkeiten"],
    ("investing", "pos", "en"): ["easy to set up an ETF savings plan"],
    ("investing", "pos", "de"): ["ETF-Sparplan ganz einfach"],
    ("general", "any", "en"): ["its ok", "does what it should", "nothing special"],
    ("general", "any", "de"): ["Alles gut", "Passt schon", "Nichts Besonderes"],
}
GIBBERISH = ["asdf", "good good good", "...", "x", "jjjjjjj", "nsdkjfh", "ok ok ok ok",
             "-", "qwertz", "fgfgfg", "1234"]
UMLAUT_PLAIN = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
STATE_TEXT_VARIANTS = {  # free-text spellings seen in W3
    9: ["Bayern", "bavaria ", "BY", "Bayern "], 5: ["NRW", "Nordrhein Westfalen", "nrw"],
    8: ["BaWü", "Baden-Württemberg", "BW"], 11: ["Berlin", "berlin ", "BE"],
    3: ["Niedersachsen", "lower saxony"], 6: ["Hessen", "hesse"], 14: ["Sachsen", "saxony"],
    2: ["Hamburg", "HH"],
}


# =============================================================================
# Helpers
# =============================================================================

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cut5(latent: np.ndarray, thresholds=SAT_THRESHOLDS) -> np.ndarray:
    """Map a continuous latent score onto a 1..5 scale."""
    return np.digitize(latent, thresholds) + 1


def norm_probs(d: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = np.array(list(d.keys()))
    p = np.array(list(d.values()), dtype=float)
    return keys, p / p.sum()


def nps_of(scores: np.ndarray) -> float:
    scores = np.asarray(scores)
    return 100.0 * ((scores >= 9).mean() - (scores <= 6).mean())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_codebook(path: Path = CODEBOOK_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Calibration:
    intercept: float = 7.6
    nps_star_mean: float = 0.0
    nps_star_sd: float = 1.0
    sat_star_mean: float = 0.0
    sat_star_sd: float = 1.0
    achieved_w1_nps: float = float("nan")


@dataclass
class WaveResult:
    wave: int
    raw: pd.DataFrame                  # raw export (clean names, before damage)
    truth: pd.DataFrame                # hidden traits + planted flags
    noncompletes: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# 1. Respondents: demographics and hidden traits
# =============================================================================

def draw_demographics(rng: np.random.Generator, n: int) -> pd.DataFrame:
    band_keys, band_p = norm_probs({b: POP_AGE_BAND[b] * SKEW_AGE_BAND[b] for b in AGE_BANDS})
    age_band = rng.choice(band_keys, size=n, p=band_p)
    lo = np.array([AGE_BANDS[b][0] for b in age_band])
    hi = np.array([AGE_BANDS[b][1] for b in age_band])
    age = rng.integers(lo, hi + 1)

    g_keys, g_p = norm_probs(POP_GENDER)
    gender = rng.choice(g_keys, size=n, p=g_p)

    s_keys, s_p = norm_probs({s: POP_STATE[s] * (SKEW_EAST if s in EAST_STATES else 1.0)
                              for s in POP_STATE})
    state = rng.choice(s_keys, size=n, p=s_p)

    return pd.DataFrame({
        "age": age, "age_band": age_band, "gender": gender, "state": state,
        "region": pd.Series(state).map(STATE_TO_REGION).to_numpy(),
    })


def draw_traits(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    age = df["age"].to_numpy()

    # Education: younger cohorts hold more Abitur / degrees
    edu_z = rng.normal(0, 1, n) - 0.012 * (age - 44)
    edu = np.digitize(edu_z, [-1.1, -0.35, 0.25, 0.8]) + 1
    # recode latent order (low..high) to codebook order:
    # 1 Hauptschule, 2 Realschule, 4 vocational, 3 Abitur, 5 university
    edu = pd.Series(edu).map({1: 1, 2: 2, 3: 4, 4: 3, 5: 5}).to_numpy()

    # Employment depends on life stage
    emp = np.empty(n, dtype=int)
    u = rng.random(n)
    young, old = age < 25, age >= 63
    mid = ~young & ~old
    emp[young] = np.select([u[young] < 0.30, u[young] < 0.40, u[young] < 0.95],
                           [1, 2, 4], default=6)
    emp[mid] = np.select([u[mid] < 0.62, u[mid] < 0.80, u[mid] < 0.90, u[mid] < 0.93],
                         [1, 2, 3, 4], default=6)
    emp[old] = np.select([u[old] < 0.15, u[old] < 0.27, u[old] < 0.32, u[old] < 0.97],
                         [1, 2, 3, 5], default=6)

    emp_effect = pd.Series(emp).map({1: 0.5, 2: -0.3, 3: 0.2, 4: -1.2, 5: -0.3, 6: -1.0}).to_numpy()
    inc_z = 0.5 * edu_z + emp_effect + 0.01 * (np.minimum(age, 55) - 40) + rng.normal(0, 0.8, n)
    income = np.digitize(inc_z, [-1.2, -0.4, 0.3, 0.9, 1.6]) + 1
    income = np.where(rng.random(n) < 0.06, -99, income)

    digital = -0.03 * (age - 44) + 0.15 * edu_z + rng.normal(0, 1, n)
    price_sens = rng.normal(0, 1, n) - 0.2 * inc_z
    trust = rng.normal(0, 1, n)

    out = df.copy()
    out["f1_income"], out["f2_employment"], out["f3_education"] = income, emp, edu
    out["digital"], out["price_sens"], out["trust"] = digital, price_sens, trust
    out["_edu_z"], out["_inc_z"] = edu_z, inc_z
    return out


# =============================================================================
# 2. Banking relationship and brand funnel
# =============================================================================

def draw_banking(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    n, dig = len(df), df["digital"].to_numpy()
    out = df.copy()

    uses = np.zeros((n, 6), dtype=int)
    for b, (a, s) in USAGE_COEF.items():
        uses[:, b - 1] = rng.random(n) < logistic(a + s * dig)
    uses[:, 5] = rng.random(n) < P_OTHER_BANK
    uses[uses.sum(axis=1) == 0, 5] = 1          # everyone passed S4 (has an account)
    for b in range(1, 7):
        out[f"a1_r{b}"] = uses[:, b - 1]
    out["a1_r7"] = 0

    # Main bank: draw among used banks, proportional to pull x digital tilt
    pull = np.column_stack([MAIN_PULL[b] * np.exp(MAIN_DIGITAL_TILT[b] * dig) for b in range(1, 7)])
    pull = pull * uses
    pull = pull / pull.sum(axis=1, keepdims=True)
    cum = pull.cumsum(axis=1)
    out["main_bank"] = (rng.random(n)[:, None] > cum).sum(axis=1) + 1

    out["mosaik_cust"] = out["a1_r1"]
    out["mosaik_main"] = (out["main_bank"] == 1).astype(int)

    # Tenure: Mosaik launched 2016; nobody under 21 holds an account 3+ years
    tenure = rng.choice([1, 2, 3, 4], size=n, p=TENURE_P)
    age = out["age"].to_numpy()
    tenure = np.where((age < 23) & (tenure == 4), 3, tenure)
    tenure = np.where((age < 21) & (tenure >= 3), 2, tenure)
    out["a3_tenure"] = np.where(out["mosaik_cust"] == 1, tenure, np.nan)

    # Funnel: users are always aware; consideration higher among users
    for b, (a, s) in AWARE_COEF.items():
        aware = (rng.random(n) < logistic(a + s * dig)) | (uses[:, b - 1] == 1)
        out[f"b1_r{b}"] = aware.astype(int)
        ca, cs = CONSIDER_COEF[b]
        p_cons = logistic(ca + cs * dig + CONSIDER_USER_BOOST * uses[:, b - 1])
        consider = (rng.random(n) < p_cons).astype(float)
        out[f"b2_r{b}"] = np.where(aware, consider, np.nan)
    return out


# =============================================================================
# 3. Experience (Mosaik customers), attitudes, paradata
# =============================================================================

def latent_scores(rng: np.random.Generator, df: pd.DataFrame, wave: int) -> pd.DataFrame:
    """Hidden satisfaction score per attribute (customers only; NaN otherwise)."""
    n = len(df)
    dig = df["digital"].to_numpy()
    cust = df["mosaik_cust"].to_numpy() == 1
    main = df["mosaik_main"].to_numpy() == 1
    under30 = df["age"].to_numpy() < 30
    s = {}
    for a in ATTRS:
        delta = np.full(n, WAVE_DELTA.get(a, [0, 0, 0, 0])[wave - 1], dtype=float)
        if a == "fees" and wave == 3:
            delta = delta + FEES_MAIN_EXTRA_W3 * main
        if a == "app" and wave == 4:
            delta = np.where(under30, APP_UNDER30_W4, delta)
        val = MU[a] + BETA[a] * dig + delta + rng.normal(0, EPS_SD, n)
        s[f"s_{a}"] = np.where(cust, val, np.nan)
    return pd.DataFrame(s, index=df.index)


def nps_star(rng: np.random.Generator, s: pd.DataFrame, trust: np.ndarray) -> np.ndarray:
    total = sum(w * s[f"s_{a}"].to_numpy() for a, w in DRIVER_WEIGHTS.items())
    return total + TRUST_WEIGHT * trust + rng.normal(0, NPS_NOISE_SD, len(s))


def score_nps(nps_star_vals: np.ndarray, intercept: float) -> np.ndarray:
    return np.clip(np.rint(intercept + NPS_SCALE * nps_star_vals), 0, 10)


def calibrate(seed: int) -> Calibration:
    """Find the 0-10 intercept so unweighted W1 NPS among customers = target.

    Uses its own random stream, so the main simulation is unaffected."""
    rng = np.random.default_rng([seed, 999])
    df = draw_banking(rng, draw_traits(rng, draw_demographics(rng, CALIBRATION_N)))
    df = df[df["mosaik_cust"] == 1].reset_index(drop=True)
    s = latent_scores(rng, df, wave=1)
    ns = nps_star(rng, s, df["trust"].to_numpy())
    sat = ns + rng.normal(0, 0.3, len(ns))

    lo, hi = 3.0, 12.0
    for _ in range(60):                      # bisection; NPS rises with intercept
        mid = (lo + hi) / 2
        if nps_of(score_nps(ns, mid)) < NPS_TARGET_W1:
            lo = mid
        else:
            hi = mid
    cal = Calibration(intercept=round((lo + hi) / 2, 4),
                      nps_star_mean=float(ns.mean()), nps_star_sd=float(ns.std()),
                      sat_star_mean=float(sat.mean()), sat_star_sd=float(sat.std()))
    cal.achieved_w1_nps = nps_of(score_nps(ns, cal.intercept))
    return cal


def add_typos(rng: np.random.Generator, text: str, lang: str) -> str:
    if lang == "de" and rng.random() < 0.25:
        for k, v in UMLAUT_PLAIN.items():
            text = text.replace(k, v)
    if rng.random() < 0.08 and len(text) > 4:
        i = int(rng.integers(0, len(text) - 1))
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if rng.random() < 0.15:
        text = text.lower()
    return text


def draw_open_end(rng: np.random.Generator, s_row: dict, c1: float, wave: int) -> tuple[str, int]:
    """Choose the reason from the attribute that drives this respondent's score."""
    lang = "de" if rng.random() < 0.65 else "en"
    valid = {a: v for a, v in s_row.items() if not np.isnan(v) and (a != "investing" or wave >= 3)}
    if rng.random() < 0.10 or not valid:
        attr, pol = "general", "any"
    elif c1 >= 9:
        attr, pol = max(valid, key=valid.get), "pos"
    else:
        attr, pol = min(valid, key=valid.get), "neg"
    options = OE_TEMPLATES[(attr, pol, lang)]
    text = options[int(rng.integers(0, len(options)))]
    return add_typos(rng, text, lang), OE_CODE[attr]


def draw_experience(rng: np.random.Generator, df: pd.DataFrame, wave: int,
                    cal: Calibration) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    cust = out["mosaik_cust"].to_numpy() == 1
    dig = out["digital"].to_numpy()
    s = latent_scores(rng, out, wave)
    ns = nps_star(rng, s, out["trust"].to_numpy())
    out = pd.concat([out, s], axis=1)
    out["nps_star"] = np.where(cust, ns, np.nan)

    # C6 products held
    out["c6_r1"] = 1
    out["c6_r2"] = (rng.random(n) < 0.45).astype(int)
    out["c6_r3"] = (rng.random(n) < 0.35).astype(int)
    out["c6_r4"] = (rng.random(n) < logistic(-1.1 + 0.5 * dig)).astype(int)
    out["c6_r5"] = (rng.random(n) < 0.15).astype(int)

    # C3 grid, with "don't know" for products not held
    for a in ATTRS:
        item = ATTR_ITEM[a]
        vals = cut5(s[f"s_{a}"].to_numpy()).astype(float)
        if a == "interest":
            vals = np.where((out["c6_r2"] == 0) & (rng.random(n) < 0.25), -97, vals)
        if a == "investing":
            vals = np.where((out["c6_r4"] == 0) & (rng.random(n) < 0.60), -97, vals)
            if wave < 3:
                vals = np.full(n, np.nan)
        out[item] = vals
    out["c3x_attn"] = 4.0

    # C1 NPS, C4 overall, C5 switching
    out["c1_nps"] = score_nps(ns, cal.intercept)
    sat_z = (ns + rng.normal(0, 0.3, n) - cal.sat_star_mean) / cal.sat_star_sd
    out["c4_overall"] = cut5(sat_z, [-1.9, -1.0, 0.0, 1.0]).astype(float)
    nps_z = (ns - cal.nps_star_mean) / cal.nps_star_sd
    switch_latent = -0.9 * nps_z + 0.4 * out["price_sens"].to_numpy() + rng.normal(0, 0.8, n)
    out["c5_switch"] = cut5(switch_latent, [-0.9, 0.0, 0.9, 1.6]).astype(float)

    # D: usage and service
    out["d1_app_freq"] = cut5(-dig + rng.normal(0, 0.8, n), [-0.9, 0.5, 1.4, 99.0]).astype(float)
    p_contact = 0.22 + (0.10 if wave == 3 else 0.03 if wave == 4 else 0.0)
    out["d2_contact"] = (rng.random(n) < p_contact).astype(float)
    serv = s["s_service"].to_numpy()
    res_latent = serv + rng.normal(0, 0.7, n)
    d3 = np.select([res_latent > -0.1, res_latent > -1.0], [1, 2], default=3).astype(float)
    out["d3_resolved"] = np.where(out["d2_contact"] == 1, d3, np.nan)

    # C2 open end (needs C1)
    reasons, codes = [], []
    s_cols = [f"s_{a}" for a in ATTRS]
    for i in range(n):
        if not cust[i]:
            reasons.append(None)
            codes.append(np.nan)
            continue
        row = {a: s.iat[i, j] for j, a in enumerate(ATTRS)}
        text, code = draw_open_end(rng, row, out["c1_nps"].iat[i], wave)
        reasons.append(text)
        codes.append(code)
    out["c2_reason"] = pd.array(reasons, dtype="object")
    out["c2_true_code"] = codes

    # Non-customers are not asked section C/D
    c_cols = ["c1_nps", "c2_reason", "c3x_attn", "c4_overall", "c5_switch",
              "d1_app_freq", "d2_contact", "d3_resolved"] + C3_ITEMS + [f"c6_r{i}" for i in range(1, 6)]
    for c in c_cols:
        out[c] = out[c].where(cust)
    return out


def draw_attitudes(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    age, dig = out["age"].to_numpy(), out["digital"].to_numpy()
    latent = {
        "e1_r1": 0.7 * out["price_sens"].to_numpy(),
        "e1_r2": -0.7 * dig,
        "e1_r3": 0.8 * dig,
        "e1_r4": 0.2 * out["_edu_z"].to_numpy() - 0.005 * (age - 44),
        "e1_r5": 0.4 * out["_inc_z"].to_numpy() + 0.3 * dig,
        "e1_r6": 0.015 * (age - 44) + 0.2 * out["trust"].to_numpy(),
    }
    for item, mean in latent.items():
        out[item] = cut5(mean + rng.normal(0, 0.9, n), [-1.3, -0.45, 0.45, 1.3])
    return out


def draw_paradata(rng: np.random.Generator, df: pd.DataFrame, wave: int) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    dig = out["digital"].to_numpy()
    p_mob = logistic(0.2 + 0.5 * dig) * 0.93
    u = rng.random(n)
    out["device"] = np.select([u < p_mob, u < p_mob + 0.07], [2, 3], default=1)

    base = np.where(out["mosaik_cust"] == 1, 840.0, 520.0)
    base = base * pd.Series(out["device"]).map({1: 1.0, 2: 1.08, 3: 1.05}).to_numpy()
    dur = base * np.exp(rng.normal(0, 0.30, n))
    out["duration_s"] = np.rint(np.maximum(dur, 0.60 * base)).astype(int)   # no accidental speeders

    start_day, n_days = WAVES[wave]
    day = pd.Timestamp(start_day) + pd.to_timedelta(rng.integers(0, n_days, n), unit="D")
    hour_p = np.array([1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 8, 9, 9, 7, 5, 3], dtype=float)
    hours = rng.choice(np.arange(7, 24), size=n, p=hour_p / hour_p.sum())
    secs = hours * 3600 + rng.integers(0, 3600, n)
    out["start_ts"] = day + pd.to_timedelta(secs, unit="s")
    out["end_ts"] = out["start_ts"] + pd.to_timedelta(out["duration_s"], unit="s")
    return out


def fix_accidental_straightlining(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """Genuine respondents should never trip the straightliner rule by chance."""
    out = df.copy()
    for items in (E1_ITEMS, C3_ITEMS):
        grid = out[items].where(out[items] > 0)
        same = (grid.nunique(axis=1) == 1) & (grid.notna().sum(axis=1) >= 6)
        for idx in out.index[same]:
            col = items[int(rng.integers(0, 6))]
            v = out.at[idx, col]
            out.at[idx, col] = v + 1 if v < 5 else v - 1
    return out


def generate_wave(rng: np.random.Generator, wave: int, n: int, cal: Calibration,
                  panel_ids: np.ndarray) -> pd.DataFrame:
    df = draw_demographics(rng, n)
    df = draw_traits(rng, df)
    df = draw_banking(rng, df)
    df = draw_experience(rng, df, wave, cal)
    df = draw_attitudes(rng, df)
    df = draw_paradata(rng, df, wave)
    df = fix_accidental_straightlining(rng, df)
    df["panel_id"] = panel_ids
    df["status"] = 1
    df["s4_account"] = 1
    df["s5_r1"], df["s5_r2"], df["s5_r3"], df["s5_r4"] = 0, 0, 0, 1
    df["wave"] = wave
    return df


# =============================================================================
# 4. Bad respondents
# =============================================================================

def _pick(rng: np.random.Generator, pool: pd.Index, k: int) -> pd.Index:
    k = int(max(0, min(k, len(pool))))
    return pd.Index(rng.choice(pool.to_numpy(), size=k, replace=False)) if k else pd.Index([])


def plant_bad_respondents(rng: np.random.Generator, df: pd.DataFrame, n_dup: int) -> pd.DataFrame:
    """Corrupt a share of rows and append duplicates. Adds true_* flag columns."""
    out = df.copy()
    N = len(out) + n_dup
    flags = ["true_speeder", "true_straightliner", "true_attention", "true_duplicate",
             "true_gibberish", "true_inconsistent"]
    for f in flags:
        out[f] = 0
    out["bad_profile"] = ""
    out["inconsistency_type"] = ""
    cust_idx = out.index[out["mosaik_cust"] == 1]
    median_dur = out.groupby("device")["duration_s"].transform("median")

    def add_profile(idx, label):
        out.loc[idx, "bad_profile"] = (out.loc[idx, "bad_profile"] + "|" + label).str.strip("|")

    def straightline(idx):
        for i in idx:
            v = int(rng.integers(1, 6))
            items = E1_ITEMS + ([c for c in C3_ITEMS if pd.notna(out.at[i, c])]
                                if out.at[i, "mosaik_cust"] == 1 else [])
            out.loc[i, items] = v
            if out.at[i, "mosaik_cust"] == 1:
                out.at[i, "c3x_attn"] = v           # a straightliner answers the check row too
        out.loc[idx, "true_straightliner"] = 1

    # --- speeders: random responders + fast-but-genuine readers ---------------
    speed_idx = _pick(rng, out.index, round(BAD_RATES["speeder"] * N))
    out.loc[speed_idx, "duration_s"] = np.rint(
        median_dur[speed_idx] * rng.uniform(0.15, 0.33, len(speed_idx))).astype(int)
    out.loc[speed_idx, "true_speeder"] = 1
    n_random = round(SPEEDER_RANDOM_SHARE * len(speed_idx))
    random_idx = speed_idx[:n_random]
    add_profile(random_idx, "speeder_random")
    add_profile(speed_idx[n_random:], "fast_reader")
    for i in random_idx:
        out.loc[i, E1_ITEMS] = rng.integers(1, 6, len(E1_ITEMS))
        if out.at[i, "mosaik_cust"] == 1:
            items = [c for c in C3_ITEMS if pd.notna(out.at[i, c])]
            out.loc[i, items] = rng.integers(1, 6, len(items))
            out.at[i, "c3x_attn"] = int(rng.integers(1, 6))
            out.at[i, "c1_nps"] = int(rng.integers(0, 11))
            out.at[i, "c4_overall"] = int(rng.integers(1, 6))
            out.at[i, "c5_switch"] = int(rng.integers(1, 6))
    # half of the random responders straightline; the other half (customers) write gibberish
    half = len(random_idx) // 2
    straightline(random_idx[:half])
    gib_from_speed = [i for i in random_idx[half:] if out.at[i, "mosaik_cust"] == 1]
    out.loc[gib_from_speed, "c2_reason"] = rng.choice(GIBBERISH, len(gib_from_speed))
    out.loc[gib_from_speed, "true_gibberish"] = 1
    out.loc[gib_from_speed, "c2_true_code"] = np.nan

    # --- straightliners (top up to target) --------------------------------------
    need = round(BAD_RATES["straightliner"] * N) - int(out["true_straightliner"].sum())
    sl_idx = _pick(rng, out.index[(out["bad_profile"] == "")], need)
    straightline(sl_idx)
    add_profile(sl_idx, "straightliner")

    # --- attention-check failures (customers; top up to target) ---------------
    out["true_attention"] = ((out["mosaik_cust"] == 1) & (out["c3x_attn"] != 4)).astype(int)
    need = round(BAD_RATES["attention"] * N) - int(out["true_attention"].sum())
    att_idx = _pick(rng, cust_idx[(out.loc[cust_idx, "bad_profile"] == "").to_numpy()], need)
    out.loc[att_idx, "c3x_attn"] = rng.choice([1, 2, 3, 5], len(att_idx))
    out.loc[att_idx, "true_attention"] = 1
    add_profile(att_idx, "attention_fail")

    # --- gibberish open ends (customers; top up to target) ---------------------
    need = round(BAD_RATES["gibberish"] * N) - int(out["true_gibberish"].sum())
    gib_idx = _pick(rng, cust_idx[(out.loc[cust_idx, "bad_profile"] == "").to_numpy()], need)
    out.loc[gib_idx, "c2_reason"] = rng.choice(GIBBERISH, len(gib_idx))
    out.loc[gib_idx, "c2_true_code"] = np.nan
    out.loc[gib_idx, "true_gibberish"] = 1
    add_profile(gib_idx, "gibberish")

    # --- inconsistent answers ---------------------------------------------------
    k = round(BAD_RATES["inconsistent"] * N)
    clean_cust = cust_idx[(out.loc[cust_idx, "bad_profile"] == "").to_numpy()]
    k_a, k_b = round(0.4 * k), round(0.3 * k)
    a_idx = _pick(rng, clean_cust, k_a)                             # too young for 5+ years
    out.loc[a_idx, "age"] = rng.integers(18, 21, len(a_idx))
    out.loc[a_idx, "age_band"] = 1
    out.loc[a_idx, "a3_tenure"] = 4
    out.loc[a_idx, "inconsistency_type"] = "age_vs_tenure"
    main_pool = clean_cust.difference(a_idx)
    main_pool = main_pool[(out.loc[main_pool, "main_bank"] == 1).to_numpy()]
    b_idx = _pick(rng, main_pool, k_b)                              # main bank not ticked in A1
    out.loc[b_idx, "a1_r1"] = 0
    out.loc[b_idx, "inconsistency_type"] = "main_bank_not_used"
    contact_pool = clean_cust.difference(a_idx).difference(b_idx)
    contact_pool = contact_pool[(out.loc[contact_pool, "d2_contact"] == 0).to_numpy()]
    c_idx = _pick(rng, contact_pool, k - len(a_idx) - len(b_idx))  # D3 answered without contact
    out.loc[c_idx, "d3_resolved"] = rng.choice([1, 2, 3], len(c_idx))
    out.loc[c_idx, "inconsistency_type"] = "d3_without_contact"
    inc_idx = a_idx.append(b_idx).append(c_idx)
    out.loc[inc_idx, "true_inconsistent"] = 1
    add_profile(inc_idx, "inconsistent")

    # --- duplicates: same panel member re-enters a few days later -------------
    src = _pick(rng, out.index[out["bad_profile"] == ""], n_dup)
    dups = out.loc[src].copy()
    dups["start_ts"] = dups["start_ts"] + pd.to_timedelta(rng.integers(1, 4, len(dups)), unit="D")
    dups["duration_s"] = np.rint(dups["duration_s"] * rng.uniform(0.8, 1.1, len(dups))).astype(int)
    dups["end_ts"] = dups["start_ts"] + pd.to_timedelta(dups["duration_s"], unit="s")
    for i in dups.index:                                            # small answer changes
        before = dups.loc[i, E1_ITEMS].copy()
        for _ in range(10):                                         # never create a straightliner
            col = E1_ITEMS[int(rng.integers(0, 6))]
            dups.at[i, col] = int(np.clip(before[col] + rng.choice([-1, 1]), 1, 5))
            if dups.loc[i, E1_ITEMS].nunique() > 1:
                break
            dups.loc[i, E1_ITEMS] = before
    dups["true_duplicate"] = 1
    dups["bad_profile"] = "duplicate"
    dups["_dup_of"] = src.to_numpy()
    out["_dup_of"] = np.nan
    out = pd.concat([out, dups], ignore_index=True)

    # --- what the documented removal rule should do with each row -------------
    soft = out[["true_speeder", "true_straightliner", "true_gibberish"]].sum(axis=1)
    hard = out[["true_attention", "true_duplicate", "true_inconsistent"]].any(axis=1)
    out["should_remove"] = (hard | (soft >= 2)).astype(int)
    out["end_ts"] = out["start_ts"] + pd.to_timedelta(out["duration_s"], unit="s")
    return out


def assign_records(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """Records are numbered in order of interview start, like a platform export."""
    out = df.copy()
    out["_row"] = out.index
    out = out.sort_values("start_ts", kind="stable").reset_index(drop=True)
    out["record"] = np.arange(1, len(out) + 1)
    rec_by_row = dict(zip(out["_row"], out["record"]))
    out["dup_of_record"] = out["_dup_of"].map(rec_by_row)    # NaN for non-duplicates
    return out.drop(columns=["_row", "_dup_of"])


def make_noncompletes(rng: np.random.Generator, wave: int, first_record: int,
                      panel_ids: np.ndarray) -> pd.DataFrame:
    """Partials (2), screen-outs (3) and over-quota (4) rows for the W1 export."""
    rows = []
    start_day, n_days = WAVES[wave]
    rec = first_record
    pid = iter(panel_ids)
    for status, k in NONCOMPLETES_W1.items():
        for _ in range(k):
            ts = pd.Timestamp(start_day) + pd.Timedelta(days=int(rng.integers(0, n_days)),
                                                         seconds=int(rng.integers(7 * 3600, 23 * 3600)))
            r = {"record": rec, "panel_id": next(pid), "status": status, "start_ts": ts,
                 "device": int(rng.choice([1, 2, 3], p=[0.4, 0.53, 0.07])),
                 "gender": int(rng.choice([1, 2])), "state": int(rng.integers(1, 17))}
            if status == 3:
                reason = rng.choice(["age", "account", "industry"])
                r["age"] = int(rng.choice([16, 17, 70, 72, 75])) if reason == "age" else int(rng.integers(18, 70))
                r["s4_account"] = 0 if reason == "account" else 1
                r["s5_r1"] = 1 if reason == "industry" else 0
                r["s5_r4"] = 0 if reason == "industry" else 1
            else:
                r["age"] = int(rng.integers(18, 70))
                r["s4_account"], r["s5_r1"], r["s5_r4"] = 1, 0, 1
            if status == 2:                                            # dropped out after section B
                banks = rng.random(6) < 0.4
                for b in range(1, 7):
                    r[f"a1_r{b}"] = int(banks[b - 1])
                r["a1_r7"] = 0
                for b in range(1, 6):
                    r[f"b1_r{b}"] = int(rng.random() < 0.6)
            rows.append(r)
            rec += 1
    return pd.DataFrame(rows)


# =============================================================================
# 5. Raw export layout and per-wave damage
# =============================================================================

def raw_columns(wave: int) -> list[str]:
    cols = list(RAW_NAMES.keys())
    if wave < 3:
        cols.remove("c3_r8")
    return cols


def to_raw(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    """Clean-name answers -> raw export frame (raw names, platform missing codes)."""
    cols = raw_columns(wave)
    raw = df.reindex(columns=cols).copy()
    raw["start_ts"] = pd.to_datetime(raw["start_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    raw["end_ts"] = pd.to_datetime(raw["end_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in cols:
        if c in ("start_ts", "end_ts", "panel_id", "c2_reason"):
            continue
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("Int64")
    return raw.rename(columns=RAW_NAMES)


def damage_w1(rng: np.random.Generator, raw: pd.DataFrame, noncompletes: pd.DataFrame) -> pd.DataFrame:
    """W1: partials / screen-outs included; '99' used alongside '-99'; blank multi-selects."""
    nc = to_raw(noncompletes.reindex(columns=raw_columns(1)), 1)
    nc.loc[nc["status"] != 2, "end_date"] = np.nan
    nc.loc[nc["status"] == 2, "end_date"] = np.nan              # partials never reached the end
    out = pd.concat([raw, nc], ignore_index=True)
    for col in ("S2", "F1"):
        refused = out.index[out[col] == -99]
        flip = refused[rng.random(len(refused)) < 0.35]
        out.loc[flip, col] = 99
    multi = [c for c in out.columns if c.startswith(("A1r", "B1r", "B2r", "C6r", "S5r"))]
    for c in multi:
        out[c] = out[c].mask(out[c] == 0)                       # 0 exported as blank
    return out.sort_values("record").reset_index(drop=True)


def damage_w2(rng: np.random.Generator, raw: pd.DataFrame) -> pd.DataFrame:
    """W2: C1 renamed QNPS; qtime as hh:mm:ss text; German date format."""
    out = raw.rename(columns={"C1": "QNPS"}).copy()
    secs = out["qtime"].astype("float")
    out["qtime"] = [f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}" for s in secs]
    for c in ("start_date", "end_date"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%d.%m.%Y %H:%M")
    return out


def damage_w3(rng: np.random.Generator, raw: pd.DataFrame) -> pd.DataFrame:
    """W3: free-text states in ~5% of rows; ages like '34 Jahre' in ~3%."""
    out = raw.copy()
    out["S3"] = out["S3"].astype("object")
    out["S1"] = out["S1"].astype("object")
    eligible = out.index[out["S3"].isin(list(STATE_TEXT_VARIANTS))]
    pick = eligible[rng.random(len(eligible)) < 0.05 * len(out) / max(len(eligible), 1)]
    for i in pick:
        variants = STATE_TEXT_VARIANTS[int(out.at[i, "S3"])]
        out.at[i, "S3"] = variants[int(rng.integers(0, len(variants)))]
    ages = out.index[rng.random(len(out)) < 0.03]
    for i in ages:
        out.at[i, "S1"] = f"{out.at[i, 'S1']} Jahre" if rng.random() < 0.7 else f"{out.at[i, 'S1']} years"
    return out


def mojibake(text: str) -> str:
    try:
        return text.encode("utf-8").decode("latin-1")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def damage_w4(rng: np.random.Generator, raw: pd.DataFrame) -> pd.DataFrame:
    """W4: trailing whitespace and garbled umlauts in open ends (labels handled at write)."""
    out = raw.copy()
    has_text = out["C2"].notna()
    ws = out.index[has_text & (rng.random(len(out)) < 0.20)]
    out.loc[ws, "C2"] = out.loc[ws, "C2"].astype(str) + rng.choice(["  ", " ", "\t"], len(ws))
    # garble ~2% of all open ends, choosing texts that contain umlauts / non-ASCII
    non_ascii = out.index[has_text & out["C2"].fillna("").map(lambda t: not t.isascii())]
    k = min(len(non_ascii), round(0.02 * int(has_text.sum())))
    garble = pd.Index(rng.choice(non_ascii.to_numpy(), size=k, replace=False)) if k else pd.Index([])
    out.loc[garble, "C2"] = out.loc[garble, "C2"].astype(str).map(mojibake)
    return out


def spss_labels(codebook: dict, raw_cols: list[str]) -> tuple[dict, dict]:
    """Variable and value labels for the W4 .sav, from the codebook (+ one relabel)."""
    sets = codebook["value_sets"]
    var_labels, value_labels = {}, {}
    for v in codebook["variables"]:
        raw = v.get("raw")
        if raw and raw in raw_cols:
            var_labels[raw] = v["label"]
            vs = v.get("values")
            if vs and v["type"] == "int":
                value_labels[raw] = {int(k): str(lbl) for k, lbl in sets[vs].items()}
    # Planted damage: a category relabelled by the fieldwork agency (code unchanged)
    if "F2" in value_labels:
        value_labels["F2"][6] = "Nicht erwerbstätig"
    return var_labels, value_labels


def write_w4(raw: pd.DataFrame, path_sav: Path, codebook: dict) -> Path:
    var_labels, value_labels = spss_labels(codebook, list(raw.columns))
    out = raw.copy()
    for c in ("start_date", "end_date"):
        out[c] = pd.to_datetime(out[c])
    for c in out.columns:
        if str(out[c].dtype) == "Int64":
            out[c] = out[c].astype("float64")                   # SPSS numerics are doubles
    try:
        import pyreadstat
    except ImportError:
        csv_path = path_sav.with_suffix(".csv")
        log.warning("pyreadstat not installed: writing %s + value labels JSON instead of .sav. "
                    "Install it (pip install pyreadstat) to get the SPSS file.", csv_path.name)
        out.to_csv(csv_path, index=False, encoding="utf-8")
        with open(path_sav.with_suffix(".labels.json"), "w", encoding="utf-8") as fh:
            json.dump({"variable_labels": var_labels, "value_labels": value_labels},
                      fh, ensure_ascii=False, indent=2)
        return csv_path
    pyreadstat.write_sav(out, str(path_sav), column_labels=var_labels,
                         variable_value_labels=value_labels)
    return path_sav


# =============================================================================
# 6. Orchestration
# =============================================================================

TRUTH_COLS = ["wave", "record", "panel_id", "status", "mosaik_cust", "mosaik_main",
              "age", "digital", "price_sens", "trust", "nps_star",
              *[f"s_{a}" for a in ATTRS], "c2_true_code",
              "true_speeder", "true_straightliner", "true_attention", "true_duplicate",
              "true_gibberish", "true_inconsistent", "inconsistency_type", "bad_profile",
              "dup_of_record", "should_remove"]


def simulate(seed: int, n_per_wave: int, out_dir: Path) -> dict:
    codebook = load_codebook()
    raw_dir, truth_dir = out_dir / "raw", out_dir / "_truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    cal = calibrate(seed)
    log.info("Calibrated NPS intercept %.3f (W1 calibration NPS %.1f)", cal.intercept, cal.achieved_w1_nps)

    rng = np.random.default_rng(seed)
    pool = np.array([f"P{n:07d}" for n in rng.choice(np.arange(1_000_000, 9_999_999),
                                                     size=200_000, replace=False)])
    pool_pos = 0
    truth_frames, answer_frames, summary, files = [], [], {}, {}

    for wave in WAVES:
        n_dup = round(BAD_RATES["duplicate"] * n_per_wave)
        n_base = n_per_wave - n_dup
        # panel members are drawn from a large pool; some recur across waves
        ids = rng.choice(pool[: 60_000], size=n_base, replace=False)
        df = generate_wave(rng, wave, n_base, cal, ids)
        df = plant_bad_respondents(rng, df, n_dup)
        df = assign_records(rng, df)

        # Undamaged answers (truth for ingest tests) and hidden traits
        answers = df.reindex(columns=["wave"] + raw_columns(wave)).copy()
        answer_frames.append(answers)
        truth_frames.append(df.reindex(columns=TRUTH_COLS))

        raw = to_raw(df, wave)
        if wave == 1:
            nc_ids = pool[60_000 + pool_pos: 60_000 + pool_pos + sum(NONCOMPLETES_W1.values())]
            pool_pos += len(nc_ids)
            nc = make_noncompletes(rng, wave, first_record=len(df) + 1, panel_ids=nc_ids)
            raw = damage_w1(rng, raw, nc)
            path = raw_dir / "W1_export.csv"
            raw.to_csv(path, index=False, encoding="utf-8")
        elif wave == 2:
            raw = damage_w2(rng, raw)
            path = raw_dir / "W2_export.csv"
            raw.to_csv(path, index=False, encoding="latin-1", errors="replace")
        elif wave == 3:
            raw = damage_w3(rng, raw)
            path = raw_dir / "W3_export.csv"
            raw.to_csv(path, index=False, encoding="utf-8")
        else:
            raw = damage_w4(rng, raw)
            path = write_w4(raw, raw_dir / "W4_export.sav", codebook)
        files[path.name] = sha256(path)

        cust = df[df["mosaik_cust"] == 1]
        genuine = cust[cust["should_remove"] == 0]
        summary[wave] = {
            "raw_rows": int(len(raw)),
            "raw_completes": int(len(df)),
            "should_remove": int(df["should_remove"].sum()),
            "mosaik_customers": int(len(cust)),
            "mosaik_share": round(float(df["mosaik_cust"].mean()), 3),
            "true_nps_unweighted_kept": round(nps_of(genuine["c1_nps"]), 1),
            "mosaik_main_given_user": round(float(cust["mosaik_main"].mean()), 3),
            "sparhaus_main_given_user": round(float((df.loc[df["a1_r4"] == 1, "main_bank"] == 4).mean()), 3),
            "flags": {f: int(df[f].sum()) for f in TRUTH_COLS if f.startswith("true_")},
        }
        log.info("W%d: %d raw rows, %d completes, %d to remove, %d customers, NPS %.1f",
                 wave, summary[wave]["raw_rows"], len(df), summary[wave]["should_remove"],
                 len(cust), summary[wave]["true_nps_unweighted_kept"])

    # --- truth files --------------------------------------------------------------
    truth = pd.concat(truth_frames, ignore_index=True)
    truth.to_csv(truth_dir / "respondents_truth.csv", index=False)
    pd.concat(answer_frames, ignore_index=True).to_csv(truth_dir / "true_answers.csv", index=False)

    drivers = pd.DataFrame([{"attribute": a, "item": ATTR_ITEM[a], "true_weight": w,
                             "true_rank": r}
                            for r, (a, w) in enumerate(sorted(DRIVER_WEIGHTS.items(),
                                                              key=lambda kv: -kv[1]), start=1)])
    drivers.to_csv(truth_dir / "driver_weights.csv", index=False)

    effects = []
    for a in ATTRS:
        for w in WAVES:
            effects.append({"attribute": a, "item": ATTR_ITEM[a], "wave": w,
                            "delta_all": WAVE_DELTA.get(a, [0, 0, 0, 0])[w - 1],
                            "delta_main_extra": FEES_MAIN_EXTRA_W3 if (a == "fees" and w == 3) else 0.0,
                            "delta_under30": APP_UNDER30_W4 if (a == "app" and w == 4) else None})
    pd.DataFrame(effects).to_csv(truth_dir / "wave_effects.csv", index=False)

    for p in sorted(truth_dir.glob("*.csv")):
        files[f"_truth/{p.name}"] = sha256(p)
    meta = {
        "seed": seed, "n_per_wave": n_per_wave,
        "calibration": cal.__dict__,
        "nps_expected_unweighted_large_sample": {"W1": 24, "W2": 24, "W3": 4, "W4": 16},
        "bad_rates": BAD_RATES,
        "removal_rule": "remove if attention OR duplicate OR inconsistent, or 2+ of speeder/straightliner/gibberish",
        "waves": summary,
        "sha256": files,
        "note": "All brands and data are simulated. The pipeline must not read data/_truth/.",
    }
    with open(truth_dir / "simulation_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-per-wave", type=int, default=N_PER_WAVE)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")
    meta = simulate(args.seed, args.n_per_wave, args.out_dir)
    log.info("Done. Files written to %s", args.out_dir)
    for w, s in meta["waves"].items():
        log.info("  W%s flags: %s", w, s["flags"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
