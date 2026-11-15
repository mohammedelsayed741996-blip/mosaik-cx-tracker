#!/usr/bin/env python3
"""
08_charts_deck.py - Mosaik Bank CX Tracker 2026: client charts, deck and README.

Builds the client presentation from the pipeline outputs:

  12 slides (16:9, 300 dpi PNG)   outputs/charts/NN_<name>.png
  chart data (one CSV per chart)   outputs/charts/data/NN_<name>.csv
  deck                             outputs/Mosaik_CX_Tracker_2026.pdf (+ .pptx)
  README                           the 3 strongest charts and headline findings
                                   between the KEY-CHARTS markers in README.md

Chart standards (applied by the helpers below, so no chart can skip them):
  * the title states the finding with a number, and is written FROM THE DATA:
    if a result does not support the expected story, the title says what the
    data says instead
  * a subtitle gives the question and scale; a footer gives base, weighting,
    significance rule and source on every chart
  * Mosaik is the one highlight colour (blue); competitors and earlier waves
    are neutral grey; text never takes a series colour
  * significance is visible (filled marker / bold label) and explained
  * bases under 50 are flagged, under 30 suppressed
  * no 3D, no pies, zero baselines for bars, thin marks, hairline grid

Inputs (run after steps 04-07):
  data/clean/tracker_weighted.csv, outputs/tables/{banners_long,nps_by_wave,
  nps_by_wave_group,drivers}.csv, outputs/tables/drivers_summary.json,
  outputs/quality_metrics.json (optional, for the data-quality appendix)

Usage
-----
    python src/08_charts_deck.py
    python src/08_charts_deck.py --no-pptx --no-readme
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402

log = logging.getLogger("deck")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean" / "tracker_weighted.csv"
TABLES = ROOT / "outputs" / "tables"
QUALITY = ROOT / "outputs" / "quality_metrics.json"
CHARTS = ROOT / "outputs" / "charts"
DECK_STEM = ROOT / "outputs" / "Mosaik_CX_Tracker_2026"
README = ROOT / "README.md"

# --- palette (validated reference palette; Mosaik = categorical slot 1) --------
MOSAIK = "#2a78d6"        # the one highlight
SECOND = "#eb6834"        # a second series that needs its own identity
NEUTRAL = "#b4b2ab"       # competitors, earlier waves, context
DETRACTOR, PASSIVE, PROMOTER = "#e34948", "#c3c2b7", "#2a78d6"   # diverging: red / grey / blue
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

W, H, DPI = 13.333, 7.5, 300                          # 16:9 at 300 dpi -> 4000 x 2250 px
SOURCE = "Source: Mosaik Bank CX Tracker 2026 (simulated data; fictional brands). Weighted to Destatis 31.12.2025."
SIG_RULE = "Significance: 95% level, z-test on effective bases."
WAVES = {1: "W1\nMar 26", 2: "W2\nJun 26", 3: "W3\nSep 26", 4: "W4\nDec 26"}
WAVES_SHORT = {1: "W1", 2: "W2", 3: "W3", 4: "W4"}
ATTR = {1: "App ease", 2: "Transfer speed", 3: "Fees / value for money", 4: "Customer service",
        5: "Security & trust", 6: "Card features", 7: "Savings interest", 8: "Investing features"}
CODES = {1: "Fees / value for money", 2: "App", 3: "Customer service", 4: "Security / trust",
         5: "Transfers", 6: "Card", 7: "Interest / investing", 8: "Other / general"}
BRANDS = {1: "Mosaik Bank", 2: "Volta", 3: "Atlas Direkt", 4: "Sparhaus", 5: "Pebble"}
SHORT = {"fees": "Value for money", "app": "App ease", "security": "Security & trust",
         "service": "Customer service", "card": "Card features", "interest": "Savings interest",
         "speed": "Transfer speed"}
FLAG_BASE, MIN_BASE = 50, 30

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 12, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 13, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


# =============================================================================
# Data
# =============================================================================

class Inputs:
    def __init__(self, data: Path, tables: Path, quality: Path):
        need = [data, tables / "banners_long.csv", tables / "nps_by_wave.csv", tables / "drivers.csv"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing inputs {missing}: run steps 04-06 first")
        self.d = pd.read_csv(data, low_memory=False)
        self.banners = pd.read_csv(tables / "banners_long.csv", keep_default_na=False,
                                   na_values=[""]).assign(sig=lambda x: x["sig"].fillna(""))
        self.nps = pd.read_csv(tables / "nps_by_wave.csv")
        grp = tables / "nps_by_wave_group.csv"
        self.nps_group = pd.read_csv(grp) if grp.exists() else None
        self.drivers = pd.read_csv(tables / "drivers.csv")
        js = tables / "drivers_summary.json"
        self.drivers_summary = json.loads(js.read_text()) if js.exists() else {}
        self.quality = json.loads(quality.read_text()) if quality.exists() else None
        self.latest = int(self.d["wave"].max())
        self.cust = self.d[self.d["mosaik_cust"] == 1]

    def wpct(self, frame: pd.DataFrame, col: str, value=None) -> float:
        g = frame[frame[col].notna()]
        x = g[col] if value is None else (g[col] == value).astype(float)
        return 100 * np.average(x, weights=g["weight"]) if len(g) else np.nan

    def base_line(self, frame: pd.DataFrame, label: str, by_wave=True) -> str:
        if by_wave:
            ns = frame.groupby("wave").size()
            return f"Base: {label}, n = " + " / ".join(f"{n:,}" for n in ns) + \
                   f" ({' / '.join(WAVES_SHORT[w] for w in ns.index)}, unweighted)."
        return f"Base: {label}, n = {len(frame):,} (unweighted)."

    def trend_cell(self, table: str, row: int, wave: int) -> pd.Series:
        b = self.banners
        return b[(b["banner"] == "trend") & (b["table_id"] == table) & (b["row_order"] == row) &
                 (b["column"].str.startswith(f"W{wave}"))].iloc[0]

    def higher_than(self, table: str, row: int, wave_a: int, wave_b: int) -> bool:
        """Is wave_a significantly higher than wave_b on the trend banner?"""
        a, b = self.trend_cell(table, row, wave_a), self.trend_cell(table, row, wave_b)
        return b["letter"] in (a["sig"] or "")


# =============================================================================
# Slide helpers (every chart goes through these, so the standards always apply)
# =============================================================================

class Deck:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        (out_dir / "data").mkdir(parents=True, exist_ok=True)
        self.slides: list[dict] = []

    def slide(self, title: str, subtitle: str = ""):
        fig = plt.figure(figsize=(W, H), dpi=100)
        t = fig.text(0.045, 0.915, title, fontsize=23, fontweight="bold", color=INK, va="bottom")
        renderer = fig.canvas.get_renderer()
        max_w = 0.91 * fig.bbox.width
        size = 23.0
        while t.get_window_extent(renderer).width > max_w and size > 18:
            size -= 0.5
            t.set_fontsize(size)
        if t.get_window_extent(renderer).width > max_w:          # last resort: two lines
            words, lines, cur = title.split(), [], ""
            for wd in words:
                trial = f"{cur} {wd}".strip()
                t.set_text(trial)
                if t.get_window_extent(renderer).width > max_w and cur:
                    lines.append(cur)
                    cur = wd
                else:
                    cur = trial
            lines.append(cur)
            t.set_text("\n".join(lines))
            t.set_y(0.905)
            log.warning("Title wrapped to %d lines, consider shortening: %s", len(lines), title)
        if subtitle:
            fig.text(0.045, 0.875, subtitle, fontsize=13.5, color=INK2, va="top")
        return fig

    def save(self, fig, name: str, title: str, base: str = "", notes: str = "", data: pd.DataFrame | None = None,
             footer: bool = True):
        n = len(self.slides) + 1
        if footer:
            lines = [s for s in (base, notes) if s] + [SOURCE + " " + SIG_RULE if notes != "-" else SOURCE]
            fig.text(0.045, 0.035, "\n".join(l for l in lines if l != "-"), fontsize=9.5, color=MUTED,
                     va="bottom", linespacing=1.45)
        fig.text(0.955, 0.035, f"{n}", fontsize=10, color=MUTED, ha="right", va="bottom")
        fig.add_artist(Line2D([0.045, 0.955], [0.135 if footer else 0.08] * 2, color=GRID, lw=0.8))
        path = self.out_dir / f"{n:02d}_{name}.png"
        fig.savefig(path, dpi=DPI)
        if data is not None:
            data.to_csv(self.out_dir / "data" / f"{n:02d}_{name}.csv", index=False)
        self.slides.append({"n": n, "name": name, "title": title, "png": path, "fig": fig,
                            "notes": "\n".join(s for s in (title, base, notes, SOURCE) if s and s != "-")})
        return path


def style_axes(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(grid_axis != "y")
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=12)


def plot_area(fig, rect=(0.07, 0.2, 0.86, 0.62)):
    return fig.add_axes(rect)


def nps_estimate(frame: pd.DataFrame, n_design: int) -> tuple[float, float, int]:
    """Weighted NPS and its linearisation SE for a domain of a design with n_design rows."""
    w, y = frame["weight"].to_numpy(), frame["nps_score"].to_numpy(float)
    if len(y) < MIN_BASE:
        return np.nan, np.nan, len(y)
    m = np.average(y, weights=w)
    se = np.sqrt(n_design / (n_design - 1) * np.sum((w * (y - m)) ** 2)) / w.sum()
    return m, se, len(y)


def p_two_sided(diff: float, se: float) -> float:
    from scipy.stats import norm
    return float(2 * norm.sf(abs(diff / se)))


def spread_labels(y: np.ndarray, gap: float, lo: float = -np.inf) -> np.ndarray:
    """Nudge label positions apart (keeping their order) so no two are closer than `gap`."""
    order = np.argsort(y)
    out = y.astype(float).copy()
    prev = lo - gap
    for i in order:
        out[i] = max(out[i], prev + gap)
        prev = out[i]
    return out


def signed(x: float) -> str:
    return f"{x:+.0f}"


# =============================================================================
# The 12 slides
# =============================================================================

def s_exec_summary(deck: Deck, I: Inputs, facts: dict):
    title = (f"The card fee cost {facts['drop']:.0f} NPS points; the app relaunch won back {facts['rec']:.0f}"
             if facts["drop"] > 0 and facts["rec"] > 0 else f"NPS is {signed(facts['nps_latest'])} in W{I.latest}")
    fig = deck.slide("Executive summary", "Mosaik Bank customer experience, four quarterly waves of 2026")
    tiles = [(signed(facts["nps_latest"]), f"NPS in W{I.latest}", f"{signed(facts['vs_w1'])} vs W1"),
             (f"{-facts['drop']:+.0f}", "NPS change after the Q3 card fee", "W2 to W3, significant" if facts["drop_sig"] else "W2 to W3"),
             (f"{facts['rec']:+.0f}", "NPS change after the Q4 app relaunch", "W3 to W4, significant" if facts["rec_sig"] else "W3 to W4")]
    for i, (big, label, sub) in enumerate(tiles):
        x = 0.045 + i * 0.31
        fig.text(x, 0.66, big, fontsize=54, fontweight="bold", color=INK)
        fig.text(x, 0.6, label, fontsize=14, color=INK)
        fig.text(x, 0.565, sub, fontsize=12, color=INK2)
    fig.text(0.045, 0.47, "What to do", fontsize=15, fontweight="bold")
    for i, rec in enumerate(facts["recommendations"]):
        fig.text(0.045, 0.415 - i * 0.075, f"{i + 1}.  {rec}", fontsize=14, color=INK, va="top")
    deck.save(fig, "executive_summary", title, I.base_line(I.cust, "Mosaik customers"),
              data=pd.DataFrame([facts]).drop(columns=["recommendations"]))


def s_nps_trend(deck: Deck, I: Inputs, facts: dict):
    t = I.nps.sort_values("wave")
    if facts["drop"] > 0 and facts["drop_sig"]:
        title = f"NPS fell {facts['drop']:.0f} points after the Q3 card fee"
        title += f" and won back {facts['rec']:.0f} in Q4" if facts["rec"] > 0 else ""
    else:
        title = f"NPS moved from {signed(t.nps.iloc[0])} to {signed(t.nps.iloc[-1])} over 2026"
    fig = deck.slide(title, "Net Promoter Score (% promoters 9-10 minus % detractors 0-6), with 95% confidence interval")
    ax = plot_area(fig)
    x = np.arange(len(t))
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(x, t.nps, color=MOSAIK, lw=2, zorder=2)
    ax.vlines(x, t.ci_low, t.ci_high, color=MOSAIK, lw=2, alpha=0.35, zorder=1)
    for xi, (_, r) in zip(x, t.iterrows()):
        sig = bool(r.get("sig_vs_prev", False))
        ax.scatter(xi, r.nps, s=130, zorder=3, color=MOSAIK if sig else SURFACE,
                   edgecolor=MOSAIK, linewidth=2)
        ax.text(xi, r.ci_high + 2.5, signed(r.nps), ha="center", fontsize=17, fontweight="bold")
        if xi > 0 and not pd.isna(r.change_vs_prev):
            ax.text(xi - 0.5, (r.nps + t.nps.iloc[xi - 1]) / 2, f"{r.change_vs_prev:+.0f}" + ("*" if sig else ""),
                    ha="center", va="center", fontsize=11.5, color=INK2,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
    ax.annotate("Card fee introduced", xy=(1.5, ax.get_ylim()[1]), xytext=(1.5, 47), ha="center",
                fontsize=11, color=INK2)
    ax.annotate("App relaunch", xy=(2.5, 47), ha="center", fontsize=11, color=INK2)
    for xv in (1.5, 2.5):
        ax.axvline(xv, color=GRID, lw=1, zorder=0)
    ax.set_xticks(x, [WAVES[w] for w in t.wave])
    ax.set_ylim(min(-10, t.ci_low.min() - 5), 50)
    ax.set_xlim(-0.5, len(t) - 0.5)
    style_axes(ax)
    ax.set_ylabel("NPS")
    deck.save(fig, "nps_trend", title, I.base_line(I.cust, "Mosaik customers"),
              "Filled marker / * = significant change vs previous wave (design-based SE, independent samples).",
              data=t)


def s_nps_composition(deck: Deck, I: Inputs):
    rows = []
    for w, g in I.cust.groupby("wave"):
        rows.append({"wave": w, "Detractors (0-6)": I.wpct(g, "nps_group", 1),
                     "Passives (7-8)": I.wpct(g, "nps_group", 2), "Promoters (9-10)": I.wpct(g, "nps_group", 3)})
    t = pd.DataFrame(rows)
    d_ch = t["Detractors (0-6)"].iloc[2] - t["Detractors (0-6)"].iloc[1]
    p_ch = t["Promoters (9-10)"].iloc[2] - t["Promoters (9-10)"].iloc[1]
    if abs(d_ch - abs(p_ch)) < 2:
        title = (f"The W3 drop was two-sided: promoters {p_ch:+.0f} points, detractors {d_ch:+.0f} points")
    elif d_ch > abs(p_ch):
        title = (f"The W3 drop came from new detractors: {t['Detractors (0-6)'].iloc[1]:.0f}% "
                 f"to {t['Detractors (0-6)'].iloc[2]:.0f}%, while promoters moved {p_ch:+.0f} points")
    else:
        title = f"The W3 drop came from fewer promoters ({p_ch:+.0f} points) more than new detractors ({d_ch:+.0f})"
    fig = deck.slide(title, "Share of Mosaik customers by likelihood to recommend (0-10), %")
    ax = plot_area(fig, (0.12, 0.2, 0.8, 0.6))
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for col, color in (("Detractors (0-6)", DETRACTOR), ("Passives (7-8)", PASSIVE), ("Promoters (9-10)", PROMOTER)):
        ax.barh(y, t[col], left=left, height=0.58, color=color, edgecolor=SURFACE, linewidth=2, label=col)
        for yi, l, v in zip(y, left, t[col]):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=13,
                        color=INK if color == PASSIVE else "#ffffff", fontweight="bold")
        left += t[col].to_numpy()
    ax.set_yticks(y, [WAVES[w].replace("\n", " ") for w in t.wave])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style_axes(ax, "x")
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=12)
    deck.save(fig, "nps_composition", title, I.base_line(I.cust, "Mosaik customers"), data=t)


def s_attributes(deck: Deck, I: Inputs):
    prev, last = I.latest - 1, I.latest
    rows = []
    for i in range(1, 9):
        col = f"c3_t2b_r{i}"
        rows.append({"attribute": ATTR[i], "row": i,
                     "W2": I.wpct(I.cust[I.cust.wave == 2], col), f"W{prev}": I.wpct(I.cust[I.cust.wave == prev], col),
                     f"W{last}": I.wpct(I.cust[I.cust.wave == last], col),
                     "up_sig": I.higher_than("T02", i, last, prev), "down_sig": I.higher_than("T02", i, prev, last)})
    t = pd.DataFrame(rows).sort_values(f"W{last}")
    fees, app = t.set_index("row").loc[3], t.set_index("row").loc[1]
    fee_drop = fees["W2"] - fees[f"W{prev}"]
    app_up = app[f"W{last}"] - app[f"W{prev}"]
    title = (f"Fees satisfaction fell {fee_drop:.0f} points after the card fee; app ease rose {app_up:.0f} after the relaunch"
             if fee_drop > 3 and app_up > 3 else
             f"App ease leads at {app[f'W{last}']:.0f}% satisfied; fees trail at {fees[f'W{last}']:.0f}%")
    fig = deck.slide(title, f"Satisfied + very satisfied (top-2-box), % of Mosaik customers, W{prev} vs W{last}")
    ax = plot_area(fig, (0.24, 0.2, 0.68, 0.62))
    y = np.arange(len(t))
    ax.hlines(y, t[f"W{prev}"], t[f"W{last}"], color=AXIS, lw=2, zorder=1)
    ax.scatter(t[f"W{prev}"], y, s=110, color=SURFACE, edgecolor=NEUTRAL, linewidth=2, zorder=2, label=f"W{prev}")
    ax.scatter(t[f"W{last}"], y, s=130, color=MOSAIK, edgecolor=SURFACE, linewidth=2, zorder=3, label=f"W{last}")
    for yi, (_, r) in zip(y, t.iterrows()):
        mark = " ▲" if r.up_sig else (" ▼" if r.down_sig else "")
        right = max(r[f"W{prev}"], r[f"W{last}"])
        ax.text(right + 2, yi, f"{r[f'W{last}']:.0f}%{mark}", va="center", fontsize=12.5,
                fontweight="bold" if mark else "normal")
    ax.set_yticks(y, t["attribute"])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style_axes(ax, "x")
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    base = I.base_line(I.cust[I.cust.wave.isin([prev, last])], "Mosaik customers") + " Don't know excluded."
    deck.save(fig, "attribute_satisfaction", title, base,
              f"▲/▼ = significant change W{prev} to W{last}. Investing features asked from W3.", data=t)


def s_drivers(deck: Deck, I: Inputs):
    t = I.drivers.sort_values("share_of_r2")
    top2 = t.sort_values("rank").head(2)
    title = (f"{SHORT[top2.attribute.iloc[0]]} and {SHORT[top2.attribute.iloc[1]].lower()} explain "
             f"{100 * top2.share_of_r2.sum():.0f}% of what drives NPS")
    fig = deck.slide(title, "Relative importance: share of explained variance in likelihood to recommend "
                            "(Shapley regression), with 95% bootstrap interval")
    ax = plot_area(fig, (0.24, 0.2, 0.68, 0.62))
    y = np.arange(len(t))
    colors = [MOSAIK if r <= 3 else NEUTRAL for r in t["rank"]]
    ax.barh(y, 100 * t.share_of_r2, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    if "share_ci_low" in t:
        ax.hlines(y, 100 * t.share_ci_low, 100 * t.share_ci_high, color=INK2, lw=1.2)
    for yi, v, hi in zip(y, 100 * t.share_of_r2, 100 * t.get("share_ci_high", t.share_of_r2)):
        ax.text(max(v, hi) + 1, yi, f"{v:.0f}%", va="center", fontsize=12.5)
    ax.set_yticks(y, t["label"])
    ax.set_xlim(0, max(55, 100 * t.get("share_ci_high", t.share_of_r2).max() + 8))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style_axes(ax, "x")
    s = I.drivers_summary
    base = (f"Base: Mosaik customers W1-W{I.latest} pooled who rated all 7 attributes, n = {s.get('n', 0):,} "
            f"(unweighted). Model R² = {s.get('r2', float('nan')):.2f}.")
    stab = s.get("bootstrap") or {}
    note = (f"Top 3 order identical in {100 * stab['share_same_top3_order']:.0f}% of {stab['reps']} bootstrap resamples."
            if stab else "")
    deck.save(fig, "key_drivers", title, base, note, data=t)


def s_priority(deck: Deck, I: Inputs):
    t = I.drivers.copy()
    t["importance"] = 100 * t.share_of_r2
    top = t.sort_values("rank").iloc[0]
    fix = t[t.quadrant == "Fix first"].sort_values("rank")
    if len(fix):
        f0 = fix.iloc[0]
        title = (f"{SHORT[f0.attribute]} is the priority: the {'top' if f0['rank'] == 1 else 'a major'} driver, "
                 f"with only {f0.perf_t2b_latest:.0f}% satisfied")
    else:
        title = f"No major driver underperforms; protect {SHORT[top.attribute].lower()}"
    fig = deck.slide(title, f"Importance (share of explained NPS variance) vs performance (top-2-box, W{I.latest})")
    ax = plot_area(fig, (0.08, 0.2, 0.84, 0.62))
    imp_cut = 100 / len(t)
    perf_cut = t.perf_t2b_latest.mean()
    ax.axhline(imp_cut, color=AXIS, lw=1)
    ax.axvline(perf_cut, color=AXIS, lw=1)
    xmax, ymax = 100, max(50, t.importance.max() + 6)
    for (tx, ty, lab, ha) in ((2, ymax - 2, "FIX FIRST", "left"), (xmax - 2, ymax - 2, "MAINTAIN", "right"),
                              (2, 1.5, "MONITOR", "left"), (xmax - 2, 1.5, "SECONDARY", "right")):
        ax.text(tx, ty, lab, ha=ha, va="top" if ty > 10 else "bottom", fontsize=11, color=MUTED, fontweight="bold")
    for _, r in t.iterrows():
        hl = r.quadrant == "Fix first"
        ax.scatter(r.perf_t2b_latest, r.importance, s=180 if hl else 120, color=MOSAIK if hl else NEUTRAL,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(r.label, (r.perf_t2b_latest, r.importance), xytext=(9, 6), textcoords="offset points",
                    fontsize=12, fontweight="bold" if hl else "normal")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(f"Performance: % satisfied, W{I.latest}")
    ax.set_ylabel("Importance: % of explained variance")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style_axes(ax, "both")
    base = (f"Base: importance from Mosaik customers W1-W{I.latest} pooled (n = {I.drivers_summary.get('n', 0):,}); "
            f"performance from W{I.latest} customers (n = {(I.cust.wave == I.latest).sum():,}).")
    deck.save(fig, "priority_matrix", title, base,
              f"Lines at average importance ({imp_cut:.0f}%) and average performance ({perf_cut:.0f}%).", data=t)


def nps_lines(ax, frame: pd.DataFrame, color, label, show_label=True, lw=2):
    f = frame.sort_values("wave")
    x = f.wave.to_numpy() - 1
    ax.plot(x, f.nps, color=color, lw=lw, label=label)
    for xi, (_, r) in zip(x, f.iterrows()):
        small = bool(r.get("small_base", False))
        ax.scatter(xi, r.nps, s=60, color=SURFACE if small else color, edgecolor=color, linewidth=2, zorder=3)
    if show_label:
        ax.text(x[-1] + 0.12, f.nps.iloc[-1], f"{label}: NPS {signed(f.nps.iloc[-1])}", va="center", fontsize=12.5)


def s_main_secondary(deck: Deck, I: Inputs):
    g = I.nps_group
    if g is None:
        return
    g = g[g.group == "mosaik_main"]
    m, s = g[g.subgroup == "Main bank"].set_index("wave"), g[g.subgroup == "Secondary"].set_index("wave")
    dm, ds = m.nps[2] - m.nps[3], s.nps[2] - s.nps[3]
    se_dd = np.sqrt(m.se[2] ** 2 + m.se[3] ** 2 + s.se[2] ** 2 + s.se[3] ** 2)
    p_dd = p_two_sided(dm - ds, se_dd)
    if dm > 0 and ds > 0 and p_dd >= 0.05:
        title = f"Main-bank and secondary customers both lost about {np.mean([dm, ds]):.0f} NPS points after the fee"
    elif dm > ds:
        title = f"Main-bank customers lost more after the fee: {dm:.0f} NPS points vs {ds:.0f} for secondary users"
    else:
        title = f"Secondary users lost more after the fee: {ds:.0f} NPS points vs {dm:.0f} for main-bank customers"
    fig = deck.slide(title, "NPS by Mosaik relationship and wave")
    ax = plot_area(fig, (0.07, 0.2, 0.76, 0.62))
    ax.axhline(0, color=AXIS, lw=1)
    nps_lines(ax, m.reset_index(), MOSAIK, "Main bank")
    nps_lines(ax, s.reset_index(), SECOND, "Secondary")
    ax.set_xticks(range(4), [WAVES[w] for w in range(1, 5)])
    ax.set_xlim(-0.3, 3.3)
    style_axes(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    sig_m = bool(m.loc[3, "sig_vs_prev"]) if "sig_vs_prev" in m else False
    sig_s = bool(s.loc[3, "sig_vs_prev"]) if "sig_vs_prev" in s else False
    base = ("Base: Mosaik customers, main bank n = " + " / ".join(f"{int(n)}" for n in m.n_unweighted) +
            "; secondary n = " + " / ".join(f"{int(n)}" for n in s.n_unweighted) + " (W1-W4, unweighted).")
    note = (f"W2 to W3 change significant for main bank: {'yes' if sig_m else 'no'}; "
            f"for secondary: {'yes' if sig_s else 'no'}. Difference between the two drops: p = {p_dd:.2f}.")
    deck.save(fig, "nps_main_secondary", title, base, note, data=g)


def s_age(deck: Deck, I: Inputs):
    """Under 30 vs 30+. Single age bands have 45-250 customers per wave (SE up to 12 points), too
    noisy to compare wave on wave, so the slide uses two groups and tests the difference in recovery."""
    N = len(I.d)
    rows = []
    for lab, keep in (("Under 30", I.cust.age < 30), ("30 and over", I.cust.age >= 30)):
        for w in sorted(I.cust.wave.unique()):
            m, se, n = nps_estimate(I.cust[keep & (I.cust.wave == w)], N)
            rows.append({"group": lab, "wave": w, "nps": m, "se": se, "n_unweighted": n,
                         "small_base": n < FLAG_BASE})
    t = pd.DataFrame(rows)
    u, o = t[t.group == "Under 30"].set_index("wave"), t[t.group == "30 and over"].set_index("wave")
    ru, ro = u.nps[4] - u.nps[3], o.nps[4] - o.nps[3]
    p_dd = p_two_sided(ru - ro, np.sqrt(u.se[3] ** 2 + u.se[4] ** 2 + o.se[3] ** 2 + o.se[4] ** 2))
    if p_dd < 0.05:
        lead = "Under-30s" if ru > ro else "Customers aged 30+"
        title = f"{lead} recovered faster in W4: {max(ru, ro):+.0f} NPS points vs {min(ru, ro):+.0f}"
    else:
        title = f"Both age groups recovered in W4 (under 30 {ru:+.0f}, 30+ {ro:+.0f}): no significant gap"
    fig = deck.slide(title, "NPS by age group and wave")
    ax = plot_area(fig, (0.07, 0.2, 0.74, 0.62))
    ax.axhline(0, color=AXIS, lw=1)
    nps_lines(ax, u.reset_index(), MOSAIK, "Under 30")
    nps_lines(ax, o.reset_index(), SECOND, "30 and over")
    ax.set_xticks(range(4), [WAVES[w] for w in range(1, 5)])
    ax.set_xlim(-0.3, 3.3)
    style_axes(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    base = ("Base: Mosaik customers, under 30 n = " + " / ".join(str(int(n)) for n in u.n_unweighted) +
            "; 30 and over n = " + " / ".join(str(int(n)) for n in o.n_unweighted) + " (W1-W4, unweighted).")
    note = (f"Difference between the two W3 to W4 changes: p = {p_dd:.2f}. "
            "Single age bands are too small for wave-on-wave reading (see banner tables).")
    deck.save(fig, "nps_by_age", title, base, note, data=t)


def s_funnel(deck: Deck, I: Inputs):
    d = I.d[I.d.wave == I.latest]
    rows = []
    for b, name in BRANDS.items():
        rows.append({"brand": name,
                     "Aware": I.wpct(d, f"b1_r{b}"),
                     "Would consider": 100 * np.average((d[f"b2_r{b}"] == 1), weights=d.weight),
                     "Uses": I.wpct(d, f"a1_r{b}"),
                     "Main bank": 100 * np.average((d.main_bank == b), weights=d.weight)})
    t = pd.DataFrame(rows)
    t["main_of_users"] = 100 * t["Main bank"] / t["Uses"]
    mo = t.set_index("brand").loc["Mosaik Bank"]
    best = t.sort_values("main_of_users").iloc[-1]
    title = (f"Mosaik turns only {mo.main_of_users / 10:.0f} in 10 users into main-bank customers; "
             f"{best.brand} turns {best.main_of_users / 10:.0f} in 10")
    fig = deck.slide(title, f"Brand funnel, % of all respondents, W{I.latest}")
    ax = plot_area(fig, (0.07, 0.2, 0.72, 0.62))
    stages = ["Aware", "Would consider", "Uses", "Main bank"]
    x = np.arange(len(stages))
    label_y = spread_labels(t["Main bank"].to_numpy(), gap=5.0, lo=1.0)
    for (_, r), ly in zip(t.iterrows(), label_y):
        mo_ = r.brand == "Mosaik Bank"
        ax.plot(x, r[stages].astype(float), color=MOSAIK if mo_ else NEUTRAL, lw=2.5 if mo_ else 1.5,
                marker="o", ms=8 if mo_ else 5, mec=SURFACE, zorder=3 if mo_ else 2)
        ax.plot([x[-1] + 0.02, x[-1] + 0.1], [r["Main bank"], ly], color=GRID, lw=0.8)
        ax.text(x[-1] + 0.12, ly, f"{r.brand}  {r['Main bank']:.0f}%  ({r.main_of_users:.0f}% of users)",
                va="center", fontsize=11.5, fontweight="bold" if mo_ else "normal", color=INK)
    for xi, v in zip(x, mo[stages]):
        ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=12.5, fontweight="bold")
    ax.set_xticks(x, stages)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.2, 3.2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style_axes(ax)
    deck.save(fig, "brand_funnel", title, I.base_line(d, f"all respondents W{I.latest}", by_wave=False),
              "Label at right: main-bank share of all respondents and of the brand's own users.", data=t)


def s_switching(deck: Deck, I: Inputs):
    c = I.cust[I.cust.wave == I.latest].assign(sw=lambda x: (x.c5_switch >= 4).astype(float).where(x.c5_switch.notna()))
    rows = [{"group": lab, "n": int((c.nps_group == k).sum()), "likely_to_switch": I.wpct(c[c.nps_group == k], "sw")}
            for k, lab in ((1, "Detractors (0-6)"), (2, "Passives (7-8)"), (3, "Promoters (9-10)"))]
    t = pd.DataFrame(rows)
    det, prom = t.likely_to_switch.iloc[0], t.likely_to_switch.iloc[2]
    words = {2: "One in two", 3: "One in three", 4: "One in four", 5: "One in five"}
    k = round(100 / det) if det > 0 else 0
    frac = f"{words[k]}" if k in words and abs(det - 100 / k) < 3 else f"{det:.0f}% of"
    title = f"{frac} detractors plan to switch their main account within a year, vs {prom:.0f}% of promoters"
    fig = deck.slide(title, "Likely or very likely to switch main account in the next 12 months, %")
    ax = plot_area(fig, (0.2, 0.22, 0.7, 0.55))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.likely_to_switch, height=0.55, color=[DETRACTOR, NEUTRAL, NEUTRAL], edgecolor=SURFACE, linewidth=2)
    for yi, v, n in zip(y, t.likely_to_switch, t.n):
        ax.text(v + 1, yi, f"{v:.0f}%" + ("*" if n < FLAG_BASE else ""), va="center", fontsize=14, fontweight="bold")
    ax.set_yticks(y, t.group)
    ax.set_xlim(0, max(60, t.likely_to_switch.max() + 10))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style_axes(ax, "x")
    base = f"Base: Mosaik customers W{I.latest}: " + ", ".join(f"{g.split(' (')[0].lower()} n = {n}" for g, n in zip(t.group, t.n)) + "."
    deck.save(fig, "switching_by_nps", title, base, data=t)


def s_reasons(deck: Deck, I: Inputs):
    det = I.cust[I.cust.nps_group == 1]
    prev, last = I.latest - 1, I.latest
    rows = [{"code": CODES[k], f"W{prev}": I.wpct(det[det.wave == prev], f"c2_code_{k}"),
             f"W{last}": I.wpct(det[det.wave == last], f"c2_code_{k}")} for k in CODES]
    t = pd.DataFrame(rows).sort_values(f"W{last}")
    fees = t.set_index("code").loc["Fees / value for money"]
    top = t.iloc[-1]
    title = (f"Fees are detractors' top complaint in W{last}: {fees[f'W{last}']:.0f}% mention them "
             f"(W{prev}: {fees[f'W{prev}']:.0f}%)" if top.code.startswith("Fees") else
             f"Detractors most often mention {top.code.lower()} ({top[f'W{last}']:.0f}%)")
    fig = deck.slide(title, "Main reason for the score, coded open ends, % of detractors (multiple codes possible)")
    ax = plot_area(fig, (0.2, 0.2, 0.42, 0.62))
    y = np.arange(len(t))
    ax.barh(y + 0.18, t[f"W{prev}"], height=0.34, color=NEUTRAL, edgecolor=SURFACE, linewidth=2, label=f"W{prev}")
    ax.barh(y - 0.18, t[f"W{last}"], height=0.34, color=MOSAIK, edgecolor=SURFACE, linewidth=2, label=f"W{last}")
    for yi, v in zip(y, t[f"W{last}"]):
        ax.text(v + 1, yi - 0.18, f"{v:.0f}%", va="center", fontsize=11)
    ax.set_yticks(y, t.code)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlim(0, max(70, t[[f"W{prev}", f"W{last}"]].max().max() + 10))
    style_axes(ax, "x")
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    # three short verbatims from the latest wave, chosen deterministically (fees, app, service)
    fig.text(0.66, 0.78, "In their words (W{})".format(last), fontsize=14, fontweight="bold")
    quotes = []
    for k, grp, who in ((1, 1, "detractor"), (3, 1, "detractor"), (2, 3, "promoter")):
        pool = I.cust[(I.cust.wave == last) & (I.cust[f"c2_code_{k}"] == 1) & (I.cust.nps_group == grp)
                      & I.cust.c2_reason.notna()]
        texts = pool.c2_reason.str.strip()
        texts = texts[texts.str.len().between(12, 60)]
        if len(texts):                                  # the most common wording, as answered
            quotes.append((f"{CODES[k]} ({who})", texts.value_counts().sort_index().sort_values(
                ascending=False, kind="stable").index[0]))
    for i, (code, q) in enumerate(quotes):
        fig.text(0.66, 0.69 - i * 0.15, f"“{q.strip()}”", fontsize=14, color=INK, style="italic", wrap=True)
        fig.text(0.66, 0.65 - i * 0.15, code, fontsize=11, color=MUTED)
    n = det.groupby("wave").size()
    base = f"Base: detractors (0-6), W{prev} n = {n.get(prev, 0)}, W{last} n = {n.get(last, 0)} (unweighted)."
    deck.save(fig, "detractor_reasons", title, base, "Quotes are simulated answers from the data.", data=t)


def s_recommendations(deck: Deck, I: Inputs, facts: dict):
    title = "Recommendations and what to track in W1 2027"
    fig = deck.slide(title, "Three actions, each with an owner and a KPI from this tracker")
    items = [(a, o, k) for a, o, k in facts["actions"]]
    heads = ["Action", "Owner", "KPI to watch next wave"]
    xs = [0.045, 0.6, 0.73]
    for x, h in zip(xs, heads):
        fig.text(x, 0.78, h, fontsize=13, color=MUTED, fontweight="bold")
    for i, row in enumerate(items):
        yy = 0.7 - i * 0.14
        fig.add_artist(Line2D([0.045, 0.955], [yy + 0.045] * 2, color=GRID, lw=0.8))
        for x, txt, wgt in zip(xs, row, ("bold", "normal", "normal")):
            fig.text(x, yy, txt, fontsize=14, fontweight=wgt, va="top", wrap=True)
    fig.text(0.045, 0.25, "Questionnaire change for 2027: add a direct question on awareness of the card fee and "
             "whether the customer cancelled the card, to separate fee anger from product use.", fontsize=13,
             color=INK2, wrap=True)
    deck.save(fig, "recommendations", title, "-", "-")


def s_appendix_method(deck: Deck, I: Inputs):
    rows = []
    for w, g in I.d.groupby("wave"):
        wt = g.weight.to_numpy()
        deff = len(wt) * (wt ** 2).sum() / wt.sum() ** 2
        c = g[g.mosaik_cust == 1].weight.to_numpy()
        rows.append({"Wave": WAVES[w].replace("\n", " "), "Interviews": f"{len(g):,}",
                     "Customers": f"{len(c):,}", "Efficiency": f"{100 / deff:.0f}%",
                     "Effective n": f"{len(wt) / deff:,.0f}",
                     "Customers eff. n": f"{c.sum() ** 2 / (c ** 2).sum():,.0f}",
                     "Weight range": f"{wt.min():.2f}-{wt.max():.2f}"})
    t = pd.DataFrame(rows)
    title = "Appendix: method, sample and weighting"
    fig = deck.slide(title, "Online panel survey (CAWI), adults 18-69 in Germany with an online current account")
    ax = fig.add_axes((0.045, 0.5, 0.91, 0.31))
    ax.axis("off")
    tb = ax.table(cellText=t.values, colLabels=t.columns, loc="upper left", cellLoc="center",
                  colWidths=[1 / len(t.columns)] * len(t.columns))
    tb.auto_set_font_size(False)
    tb.set_fontsize(12.5)
    tb.scale(1, 2.1)
    for (r, _), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK2)
    bullets = ["Quarterly waves of about 3,000 interviews; about 5% removed by quality checks (see next slide).",
               "Weighting: raking per wave to Destatis 12411-0013 (31.12.2025), age band x gender x region, trimmed 0.3-3.0.",
               "Significance: z-tests on effective bases, 95% level, no multiple-comparison correction.",
               "NPS standard errors are design-based (R survey package); waves are independent samples."]
    for i, b in enumerate(bullets):
        fig.text(0.045, 0.45 - i * 0.055, "•  " + b, fontsize=13, color=INK)
    deck.save(fig, "appendix_method", title, "-", "-", data=t)


def s_appendix_quality(deck: Deck, I: Inputs):
    q = I.quality
    if not q:
        return
    labels = {"qc_speeder": "Speeder (soft)", "qc_straightliner": "Straightliner (soft)",
              "qc_gibberish": "Gibberish open end (soft)", "qc_attention": "Attention check failed (hard)",
              "qc_duplicate": "Duplicate panel ID (hard)", "qc_inconsistent": "Inconsistent answers (hard)",
              "qc_remove": "Removed"}
    t = pd.DataFrame(q["flags_by_wave"]).T[list(labels)].rename(columns=labels)
    t.insert(0, "Completes", [3000] * len(t)) if "completes" not in q else None
    t = t.T
    title = f"Appendix: {q['removed']:,} of {q['completes']:,} interviews ({100 * q['removed'] / q['completes']:.1f}%) removed by quality checks"
    fig = deck.slide(title, "Respondents flagged per check and wave; removed = any hard flag or 2+ soft flags")
    ax = fig.add_axes((0.045, 0.3, 0.91, 0.5))
    ax.axis("off")
    cells = [[r] + [f"{int(v):,}" for v in t.loc[r]] for r in t.index]
    tb = ax.table(cellText=cells, colLabels=["Check"] + list(t.columns), loc="upper left", cellLoc="center",
                  colWidths=[0.36] + [0.64 / len(t.columns)] * len(t.columns))
    tb.auto_set_font_size(False)
    tb.set_fontsize(12.5)
    tb.scale(1, 2.0)
    for (r, cidx), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        if r == 0 or (r == len(cells)):
            cell.set_text_props(fontweight="bold")
        if cidx == 0:
            cell.set_text_props(ha="left")
    deck.save(fig, "appendix_quality", title, "-", "One respondent can carry several flags. Full report: outputs/quality_report.md.")


# =============================================================================
# Facts used across slides (all computed, none typed)
# =============================================================================

def compute_facts(I: Inputs) -> dict:
    t = I.nps.set_index("wave")
    drop = t.nps[2] - t.nps[3]
    rec = t.nps[4] - t.nps[3]
    drv = I.drivers.sort_values("rank")
    fees = drv[drv.attribute == "fees"].iloc[0]
    app = drv[drv.attribute == "app"].iloc[0]
    service = drv[drv.attribute == "service"].iloc[0]
    facts = {
        "nps_latest": t.nps[I.latest], "vs_w1": t.nps[I.latest] - t.nps[1], "drop": drop, "rec": rec,
        "drop_sig": bool(t.sig_vs_prev[3]), "rec_sig": bool(t.sig_vs_prev[4]),
        "fees_t2b": fees.perf_t2b_latest, "fees_share": 100 * fees.share_of_r2,
        "app_t2b": app.perf_t2b_latest, "service_t2b": service.perf_t2b_latest,
    }
    facts["recommendations"] = [
        f"Fix the fee story first: value for money drives {facts['fees_share']:.0f}% of NPS and only "
        f"{facts['fees_t2b']:.0f}% are satisfied.",
        f"Keep the relaunched app as the advantage ({facts['app_t2b']:.0f}% satisfied) and use it to reach "
        f"main-bank customers.",
        f"Watch customer service ({facts['service_t2b']:.0f}% satisfied): contacts rose with the fee.",
    ]
    facts["actions"] = [
        ("Waive or bundle the €4.90 card fee for main-bank customers", "Product & pricing",
         f"Fees top-2-box (now {facts['fees_t2b']:.0f}%)"),
        ("Promote the new app to customers who use Mosaik as a secondary account", "Marketing / CRM",
         "Main-bank share among users"),
        ("Add service capacity and a fee FAQ in the app", "Customer service",
         f"Service top-2-box (now {facts['service_t2b']:.0f}%)"),
    ]
    return facts


# =============================================================================
# Deck outputs: PDF, PPTX, README
# =============================================================================

def title_slide(deck: Deck, I: Inputs):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.add_artist(Line2D([0.045, 0.2], [0.62, 0.62], color=MOSAIK, lw=6))
    fig.text(0.045, 0.52, "Mosaik Bank Customer Experience Tracker", fontsize=34, fontweight="bold")
    fig.text(0.045, 0.45, f"Results to W{I.latest} (December 2026)", fontsize=20, color=INK2)
    fig.text(0.045, 0.12, "Portfolio project. All survey data is simulated and all brands are fictional; "
             "population targets are official Destatis figures.", fontsize=11, color=MUTED)
    return fig


def write_pdf(deck: Deck, cover, path: Path):
    with PdfPages(path) as pdf:
        pdf.savefig(cover)
        for s in deck.slides:
            pdf.savefig(s["fig"])
        info = pdf.infodict()
        info["Title"] = "Mosaik Bank CX Tracker 2026"
        info["Subject"] = "Portfolio project with simulated data"


def write_pptx(deck: Deck, cover_png: Path, path: Path) -> bool:
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError:
        log.warning("python-pptx not installed: skipping %s (pip install python-pptx)", path.name)
        return False
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    for png, notes in [(cover_png, "Title")] + [(s["png"], s["notes"]) for s in deck.slides]:
        sl = prs.slides.add_slide(blank)
        sl.shapes.add_picture(str(png), 0, 0, width=prs.slide_width, height=prs.slide_height)
        sl.notes_slide.notes_text_frame.text = notes
    prs.save(path)
    return True


def update_readme(deck: Deck, readme: Path, facts: dict):
    pick = {s["name"]: s for s in deck.slides}
    chosen = [pick[n] for n in ("nps_trend", "attribute_satisfaction", "priority_matrix") if n in pick]
    rel = lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()  # noqa: E731
    block = ["<!-- KEY-CHARTS:START (written by src/08_charts_deck.py; do not edit by hand) -->", "",
             "## Headline findings", ""]
    block += [f"- **{s['title']}.**" for s in chosen]
    block += ["", *[f"![{s['title']}]({rel(s['png'])})" for s in chosen], "",
              f"Full deck: [`outputs/{DECK_STEM.name}.pdf`](outputs/{DECK_STEM.name}.pdf)", "",
              "<!-- KEY-CHARTS:END -->"]
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Mosaik Bank CX Tracker\n"
    pattern = re.compile(r"<!-- KEY-CHARTS:START.*?<!-- KEY-CHARTS:END -->", re.S)
    if pattern.search(text):
        text = pattern.sub("\n".join(block), text)
    else:
        lines = text.splitlines()
        insert_at = next((i + 1 for i, l in enumerate(lines) if l.startswith("# ")), 0)
        text = "\n".join(lines[:insert_at] + [""] + block + [""] + lines[insert_at:]) + "\n"
    readme.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the client charts, deck and README block.")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tables-dir", type=Path, default=TABLES)
    ap.add_argument("--quality", type=Path, default=QUALITY)
    ap.add_argument("--charts-dir", type=Path, default=CHARTS)
    ap.add_argument("--deck", type=Path, default=DECK_STEM, help="deck path without extension")
    ap.add_argument("--readme", type=Path, default=README)
    ap.add_argument("--no-pptx", action="store_true")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    I = Inputs(args.data, args.tables_dir, args.quality)
    facts = compute_facts(I)
    deck = Deck(args.charts_dir)
    s_exec_summary(deck, I, facts)
    s_nps_trend(deck, I, facts)
    s_nps_composition(deck, I)
    s_attributes(deck, I)
    s_drivers(deck, I)
    s_priority(deck, I)
    s_main_secondary(deck, I)
    s_age(deck, I)
    s_funnel(deck, I)
    s_switching(deck, I)
    s_reasons(deck, I)
    s_recommendations(deck, I, facts)
    s_appendix_method(deck, I)
    s_appendix_quality(deck, I)

    cover = title_slide(deck, I)
    cover_png = args.charts_dir / "00_title.png"
    cover.savefig(cover_png, dpi=DPI)
    args.deck.parent.mkdir(parents=True, exist_ok=True)
    write_pdf(deck, cover, args.deck.with_suffix(".pdf"))
    pptx_ok = not args.no_pptx and write_pptx(deck, cover_png, args.deck.with_suffix(".pptx"))
    if not args.no_readme:
        update_readme(deck, args.readme, facts)

    log.info("Slides:")
    for s in deck.slides:
        log.info("  %02d  %s", s["n"], s["title"])
    log.info("Wrote %d charts to %s, deck %s.pdf%s%s", len(deck.slides), args.charts_dir, args.deck,
             " + .pptx" if pptx_ok else "", "" if args.no_readme else f", README block in {args.readme.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
