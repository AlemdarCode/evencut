"""The README figures, drawn from the result files in bench/results (nothing is typed in by hand
except the vast.ai listing column of the link figure, quoted from the offers at rental time).

Every figure is written twice, for GitHub's light and dark themes (docs/figures/<name>-light.svg
and -dark.svg; the README picks one with <picture>), and again in Turkish for README.tr.md
(docs/figures/tr/, with Turkish number formatting). Colors, marks and spacing follow one method:
categorical slots in a fixed, validated order (checked for colour-vision deficiency and contrast
in both themes), one series in one colour with the story highlighted, thin marks, recessive
hairline grids, values labelled selectively, and a data table next to every figure in the README.

Run:   python bench/make_figures.py            (needs matplotlib; no GPU, no torch)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
OUT = ROOT / "docs" / "figures"
sys.path.insert(0, str(ROOT / "python"))
LENGTHS = Path.home() / ".cache" / "evencut" / "data" / "lengths"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", other="#8f8d86", inside="#ffffff",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", other="#77756f", inside="#ffffff",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}
# a label inside a fill: white, except near-black on the yellow slot (light theme)
INSIDE_DARK_ON = {"#eda100"}
BAR_PX = 20                # bar thickness in pixels (<= 24)
DPI = 96
WIDTH_IN = 9.6


# ---------------------------------------------------------------------------------------------
# languages: every figure is drawn in English (docs/figures) and in Turkish (docs/figures/tr)

LANGS = ("en", "tr")
LANG = "en"                # the language being drawn; main() sets it

TR = {
    # 1. step time
    "Time per optimizer step, lower is better": "Optimizer adımı başına süre, düşük olan daha iyi",
    "RTX 4060, 8 GPUs simulated\nWikipedia-TR, 48 sequences per step":
        "RTX 4060, 8 GPU taklit edildi\nWikipedia-TR, adım başına 48 dizi",
    "RTX 4060, 8 GPUs simulated\nDolly-15k, 128 sequences x accumulation 8":
        "RTX 4060, 8 GPU taklit edildi\nDolly-15k, 128 dizi × biriktirme 8",
    "2x A100 SXM4, rented (vast.ai)\nDolly-15k, 32 sequences x accumulation 8":
        "2x A100 SXM4, kiralık (vast.ai)\nDolly-15k, 32 dizi × biriktirme 8",
    "2x RTX 4090, rented (vast.ai)\nDolly-15k, 32 sequences x accumulation 8":
        "2x RTX 4090, kiralık (vast.ai)\nDolly-15k, 32 dizi × biriktirme 8",
    "DistributedSampler (default)": "DistributedSampler (varsayılan)",
    "HF group_by_length": "HF group_by_length",
    "HF batch_rebalance": "HF batch_rebalance",
    "Evencut": "Evencut",
    "ms per step": "adım başına ms",
    "not measured (no transformers on that machine)": "ölçülmedi (o makinede transformers yoktu)",
    # 2. where the time goes
    "Where the GPU time of a step goes (predicted from the machine's probe)":
        "Bir adımın GPU zamanı nereye gidiyor (makinenin probe'undan öngörü)",
    "2x A100 SXM4: Dolly-15k, 16 per GPU x accumulation 8": "2x A100 SXM4: Dolly-15k, GPU başına 16 × biriktirme 8",
    "2x RTX 4090: Wikipedia-TR, 6 per GPU": "2x RTX 4090: Wikipedia-TR, GPU başına 6",
    "useful work": "yararlı iş",
    "padding": "dolgu",
    "small batches": "küçük batch",
    "waiting for the slowest GPU": "en yavaş GPU'yu bekleme",
    "gradient sync + optimizer": "gradyan senkronu + optimizer",
    "{ms} ms (measured {m})": "{ms} ms (ölçülen {m})",
    # 3. probe accuracy
    "predicted = measured": "öngörülen = ölçülen",
    "datasheet estimate from an RTX 4060 (median error {e})": "RTX 4060'tan datasheet tahmini (ortanca hata {e})",
    "probe of the machine itself (median error {e})": "makinenin kendi probe'u (ortanca hata {e})",
    "measured step time, ms (real DDP training)": "ölçülen adım süresi, ms (gerçek DDP eğitimi)",
    "predicted step time, ms": "öngörülen adım süresi, ms",
    "Predicted vs measured, {n} rented machines\nshaded: within 10%":
        "Öngörülen ve ölçülen, kiralık {n} makine\ngölgeli alan: %10 içinde",
    # 4. cost
    "Dolly-15k, base model, accumulation 4": "Dolly-15k, base model, biriktirme 4",
    "measured (real training)": "ölçülen (gerçek eğitim)",
    "probe of that machine": "o makinenin probe'u",
    "datasheet estimate (no probe)": "datasheet tahmini (probe yok)",
    "dollars per million training sequences (lower is better)":
        "milyon eğitim dizisi başına dolar (düşük olan daha iyi)",
    "Which rented machine trains this job for less? {job}": "Bu işi hangi kiralık makine daha ucuza eğitir? {job}",
    # 5. datasheet ratio by model size
    "datasheet ratio": "datasheet oranı",
    "{name} (datasheet: {x} an RTX 4060)": "{name} (datasheet'e göre RTX 4060'ın {k} katı)",
    "30M": "30M", "91M": "91M", "310M": "310M", "7B shape": "7B şekli",
    "model size (8k-token batches)": "model boyutu (8k token'lık batch'ler)",
    "measured speed-up / datasheet speed-up": "ölçülen hızlanma / datasheet hızlanması",
    "How much of its datasheet advantage over an RTX 4060\na GPU really delivers":
        "Bir GPU, RTX 4060'a karşı datasheet üstünlüğünün\nne kadarını gerçekten veriyor",
    # 6. gradient sync
    "ms to all-reduce the 120 MB of fp32 gradients of a 30M model (every optimizer step)":
        "30M'lik bir modelin 120 MB fp32 gradyanını all-reduce etmenin süresi, ms (her optimizer adımında)",
    "Gradient sync is where the GPU link shows": "GPU bağlantısının farkı gradyan senkronunda görünür",
    "vast.ai listed NVLink 300 GB/s": "vast.ai NVLink 300 GB/s listeledi",
    "NVLink between the GPUs": "GPU'lar arasında NVLink",
    "PCIe through the CPUs (no NVLink)": "CPU'lar üzerinden PCIe (NVLink yok)",
    "GeForce: no peer-to-peer": "GeForce: eşler arası erişim yok",
}


def T(s: str) -> str:
    """``s`` in the language being drawn (a Turkish string missing from TR is an error, not English)."""
    return s if LANG == "en" else TR[s]


def num(x: float, spec: str) -> str:
    """``x`` formatted by ``spec``: 1,365.5 in English, 1.365,5 in Turkish."""
    s = format(x, spec)
    return s if LANG == "en" else s.translate(str.maketrans(",.", ".,"))


def pct(x: float) -> str:
    """A share as a whole percent: 25% in English, %25 in Turkish."""
    return f"{x:.0%}" if LANG == "en" else "%" + num(100 * x, ".0f")


def times(x: float, spec: str) -> str:
    """A speed ratio: 1.69x in English, 1,69× in Turkish."""
    return f"{x:{spec}}x" if LANG == "en" else num(x, spec) + "×"


def box(label: str) -> str:
    """A machine label (``2x RTX 4090, 2 Nov, box 3``) in the language being drawn."""
    return label if LANG == "en" else label.replace(" Nov", " Kasım").replace("box ", "kutu ")


class TurkishScalarFormatter(matplotlib.ticker.ScalarFormatter):
    """Matplotlib's default tick labels with a decimal comma (0,2) and a thousands dot."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).translate(str.maketrans(",.", ".,"))


# ---------------------------------------------------------------------------------------------
# colour checks (computed, not eyeballed)

def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def check_themes() -> None:
    for name, t in THEMES.items():
        for role in ("ink", "ink2", "muted"):
            assert contrast(t[role], t["surface"]) >= 3.0, (name, role)
        assert contrast(t["other"], t["surface"]) >= 3.0, (name, "other")   # de-emphasised bars are still data


# ---------------------------------------------------------------------------------------------
# drawing helpers

def setup(theme: str):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "svg.fonttype": "path", "svg.hashsalt": "evencut",
        "axes.facecolor": t["surface"], "figure.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "text.color": t["ink"], "axes.titlecolor": t["ink"], "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


def style_axes(ax, t, grid_axis: str = "x") -> None:
    ax.grid(True, axis=grid_axis, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    if LANG == "tr":
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is matplotlib.ticker.ScalarFormatter:
                axis.set_major_formatter(TurkishScalarFormatter())


def px_to_data(ax, px: float, axis: str) -> float:
    """Data units spanned by ``px`` pixels along ``axis`` (after the layout is final)."""
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (px, px)])
    return abs(x1 - x0) if axis == "x" else abs(y1 - y0)


def hbar(ax, y: float, x0: float, width: float, color: str, rounded: bool = True) -> None:
    """A horizontal bar BAR_PX thick, 4 px rounded at its data end, square at the baseline."""
    h = px_to_data(ax, BAR_PX, "y")
    if width <= 0:
        return
    rx, ry = px_to_data(ax, 4, "x"), px_to_data(ax, 4, "y")       # a 4 px radius, round on screen
    y0, y1, x1 = y - h / 2, y + h / 2, x0 + width
    if not rounded or width <= 2 * rx:
        ax.add_patch(Rectangle((x0, y0), width, h, facecolor=color, edgecolor="none", zorder=3))
        return
    # one closed outline: square at the baseline, both corners of the data end rounded
    verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0, y1),
             (x0, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def finish(fig, name: str, theme: str) -> Path:
    out = OUT if LANG == "en" else OUT / LANG
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{theme}.svg"
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "bench/make_figures.py"})
    preview = os.environ.get("EVENCUT_FIGURE_PREVIEW")      # a folder for PNG copies, to look at them
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        suffix = "" if LANG == "en" else f"-{LANG}"
        fig.savefig(Path(preview) / f"{name}-{theme}{suffix}.png", format="png", dpi=DPI)
    plt.close(fig)
    return path


def load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def row_ms(run: dict, order: str, split: str):
    rows = run.get("part_b") or run.get("rows")
    return next((r["step_ms"] for r in rows if r["order"] == order and r["split"] == split), None)


# ---------------------------------------------------------------------------------------------
# 1. step time by batching strategy

DEFAULT = ("random", "equal counts")
HF_GBL = ("pooled (like group_by_length)", "Hugging Face group_by_length")
HF_BR = ("random", "Hugging Face batch_rebalance")
WINDOW = ("random", "Evencut, whole optimizer step balanced (window)")
STEP_ATTN = ("length-grouped", "Evencut, padded size + attention")
STRATEGY_PANELS = [
    ("RTX 4060, 8 GPUs simulated\nWikipedia-TR, 48 sequences per step", "gpu_batches_wikipedia_v2.json", STEP_ATTN),
    ("RTX 4060, 8 GPUs simulated\nDolly-15k, 128 sequences x accumulation 8", "gpu_batches_dolly_accum8.json", WINDOW),
    ("2x A100 SXM4, rented (vast.ai)\nDolly-15k, 32 sequences x accumulation 8", "vast_a100x2_v3/ddp_dolly_accum8.json",
     WINDOW),
    ("2x RTX 4090, rented (vast.ai)\nDolly-15k, 32 sequences x accumulation 8", "vast_rtx4090x2_v3/ddp_dolly_accum8.json",
     WINDOW),
]
STRATEGY_LABELS = ["DistributedSampler (default)", "HF group_by_length", "HF batch_rebalance", "Evencut"]


def strategy_data() -> list:
    out = []
    for title, rel, evencut_key in STRATEGY_PANELS:
        run = load(rel)
        vals = [row_ms(run, *DEFAULT), row_ms(run, *HF_GBL), row_ms(run, *HF_BR), row_ms(run, *evencut_key)]
        out.append(dict(title=title, source=f"bench/results/{rel}", values=vals))
    return out


def fig_strategies(theme: str) -> Path:
    t = setup(theme)
    data = strategy_data()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.6), dpi=DPI, layout="constrained")
    fig.suptitle(T("Time per optimizer step, lower is better"), x=0.01, ha="left", fontsize=13, color=t["ink"])
    ys = list(range(len(STRATEGY_LABELS)))[::-1]
    for ax, panel in zip(axes.flat, data):
        vals = panel["values"]
        top = max(v for v in vals if v is not None)
        ax.set_xlim(0, top * 1.42)
        ax.set_ylim(-0.6, len(ys) - 0.4)
        ax.set_yticks(ys, [T(label) for label in STRATEGY_LABELS])
        ax.set_title(T(panel["title"]), loc="left", fontsize=10, color=t["ink2"], linespacing=1.3)
        ax.set_xlabel(T("ms per step"), fontsize=9)
        style_axes(ax, t)
    fig.canvas.draw()                                    # the layout is final: bars are sized in pixels
    for ax, panel in zip(axes.flat, data):
        vals = panel["values"]
        base = vals[0]
        for y, v, label in zip(ys, vals, STRATEGY_LABELS):
            if v is None:
                ax.text(0.01 * ax.get_xlim()[1], y, T("not measured (no transformers on that machine)"),
                        va="center", fontsize=8.5, color=t["muted"])
                continue
            color = t["series"][0] if label == "Evencut" else t["other"]
            hbar(ax, y, 0, v, color)
            speed = base / v
            text = num(v, ",.0f") + " ms" if v >= 100 else num(v, ".1f") + " ms"
            if label != STRATEGY_LABELS[0]:
                text += "  " + times(speed, ".2f")
            ax.text(v + px_to_data(ax, 6, "x"), y, text, va="center", fontsize=9,
                    color=t["ink"] if label == "Evencut" else t["ink2"],
                    fontweight="bold" if label == "Evencut" else "normal")
    return finish(fig, "step-time", theme)


# ---------------------------------------------------------------------------------------------
# 2. where the GPU time goes (diagnose with a probe of the machine)

TIME_PARTS = [("useful", "useful work"), ("padding", "padding"), ("small_batches", "small batches"),
              ("waiting", "waiting for the slowest GPU"), ("fixed", "gradient sync + optimizer")]
TIME_CASES = [
    dict(title="2x A100 SXM4: Dolly-15k, 16 per GPU x accumulation 8", lengths="dolly15k_gpt2.txt", max_len=1024,
         probe="vast_a100x2_v3/probe_small.json", per_gpu=16, accum=8, fix="evencut_window",
         measured="vast_a100x2_v3/ddp_dolly_accum8.json", measured_fix=WINDOW),
    dict(title="2x RTX 4090: Wikipedia-TR, 6 per GPU", lengths="wikipedia_tr.txt", max_len=2048,
         probe="vast_rtx4090x2_v3/probe_small.json", per_gpu=6, accum=1, fix="evencut_step",
         measured="vast_rtx4090x2_v3/ddp_wikipedia.json", measured_fix=("random", "Evencut, balanced padded size")),
]


def time_data() -> list:
    from evencut.train.cli import read_lengths
    from evencut.train.diagnose import Setup, diagnose
    from evencut.train.timing import Probe

    out = []
    for case in TIME_CASES:
        probe = Probe.load(RESULTS / case["probe"])
        d = diagnose(read_lengths(str(LENGTHS / case["lengths"])),
                     Setup(gpus=probe.gpus, global_batch=probe.gpus * case["per_gpu"], accum=case["accum"],
                           max_len=case["max_len"]), probe=probe, sample_steps=300)
        run = load(case["measured"])
        bars = []
        for key, label in (("default", "DistributedSampler"), (case["fix"], "Evencut")):
            r = d.row(key)
            bars.append(dict(label=label, step_ms=r.step_ms, parts={k: r.shares[k] * r.step_ms for k, _ in TIME_PARTS}))
        out.append(dict(title=case["title"], bars=bars, probe=f"bench/results/{case['probe']}",
                        measured=[row_ms(run, *DEFAULT), row_ms(run, *case["measured_fix"])],
                        measured_source=f"bench/results/{case['measured']}"))
    return out


def fig_time_parts(theme: str, data: list) -> Path:
    t = setup(theme)
    fig, axes = plt.subplots(len(data), 1, figsize=(WIDTH_IN, 1.75 * len(data) + 1.1), dpi=DPI, layout="constrained")
    fig.suptitle(T("Where the GPU time of a step goes (predicted from the machine's probe)"), x=0.01, ha="left",
                 fontsize=13, color=t["ink"])
    for ax, case in zip(axes, data):
        top = max(b["step_ms"] for b in case["bars"])
        ax.set_xlim(0, top * 1.18)
        ax.set_ylim(-0.7, 1.7)
        ax.set_yticks([1, 0], [b["label"] for b in case["bars"]])
        ax.set_title(T(case["title"]), loc="left", fontsize=10, color=t["ink2"])
        ax.set_xlabel(T("ms per step"), fontsize=9)
        style_axes(ax, t)
    fig.canvas.draw()
    for ax, case in zip(axes, data):
        gap = px_to_data(ax, 2, "x")
        for y, bar, meas in zip((1, 0), case["bars"], case["measured"]):
            x = 0.0
            for i, (key, _) in enumerate(TIME_PARTS):
                w = bar["parts"][key]
                color = t["series"][i]
                hbar(ax, y, x, max(0.0, w - gap), color, rounded=False)
                share = w / bar["step_ms"]
                if share >= 0.09 and w > px_to_data(ax, 34, "x"):
                    ink = "#0b0b0b" if color in INSIDE_DARK_ON else t["inside"]
                    ax.text(x + w / 2, y, pct(share), ha="center", va="center", fontsize=8.5, color=ink, zorder=4)
                x += w
            ax.text(x + px_to_data(ax, 6, "x"), y,
                    T("{ms} ms (measured {m})").format(ms=num(bar["step_ms"], ".0f"), m=num(meas, ".0f")),
                    va="center", fontsize=9, color=t["ink2"])
    handles = [Rectangle((0, 0), 1, 1, color=t["series"][i]) for i in range(len(TIME_PARTS))]
    fig.legend(handles, [T(label) for _, label in TIME_PARTS], loc="outside lower left", ncols=5, fontsize=9,
               handlelength=1.0, handleheight=1.0, labelcolor=t["ink2"])
    return finish(fig, "time-parts", theme)


# ---------------------------------------------------------------------------------------------
# 3. probe accuracy: predicted vs measured step time on rented machines

def accuracy_data() -> dict:
    report = load("backtest_probe.json")
    probe, est = [], []
    for mach in report["machines"]:
        for run in mach["runs"]:
            for row in run["rows"]:
                probe.append((row["measured"], row["probe"]))
                if "estimate" in row:
                    est.append((row["measured"], row["estimate"]))
    med = lambda pts: statistics.median(abs(p / m - 1) for m, p in pts)  # noqa: E731
    return dict(probe=probe, estimate=est, probe_median=med(probe), estimate_median=med(est),
                machines=len(report["machines"]), source="bench/results/backtest_probe.json")


def fig_accuracy(theme: str) -> Path:
    t = setup(theme)
    d = accuracy_data()
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.62, 5.0), dpi=DPI, layout="constrained")
    allv = [v for pts in (d["probe"], d["estimate"]) for pair in pts for v in pair]
    lo, hi = min(allv) / 1.3, max(allv) * 1.3
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    xs = [lo, hi]
    ax.fill_between(xs, [x / 1.1 for x in xs], [x * 1.1 for x in xs], color=t["muted"], alpha=0.12, lw=0, zorder=1)
    ax.plot(xs, xs, color=t["muted"], lw=1, zorder=2)
    ax.text(lo * 1.5, lo * 1.5 * 1.18, T("predicted = measured"), rotation=45, rotation_mode="anchor", fontsize=8.5,
            color=t["ink2"], ha="left")
    ring = dict(edgecolors=t["surface"], linewidths=1.2, zorder=3)
    ax.scatter(*zip(*d["estimate"]), s=30, marker="^", color=t["series"][1], alpha=0.85,
               label=T("datasheet estimate from an RTX 4060 (median error {e})").format(e=pct(d["estimate_median"])),
               **ring)
    ax.scatter(*zip(*d["probe"]), s=30, marker="o", color=t["series"][0],
               label=T("probe of the machine itself (median error {e})").format(e=pct(d["probe_median"])), **ring)
    ax.set_xlabel(T("measured step time, ms (real DDP training)"))
    ax.set_ylabel(T("predicted step time, ms"))
    ax.set_title(T("Predicted vs measured, {n} rented machines\nshaded: within 10%").format(n=d["machines"]),
                 loc="left", fontsize=11, color=t["ink"])
    style_axes(ax, t, "both")
    ax.legend(loc="upper left", fontsize=8.5, labelcolor=t["ink2"], handletextpad=0.3)
    return finish(fig, "probe-accuracy", theme)


# ---------------------------------------------------------------------------------------------
# 4. which machine is cheaper: $ per million sequences

COST_JOB = ("Dolly-15k, base model, accumulation 4", "Evencut")
BOX = {"vast_a100x2": "2x A100 SXM4 40GB, box 1", "vast_a100x2_v2": "2x A100 SXM4 40GB, box 2",
       "vast_a100x2_v3": "2x A100 SXM4 80GB, box 3", "vast_rtx4090x2": "2x RTX 4090, box 1",
       "vast_rtx4090x2_v2": "2x RTX 4090, box 2", "vast_rtx4090x2_v3": "2x RTX 4090, box 3"}


def cost_data() -> dict:
    report = load("backtest_probe.json")
    case = next(c for c in report["machine_ranking"] if (c["job"], c["split"]) == COST_JOB)
    links = {m["dir"]: (m["probes"].get("base") or m["probes"].get("small") or {}).get("links") for m in report["machines"]}
    prices = {m["dir"]: m["price"] for m in report["machines"]}
    rows = sorted(case["cost"]["measured"], key=case["cost"]["measured"].get)
    price = (lambda p: f"${p:.2f}/h") if LANG == "en" else (lambda p: f"{num(p, '.2f')} $/sa")
    return dict(job=COST_JOB, rows=[dict(machine=n, label=f"{box(BOX.get(n, n))} ({'/'.join(links.get(n) or ['?'])}, "
                                                           f"{price(prices[n])})",
                                         **{k: case["cost"][k].get(n) for k in ("measured", "probe", "estimate")})
                                    for n in rows], source="bench/results/backtest_probe.json")


def fig_cost(theme: str) -> Path:
    t = setup(theme)
    d = cost_data()
    n = len(d["rows"])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.42 * n + 1.6), dpi=DPI, layout="constrained")
    ys = list(range(n))[::-1]
    allv = [r[k] for r in d["rows"] for k in ("measured", "probe", "estimate") if r[k] is not None]
    ax.set_xlim(0, max(allv) * 1.12)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys, [r["label"] for r in d["rows"]])
    # measured: a large open ring, so a probe square right on top of it stays visible inside
    ax.scatter([r["measured"] for r in d["rows"]], ys, s=170, marker="o", facecolors="none",
               edgecolors=t["series"][0], linewidths=2.2, zorder=4, label=T("measured (real training)"))
    ring = dict(edgecolors=t["surface"], linewidths=1.2, zorder=5)
    ax.scatter([r["probe"] for r in d["rows"]], ys, s=42, marker="s", color=t["series"][2],
               label=T("probe of that machine"), **ring)
    ax.scatter([r["estimate"] for r in d["rows"]], ys, s=56, marker="^", color=t["series"][1],
               label=T("datasheet estimate (no probe)"), **ring)
    for y, r in zip(ys, d["rows"]):
        ax.plot([min(r["measured"], r["estimate"]), max(r["measured"], r["estimate"])], [y, y], color=t["grid"],
                lw=1.5, zorder=2)
    ax.set_xlabel(T("dollars per million training sequences (lower is better)"))
    ax.set_title(T("Which rented machine trains this job for less? {job}").format(job=T(d["job"][0])), loc="left",
                 fontsize=11, color=t["ink"])
    style_axes(ax, t)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower left", ncols=3, fontsize=9, labelcolor=t["ink2"])
    return finish(fig, "machine-cost", theme)


# ---------------------------------------------------------------------------------------------
# 5. share of the datasheet speed-up reached, by model size

SHAPES = [("small", "30M"), ("base", "91M"), ("large", "310M"), ("7b", "7B shape")]


def scaling_data() -> dict:
    report = load("backtest_probe.json")
    rows = [r for r in report["scaling"] if r["tokens"] == 8192]
    out = {}
    for gpu_key, match in (("A100", "A100"), ("RTX 4090", "4090")):
        pts = [(i, r["measured_speedup"] / r["datasheet_speedup"]) for i, (shape, _) in enumerate(SHAPES)
               for r in rows if r["shape"] == shape and match in r["gpu"]]
        out[gpu_key] = dict(points=pts, datasheet=next(r["datasheet_speedup"] for r in rows if match in r["gpu"]))
    return dict(series=out, source="bench/results/backtest_probe.json (section F)")


def fig_scaling(theme: str) -> Path:
    t = setup(theme)
    d = scaling_data()
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.62, 4.4), dpi=DPI, layout="constrained")
    ax.set_xlim(-0.4, len(SHAPES) - 0.6)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color=t["muted"], lw=1, zorder=1)
    ax.text(len(SHAPES) - 0.62, 1.02, T("datasheet ratio"), ha="right", va="bottom", fontsize=8.5, color=t["ink2"])
    for slot, (name, marker) in enumerate((("A100", "o"), ("RTX 4090", "s"))):
        s = d["series"][name]
        xs = sorted({x for x, _ in s["points"]})
        med = [statistics.median(v for x2, v in s["points"] if x2 == x) for x in xs]
        ax.plot(xs, med, color=t["series"][slot], lw=2, zorder=2)
        ax.scatter([x for x, _ in s["points"]], [v for _, v in s["points"]], s=40, marker=marker,
                   color=t["series"][slot], edgecolors=t["surface"], linewidths=1.2, zorder=3,
                   label=T("{name} (datasheet: {x} an RTX 4060)").format(
                       name=name, x=times(s["datasheet"], ".1f"), k=num(s["datasheet"], ".1f")))
        ax.text(xs[-1] + 0.08, med[-1], pct(med[-1]), va="center", fontsize=9, color=t["ink"])
    ax.set_xticks(range(len(SHAPES)), [T(label) for _, label in SHAPES])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0) if LANG == "en" else
                                 matplotlib.ticker.FuncFormatter(lambda v, _: "%" + num(100 * v, ".0f")))
    ax.set_xlabel(T("model size (8k-token batches)"))
    ax.set_ylabel(T("measured speed-up / datasheet speed-up"))
    ax.set_title(T("How much of its datasheet advantage over an RTX 4060\na GPU really delivers"), loc="left",
                 fontsize=11, color=t["ink"])
    style_axes(ax, t, "y")
    ax.legend(loc="lower right", fontsize=8.5, labelcolor=t["ink2"])
    return finish(fig, "datasheet-by-size", theme)


# ---------------------------------------------------------------------------------------------
# 6. gradient sync: what the link costs

# vast.ai's bw_nvlink field for each rented box, read from the offer list when it was rented
# (bench/results does not record offers). 0 = vast listed no NVLink.
VAST_LISTED_NVLINK = {"ddp_a100x4": 300, "vast_a100x2": 0, "vast_a100x2_v2": 300, "vast_a100x2_v3": 300,
                      "ddp_batches": 0, "vast_rtx4090x2": 0, "vast_rtx4090x2_v2": 0, "vast_rtx4090x2_v3": 0}


def link_class(links) -> int:
    s = " ".join(links or [])
    if "NV" in s:
        return 0
    return 2 if "4090" in s else 1


def sync_data() -> list:
    from evencut.train.timing import Probe

    rows = []
    for key, rel, label in (("ddp_a100x4", "ddp_a100x4.json", "4x A100 SXM4, 1 Nov"),
                            ("ddp_batches", "ddp_batches.json", "8x RTX 4090, 1 Nov")):
        run = load(rel)
        links = sorted({c for line in run["topology"].splitlines() if line.startswith("GPU")
                        for c in line.split()[1:run["gpus"] + 1] if c != "X"})
        rows.append(dict(key=key, label=label, ms=run["allreduce_ms"], links=links, gpus=run["gpus"],
                         geforce="4090" in run["gpu"], source=f"bench/results/{rel}"))
    for d in ("vast_a100x2", "vast_a100x2_v2", "vast_a100x2_v3", "vast_rtx4090x2", "vast_rtx4090x2_v2",
              "vast_rtx4090x2_v3"):
        p = Probe.load(RESULTS / d / "probe_small.json")
        rows.append(dict(key=d, label=BOX[d].replace(", box", ", 2 Nov, box"), ms=p.allreduce_ms(), links=p.machine.get("links"),
                         gpus=p.gpus, geforce="4090" in p.gpu, source=f"bench/results/{d}/probe_small.json"))
    for r in rows:
        r["klass"] = 0 if any(str(x).startswith("NV") for x in (r["links"] or [])) else (2 if r["geforce"] else 1)
        r["listed_nvlink"] = VAST_LISTED_NVLINK[r["key"]]
    return sorted(rows, key=lambda r: r["ms"])


SYNC_CLASSES = ["NVLink between the GPUs", "PCIe through the CPUs (no NVLink)", "GeForce: no peer-to-peer"]


def fig_sync(theme: str) -> Path:
    t = setup(theme)
    rows = sync_data()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.44 * n + 1.7), dpi=DPI, layout="constrained")
    ys = list(range(n))[::-1]
    top = max(r["ms"] for r in rows)
    ax.set_xlim(0, top * 1.55)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys, [f"{box(r['label'])} ({'/'.join(r['links'] or ['?'])})" for r in rows])
    ax.set_xlabel(T("ms to all-reduce the 120 MB of fp32 gradients of a 30M model (every optimizer step)"))
    ax.set_title(T("Gradient sync is where the GPU link shows"), loc="left", fontsize=11, color=t["ink"])
    style_axes(ax, t)
    fig.canvas.draw()
    for y, r in zip(ys, rows):
        hbar(ax, y, 0, r["ms"], t["series"][r["klass"]])
        listed = bool(r["listed_nvlink"]) and r["klass"] != 0      # vast listed NVLink, the box had none
        note = num(r["ms"], ".1f") + " ms" + ("   " + T("vast.ai listed NVLink 300 GB/s") if listed else "")
        ax.text(r["ms"] + px_to_data(ax, 6, "x"), y, note, va="center", fontsize=9,
                color=t["ink"] if listed else t["ink2"], fontweight="bold" if listed else "normal")
    handles = [Rectangle((0, 0), 1, 1, color=t["series"][i]) for i in range(3)]
    fig.legend(handles, [T(c) for c in SYNC_CLASSES], loc="outside lower left", ncols=3, fontsize=9,
               labelcolor=t["ink2"],
               handlelength=1.0, handleheight=1.0)
    return finish(fig, "gradient-sync", theme)


# ---------------------------------------------------------------------------------------------


def main() -> None:
    global LANG
    check_themes()
    parts = time_data()
    LANG = "en"                                          # figures.json keeps the English labels
    tables = dict(step_time=strategy_data(), time_parts=parts, accuracy={k: v for k, v in accuracy_data().items()
                                                                         if k not in ("probe", "estimate")},
                  cost=cost_data(), scaling=scaling_data(), sync=sync_data())
    written = []
    for lang in LANGS:
        LANG = lang
        for theme in THEMES:
            written += [fig_strategies(theme), fig_time_parts(theme, parts), fig_accuracy(theme), fig_cost(theme),
                        fig_scaling(theme), fig_sync(theme)]
    LANG = "en"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures.json").write_text(json.dumps(tables, indent=1), encoding="utf-8")
    for p in written:
        print(f"{p.relative_to(ROOT)}  {os.path.getsize(p) / 1024:.0f} KB")
    print(f"{(OUT / 'figures.json').relative_to(ROOT)}  (the numbers behind every figure, for the README tables)")


if __name__ == "__main__":
    main()
