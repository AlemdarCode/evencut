"""How much GPU time does data-parallel training lose to padding and to waiting? No GPU needed.

Give it every training sequence's length and the training setup. It replays a sample of your
run with the batching you use now and with the alternatives, costs every GPU's work per step (a
padded batch costs members x longest, plus the attention term when the model shape is given),
and splits the GPU time into useful work, padding, waiting for the slowest GPU and, when given, the
fixed time every step takes regardless of its batch (gradient sync, optimizer). With a run time, a
price and a power figure it turns the waste into GPU-hours, money and energy.

With a probe of the machine (``evencut-train probe``, :class:`~evencut.train.timing.Probe`) every
batch is priced by its measured time instead, so the step times, hours and money are predicted in
absolute terms, and the time small batches waste (a GPU is not proportionally faster on them) shows
up as its own share.

    from evencut.train.diagnose import Setup, diagnose
    d = diagnose(lengths, Setup(gpus=8, global_batch=256, accum=4))
    print(d.report())
"""

from __future__ import annotations

import dataclasses
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from .. import EvencutError
from . import _plans as P
from .timing import Probe

STRATEGIES = ("default", "dynamic_padding", "grouped_equal", "group_by_length", "batch_rebalance",
              "evencut_step", "evencut_window", "evencut_pooled", "sorted")
CURRENT_CHOICES = ("default", "grouped_equal", "group_by_length", "batch_rebalance", "evencut_step",
                   "evencut_window", "evencut_pooled", "sorted", "packed")
LABELS = {
    "default": "DistributedSampler (random order, equal counts per GPU)",
    "dynamic_padding": "same, but padded to each batch's longest",
    "grouped_equal": "global batch sorted by length, equal counts per GPU",
    "group_by_length": "Hugging Face Trainer group_by_length",
    "batch_rebalance": "Hugging Face Trainer batch_rebalance",
    "evencut_step": "Evencut, same steps as now, each micro-step balanced",
    "evencut_window": "Evencut, same steps as now, each optimizer step balanced",
    "evencut_pooled": "Evencut, length-grouped steps (like group_by_length), balanced",
    "sorted": "whole sample sorted, max-token batches (fairseq-style = Evencut)",
    "ceiling": "ceiling: no padding, perfect balance (packing)",
}

# Board power (TDP) in watts, the upper bound of what a busy card draws. Matched by substring,
# longest key first. Waiting GPUs draw less, so energy figures from this table are upper bounds.
GPU_WATTS = {
    "b200": 1000, "h200": 700, "h100 sxm": 700, "h100 nvl": 400, "h100 pcie": 350, "h100": 700,
    "gh200": 900, "a100 sxm": 400, "a100 pcie": 300, "a100": 400, "a10g": 300, "a10": 150,
    "l40s": 350, "l40": 300, "l4": 72, "a6000": 300, "rtx 6000 ada": 300, "a40": 300, "v100": 300,
    "t4": 70, "mi300x": 750, "mi250": 500, "rtx 5090": 575, "rtx 4090": 450, "rtx 4080": 320,
    "rtx 3090": 350, "rtx 4070": 200, "rtx 4060 ti": 165, "rtx 4060": 115,
}


def gpu_watts(name: str) -> Optional[int]:
    """Board power of a GPU by name: whole words only, never a "Ti"/"Super" variant of a listed card
    (an RTX A4000 is not an A40, an RTX 4060 Ti not an RTX 4060)."""
    key = " ".join(name.lower().replace("nvidia", " ").replace("geforce", " ").replace("-", " ").replace("_", " ").split())
    for k in sorted(GPU_WATTS, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])(?!\s*(?:ti|super)\b)", key):
            return GPU_WATTS[k]
    return None


@dataclass(frozen=True)
class Setup:
    """The training run as it is configured now.

    ``global_batch`` counts sequences per micro-step over all GPUs (per-device batch x GPUs);
    an optimizer step takes ``accum`` micro-steps. ``sync="window"``: the GPUs wait for each
    other once per optimizer step (DDP with ``no_sync`` while accumulating, ZeRO-1/2);
    ``"microstep"``: at every micro-step (FSDP, ZeRO-3). ``pad="max_length"`` pads every batch
    to ``max_len``; ``"longest"`` to the batch's longest sequence, rounded up to ``pad_multiple``.
    """

    gpus: int
    global_batch: int
    accum: int = 1
    max_len: Optional[int] = None
    pad: str = "longest"
    pad_multiple: int = 1
    current: str = "default"
    sync: str = "window"
    seed: int = 0
    pool_steps: Optional[int] = None

    @property
    def pool(self) -> int:
        """Optimizer steps sorted together by the pooled strategy (default: as group_by_length)."""
        return self.pool_steps or P.default_pool_steps(self.gpus)

    def validate(self) -> None:
        for name in ("gpus", "global_batch", "accum", "pad_multiple"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise EvencutError(f"evencut: {name} must be an integer >= 1")
        if self.global_batch < self.gpus:
            raise EvencutError("evencut: the global batch needs at least one sequence per GPU")
        if self.max_len is not None and (isinstance(self.max_len, bool) or not isinstance(self.max_len, int)
                                         or self.max_len < 1):
            raise EvencutError("evencut: max_len must be an integer >= 1")
        if self.pad not in ("longest", "max_length"):
            raise EvencutError("evencut: pad must be 'longest' or 'max_length'")
        if self.pad == "max_length" and self.max_len is None:
            raise EvencutError("evencut: pad='max_length' needs max_len")
        if self.current not in CURRENT_CHOICES:
            raise EvencutError(f"evencut: current must be one of {', '.join(CURRENT_CHOICES)}")
        if self.sync not in ("window", "microstep"):
            raise EvencutError("evencut: sync must be 'window' or 'microstep'")
        if self.pool_steps is not None and (isinstance(self.pool_steps, bool) or not isinstance(self.pool_steps, int)
                                            or self.pool_steps < 1):
            raise EvencutError("evencut: pool_steps must be an integer >= 1")
        if self.current in ("default", "group_by_length") and self.global_batch % self.gpus:
            raise EvencutError("evencut: with DistributedSampler or group_by_length the global batch is "
                               "per-device batch x GPUs, so it must be a multiple of gpus")


@dataclass(frozen=True)
class Model:
    """The model shape, to price causal attention (long sequences cost more than their length)."""

    params: int
    layers: int
    d_model: int

    @property
    def attention_constant(self) -> int:
        return P.attention_constant(self.params, self.layers, self.d_model)


@dataclass(frozen=True)
class Calibration:
    """Optional absolute figures for the run as it is now. ``step_ms`` is the measured time of
    one optimizer step; ``run_hours`` the wall time of the whole run (over ``epochs``).
    ``fixed_ms`` is the part of an optimizer step that does not grow with the batch: gradient
    sync, the optimizer, kernel launches. Without it every step is taken to be compute-bound,
    which overstates the gains of small models on fast GPUs (bench/results/backtest_diagnose.md:
    a 30M model on A100s spends ~10 ms per micro-step that no batching changes)."""

    step_ms: Optional[float] = None
    run_hours: Optional[float] = None
    epochs: float = 1.0
    fixed_ms: float = 0.0
    price_per_gpu_hour: Optional[float] = None
    watts: Optional[float] = None
    watts_source: str = ""


@dataclass(frozen=True)
class Row:
    key: str
    label: str
    stats: Optional[P.PlanStats]
    speedup: Optional[float] = None          # current step time / this step time
    shares: dict = field(default_factory=dict)
    fixed_global_batch: bool = True
    unavailable: str = ""
    step_ms: Optional[float] = None          # absolute, when a probe or a measured step time sets the scale
    over_memory: bool = False                # largest batch above what fit in the probe

    @property
    def available(self) -> bool:
        return self.stats is not None


@dataclass
class Diagnosis:
    setup: Setup
    data: dict
    cost_model: str
    sample_steps: int
    rows: list
    calibration: Calibration
    notes: list
    current_key: str
    model: Optional[Model] = None
    money: dict = field(default_factory=dict)
    recommendation: str = ""
    probe: Optional[Probe] = None

    def row(self, key: str) -> Row:
        return next(r for r in self.rows if r.key == key)

    def to_dict(self) -> dict:
        def row_dict(r: Row) -> dict:
            d = {"key": r.key, "label": r.label, "speedup": r.speedup, "shares": r.shares, "step_ms": r.step_ms,
                 "fixed_global_batch": r.fixed_global_batch, "unavailable": r.unavailable or None,
                 "over_memory": r.over_memory}
            if r.stats is not None:
                s = r.stats
                d["stats"] = dict(asdict(s), idle_share=s.idle_share, padding_share=s.padding_share,
                                  useful_share=s.useful_share, padded_fraction=s.padded_fraction)
            return d
        return {"setup": asdict(self.setup), "model": asdict(self.model) if self.model else None,
                "data": self.data, "cost_model": self.cost_model, "sample_steps": self.sample_steps,
                "calibration": asdict(self.calibration), "current": self.current_key,
                "rows": [row_dict(r) for r in self.rows], "money": self.money,
                "recommendation": self.recommendation, "notes": self.notes,
                "probe": ({"source": self.probe.source, "gpu": self.probe.gpu, "gpus": self.probe.gpus,
                           "date": self.probe.date} if self.probe else None)}

    def report(self) -> str:
        return format_report(self)


# ---------------------------------------------------------------------------------------------


def _lengths(lengths: Sequence[int], max_len: Optional[int]) -> list[int]:
    try:
        out = list(lengths)
    except TypeError:
        raise EvencutError("evencut: lengths must be a sequence of integers") from None
    if not out:
        raise EvencutError("evencut: lengths must not be empty")
    for i, v in enumerate(out):
        if isinstance(v, bool) or not isinstance(v, int):
            try:
                if float(v) != int(v):
                    raise ValueError
                v = int(v)
            except (TypeError, ValueError, OverflowError):
                raise EvencutError(f"evencut: lengths[{i}] = {v!r} is not an integer") from None
        if v < 0:
            raise EvencutError(f"evencut: lengths[{i}] = {v} is negative")
        out[i] = min(v, max_len) if max_len is not None else v
    return out


def _summary(raw: Sequence[int], cut: list[int], max_len: Optional[int]) -> dict:
    s = sorted(cut)
    n = len(s)

    def q(p: float) -> int:
        return s[min(n - 1, int(p * (n - 1)))]
    return {"sequences": n, "tokens": sum(s), "mean": statistics.fmean(s), "median": q(0.5), "p90": q(0.9),
            "p99": q(0.99), "max": s[-1], "max_len": max_len,
            "truncated": sum(1 for x in raw if max_len is not None and x > max_len)}


def _plan(key: str, order: list[int], lens: list[int], setup: Setup, cost: Optional[P.CostFn],
          dataset_size: int) -> P.Plan:
    g, k, m = setup.global_batch, setup.gpus, setup.accum
    if key in ("default", "dynamic_padding"):
        return P.per_step(order, lens, k, g, "strided")
    if key == "grouped_equal":
        return P.per_step(order, lens, k, g, "equal", grouped=True)
    if key == "group_by_length":
        b = g // k
        return P.hf_group_by_length(order, lens, k, b, m, mega_batch_mult=P.hf_mega_batch_mult(dataset_size, b * m))
    if key == "batch_rebalance":
        return P.hf_batch_rebalance(order, lens, k, g, m)
    if key == "evencut_step":
        return P.per_step(order, lens, k, g, "padded", grouped=True, cost=cost)
    if key == "evencut_window":
        return P.window(order, lens, k, g, m, cost=cost)
    if key == "evencut_pooled":
        balance = "window" if m > 1 and setup.sync == "window" else "step"
        return P.pooled(order, lens, k, g, m, pool_steps=setup.pool, seed=P.epoch_seed(setup.seed, 0),
                        balance=balance, cost=cost)
    if key == "sorted":
        return P.sorted_dataset(order, lens, k, len(order) // g, P.epoch_seed(setup.seed, 0), cost=cost)
    raise EvencutError(f"evencut: unknown strategy {key!r}")


def probe_model(probe: Probe, model: Optional[Model]) -> Model:
    """The model a probe measured; an explicitly given model must be the same shape."""
    pm = Model(int(probe.model["params"]), int(probe.model["layers"]), int(probe.model["d_model"]))
    if model is not None and model.attention_constant != pm.attention_constant:
        raise EvencutError(f"evencut: the probe measured a different model ({pm.params:,} weights, {pm.layers} layers, "
                           f"width {pm.d_model}); probe your model's shape or leave --params/--layers/--d-model out")
    if probe.curve.a != pm.attention_constant:
        raise EvencutError("evencut: damaged probe file: its time curve does not match its model")
    return pm


def diagnose(lengths: Sequence[int], setup: Setup, *, model: Optional[Model] = None,
             calibration: Optional[Calibration] = None, sample_steps: int = 400,
             strategies: Sequence[str] = STRATEGIES, probe: Optional[Probe] = None) -> Diagnosis:
    """Replays up to ``sample_steps`` optimizer steps of a random epoch order (``setup.seed``)
    under every strategy and compares them with ``setup.current``. With ``probe`` every batch is
    timed by the measured curve of that machine."""
    setup.validate()
    cal = calibration or Calibration()
    if probe is not None:
        model = probe_model(probe, model)
        if cal.watts is None and probe.mean_watts():
            cal = dataclasses.replace(cal, watts=probe.mean_watts(),
                                      watts_source="measured under full load by the probe")
        if cal.price_per_gpu_hour is None and probe.price_per_hour:
            cal = dataclasses.replace(cal, price_per_gpu_hour=probe.price_per_hour / probe.gpus)
    raw = list(lengths)
    lens = _lengths(raw, setup.max_len)
    n = len(lens)
    per_step = setup.global_batch * setup.accum
    if n < per_step:
        raise EvencutError(f"evencut: {n} sequences do not fill one optimizer step of {per_step}")
    steps = max(1, min(int(sample_steps), n // per_step))
    order = list(range(n))
    random.Random(P.epoch_seed(setup.seed, 0)).shuffle(order)
    order = order[: steps * per_step]

    cost = P.AttentionCost(model.attention_constant) if model else None
    cost_model = (f"dense layers + causal attention: members x longest x ({model.attention_constant} + longest)"
                  if model else "padded tokens (members x longest); attention not priced")
    if probe is not None:
        cost_model = (f"measured on {probe.label()}: time of one micro-batch by its size (forward + backward, "
                      f"slowest GPU)")
    timing = probe.curve if probe is not None else None
    exposed = None                 # gradient sync each step cannot hide (probe, multi-GPU, no --fixed-ms)
    notes_sync: list[str] = []
    if probe is not None and not cal.fixed_ms and setup.gpus > 1:
        try:
            exposed = probe.sync(setup.gpus)
            if setup.gpus != probe.gpus:
                notes_sync.append(f"gradient sync scaled from the probe's {probe.gpus} GPUs to {setup.gpus}")
            if not probe.measured_sync:
                notes_sync.append("this probe did not measure how much sync DDP hides behind the backward pass "
                                  "(older probe): sync counted in full, so steps are predicted a little slow")
        except EvencutError:
            notes_sync.append(f"the probe ran on {probe.gpus} GPU(s): gradient sync of {setup.gpus} GPUs is not "
                              f"included (probe with torchrun on all GPUs, or pass --fixed-ms)")
    pad_to = setup.max_len if setup.pad == "max_length" else None
    notes: list[str] = []

    current = "default" if setup.current == "packed" else setup.current
    keys = [k for k in strategies if k in STRATEGIES]
    if current not in keys:
        keys.insert(0, current)
    if setup.pad != "max_length" and "dynamic_padding" in keys:
        keys.remove("dynamic_padding")            # identical to the default then
    if setup.accum == 1 and "evencut_window" in keys:
        keys.remove("evencut_window")             # one micro-step per optimizer step: same as per step
    if setup.global_batch % setup.gpus:
        for k in ("default", "dynamic_padding", "group_by_length"):
            if k in keys and k != current:
                keys.remove(k)

    rows: list[Row] = []
    for key in keys:
        # the user's own padding applies to what they run now; every alternative pads to the longest
        own = key == current or key == "default"
        try:
            plan = _plan(key, order, lens, setup, cost, n)
        except ImportError:
            rows.append(Row(key, LABELS[key], None,
                            unavailable="needs transformers>=5.17 with torch installed to compare"))
            continue
        stats = P.evaluate(plan, lens, setup.gpus, setup.accum, cost=cost,
                           pad_to=pad_to if own and key != "dynamic_padding" else None,
                           pad_multiple=setup.pad_multiple, sync=setup.sync, timing=timing, exposed=exposed)
        rows.append(Row(key, LABELS[key], stats, fixed_global_batch=key != "sorted"))

    cur = next(r for r in rows if r.key == current)
    u = cur.stats.useful_ms
    ceiling = P.PlanStats(setup.gpus, cur.stats.steps, cur.stats.microsteps, cur.stats.useful_cost,
                          cur.stats.useful_cost, cur.stats.useful_cost, cur.stats.real_tokens,
                          cur.stats.real_tokens, cur.stats.sequences, 0, 0, u, u, u, u, cur.stats.sync_ms)
    rows.append(Row("ceiling", LABELS["ceiling"], ceiling, fixed_global_batch=True))

    total_steps = n * cal.epochs / per_step
    step_ms = cal.step_ms
    if step_ms is None and cal.run_hours is not None:
        step_ms = cal.run_hours * 3.6e6 / total_steps
    predicted = False
    if probe is not None:
        # time per optimizer step = k * measured-curve time + fixed; k = 1 unless the run's real step
        # time is known, which then corrects the probe's compute time
        # (fixed = optimizer, or --fixed-ms; the exposed sync of each row comes from evaluate)
        fixed = float(cal.fixed_ms) if cal.fixed_ms else probe.optimizer_ms
        notes += notes_sync
        unit = cur.stats.step_ms / cur.stats.steps
        cur_sync = cur.stats.sync_ms / cur.stats.steps
        k = 1.0
        if step_ms is not None:
            if step_ms <= fixed + cur_sync:
                raise EvencutError("evencut: the step time must be larger than the fixed time per step")
            k = (step_ms - fixed - cur_sync) / unit
            notes.append(f"the probe predicts {unit + fixed + cur_sync:.1f} ms per optimizer step as you run now; you "
                         f"measured {step_ms:.1f} ms, so the probe's compute times were scaled by {k:.2f}")
        else:
            step_ms, predicted = unit + fixed + cur_sync, True
    else:
        # time per optimizer step = k * cost + fixed (ms); k from the calibration, else relative only
        fixed = float(cal.fixed_ms or 0.0)
        unit = cur.stats.step_cost / cur.stats.steps
        k = None
        if step_ms is not None:
            if step_ms <= fixed:
                raise EvencutError("evencut: the step time must be larger than the fixed time per step")
            k = (step_ms - fixed) / unit
        elif fixed:
            notes.append("fixed_ms was ignored: it needs step_ms or run_hours to set the scale")
            fixed = 0.0

    def step_time(st: P.PlanStats) -> float:
        if probe is not None:
            return k * st.step_ms / st.steps + st.sync_ms / st.steps + fixed
        c = st.step_cost / st.steps
        return k * c + fixed if k is not None else c

    def shares(st: P.PlanStats, t: float) -> dict:
        if probe is not None:
            per = k / st.steps
            return {"useful": per * st.useful_ms / t, "padding": per * (st.linear_ms - st.useful_ms) / t,
                    "small_batches": per * (st.busy_ms - st.linear_ms) / t,
                    "waiting": per * (st.step_ms - st.busy_ms) / t, "fixed": (fixed + st.sync_ms / st.steps) / t}
        scale = (k if k is not None else 1.0) * st.step_cost / st.steps
        return {"useful": scale * st.useful_share / t, "padding": scale * st.padding_share / t,
                "small_batches": 0.0, "waiting": scale * st.idle_share / t,
                "fixed": fixed / t if k is not None else 0.0}

    full = probe is not None and probe.model.get("probe_layers") == probe.model.get("layers")
    fit = probe.max_tokens_fit if full else None
    absolute = probe is not None or k is not None
    t_cur = step_time(cur.stats)
    final_rows = []
    for r in rows:
        if r.stats is None:
            final_rows.append(r)
            continue
        t = step_time(r.stats)
        over = bool(fit and r.key != "ceiling" and r.stats.peak_padded > fit)
        final_rows.append(Row(r.key, r.label, r.stats, t_cur / t, shares(r.stats, t), r.fixed_global_batch,
                              r.unavailable, t if absolute else None, over))

    d = Diagnosis(setup, _summary(raw, lens, setup.max_len), cost_model, steps, final_rows, cal, notes, current,
                  model, probe=probe)
    if setup.current == "packed":
        d.notes.append("packed: you already train without padding, so there is nothing for Evencut to fix; "
                       "the table shows what padded batching would cost instead")
    if steps < n // per_step:
        d.notes.append(f"replayed {steps} of {n // per_step} optimizer steps ({steps * per_step:,} sequences); "
                       "raise sample_steps to replay more")
    if any(r.over_memory for r in final_rows):
        d.notes.append(f"! = its largest batch is above the {fit:,} padded tokens per GPU that fit in the probe")
    d.money = _money(d, step_ms, total_steps)
    if d.money:
        d.money["predicted"] = predicted
    d.recommendation = _recommend(d)
    return d


def _money(d: Diagnosis, step_ms: Optional[float], total_steps: float) -> dict:
    if step_ms is None:
        return {}
    cur = d.row(d.current_key)
    hours = step_ms * total_steps / 3.6e6
    gpu_hours = hours * d.setup.gpus
    lost = cur.shares["padding"] + cur.shares["waiting"] + cur.shares.get("small_batches", 0.0)
    out = {"run_hours": hours, "gpu_hours": gpu_hours, "wasted_gpu_hours": gpu_hours * lost,
           "fixed_gpu_hours": gpu_hours * cur.shares["fixed"]}
    cal = d.calibration
    if cal.price_per_gpu_hour is not None:
        out["price_per_gpu_hour"] = cal.price_per_gpu_hour
        out["wasted_cost"] = out["wasted_gpu_hours"] * cal.price_per_gpu_hour
        out["run_cost"] = gpu_hours * cal.price_per_gpu_hour
    if cal.watts is not None:
        out["watts"] = cal.watts
        out["watts_source"] = cal.watts_source
        out["wasted_kwh"] = out["wasted_gpu_hours"] * cal.watts / 1000
    saves = {}
    for r in d.rows:
        if r.speedup and r.key != d.current_key:
            h = hours * (1 - 1 / r.speedup)
            saves[r.key] = {"run_hours": hours / r.speedup, "hours_saved": h, "gpu_hours_saved": h * d.setup.gpus}
            if cal.price_per_gpu_hour is not None:
                saves[r.key]["cost_saved"] = h * d.setup.gpus * cal.price_per_gpu_hour
            if cal.watts is not None:
                saves[r.key]["kwh_saved"] = h * d.setup.gpus * cal.watts / 1000
    out["saves"] = saves
    return out


SIGNIFICANT = 1.03     # below a 3% gain the advice is "leave it"


# Strategies that keep every optimizer step's sequences as a shuffled DistributedSampler run has
# them (training statistics unchanged) versus ones that group similar lengths into the same steps.
SAME_STEPS = ("default", "dynamic_padding", "batch_rebalance", "evencut_step", "evencut_window")
GROUPING = ("grouped_equal", "group_by_length", "evencut_pooled")
GROUPING_CAUTION = ("Grouping lengths forms many steps of short sequences whose per-GPU batches can be too small to "
                    "keep a GPU busy, which this model does not price: on an RTX 4060 (30M model) Evencut's grouped "
                    "mode was slower than its same-steps mode on both Wikipedia (70.6 vs 65.7 ms) and Dolly (370 vs "
                    "337 ms per step) although predicted faster. Measure before choosing it.")
GROUPING_NOTE = ("The probe's measured times already include how slowly a GPU runs small batches; check that your "
                 "model trains as well with grouped steps before switching.")


def _best(rows: list, keys: Sequence[str]) -> Optional[Row]:
    pick = [r for r in rows if r.key in keys]
    return max(pick, key=lambda r: r.speedup, default=None)


def _recommend(d: Diagnosis) -> str:
    """Advice: first the best option that keeps every step's sequences (measured to be the safest
    gain), then length grouping and whole-dataset sorting as larger, less certain changes."""
    if d.setup.current == "packed":
        return "You pack sequences already: no padding to remove. Keep packing."
    cur = d.row(d.current_key)
    avail = [r for r in d.rows if r.available and r.key != "ceiling"]
    best = _best(avail, SAME_STEPS) or cur
    group = _best(avail, GROUPING)
    best_any = max(avail, key=lambda r: r.speedup)
    ceiling = d.row("ceiling")
    lines = []
    if best_any.speedup < SIGNIFICANT:
        return (f"Your batching is already within {100 * (best_any.speedup - 1):.0f}% of the best option; Evencut "
                f"would not pay off here.")
    if best.key == d.current_key or best.speedup < SIGNIFICANT:
        lines.append("Among splits that keep every step's sequences, yours is already the best.")
    else:
        lines.append(f"Switch to: {best.label}. Predicted {best.speedup:.2f}x faster steps "
                     f"({_pct(1 - 1 / best.speedup)} less GPU time); every step keeps the sequences it has now.")
        hf = next((r for r in d.rows if r.key == "batch_rebalance" and r.available), None)
        if best.key.startswith("evencut") and hf is not None:
            gain = best.speedup / hf.speedup
            if gain < SIGNIFICANT:
                lines.append(f"{hf.label} gets almost the same ({hf.speedup:.2f}x); if you use the Hugging Face "
                             f"Trainer, that built-in option is enough.")
            else:
                lines.append(f"That is {gain:.2f}x faster than Hugging Face batch_rebalance ({hf.speedup:.2f}x).")
        if d.setup.pad == "max_length":
            dyn = next((r for r in d.rows if r.key == "dynamic_padding" and r.available), None)
            if dyn:
                lines.append(f"Most of it comes from padding to the batch's longest sequence instead of "
                             f"max_length: that switch alone gives {dyn.speedup:.2f}x.")
        if best.stats.peak_padded > 1.1 * cur.stats.peak_padded:
            lines.append(f"Check memory: its largest batch holds {best.stats.peak_padded:,} padded tokens, "
                         f"{best.stats.peak_padded / max(cur.stats.peak_padded, 1):.1f}x today's.")
    if group is not None and group.speedup > best.speedup * SIGNIFICANT:
        lines.append(f"Grouping similar lengths into the same steps ({group.label}) is predicted at "
                     f"{group.speedup:.2f}x, but it changes which sequences train together. "
                     + (GROUPING_NOTE if d.probe is not None else GROUPING_CAUTION))
    sorted_row = next((r for r in avail if r.key == "sorted"), None)
    if sorted_row is not None and sorted_row.speedup > best.speedup * SIGNIFICANT:
        lines.append(f"If your training can take a varying number of sequences per step, sorting the whole "
                     f"dataset into max-token batches is predicted at {sorted_row.speedup:.2f}x.")
    left = ceiling.speedup / max(best_any.speedup, best.speedup)
    if left > 1.15:
        lost = "padding, small batches or waiting" if d.probe is not None else "padding or waiting"
        lines.append(f"Even then {_pct(1 - 1 / left)} of the GPU time stays {lost}; only packing "
                     f"(no padding at all, full batches) recovers it.")
    if d.setup.sync == "microstep" and best.key == "evencut_window":
        lines.append("Note: with FSDP/ZeRO-3 every micro-step synchronises; prefer the per-step split.")
    return " ".join(lines)


SAMPLER_MODE = {"evencut_pooled": "pooled", "evencut_window": "window", "evencut_step": "step", "sorted": "sorted"}


def how_to(d: Diagnosis) -> str:
    """The sampler line for the advised Evencut option (the best one that keeps every step's
    sequences), or "" when none is worth switching to."""
    avail = [r for r in d.rows if r.available and r.key in ("evencut_step", "evencut_window")]
    if not avail or d.setup.current == "packed":
        return ""
    best = max(avail, key=lambda r: r.speedup)
    if best.speedup < SIGNIFICANT or best.key == d.current_key:
        return ""
    s = d.setup
    per = f"per_device_batch={s.global_batch // s.gpus}" if s.global_batch % s.gpus == 0 else f"global_batch={s.global_batch}"
    args = [per]
    if s.accum > 1:
        args.append(f"accum={s.accum}")
    args.append(f"mode={SAMPLER_MODE[best.key]!r}")
    if best.key == "evencut_pooled" and s.pool_steps:
        args.append(f"pool_steps={s.pool_steps}")
    if s.max_len:
        args.append(f"max_len={s.max_len}")
    if s.sync == "microstep":
        args.append("sync_every_microstep=True")
    return ("DataLoader(dataset, batch_sampler=BalancedBatchSampler(lengths, " + ", ".join(args)
            + "), collate_fn=pad_to_longest)")


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------------------------------------
# text report


def _bar(x: float, width: int = 24) -> str:
    n = max(0, min(width, round(x * width)))
    return "#" * n + "." * (width - n)


def _num(x: float) -> str:
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def format_report(d: Diagnosis) -> str:
    s, data = d.setup, d.data
    cur = d.row(d.current_key)
    per = s.global_batch // s.gpus if s.global_batch % s.gpus == 0 else None
    sync = ("GPUs wait for each other once per optimizer step (DDP)" if s.sync == "window"
            else "GPUs wait for each other every micro-step (FSDP/ZeRO-3)")
    pad = (f"padded to max_length {s.max_len}" if s.pad == "max_length"
           else "padded to each batch's longest" + (f", multiple of {s.pad_multiple}" if s.pad_multiple > 1 else ""))
    out = [
        "Evencut diagnosis",
        f"  data     {data['sequences']:,} sequences, {data['tokens']:,} tokens; length mean {data['mean']:.0f}, "
        f"median {data['median']}, p90 {data['p90']}, p99 {data['p99']}, max {data['max']}"
        + (f" (cut at {data['max_len']}, {data['truncated']:,} cut)" if data["max_len"] else ""),
        f"  setup    {s.gpus} GPUs x {per if per else '~' + str(s.global_batch // s.gpus)} sequences = "
        f"{s.global_batch} per micro-step, accumulation {s.accum}; {pad}; {sync}",
        f"  cost     {d.cost_model}",
        f"  replay   {d.sample_steps:,} optimizer steps, seed {s.seed}; length-grouped strategies sort "
        f"{s.pool} optimizer steps together",
        "",
        f"Where the GPU time goes now ({cur.label}):",
    ]
    names = (("useful", "useful work"), ("padding", "padding"), ("small_batches", "small batches (GPU not full)"),
             ("waiting", "waiting for the slowest GPU"), ("fixed", "fixed time (sync, optimizer)"))
    for key, name in names:
        if key in ("fixed", "small_batches") and not cur.shares.get(key):
            continue
        out.append(f"  {name:28} {cur.shares[key]:6.1%}  {_bar(cur.shares[key])}")
    m = d.money
    if m:
        line = (f"  {'predicted from the probe: ' if m.get('predicted') else ''}a run of {_num(m['run_hours'])} h on "
                f"{s.gpus} GPUs = {_num(m['gpu_hours'])} GPU-h, of which {_num(m['wasted_gpu_hours'])} GPU-h go to "
                f"padding, small batches and waiting")
        extras = []
        if "wasted_cost" in m:
            extras.append(f"${_num(m['wasted_cost'])} at ${m['price_per_gpu_hour']:g}/GPU-h")
        if "wasted_kwh" in m:
            extras.append(f"up to {_num(m['wasted_kwh'])} kWh at {m['watts']:.0f} W/GPU, {m['watts_source']}")
        out.append(line + (" (" + "; ".join(extras) + ")" if extras else ""))
    timed = d.probe is not None
    absolute = cur.step_ms is not None
    head = f"  {'':2}{'strategy':68} {'speed':>6}" + (f" {'step ms':>8}" if absolute else "")
    head += f" {'useful':>7} {'padding':>8}" + (f" {'small':>6}" if timed else "") + f" {'waiting':>8} {'peak batch':>11}"
    out += ["", "Alternatives, same sequences (step time vs now; shares of GPU time):", head]
    for r in d.rows:
        mark = "->" if r.key == d.current_key else "  "
        if not r.available:
            out.append(f"  {mark}{r.label:68} ({r.unavailable})")
            continue
        peak = (f"{r.stats.peak_padded:,}" + ("!" if r.over_memory else "")) if r.key != "ceiling" else "-"
        label = r.label + ("" if r.fixed_global_batch else " *")
        line = f"  {mark}{label:68} {r.speedup:5.2f}x" + (f" {r.step_ms:8.1f}" if absolute else "")
        line += f" {r.shares['useful']:7.1%} {r.shares['padding']:8.1%}"
        line += (f" {r.shares['small_batches']:6.1%}" if timed else "") + f" {r.shares['waiting']:8.1%} {peak:>11}"
        out.append(line)
    if any(not r.fixed_global_batch for r in d.rows if r.available):
        out.append("  * the number of sequences per optimizer step varies")
    out.append("  peak batch = largest padded batch (tokens) one GPU holds: a memory check")
    if m and m.get("saves"):
        best = max((r for r in d.rows if r.available and r.key not in (d.current_key, "ceiling")),
                   key=lambda r: r.speedup, default=None)
        if best is not None and best.speedup > 1:
            sv = m["saves"][best.key]
            line = f"  best alternative saves {_num(sv['hours_saved'])} h of the run ({_num(sv['gpu_hours_saved'])} GPU-h"
            if "cost_saved" in sv:
                line += f", ${_num(sv['cost_saved'])}"
            if "kwh_saved" in sv:
                line += f", up to {_num(sv['kwh_saved'])} kWh"
            out.append(line + ")")
    out += ["", "Advice: " + d.recommendation]
    if not d.calibration.fixed_ms and d.probe is None:
        out.append("  Caution: this assumes every step is compute-bound, so the speed-ups are upper bounds. The "
                   "ranking of the options holds up in our measurements; the size of the gain needs the fixed time "
                   "per step (gradient sync, optimizer, kernel launches), which is large for small models on fast "
                   "GPUs. Pass --step-ms with --fixed-ms to include it.")
    if not m:
        out.append("  (relative figures; pass --probe, --step-ms or --run-hours for hours, money and energy)")
    if timed:
        out.append("  small = time lost because a batch is too small to keep the GPU busy (measured by the probe)")
    for note in d.notes:
        out.append(f"  note: {note}")
    code = how_to(d)
    if code:
        out += ["", "How (PyTorch DataLoader; see evencut.train.hf for the Hugging Face Trainer):", "  " + code]
    return "\n".join(out)

