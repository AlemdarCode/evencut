"""Backtest of `evencut-train diagnose`: does the no-GPU cost model predict real step times?

Replays every measured data-parallel run in bench/results with the plan code the diagnosis uses
(evencut.train._plans), and compares predicted with measured speed-ups over the default split
(random order, equal counts):

* 4060 simulation (gpu_batches*.json): 8 GPUs' slices timed one by one on an RTX 4060, step time =
  the slowest slice. No communication, so this tests the compute cost model alone.
* real multi-GPU runs (ddp_*.json): 8x RTX 4090 (PCIe, no P2P) and 4x A100 SXM (NVLink), DDP over
  NCCL, wall time per optimizer step with communication included.

Predictions, from least to most information:
  padded     cost = padded tokens, compute-bound (what `diagnose` says with no model and no times)
  attention  cost = padded tokens + causal attention for the model shape (--params/--layers/--d-model)
  +comm      attention cost, plus the measured all-reduce time as a fixed cost per optimizer step,
             scaled by the measured default step time (--step-ms and --comm-ms)

Run: python bench/backtest_diagnose.py   (no GPU, no torch; needs the lengths files of the runs)
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from evencut.train import _plans as P  # noqa: E402

LENGTHS = Path.home() / ".cache" / "evencut" / "data" / "lengths"
MODELS = {"small": dict(layers=8, d=512), "base": dict(layers=12, d=768), "large": dict(layers=24, d=1024)}


def model_constant(layers: int, d: int, vocab: int = 8192) -> int:
    ffn = 64 * math.ceil(8 * d / 3 / 64)
    return P.attention_constant(layers * (4 * d * d + 3 * d * ffn) + d * vocab, layers, d)


POOLED = "pooled (like group_by_length)"


def builder(order_name: str, split: str, attention: int, run: dict):
    attn = P.AttentionCost(attention)
    # the phase-2 rows: built with the attention cost only when the run balanced on it
    plan_cost = attn if run.get("plans_use_attention") else None
    m, size = run["accum"], run["dataset_size"]
    if (order_name, split) == ("random", "Evencut, whole optimizer step balanced (window)"):
        return lambda o, s, g, gb, steps, seed: P.window(o, s, g, gb, m, cost=plan_cost)
    if (order_name, split) == ("random", "Hugging Face batch_rebalance"):
        return lambda o, s, g, gb, steps, seed: P.hf_batch_rebalance(o, s, g, gb, m)
    if (order_name, split) == (POOLED, "Hugging Face group_by_length"):
        return lambda o, s, g, gb, steps, seed: P.hf_group_by_length(
            o, s, g, gb // g, m, mega_batch_mult=P.hf_mega_batch_mult(size, (gb // g) * m))
    if (order_name, split) == (POOLED, "Evencut, length-grouped steps + balanced"):
        return lambda o, s, g, gb, steps, seed: P.pooled(
            o, s, g, gb, m, pool_steps=P.default_pool_steps(g), seed=P.epoch_seed(seed, 0),
            balance="window" if m > 1 else "step", cost=plan_cost)
    if order_name == "noise floor":            # every GPU gets the same first slice
        return lambda o, s, g, gb, steps, seed: [[list(range(gb // g))] * g for _ in range(steps)]
    per = {"equal counts": ("equal", None), "Evencut, balanced tokens": ("tokens", None),
           "Evencut, balanced padded size": ("padded", None), "Evencut, padded size + attention": ("padded", attn)}
    if order_name in ("random", "length-grouped") and split in per:
        how, cost = per[split]
        return lambda o, s, g, gb, steps, seed: P.per_step(o, s, g, gb, how, grouped=order_name == "length-grouped",
                                                           cost=cost)
    if order_name == "any order":
        return lambda o, s, g, gb, steps, seed: P.lpt_any_order(o, s, g, gb)
    if order_name == "sorted dataset":
        kw = {"token budget per GPU (fairseq-style)": dict(method="fairseq"),
              "Evencut, padded_slices over the sorted data": dict(method="evencut"),
              "Evencut, padded + attention over the sorted data": dict(method="evencut", cost=attn)}[split]
        return lambda o, s, g, gb, steps, seed: P.sorted_dataset(o, s, g, steps, seed, **kw)
    return None


def load_run(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    a = r["args"]
    seed = int(a["seed"])
    if path.name.startswith("gpu_batches"):
        kind, gpus, gb, steps = "4060 sim", int(a["gpus"]), int(a["global_batch"]), int(a["steps_b"])
        accum = int(a.get("accum") or 1)
        seq = int(a["seq_len"])
        if a.get("lengths") not in (None, "None"):          # read, then shuffled by gpu_batches.py
            lens = [min(int(x), seq) for x in (LENGTHS / Path(a["lengths"]).name).read_text().split()]
            random.Random(seed).shuffle(lens)
        else:                                               # the saved lengths are already shuffled
            name = "tinystories_tr.txt" if a.get("tokens") not in (None, "None") else "dolly15k_gpt2.txt"
            lens = [min(int(x), seq) for x in (LENGTHS / name).read_text().split()]
        first = 0 if a["skip_a"] == "True" else int(a["stories_a"])
        rows, comm, model = r["part_b"], 0.0, "small"
        attention = r.get("attention_constant") or model_constant(**MODELS["small"])
    else:
        kind = f"{r['gpus']}x {r['gpu'].replace('NVIDIA ', '').replace('GeForce ', '')}"
        gpus, gb, steps, seq = r["gpus"], r["global_batch"], r["steps"], r["seq_len"]
        accum = r.get("accum") or 1
        lens = [min(int(x), seq) for x in (LENGTHS / Path(a["lengths"]).name).read_text().split()]
        first = int(a["offset"])
        rows, comm, model = r["rows"], r["allreduce_ms"], r["model"]
        attention = model_constant(**MODELS[model])
    stream = lens[first: first + steps * gb]
    return dict(name=path.stem, kind=kind, dataset=r["dataset"], model=model, gpus=gpus, gb=gb, steps=steps,
                accum=accum, seq=seq, stream=stream, rows=rows, comm=comm, attention=attention, seed=seed, dataset_size=len(lens),
                plans_use_attention=path.name.startswith("gpu_batches") and r.get("attention_constant") is not None)


def predict(run: dict) -> list[dict]:
    s, order = run["stream"], range(len(run["stream"]))
    attn = P.AttentionCost(run["attention"])
    out = []
    for row in run["rows"]:
        b = builder(row["order"], row["split"], run["attention"], run)
        if b is None:
            continue
        plan = b(order, s, run["gpus"], run["gb"], run["steps"], run["seed"])
        pad = P.evaluate(plan, s, run["gpus"], run["accum"])
        att = P.evaluate(plan, s, run["gpus"], run["accum"], cost=attn)
        out.append(dict(order=row["order"], split=row["split"], step_ms=row["step_ms"],
                        pad=pad.step_cost / pad.steps, att=att.step_cost / att.steps))
    return out


def compare(run: dict) -> list[dict]:
    rows = predict(run)
    base = next(r for r in rows if r["order"] == "random" and r["split"] == "equal counts")
    floor = next(r for r in rows if r["order"] == "noise floor")
    c = run["comm"]
    k = (base["step_ms"] - c) / base["att"] if base["step_ms"] > c else None
    # two measured points (a probe times two batch sizes): step = a + b * cost, a = fixed time per step
    b2 = (base["step_ms"] - floor["step_ms"]) / (base["att"] - floor["att"]) if base["att"] > 1.1 * floor["att"] else None
    a2 = base["step_ms"] - b2 * base["att"] if b2 and b2 > 0 else None
    run["fixed_ms"] = a2
    for r in rows:
        r["measured"] = base["step_ms"] / r["step_ms"]
        r["p_pad"] = base["pad"] / r["pad"]
        r["p_att"] = base["att"] / r["att"]
        r["p_comm"] = base["step_ms"] / (k * r["att"] + c) if k else None
        r["p_two"] = base["step_ms"] / (a2 + b2 * r["att"]) if a2 is not None else None
    return rows


def err(p: float, m: float) -> float:
    return p / m - 1.0


def main() -> None:
    files = sorted(HERE.glob("results/gpu_batches*.json")) + sorted(HERE.glob("results/ddp_*.json"))
    lines = ["# Backtest: `evencut-train diagnose` against measured runs", "",
             "Predicted vs measured speed-up over the default split (random order, equal counts), per data-parallel "
             "step. Produced by `bench/backtest_diagnose.py` from the runs in `bench/results` (the same lengths, "
             "seeds and plan code). Error = predicted / measured - 1.", "",
             "* **padded**: cost = padded tokens, compute-bound (no model shape, no times given).",
             "* **attention**: padded tokens plus causal attention for the run's model shape.",
             "* **+comm**: attention cost, plus the run's measured all-reduce as a fixed cost per optimizer step, "
             "scaled by the measured default step time (what `--step-ms` and `--fixed-ms` add, with the all-reduce as the fixed part).",
             "* **2 points**: attention cost, with step time = a + b x cost fitted to two measured rows, the default "
             "and the noise floor (all GPUs get the same slice), and the other rows predicted from it. A probe that "
             "times two batch sizes on the machine gives the same two points; `a` is the fixed time per step "
             "(kernel launches, optimizer, communication).", "",
             "The noise floor and default rows are the calibration points and are left out of the summary.", ""]
    summary: dict[str, dict[str, list[float]]] = {}
    for f in files:
        run = load_run(f)
        rows = compare(run)
        lines += [f"## {run['dataset']}, {run['kind']}, {run['model']} model, accumulation {run['accum']} "
                  f"(`{f.name}`)", "",
                  f"{run['gpus']} GPUs, {run['gb']} sequences per micro-step, {run['steps']} micro-steps, cut at "
                  f"{run['seq']}; attention constant {run['attention']}; all-reduce {run['comm']:.1f} ms"
                  + (f"; fitted fixed time a = {run['fixed_ms']:.1f} ms per optimizer step." if run.get('fixed_ms') is not None else "."), "",
                  "| order | split | measured | padded | attention | +comm | 2 points |", "|---|---|--:|--:|--:|--:|--:|"]
        group = summary.setdefault(run["kind"] + (" small" if "4090" in run["kind"] and run["model"] == "small"
                                                  and run["accum"] == 1 else ""), {})
        for r in rows:
            cells = [f"{r['measured']:.2f}x"]
            for key in ("p_pad", "p_att", "p_comm", "p_two"):
                if r[key] is None:
                    cells.append("")
                    continue
                e = err(r[key], r["measured"])
                cells.append(f"{r[key]:.2f}x ({e:+.0%})")
                if not (r["order"] == "random" and r["split"] == "equal counts") and r["order"] != "noise floor":
                    group.setdefault(key, []).append(abs(e))
            lines.append(f"| {r['order']} | {r['split']} | " + " | ".join(cells) + " |")
        lines.append("")
        cands = [r for r in rows if r["order"] != "noise floor" and r["split"] != "equal counts"]
        best = max(r["measured"] for r in cands)
        # what the diagnosis advises now: the best split that keeps every step's sequences
        same = [r for r in cands if r["order"] in ("random", "length-grouped", "any order")]
        best_same = max(r["measured"] for r in same)
        for key in ("p_pad", "p_att", "p_comm", "p_two"):
            if all(r[key] is not None for r in cands):
                pick = max(cands, key=lambda r: r[key])
                group.setdefault("pick_" + key, []).append(pick["measured"] / best)
                pick = max(same, key=lambda r: r[key])
                group.setdefault("same_" + key, []).append(pick["measured"] / best_same)
    lines += ["## Summary: absolute error of the predicted speed-up (median / worst)", "",
              "| runs | rows | padded | attention | +comm | 2 points |", "|---|--:|--:|--:|--:|--:|"]
    for kind, errs in summary.items():
        cells = []
        for key in ("p_pad", "p_att", "p_comm", "p_two"):
            v = errs.get(key)
            cells.append(f"{statistics.median(v):.0%} / {max(v):.0%}" if v else "")
        lines.append(f"| {kind} | {len(errs.get('p_pad', []))} | " + " | ".join(cells) + " |")
    lines += ["", "## Does the prediction pick the right split?", "",
              "For every run, the split with the highest predicted speed-up is taken; the table gives its measured "
              "speed as a share of the best measured split (worst run / runs within 3% of the best).", "",
              "| runs | runs | padded | attention | +comm | 2 points |", "|---|--:|--:|--:|--:|--:|"]
    for kind, errs in summary.items():
        cells = []
        for key in ("p_pad", "p_att", "p_comm", "p_two"):
            v = errs.get("pick_" + key)
            cells.append(f"{min(v):.0%} / {sum(x >= 0.97 for x in v)} of {len(v)}" if v else "")
        lines.append(f"| {kind} | {len(errs.get('pick_p_pad', []))} | " + " | ".join(cells) + " |")
    lines += ["", "Among splits that keep every step's sequences (what `diagnose` advises first; it shows length "
              "grouping and whole-dataset sorting only as cautioned extras):", "",
              "| runs | runs | padded | attention | +comm | 2 points |", "|---|--:|--:|--:|--:|--:|"]
    for kind, errs in summary.items():
        cells = []
        for key in ("p_pad", "p_att", "p_comm", "p_two"):
            v = errs.get("same_" + key)
            cells.append(f"{min(v):.0%} / {sum(x >= 0.97 for x in v)} of {len(v)}" if v else "")
        lines.append(f"| {kind} | {len(errs.get('same_p_pad', []))} | " + " | ".join(cells) + " |")
    out = HERE / "results" / "backtest_diagnose.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-(len(summary) + 3):]))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
