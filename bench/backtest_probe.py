"""Backtest of `evencut-train probe`: does a minute of measurement predict real step times?

A. Same machine. The probe of this PC's RTX 4060 (forward + backward of synthetic batches, one
   to 16k tokens) predicts every row of the 4060 simulation runs (bench/results/gpu_batches*.json:
   real data-parallel plans, each GPU's slices timed on the 4060, step = slowest GPU; no sync, no
   optimizer). The probe never saw those batches and ran on another day.
B. Machines that were not probed. What `evencut-train pick` does for a vast.ai offer: the 4060
   probe's time curve scaled by datasheet bf16 TFLOPS (evencut.train.pick.GPU_SPECS) to an RTX 4090
   or an A100, against the real DDP runs on 8x RTX 4090 and 4x A100 SXM (bench/results/ddp_*.json).
   The run's measured all-reduce is added, so this tests the compute estimate alone (the link
   constants in pick.LINK_EFFICIENCY were fitted to these two machines' all-reduce times).
D. Probed machines (--machine DIR=PRICE, written by bench/run_probe_check.sh on a rented box): each
   machine's own probe (time curve of its slowest GPU, its measured all-reduce, its optimizer step)
   against the real DDP runs on the same machine, measured right after the probe.
E. For the jobs run on every probed machine: which one is cheaper, measured, predicted from the
   machines' own probes, and estimated without probing them (the RTX 4060 probe scaled by datasheet
   TFLOPS, what `pick` says for an offer nobody probed).
F. Compute alone, by model size: on every probed machine, the time of a batch of 2k and 8k tokens
   measured by its probe against the datasheet estimate from the RTX 4060 probe of the same model
   shape (small 30M, base 91M, large 310M, and a 7B shape of which 2 layers were timed). Is the
   datasheet ratio closer for big models, whose batches keep a GPU busy?
C. The question `pick` answers: for the jobs run on both machines (same data, model and
   accumulation; the A100 runs used half the sequences per step), which machine finishes them for
   less money at the prices paid (8x RTX 4090 $2.90/h, 4x A100 SXM $4.28/h)? Measured against the
   section-B estimates, with the default split and with Evencut's.

Baseline in both: the no-probe cost model of `diagnose`, scaled so the default row is exact
(what `diagnose --step-ms --fixed-ms` does). It needs a measured step; the probe does not.

Run: python bench/backtest_probe.py --probe small.json --probe-base base.json
     (probe files: evencut-train probe --layers 8 --d-model 512 --vocab 8192 --max-len 2048, and
      --layers 12 --d-model 768 for base; no GPU needed to run this script)
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))

import backtest_diagnose as B  # noqa: E402
from evencut.train import _plans as P  # noqa: E402
from evencut.train import pick as PK  # noqa: E402
from evencut.train.timing import Probe, Sync  # noqa: E402

SAME_STEPS = {("random", "equal counts"), ("random", "Evencut, balanced padded size"),
              ("random", "Evencut, whole optimizer step balanced (window)"), ("random", "Hugging Face batch_rebalance")}


def gpu_of(run_json: dict) -> str:
    return run_json.get("gpu", "")


def offer_estimate(gpu: str, gpus: int, ref):
    """pick's estimate of a machine of ``gpus`` x ``gpu`` from the probe ``ref``, or None when the
    card has no bf16 figures."""
    try:
        return PK.from_offer({"gpu_name": gpu, "num_gpus": gpus, "dph_total": 0}, ref)
    except PK.EvencutError:
        return None


def predict_rows(run: dict, curve, fixed_ms: float, exposed=None) -> list[dict]:
    s, order = run["stream"], range(len(run["stream"]))
    attn = P.AttentionCost(run["attention"])
    out = []
    for row in run["rows"]:
        b = B.builder(row["order"], row["split"], run["attention"], run)
        if b is None:
            continue
        plan = b(order, s, run["gpus"], run["gb"], run["steps"], run["seed"])
        st = P.evaluate(plan, s, run["gpus"], run["accum"], cost=attn, timing=curve, exposed=exposed)
        out.append(dict(order=row["order"], split=row["split"], measured=row["step_ms"],
                        probe=(st.step_ms + st.sync_ms) / st.steps + fixed_ms, cost=st.step_cost / st.steps))
    base = next(r for r in out if r["order"] == "random" and r["split"] == "equal counts")
    # baseline: cost model scaled by the measured default row, the run's sync as its fixed part
    k = (base["measured"] - run["comm"]) / base["cost"]
    for r in out:
        r["costmodel"] = k * r["cost"] + run["comm"]
    return out


def err(p: float, m: float) -> float:
    return p / m - 1.0


def regret(rows: list[dict], key: str, pool) -> float:
    """Measured time of the predicted-best row over the measured-best row, among ``pool`` rows."""
    cand = [r for r in rows if pool(r)]
    pick = min(cand, key=lambda r: r[key])
    return pick["measured"] / min(r["measured"] for r in cand) - 1.0


def summarize(rows: list[dict]) -> dict:
    base = next(r for r in rows if r["order"] == "random" and r["split"] == "equal counts")
    skip = lambda r: r is base  # noqa: E731
    out = {}
    for key in ("probe", "costmodel"):
        abs_e = [abs(err(r[key], r["measured"])) for r in rows if not (key == "costmodel" and skip(r))]
        sp_e = [abs(err(base[key] / r[key], base["measured"] / r["measured"])) for r in rows if not skip(r)]
        out[key] = dict(abs=statistics.median(abs_e), abs_worst=max(abs_e), speed=statistics.median(sp_e),
                        speed_worst=max(sp_e),
                        regret_all=regret(rows, key, lambda r: r["order"] != "noise floor"),
                        regret_same=regret(rows, key, lambda r: (r["order"], r["split"]) in SAME_STEPS))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="RTX 4060 probe of the small model (8 x 512, vocab 8192)")
    ap.add_argument("--probe-base", help="RTX 4060 probe of the base model (12 x 768, vocab 8192)")
    ap.add_argument("--probe-large", help="RTX 4060 probe of the large model (24 x 1024, vocab 8192)")
    ap.add_argument("--probe-7b", help="RTX 4060 probe of the 7B shape (32 x 4096, vocab 32000, 2 layers timed)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "backtest_probe")
    ap.add_argument("--machine", action="append", default=[], metavar="DIR=PRICE",
                    help="output of bench/run_probe_check.sh on a rented machine and its price per hour")
    args = ap.parse_args()
    probes = {"small": Probe.load(args.probe)}
    if args.probe_base:
        probes["base"] = Probe.load(args.probe_base)
    if args.probe_large:
        probes["large"] = Probe.load(args.probe_large)
    if args.probe_7b:
        probes["7b"] = Probe.load(args.probe_7b)
    for name, p in probes.items():
        if name not in B.MODELS:
            continue
        want = B.model_constant(**B.MODELS[name])
        if p.curve.a != want:
            raise SystemExit(f"{name} probe measured another model (a={p.curve.a}, expected {want})")

    report = {"date": dt.date.today().isoformat(), "probes": {k: dict(gpu=p.gpu, date=p.date, source=p.source)
                                                               for k, p in probes.items()}, "runs": []}
    for path in sorted(glob.glob(str(HERE / "results" / "gpu_batches*.json"))) + \
            sorted(glob.glob(str(HERE / "results" / "ddp_*.json"))):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "part_b" in raw and not raw["part_b"]:
            continue
        run = B.load_run(Path(path))
        probe = probes.get(run["model"])
        if probe is None:
            report["runs"].append(dict(name=run["name"], skipped=f"no probe of the {run['model']} model"))
            continue
        if run["kind"] == "4060 sim":
            curve, fixed, how = probe.curve, 0.0, "measured (same GPU)"
        else:
            cand = offer_estimate(gpu_of(raw), run["gpus"], probe)
            if cand is None:
                report["runs"].append(dict(name=run["name"], skipped=f"no bf16 figures for {gpu_of(raw)}"))
                continue
            curve = cand.curve
            fixed = run["comm"] + cand.optimizer_ms
            how = f"estimated as pick does for an offer ({', '.join(cand.notes)}), floor kept"
        rows = predict_rows(run, curve, fixed)
        report["runs"].append(dict(name=run["name"], kind=run["kind"], dataset=run["dataset"], model=run["model"],
                                   gpus=run["gpus"], gb=run["gb"], accum=run["accum"], how=how, rows=rows,
                                   summary=summarize(rows)))
        s = report["runs"][-1]["summary"]
        print(f"{run['name']:34} {run['kind']:14} probe |err| {s['probe']['abs']:5.0%} (speed-up {s['probe']['speed']:5.0%})"
              f"   cost model speed-up {s['costmodel']['speed']:5.0%}   regret probe {s['probe']['regret_same']:4.0%}"
              f"/{s['probe']['regret_all']:4.0%} cost {s['costmodel']['regret_same']:4.0%}/{s['costmodel']['regret_all']:4.0%}")
    report["ranking"] = ranking(report["runs"])
    report["machines"] = [probed_machine(m, probes) for m in args.machine]
    report["scaling"] = scaling(probes, [m.rpartition("=")[0] for m in args.machine])
    if report["machines"]:
        report["machine_ranking"] = machine_ranking(report["machines"])
    for c in report["ranking"]:
        print(f"{c['job']:40} {c['split']:8} measured cheaper: {c['measured_winner']:5} predicted: {c['predicted_winner']:5}"
              f"  $/1M seq measured {c['measured']['4090']:.3f} / {c['measured']['A100']:.3f}"
              f"  predicted {c['predicted']['4090']:.3f} / {c['predicted']['A100']:.3f}")
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, probes, args.out.with_suffix(".md"))
    print(f"written to {args.out.with_suffix('.md')}")


def probed_machine(arg: str, ref_probes: dict) -> dict:
    """Section D for one rented machine, and the inputs of section E."""
    path, _, price = arg.rpartition("=")
    folder = Path(path)
    own = {m: Probe.load(folder / f"probe_{m}.json") for m in ("small", "base", "large")
           if (folder / f"probe_{m}.json").exists()}
    out = {"dir": folder.name, "price": float(price), "runs": [], "probes": {}}
    for m, p in own.items():
        out["probes"][m] = dict(gpu=p.gpu, gpus=p.gpus, links=p.machine.get("links"), p2p=p.machine.get("p2p"),
                                overlap=p.overlap, sync_points=p.sync_points,
                                tflops=p.tflops, watts=p.watts, launch_ms=p.launch_ms, optimizer_ms=p.optimizer_ms,
                                allreduce_ms=p.allreduce_ms(), bus_gbps=p.allreduce.bus_gbps() if p.allreduce else None,
                                gpu_speed=p.gpu_speed, max_tokens_fit=p.max_tokens_fit)
    for f in sorted(folder.glob("ddp_*.json")):
        run = B.load_run(f)
        raw = json.loads(f.read_text(encoding="utf-8"))
        probe = own.get(run["model"])
        if probe is None:
            continue
        rows = predict_rows(run, probe.curve, probe.optimizer_ms, exposed=probe.sync(run["gpus"]))
        if probe.measured_sync:                  # the same prediction with the sync counted in full
            for r, full in zip(rows, predict_rows(run, probe.curve, probe.optimizer_ms,
                                                  exposed=Sync(probe.allreduce_ms(run["gpus"]), 0.0))):
                r["no_overlap"] = full["probe"]
        ref = ref_probes.get(run["model"])
        est = None
        cand = offer_estimate(raw.get("gpu", ""), run["gpus"], ref) if ref is not None else None
        if cand is not None:                     # what pick would estimate without this machine's probe
            est = predict_rows(run, cand.curve, run["comm"] + cand.optimizer_ms)
            for r, e in zip(rows, est):
                r["estimate"] = e["probe"]
        out["runs"].append(dict(name=run["name"], kind=run["kind"], dataset=run["dataset"], model=run["model"],
                                gpus=run["gpus"], gb=run["gb"], accum=run["accum"], rows=rows,
                                summary=summarize(rows), allreduce_measured=run["comm"],
                                allreduce_probe=probe.allreduce_ms(run["gpus"])))
    return out


def machine_ranking(machines: list[dict]) -> list[dict]:
    """$ per million sequences for every job that ran on all probed machines."""
    jobs: dict = {}
    for mach in machines:
        for x in mach["runs"]:
            jobs.setdefault((x["dataset"], x["model"], x["accum"]), {})[mach["dir"]] = (mach, x)
    out = []
    for (data, model, accum), per in sorted(jobs.items()):
        if len(per) < 2:                         # compare the machines that ran this job
            continue
        for split, key in SPLITS.items():
            cost = {"measured": {}, "probe": {}, "estimate": {}}
            for name, (mach, x) in per.items():
                row = next(r for r in x["rows"] if (r["order"], r["split"]) == key)
                for kind in cost:
                    if kind in row:
                        cost[kind][name] = row[kind] / 1000 / (x["gb"] * x["accum"]) * 1e6 / 3600 * mach["price"]
            win = {k: min(v, key=v.get) for k, v in cost.items() if len(v) == len(per)}
            out.append(dict(job=f"{data}, {model} model, accumulation {accum}", split=split, cost=cost, winner=win))
    return out


SCALING_SHAPES = ("small", "base", "large", "7b")


def scaling(ref: dict, machine_dirs: list) -> list:
    """Section F: datasheet estimate vs probe measurement of one batch, by model shape."""
    out = []
    for d in machine_dirs:
        for shape in SCALING_SHAPES:
            f = Path(d) / f"probe_{shape}.json"
            if shape not in ref or not f.exists():
                continue
            p, r = Probe.load(f), ref[shape]
            target, base = PK.gpu_spec(p.gpu), PK.gpu_spec(r.gpu)
            if target is None or base is None:
                continue                         # no datasheet figure to scale by
            ratio = base[0] / target[0]          # target time / 4060 time, by datasheet
            est = PK.estimate_curve(r, ratio)
            for tokens in (2048, 8192):
                c = p.curve.cost(tokens // 512, 512)
                meas = p.curve.ms(c)
                out.append(dict(machine=Path(d).name, gpu=p.gpu, shape=shape, tokens=tokens, measured_ms=meas,
                                estimate_ms=est.ms(c), datasheet_speedup=1 / ratio,
                                measured_speedup=r.curve.ms(c) / meas))
    return out


PRICES = {"4090": 2.90, "A100": 4.28}
SPLITS = {"default": ("random", "equal counts"), "Evencut": ("random", "Evencut, balanced padded size")}


def ranking(runs: list[dict]) -> list[dict]:
    """$ per million sequences on each machine, measured and estimated, for jobs run on both."""
    jobs: dict = {}
    for x in runs:
        if "rows" not in x or x["kind"] == "4060 sim" or x["name"].endswith("lpt"):
            continue
        machine = "A100" if "A100" in x["kind"] else "4090" if "4090" in x["kind"] else None
        if machine is None:                      # only the two machine types with known prices here
            continue
        jobs.setdefault((x["dataset"], x["model"], x["accum"]), {})[machine] = x
    out = []
    for (data, model, accum), both in sorted(jobs.items()):
        if len(both) < 2:
            continue
        for split, key in SPLITS.items():
            per = {}
            for kind in ("measured", "probe"):
                per[kind] = {}
                for machine, x in both.items():
                    row = next(r for r in x["rows"] if (r["order"], r["split"]) == key)
                    seq_per_step = x["gb"] * x["accum"]
                    per[kind][machine] = row[kind] / 1000 / seq_per_step * 1e6 / 3600 * PRICES[machine]
            win = lambda d: min(d, key=d.get)  # noqa: E731
            out.append(dict(job=f"{data}, {model} model, accumulation {accum}", split=split,
                            measured=per["measured"], predicted=per["probe"], measured_winner=win(per["measured"]),
                            predicted_winner=win(per["probe"])))
    return out


def pct(x: float) -> str:
    return f"{100 * x:+.0f}%"


def _med(runs: list, model: str, field: str) -> float:
    return statistics.median(x["summary"][model][field] for x in runs)


def write_markdown(r: dict, probes: dict, path: Path) -> None:
    runs = [x for x in r["runs"] if "rows" in x]
    groups = (("A. Same machine: RTX 4060 probe vs the RTX 4060 simulation runs", True),
              ("B. Not probed: the 4060 probe scaled by datasheet TFLOPS vs real DDP runs", False))
    lines = ["# Backtest: does a one-minute probe predict step times?", "",
             f"{r['date']}. Produced by `bench/backtest_probe.py` from the probe files "
             + ", ".join(f"{k}: {v['gpu']} ({v['date']})" for k, v in r["probes"].items())
             + ". *probe* = predicted from the probe's time curve with no measurement of the run; *cost model* = the "
               "no-probe model scaled so the default row is exact (it needs one measured step). Errors are medians "
               "over the rows of a run; *speed-up* = error of the predicted speed-up over the default row; "
               "*regret* = how much slower the predicted-best split measured than the measured-best one "
               "(same-steps splits / all splits).", ""]
    for title, same in groups:
        sel = [x for x in runs if (x["kind"] == "4060 sim") == same]
        if not sel:
            continue
        lines += [f"## {title}", "",
                  "| run | GPUs | model | step time error (probe) | speed-up error: probe | cost model | regret: probe | cost model |",
                  "|---|--:|---|--:|--:|--:|--:|--:|"]
        for x in sel:
            s = x["summary"]
            lines.append(f"| {x['name']} | {x['gpus']} x {x['kind'].split('x ')[-1] if 'x ' in x['kind'] else '4060'} "
                         f"| {x['model']} | {s['probe']['abs']:.0%} (worst {s['probe']['abs_worst']:.0%}) "
                         f"| {s['probe']['speed']:.0%} | {s['costmodel']['speed']:.0%} "
                         f"| {s['probe']['regret_same']:.0%} / {s['probe']['regret_all']:.0%} "
                         f"| {s['costmodel']['regret_same']:.0%} / {s['costmodel']['regret_all']:.0%} |")
        lines += ["", f"Median over runs: step time error {_med(sel, 'probe', 'abs'):.0%}; speed-up error probe "
                  f"{_med(sel, 'probe', 'speed'):.0%} vs cost model {_med(sel, 'costmodel', 'speed'):.0%}.", ""]
    rk = r.get("ranking") or []
    if rk:
        right = sum(c["measured_winner"] == c["predicted_winner"] for c in rk)
        lines += ["## C. Which machine is cheaper for the job?", "",
                  "Dollars per million training sequences at the prices paid (8x RTX 4090 $2.90/h, 4x A100 SXM "
                  "$4.28/h); *estimated* = section B (4060 probe scaled by datasheet TFLOPS, measured all-reduce).", "",
                  "| job | split | 8x 4090 measured | estimated | 4x A100 measured | estimated | cheaper (measured) | cheaper (estimated) |",
                  "|---|---|--:|--:|--:|--:|---|---|"]
        for c in rk:
            m, p = c["measured"], c["predicted"]
            lines.append(f"| {c['job']} | {c['split']} | ${m['4090']:.3f} | ${p['4090']:.3f} | ${m['A100']:.3f} | "
                         f"${p['A100']:.3f} | {c['measured_winner']} | {c['predicted_winner']} |")
        lines += ["", f"The estimate names the cheaper machine in {right} of {len(rk)} cases.", ""]
    for mach in r.get("machines") or []:
        runs_m = mach["runs"]
        pr = mach["probes"].get("small") or next(iter(mach["probes"].values()), {})
        lines += [f"## D. Probed: {pr.get('gpus')} x {pr.get('gpu')} (${mach['price']:.3f}/h, `{mach['dir']}`)", "",
                  f"Links {pr.get('links')}, peer-to-peer {pr.get('p2p')}; bf16 matmul "
                  + ", ".join(f"{t:.0f}" for t in (pr.get("tflops") or []) if t) + " TFLOPS per GPU; launch floor "
                  f"{(pr.get('launch_ms') or 0):.1f} ms (small model); DDP hides "
                  + (("part of the sync; still exposed: " + ", ".join(
                         f"{100 * f:.0f}% after a {t:.0f} ms batch" for t, f in pr["sync_points"]))
                     if pr.get("sync_points") else
                     f"{pr['overlap']:.0%} of the last micro-batch's time worth of sync (proportional model)"
                     if pr.get("overlap") is not None
                     else "an unmeasured part of the sync (older probe: sync counted in full)") + "; power "
                  + (", ".join(f"{w:.0f} W" for w in pr["watts"] if w) if pr.get("watts") else "not readable") + ".", "",
                  "| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) "
                  "| same, sync counted in full | speed-up error | regret | step time error without the probe "
                  "(datasheet estimate) |",
                  "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
        for x in runs_m:
            sm = x["summary"]["probe"]
            est = [abs(err(row["estimate"], row["measured"])) for row in x["rows"] if "estimate" in row]
            est_txt = f"{statistics.median(est):.0%}" if est else "-"
            full = [abs(err(row["no_overlap"], row["measured"])) for row in x["rows"] if "no_overlap" in row]
            full_txt = f"{statistics.median(full):.0%}" if full else "(same)"
            lines.append(f"| {x['name']} | {x['model']} | {x['allreduce_measured']:.1f} ms | {x['allreduce_probe']:.1f} ms "
                         f"| {sm['abs']:.0%} (worst {sm['abs_worst']:.0%}) | {full_txt} | {sm['speed']:.0%} "
                         f"| {sm['regret_same']:.0%} / {sm['regret_all']:.0%} | {est_txt} |")
        if runs_m:
            lines += ["", f"Median over runs: step time error {_med(runs_m, 'probe', 'abs'):.0%}.", ""]
        for x in runs_m:
            lines += [f"#### {x['name']} ({x['dataset']}, {x['model']}, {x['gb']} sequences per micro-step, accumulation "
                      f"{x['accum']})", "", "| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |",
                      "|---|---|--:|--:|--:|--:|--:|"]
            for row in x["rows"]:
                e = row.get("estimate")
                lines.append(f"| {row['order']} | {row['split']} | {row['measured']:.1f} | {row['probe']:.1f} | "
                             f"{pct(err(row['probe'], row['measured']))} | "
                             + (f"{e:.1f} | {pct(err(e, row['measured']))} |" if e else "- | - |"))
            lines.append("")
    mr = r.get("machine_ranking") or []
    if mr:
        lines += ["## E. Which probed machine is cheaper for the job?", "",
                  "Dollars per million training sequences at the prices paid, for every job run on more than one "
                  "probed machine; *probe* = predicted from each machine's own probe, *estimate* = from the RTX 4060 "
                  "probe and datasheets, without probing the machine. Cheapest in bold.", ""]
        for job in dict.fromkeys(c["job"] for c in mr):
            cases = [c for c in mr if c["job"] == job]
            lines += [f"**{job}**", "", "| split | machine | measured | probe | estimate |", "|---|---|--:|--:|--:|"]
            for c in cases:
                for n in sorted(c["cost"]["measured"], key=c["cost"]["measured"].get):
                    cells = []
                    for k in ("measured", "probe", "estimate"):
                        v = c["cost"][k].get(n)
                        txt = f"${v:.3f}" if v is not None else "-"
                        cells.append(f"**{txt}**" if c["winner"].get(k) == n else txt)
                    lines.append(f"| {c['split']} | {n} | " + " | ".join(cells) + " |")
            lines.append("")
        ok_p = sum(c["winner"].get("probe") == c["winner"]["measured"] for c in mr)
        ok_e = sum(c["winner"].get("estimate") == c["winner"]["measured"] for c in mr)
        lines += ["", f"Probes name the cheaper machine in {ok_p} of {len(mr)} cases; datasheet estimates in {ok_e}.", ""]
    sc = r.get("scaling") or []
    if sc:
        lines += ["## F. Compute alone, by model size: datasheet estimate vs measurement", "",
                  "One micro-batch of 512-token sequences, forward + backward, on the slowest GPU. *Speed-up* = how "
                  "much faster than the RTX 4060 (same model shape): by datasheet bf16 TFLOPS, and as measured. "
                  "*Error* = datasheet estimate (the 4060 probe scaled, with its launch floor) over the measurement.",
                  "", "| model | machine | GPU | datasheet speed-up | measured, 2k tokens | measured, 8k tokens "
                  "| error, 2k tokens | error, 8k tokens |", "|---|---|---|--:|--:|--:|--:|--:|"]
        for shape in SCALING_SHAPES:
            for mach in dict.fromkeys(x["machine"] for x in sc if x["shape"] == shape):
                two = {x["tokens"]: x for x in sc if x["shape"] == shape and x["machine"] == mach}
                a, b = two[2048], two[8192]
                lines.append(f"| {shape} | {mach} | {a['gpu'].replace('NVIDIA ', '').replace('GeForce ', '')} "
                             f"| {a['datasheet_speedup']:.1f}x | {a['measured_speedup']:.1f}x | {b['measured_speedup']:.1f}x "
                             f"| {pct(err(a['estimate_ms'], a['measured_ms']))} | {pct(err(b['estimate_ms'], b['measured_ms']))} |")
        lines.append("")
        for shape in SCALING_SHAPES:
            e = [abs(err(x["estimate_ms"], x["measured_ms"])) for x in sc if x["shape"] == shape]
            if e:
                lines.append(f"- {shape}: median |error| of the datasheet estimate {statistics.median(e):.0%} "
                             f"(worst {max(e):.0%}, {len(e)} cases)")
        lines.append("")
    lines += ["## Every row", ""]
    for x in runs:
        lines += [f"### {x['name']} ({x['kind']}, {x['dataset']}, {x['model']} model, {x['gpus']} GPUs, "
                  f"{x['gb']} sequences per micro-step, accumulation {x['accum']})", "", f"Probe: {x['how']}.", "",
                  "| data order | split | measured ms | probe ms | error | cost model ms | error |",
                  "|---|---|--:|--:|--:|--:|--:|"]
        for row in x["rows"]:
            lines.append(f"| {row['order']} | {row['split']} | {row['measured']:.1f} | {row['probe']:.1f} | "
                         f"{pct(err(row['probe'], row['measured']))} | {row['costmodel']:.1f} | "
                         f"{pct(err(row['costmodel'], row['measured']))} |")
        lines.append("")
    skipped = [x for x in r["runs"] if "skipped" in x]
    for x in skipped:
        lines.append(f"- {x['name']}: skipped, {x['skipped']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
