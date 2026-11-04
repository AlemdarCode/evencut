"""``evencut-train``: find wasted GPU time in data-parallel training.

    evencut-train diagnose lengths.txt --gpus 8 --global-batch 256 --accum 4 --max-len 2048 \\
        --run-hours 40 --price-per-gpu-hour 2.5 --gpu "A100 SXM"
    evencut-train lengths train.jsonl --fields prompt,response --tokenizer gpt2 -o lengths.txt
    torchrun --standalone --nproc-per-node=gpu -m evencut.train probe --layers 32 --d-model 4096         --vocab 32000 --max-len 4096 --probe-layers 4 -o machine.json
    evencut-train diagnose lengths.txt --gpus 8 --per-device-batch 4 --probe machine.json
    evencut-train pick lengths.txt --sequences-per-step 256 --per-device-batch 8 --probe a.json b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .. import EvencutError, __version__
from . import diagnose as dg
from .timing import Probe

_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_count(text: str) -> int:
    """'29.9M' -> 29900000, '7B' -> 7000000000, '91245312' -> 91245312."""
    t = text.strip().lower().replace("_", "").replace(",", "")
    mult = 1.0
    if t and t[-1] in _SUFFIX:
        mult, t = _SUFFIX[t[-1]], t[:-1]
    try:
        v = float(t) * mult
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a count: {text!r}") from None
    if v < 1 or v != v:
        raise argparse.ArgumentTypeError(f"not a positive count: {text!r}")
    return round(v)


def read_lengths(path: str, field: Optional[str] = None) -> list[int]:
    """One length per whitespace-separated token (.txt and anything else), a JSON list (.json),
    one record per line with ``field`` (.jsonl: an integer, or a list whose size is the length),
    or a NumPy array (.npy, needs numpy). ``-`` reads text from standard input."""
    if path == "-":
        text = sys.stdin.read()
        return [int(x) for x in text.lstrip("﻿").split()]
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        try:
            import numpy as np
        except ImportError:
            raise EvencutError("evencut: reading .npy needs numpy") from None
        return [int(x) for x in np.load(p, allow_pickle=False).ravel().tolist()]
    text = p.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and field:
            data = data[field]
        return [len(x) if isinstance(x, (list, str)) else int(x) for x in data]
    if suffix == ".jsonl":
        if not field:
            raise EvencutError("evencut: a .jsonl file needs --field (an integer length, or a token list)")
        out = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                v = json.loads(line).get(field)
                if v is None:
                    raise EvencutError(f"evencut: line {n} has no field {field!r}")
                out.append(len(v) if isinstance(v, list) else int(v))
        return out
    try:
        return [int(x) for x in text.split()]
    except ValueError as e:
        raise EvencutError(f"evencut: {path} must hold whole numbers: {e}") from None


def tokenize_jsonl(path: str, fields: list[str], tokenizer: str, add: int = 1) -> list[int]:
    """Token count (+``add`` for an end-of-text token) of every record, fields joined by newlines.
    Needs the ``tokenizers`` package (``pip install evencut[train]``)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise EvencutError("evencut: tokenizing needs the tokenizers package (pip install tokenizers)") from None
    tok = Tokenizer.from_file(tokenizer) if Path(tokenizer).is_file() else Tokenizer.from_pretrained(tokenizer)
    texts = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                texts.append("\n".join(str(rec[k]) for k in fields if rec.get(k)))
    return [len(e.ids) + add for e in tok.encode_batch(texts)]


def _positive(kind):
    def parse(text: str):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not v > 0:
            raise argparse.ArgumentTypeError(f"must be > 0: {text!r}")
        return v
    return parse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evencut-train", description="Find and fix wasted GPU time in "
                                 "data-parallel training.")
    ap.add_argument("--version", action="version", version=f"evencut {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diagnose", help="how much GPU time padding and waiting cost (no GPU needed)",
                       description="Replays your batching on your sequence lengths and splits the GPU time "
                                   "into useful work, padding, waiting for the slowest GPU and (if given) "
                                   "gradient sync, for the setup you run now and for the alternatives.")
    d.add_argument("lengths", help="sequence lengths: .txt (one per line), .json, .jsonl (with --field), .npy, or -")
    d.add_argument("--field", help="field holding the length (or token list) in .jsonl/.json input")
    g = d.add_argument_group("training setup")
    g.add_argument("--gpus", type=_positive(int), required=True, help="data-parallel GPUs (ranks)")
    b = g.add_mutually_exclusive_group(required=True)
    b.add_argument("--global-batch", type=_positive(int), help="sequences per micro-step over all GPUs")
    b.add_argument("--per-device-batch", type=_positive(int), help="sequences per micro-step per GPU")
    g.add_argument("--accum", type=_positive(int), default=1, help="gradient accumulation steps (default 1)")
    g.add_argument("--max-len", type=_positive(int), help="sequences are cut at this length")
    g.add_argument("--pad", choices=("longest", "max_length"), default="longest",
                   help="pad to each batch's longest (default) or always to --max-len")
    g.add_argument("--pad-multiple", type=_positive(int), default=1, help="pad lengths up to a multiple of this")
    g.add_argument("--current", choices=dg.CURRENT_CHOICES, default="default",
                   help="the batching you use now (default: DistributedSampler)")
    g.add_argument("--fsdp", action="store_true",
                   help="GPUs synchronise every micro-step (FSDP, ZeRO-3) instead of once per optimizer step")
    g.add_argument("--pool-steps", type=_positive(int), help="optimizer steps sorted together by the "
                   "length-grouped strategies (default: as Hugging Face group_by_length, 50 / GPUs)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--sample-steps", type=_positive(int), default=400, help="optimizer steps to replay (default 400)")
    m = d.add_argument_group("model (prices attention; leave out for short sequences)")
    m.add_argument("--params", type=parse_count, help="weights, e.g. 125M or 7B")
    m.add_argument("--layers", type=_positive(int))
    m.add_argument("--d-model", type=_positive(int), help="hidden width")
    c = d.add_argument_group("hours, money, energy")
    t = c.add_mutually_exclusive_group()
    t.add_argument("--run-hours", type=_positive(float), help="wall time of the whole run as configured now")
    t.add_argument("--step-ms", type=_positive(float), help="measured time of one optimizer step now")
    c.add_argument("--epochs", type=_positive(float), default=1.0, help="epochs the run covers (default 1)")
    c.add_argument("--fixed-ms", type=float, default=0.0, help="part of an optimizer step that does not grow with "
                   "the batch: gradient sync, optimizer (needs --step-ms or --run-hours, or --probe)")
    c.add_argument("--probe", help="probe file of the machine (evencut-train probe): times every batch by measurement "
                   "and predicts hours without --step-ms; sets the model shape, power and price it recorded")
    pr = c.add_mutually_exclusive_group()
    pr.add_argument("--price-per-gpu-hour", type=_positive(float))
    pr.add_argument("--price-per-hour", type=_positive(float), help="price of the whole machine per hour")
    w = c.add_mutually_exclusive_group()
    w.add_argument("--gpu", help="GPU model for its power figure, e.g. 'A100 SXM', 'RTX 4090'")
    w.add_argument("--watts", type=_positive(float), help="average power per GPU")
    o = d.add_argument_group("output")
    o.add_argument("--json", type=Path, help="also write the full result as JSON")
    o.add_argument("--quiet", action="store_true", help="no report on standard output")

    pb = sub.add_parser("probe", help="measure this machine for your model shape (needs torch; run with torchrun)",
                        description="Times forward + backward of a synthetic decoder of your model's shape on "
                                    "batch sizes from one short sequence up to --max-tokens, on every GPU at once, "
                                    "plus gradient all-reduce, the optimizer step, matmul speed, power and GPU "
                                    "links. Run it with torchrun --standalone --nproc-per-node=gpu -m evencut.train "
                                    "probe ... to include all GPUs. Writes a file for diagnose --probe and pick.")
    pm = pb.add_argument_group("model shape")
    pm.add_argument("--layers", type=_positive(int), required=True)
    pm.add_argument("--d-model", type=_positive(int), required=True, help="hidden width")
    pm.add_argument("--vocab", type=_positive(int), required=True, help="vocabulary size")
    pm.add_argument("--heads", type=_positive(int), help="attention heads (default: width / 64)")
    pm.add_argument("--probe-layers", type=_positive(int), help="time only this many layers and scale up "
                    "(for models that do not fit one GPU)")
    ps = pb.add_argument_group("batches")
    ps.add_argument("--max-len", type=_positive(int), required=True, help="longest sequence you train on")
    ps.add_argument("--max-tokens", type=_positive(int), help="largest padded batch per GPU to time "
                    "(default: max(16 x max-len, 16384) capped at 65536)")
    ps.add_argument("--reps", type=_positive(int), default=3, help="timed repeats per shape and round (default 3)")
    ps.add_argument("--rounds", type=_positive(int), default=3, help="passes over all shapes, each in a new "
                    "order (default 3)")
    ps.add_argument("--power-seconds", type=float, default=4.0, help="seconds of full load to read power (0: skip)")
    pb.add_argument("--price-per-hour", type=_positive(float), help="what this machine costs, kept in the file")
    pb.add_argument("--device", help="torch device (default: this rank's GPU)")
    pb.add_argument("--backend", choices=("nccl", "gloo"), help="torch.distributed backend (default: NCCL on Linux GPUs)")
    pb.add_argument("-o", "--out", default="probe.json", help="file to write (default probe.json)")
    pb.add_argument("--quiet", action="store_true")

    pk = sub.add_parser("pick", help="rank machines by the money (or time) your job takes on them",
                        description="Replays your job on every candidate machine with its measured time curve "
                                    "(probe files) or an estimate from one (vast.ai offers), with the GPU count, "
                                    "per-GPU batch and accumulation that keep your sequences per optimizer step, "
                                    "and ranks them by total cost or wall time.")
    pk.add_argument("lengths", help="sequence lengths, as for diagnose")
    pk.add_argument("--field", help="field holding the length in .jsonl/.json input")
    pj = pk.add_argument_group("job")
    pj.add_argument("--sequences-per-step", type=_positive(int), required=True,
                    help="sequences per optimizer step over all GPUs (global batch x accumulation)")
    pj.add_argument("--per-device-batch", type=_positive(int), required=True,
                    help="most sequences one GPU holds per micro-step (memory)")
    pj.add_argument("--max-len", type=_positive(int), help="sequences are cut at this length")
    pj.add_argument("--pad-multiple", type=_positive(int), default=1)
    pj.add_argument("--epochs", type=_positive(float), default=1.0)
    pj.add_argument("--seed", type=int, default=0)
    pj.add_argument("--sample-steps", type=_positive(int), default=200, help="optimizer steps to replay (default 200)")
    pc = pk.add_argument_group("machines")
    pc.add_argument("--probe", nargs="+", required=True, help="probe files of measured machines (same model shape)")
    pc.add_argument("--offers", help="vast.ai offers to estimate: the JSON of `vastai search offers --raw`")
    pc.add_argument("--price", action="append", default=[], metavar="FILE=DOLLARS",
                    help="price per hour of a probed machine, by probe file name (overrides the file)")
    pc.add_argument("--reference", help="probe file the offer estimates scale from (default: the largest tested)")
    pk.add_argument("--by", choices=("cost", "time"), default="cost")
    pk.add_argument("--top", type=_positive(int), default=15, help="rows to show (default 15)")
    pk.add_argument("--json", type=Path, help="also write the ranking as JSON")

    ln = sub.add_parser("lengths", help="token lengths of a JSON-lines dataset (needs tokenizers)")
    ln.add_argument("jsonl", help="one JSON record per line")
    ln.add_argument("--fields", default="text", help="comma-separated text fields, joined by newlines")
    ln.add_argument("--tokenizer", default="gpt2", help="tokenizer.json path or Hugging Face tokenizer name")
    ln.add_argument("--add", type=int, default=1, help="tokens added per record, e.g. end-of-text (default 1)")
    ln.add_argument("-o", "--out", type=Path, help="write here instead of standard output")
    return ap


def run_diagnose(a: argparse.Namespace) -> int:
    if a.pad == "max_length" and not a.max_len:
        raise EvencutError("evencut: --pad max_length needs --max-len")
    global_batch = a.global_batch or a.per_device_batch * a.gpus
    model = None
    given = [x is not None for x in (a.params, a.layers, a.d_model)]
    if any(given):
        if not all(given):
            raise EvencutError("evencut: --params, --layers and --d-model go together")
        model = dg.Model(a.params, a.layers, a.d_model)
    price = a.price_per_gpu_hour or (a.price_per_hour / a.gpus if a.price_per_hour else None)
    watts, source = a.watts, "given" if a.watts else ""
    if a.gpu:
        watts = dg.gpu_watts(a.gpu)
        if watts is None:
            raise EvencutError(f"evencut: no power figure for {a.gpu!r}; pass --watts instead")
        source = f"{a.gpu} board power, an upper bound"
    cal = dg.Calibration(step_ms=a.step_ms, run_hours=a.run_hours, epochs=a.epochs, fixed_ms=a.fixed_ms,
                         price_per_gpu_hour=price, watts=watts, watts_source=source)
    setup = dg.Setup(gpus=a.gpus, global_batch=global_batch, accum=a.accum, max_len=a.max_len, pad=a.pad,
                     pad_multiple=a.pad_multiple, current=a.current, sync="microstep" if a.fsdp else "window",
                     seed=a.seed, pool_steps=a.pool_steps)
    lengths = read_lengths(a.lengths, a.field)
    probe = Probe.load(a.probe) if a.probe else None
    d = dg.diagnose(lengths, setup, model=model, calibration=cal, sample_steps=a.sample_steps, probe=probe)
    if not a.quiet:
        print(d.report())
    if a.json:
        a.json.write_text(json.dumps(d.to_dict(), indent=1), encoding="utf-8")
    return 0


def run_probe(a: argparse.Namespace) -> int:
    try:
        from . import probe
    except ImportError as e:
        raise EvencutError(f"evencut: probe needs torch ({e})") from None
    max_tokens = a.max_tokens or min(max(16 * a.max_len, 16384), 65536)
    probe.run_probe(layers=a.layers, d_model=a.d_model, vocab=a.vocab, heads=a.heads, probe_layers=a.probe_layers,
                    max_len=a.max_len, max_tokens=max(max_tokens, a.max_len), reps=a.reps, rounds=a.rounds,
                    power_seconds=a.power_seconds, price_per_hour=a.price_per_hour, device=a.device, out=a.out,
                    quiet=a.quiet, backend=a.backend)
    return 0


def run_pick(a: argparse.Namespace) -> int:
    from . import pick as pk
    probes = [Probe.load(f) for f in a.probe]
    paths = [Path(f).resolve() for f in a.probe]
    names = [p.name for p in paths]
    for probe, path in zip(probes, paths):              # two files of one name: tell them apart by folder
        if names.count(path.name) > 1:
            probe.source = f"{path.parent.name}/{path.name}"
    for item in a.price:
        name, sep, value = item.rpartition("=")
        if not sep or not name:
            raise EvencutError(f"evencut: --price wants FILE=DOLLARS, not {item!r}")
        target = Path(name).resolve()
        hits = [i for i, path in enumerate(paths) if path == target]
        if not hits:
            hits = [i for i, n in enumerate(names) if n == Path(name).name]
            if len(hits) > 1:
                raise EvencutError(f"evencut: --price {name}: several probe files have that name; give its path")
        if not hits:
            raise EvencutError(f"evencut: --price {name}: no such --probe file")
        probes[hits[0]].price_per_hour = float(value)
    prices: dict = {}
    offers = pk.load_offers(a.offers) if a.offers else []
    ref = None
    if a.reference:
        target = Path(a.reference).resolve()
        ref = next((p for p, path in zip(probes, paths) if path == target), None) or Probe.load(a.reference)
    job = pk.Job(read_lengths(a.lengths, a.field), a.sequences_per_step, a.per_device_batch, max_len=a.max_len,
                 epochs=a.epochs, pad_multiple=a.pad_multiple, seed=a.seed, sample_steps=a.sample_steps)
    result = pk.pick(job, probes, offers, prices=prices, by=a.by, reference=ref)
    shown = pk.Pick(result.job, result.results[: a.top], result.skipped, result.model, result.by)
    print(shown.report())
    if a.json:
        rows = [{"machine": r.candidate.name, "gpus": r.candidate.gpus, "measured": r.candidate.measured,
                 "price_per_hour": r.candidate.price_per_hour, "per_device_batch": r.per_device, "accum": r.accum,
                 "mode": r.mode, "step_ms": r.step_ms, "default_step_ms": r.default_step_ms, "hours": r.hours,
                 "cost": r.cost, "kwh": r.kwh, "peak_padded": r.peak_padded, "over_memory": r.over_memory,
                 "notes": r.candidate.notes} for r in result.results]
        a.json.write_text(json.dumps({"by": result.by, "model": result.model, "rows": rows,
                                      "skipped": result.skipped}, indent=1), encoding="utf-8")
    return 0


def run_lengths(a: argparse.Namespace) -> int:
    lens = tokenize_jsonl(a.jsonl, [f for f in a.fields.split(",") if f], a.tokenizer, a.add)
    text = "\n".join(map(str, lens)) + "\n"
    if a.out:
        a.out.write_text(text, encoding="ascii")
        print(f"{len(lens):,} lengths written to {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return {"diagnose": run_diagnose, "lengths": run_lengths, "probe": run_probe, "pick": run_pick}[a.command](a)
    except EvencutError as e:
        print(str(e), file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"evencut-train: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
