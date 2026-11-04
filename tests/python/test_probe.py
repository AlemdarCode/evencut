"""evencut.train.probe on CPU (tiny model; timings mean nothing, the bookkeeping is checked) and with
two gloo processes for the all-reduce path."""

import json
import os
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.multiprocessing as mp  # noqa: E402

from evencut import EvencutError  # noqa: E402
from evencut.train import _model as M  # noqa: E402
from evencut.train import probe as PR  # noqa: E402
from evencut.train._plans import attention_constant  # noqa: E402
from evencut.train.timing import Probe  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "bench" / "results"
TINY = dict(layers=2, d_model=64, vocab=128, max_len=64, max_tokens=256, reps=1, rounds=2, power_seconds=0,
            device="cpu", quiet=True)


def test_shapes_cover_one_sequence_to_the_token_budget():
    grid = PR.shapes(2048, 16384)
    assert grid == sorted(grid, key=lambda s: (s[0] * s[1], s[1]))
    assert (1, 32) in grid and (8, 2048) in grid
    assert max(n * L for n, L in grid) == 16384 and all(n * L <= 16384 for n, L in grid)
    assert PR.shapes(20, 40) == [(1, 20), (2, 20)]


def test_sync_shapes_spread_over_distinct_sizes():
    grid = PR.shapes(2048, 16384)
    picks = PR.sync_shapes(grid)
    sizes = [n * L for n, L in picks]
    assert sizes == sorted(set(sizes)) and len(sizes) == 5 and sizes[-1] == 16384
    assert sizes[0] >= 1024                                   # training-sized batches, not single short ones
    assert all(p in grid for p in picks)
    assert PR.sync_shapes([(1, 32)]) == [(1, 32)]


def test_cpu_probe_writes_a_consistent_file(tmp_path):
    out = tmp_path / "cpu.json"
    p = PR.run_probe(out=str(out), **TINY)
    q = Probe.load(out)
    assert q.to_dict() == Probe.from_dict(p.to_dict()).to_dict()
    w = M.weights(128, 64, 2)
    assert q.model["params"] == w and q.curve.a == attention_constant(w, 2, 64)
    assert q.gpus == 1 and q.allreduce is None and q.optimizer_ms > 0 and q.launch_ms > 0
    assert len(q.machine["shapes"]) == len(PR.shapes(64, 256))
    assert q.max_tokens_tested == 256 and q.max_tokens_fit is None
    assert "Evencut probe: 1x CPU" in PR.format_probe(q)


def test_fewer_probe_layers_scale_by_the_flop_ratio(monkeypatch):
    """With a fake GPU whose time is exactly proportional to the probed model's FLOPs, the scaled curve
    must be the full model's FLOPs at the same rate."""
    layers, pl, d, vocab = 6, 2, 64, 128
    a_pl = attention_constant(M.weights(vocab, d, pl), pl, d)
    a_full = attention_constant(M.weights(vocab, d, layers), layers, d)
    rate = 1e-6

    def fake(model, n, L, vocab_, dev, reps, warmup=1):
        assert len(model.blocks) == pl
        return [rate * pl * n * L * (a_pl + L)] * reps
    monkeypatch.setattr(PR, "batch_samples", fake)
    p = PR.run_probe(**dict(TINY, layers=layers, probe_layers=pl))
    assert p.curve.a == a_full and p.model["probe_layers"] == pl
    for n, L, _ in p.machine["shapes"]:
        c = n * L * (a_full + L)
        assert p.curve.ms(c) == pytest.approx(rate * layers * c)
    assert any("2 of 6 layers" in note for note in p.notes)


def test_out_of_memory_ends_the_sweep_and_other_errors_propagate(monkeypatch):
    real = PR.batch_samples
    too_big = []

    def oom_above_128(model, n, L, vocab, dev, reps, warmup=1):
        if n * L >= 128:
            too_big.append((n, L))
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(model, n, L, vocab, dev, reps, warmup)
    monkeypatch.setattr(PR, "batch_samples", oom_above_128)
    p = PR.run_probe(**TINY)
    assert p.max_tokens_fit == 64 and all(n * L < 128 for n, L, _ in p.machine["shapes"])
    assert len(too_big) == 1          # after running out of memory, no larger batch is tried again
    assert "out of memory above 64" in PR.format_probe(p)

    def broken(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(PR, "batch_samples", broken)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        PR.run_probe(**TINY)


def test_exposed_share_is_the_part_of_the_allreduce_a_step_waited_for():
    assert PR.exposed_share(43.3, 50.2, 16.0) == pytest.approx(6.9 / 16.0)   # 6.9 of a 16 ms all-reduce exposed
    assert PR.exposed_share(35.1, 34.1, 8.7) == 0.0                          # fully hidden (noise below 0)
    assert PR.exposed_share(20.0, 80.0, 47.0) == 1.0                         # capped
    assert PR.exposed_share(20.0, 30.0, 0.0) == 0.0


def test_is_oom_knows_the_driver_message():
    assert PR.is_oom(RuntimeError("CUDA driver error: out of memory"))
    assert PR.is_oom(torch.cuda.OutOfMemoryError("x"))
    assert not PR.is_oom(RuntimeError("CUDA driver error: device not ready"))


def test_bad_probe_arguments():
    with pytest.raises(EvencutError):
        PR.run_probe(**dict(TINY, max_tokens=32))            # below one full-length sequence
    with pytest.raises(EvencutError):
        PR.run_probe(**dict(TINY, reps=0))


@pytest.mark.parametrize("name,gpus,worst,links", [
    ("ddp_a100x4.json", 4, "NV12", ["NV12"]),
    ("ddp_batches.json", 8, "NODE", ["PIX", "PXB", "NODE"]),
])
def test_parse_topology_of_the_rented_machines(name, gpus, worst, links):
    topo = json.loads((RESULTS / name).read_text(encoding="utf-8"))["topology"]
    t = PR.parse_topology(topo, gpus)
    assert t["worst"] == worst and t["links"] == links
    assert PR.parse_topology("", 4) == {"links": None, "worst": None}


def _worker(rank, port, out, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE="2")
    p = PR.run_probe(out=out if rank == 0 else None, **TINY)
    q.put((rank, None if p is None else p.to_dict()))


def test_two_process_probe_measures_allreduce(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    out = str(tmp_path / "two.json")
    procs = [ctx.Process(target=_worker, args=(r, port, out, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0
    results = dict(q.get() for _ in range(2))
    assert results[1] is None
    probe = Probe.load(out)
    assert probe.gpus == 2 and len(probe.gpu_speed) == 2 and len(probe.tflops) == 2
    grad_mb = 4 * probe.model["params"] / 1e6
    assert [m for m, _ in probe.allreduce.points][-1] == pytest.approx(grad_mb)
    assert probe.allreduce_ms() > 0 and probe.fixed_ms() > probe.optimizer_ms
    assert probe.overlap is None and probe.measured_sync                      # measured as points, not a proportion
    assert 1 <= len(probe.sync_points) == len(probe.sync_raw) <= 5
    assert probe.sync_points == sorted(probe.sync_points)
    for (t, f), (_n, _L, free, synced, ar) in zip(probe.sync_points, sorted(probe.sync_raw, key=lambda r: r[2])):
        assert free > 0 and synced > 0 and ar > 0 and 0.0 <= f <= 1.0
        assert t == pytest.approx(free, rel=1e-3)                            # all layers probed: no scaling
        # the file keeps times rounded to 0.001 ms, so a share recomputed from them can differ by about
        # 0.001 / all-reduce ms (tiny all-reduces on CPU make that visible)
        assert f == pytest.approx(PR.exposed_share(free, synced, ar), abs=2e-3 / ar + 1e-3)
    assert probe.sync().points == tuple(tuple(p) for p in probe.sync_points)


Q_POWER = """GPU 00000000:0A:00.0
    GPU Power Readings
        Average Power Draw                             : N/A
        Instantaneous Power Draw                       : N/A
    Power Samples
        Duration                                       : 2.37 sec
        Number of Samples                              : 119
        Max                                            : 116.20 W
        Min                                            : 113.90 W
        Avg                                            : 115.03 W
"""


def test_power_sampler_reads_power_draw_and_falls_back_to_the_driver_samples(monkeypatch):
    rounds = []

    def smi(args):
        if args[0].startswith("--query-gpu"):
            rounds.append(1)
            if len(rounds) >= 2:
                s._stop.set()
            return "0, 251.5\n1, [N/A]\n" if len(rounds) == 1 else "0, 248.5\n1, [N/A]\n"
        assert args == ["-q", "-d", "POWER", "-i", "1"]          # only the GPU that said N/A
        return Q_POWER
    s = PR.PowerSampler([0, 1], every=0.0)
    monkeypatch.setattr(PR, "_smi", smi)
    s._run()
    assert s.watts() == [pytest.approx(250.0), pytest.approx(115.03)]
    assert PR.samples_average(Q_POWER) == 115.03 and PR.samples_average("no samples") is None


def test_power_sampler_without_nvidia_smi_reports_nothing(monkeypatch):
    monkeypatch.setattr(PR, "_smi", lambda args: None)
    s = PR.PowerSampler([0], every=0.0)
    s._run()
    assert s.watts() is None
    monkeypatch.setattr(PR, "_smi", lambda args: "0, [N/A]\n" if args[0].startswith("--query") else "N/A")
    s = PR.PowerSampler([0], every=0.0)
    s._stop.set()
    s._run()
    assert s.watts() is None


def test_optimizer_step_falls_back_to_one_block_when_it_does_not_fit(monkeypatch):
    model = M.GPT(128, 64, layers=3, d=64, heads=1)
    full = sum(p.numel() for p in model.parameters())
    block = sum(p.numel() for p in model.blocks[0].parameters())
    calls = []

    def fake(fn, reps, warmup, dev):
        calls.append(1)
        if len(calls) == 1:                       # the whole model's optimizer state does not fit
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return 2.0
    monkeypatch.setattr(PR, "_ms_median", fake)
    assert PR.optimizer_ms(model, torch.device("cpu")) == pytest.approx(2.0 * full / block)
    assert all(p.grad is None for p in model.parameters())          # nothing left behind

    def broken(fn, reps, warmup, dev):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(PR, "_ms_median", broken)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        PR.optimizer_ms(model, torch.device("cpu"))


def test_amd_power_reads_both_json_layouts(monkeypatch):
    """Untested on AMD hardware: these are the documented amd-smi and rocm-smi JSON shapes."""
    amd_smi = '[{"gpu": 0, "power": {"socket_power": {"value": 612, "unit": "W"}, "gfx_voltage": {"value": 800, "unit": "mV"}}}]'
    rocm_smi = '{"card0": {"Average Graphics Package Power (W)": "297.0"}}'
    answers = {"amd-smi": amd_smi, "rocm-smi": rocm_smi}
    monkeypatch.setattr(PR, "_run", lambda cmd: answers.get(cmd[0]))
    assert PR.amd_power(0) == 612.0
    answers["amd-smi"] = None                                  # no amd-smi: fall back to rocm-smi
    assert PR.amd_power(0) == 297.0
    answers["rocm-smi"] = "not json"
    assert PR.amd_power(0) is None
    s = PR.PowerSampler([0], every=0.0, amd=True)
    answers.update({"amd-smi": amd_smi})
    calls = []

    def once(i):
        calls.append(i)
        if len(calls) > 1:
            s._stop.set()
        return 612.0
    monkeypatch.setattr(PR, "amd_power", once)
    s._run()
    assert s.watts() == [612.0]


def test_vendor_is_amd_on_a_rocm_build(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4.1", raising=False)
    assert PR.vendor() == "amd"
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert PR.vendor() in ("nvidia", "cpu")


def test_power_samples_of_several_gpus_come_from_one_call(monkeypatch):
    two = Q_POWER + Q_POWER.replace("0A:00.0", "0B:00.0").replace("115.03", "98.50")
    assert PR.samples_per_gpu(two) == [115.03, 98.50]
    calls = []
    s = PR.PowerSampler([0, 1], every=0.0)

    def smi(args):
        calls.append(args)
        if args[0].startswith("--query-gpu"):
            if sum(c[0].startswith("--query") for c in calls) >= 2:
                s._stop.set()
            return "0, [N/A]\n1, [N/A]\n"
        return two
    monkeypatch.setattr(PR, "_smi", smi)
    s._run()
    assert s.watts() == [115.03, 98.50]
    assert [c for c in calls if c[0] == "-q"] == [["-q", "-d", "POWER", "-i", "0,1"]] * 2   # one call per tick


def test_launch_floor_is_scaled_like_the_curve(monkeypatch):
    layers, pl, d, vocab = 6, 2, 64, 128
    a_pl = attention_constant(M.weights(vocab, d, pl), pl, d)
    a_full = attention_constant(M.weights(vocab, d, layers), layers, d)
    monkeypatch.setattr(PR, "batch_samples", lambda model, n, L, vocab_, dev, reps, warmup=1: [5.0] * reps)
    p = PR.run_probe(**dict(TINY, layers=layers, probe_layers=pl))
    tiny = [L for n, L, _ in p.machine["shapes"] if n * L <= 512]
    assert p.launch_ms == pytest.approx(min(5.0 * layers * (a_full + L) / (pl * (a_pl + L)) for L in tiny))
