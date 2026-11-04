"""evencut.train.timing (probe files, time curves, all-reduce scaling), the diagnosis with a probe,
and evencut-train pick. Pure Python: probes are built here, not measured."""

import json
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evencut import EvencutError
from evencut.train import _plans as P
from evencut.train import cli
from evencut.train import pick as PK
from evencut.train.diagnose import Calibration, Model, Setup, diagnose
from evencut.train.timing import AllReduce, Probe, Sync, TimeCurve, isotonic, ring_factor

PROPS = settings(max_examples=40, deadline=None)
MODEL = {"params": 29_884_416, "layers": 8, "d_model": 512, "heads": 8, "vocab": 8192, "probe_layers": 8,
         "max_len": 2048, "attention_constant": 7296}
A = 7296


def long_tail(n, seed=1, cap=2048):
    rng = random.Random(seed)
    return [max(1, min(cap, int(rng.paretovariate(1.1) * 40))) for _ in range(n)]


def floor_curve(floor=20.0, per=2e-6, a=A):
    """A GPU that takes ``floor`` ms for any batch up to floor / per cost units, then grows linearly."""
    knee = floor / per
    return TimeCurve(a, ((knee / 100, floor), (knee, floor), (knee * 10, floor * 10)))


def linear_curve(per=2e-6, a=A):
    return TimeCurve(a, ((1.0, per), (1e15, per * 1e15)))


def make_probe(curve=None, gpus=1, allreduce=None, optimizer_ms=1.0, **kw):
    machine = {"gpu": kw.pop("gpu", "NVIDIA GeForce RTX 4060"), "gpus": gpus}
    return Probe(machine=machine, model=dict(MODEL, **kw.pop("model", {})), curve=curve or floor_curve(),
                 optimizer_ms=optimizer_ms, allreduce=allreduce, max_tokens_tested=16384, **kw)


# ---------------------------------------------------------------------------------------------
# isotonic fit and the time curve


@PROPS
@given(st.lists(st.floats(0.1, 1e3), min_size=1, max_size=30), st.data())
def test_isotonic_is_monotone_and_keeps_the_weighted_mean(ys, data):
    ws = data.draw(st.lists(st.floats(0.5, 5), min_size=len(ys), max_size=len(ys)))
    fit = isotonic(ys, ws)
    assert all(a <= b + 1e-9 for a, b in zip(fit, fit[1:]))
    assert sum(w * f for w, f in zip(ws, fit)) == pytest.approx(sum(w * y for w, y in zip(ws, ys)))
    if ys == sorted(ys):
        assert fit == pytest.approx(ys)


@PROPS
@given(st.lists(st.tuples(st.floats(1e3, 1e9), st.floats(0.5, 500)), min_size=1, max_size=25),
       st.floats(1.0, 1e10))
def test_fitted_curve_is_monotone_and_never_below_its_best_rate(samples, c):
    curve = TimeCurve.fit(A, samples)
    xs = sorted([c, c * 1.5] + [p[0] for p in curve.points])
    times = [curve.ms(x) for x in xs]
    assert all(a <= b + 1e-9 for a, b in zip(times, times[1:]))
    for x in xs:
        assert curve.ms(x) >= curve.best * x * (1 - 1e-9)


def test_curve_floor_interpolation_and_proportional_tail():
    curve = TimeCurve.fit(A, [(100.0, 10.0), (200.0, 12.0), (200.0, 14.0), (400.0, 26.0)])
    assert curve.points == ((100.0, 10.0), (200.0, 13.0), (400.0, 26.0))     # equal costs averaged
    assert curve.ms(1.0) == 10.0                                             # floor
    assert curve.ms(300.0) == pytest.approx(19.5)
    assert curve.ms(800.0) == pytest.approx(52.0)                            # proportional beyond
    assert curve.best == pytest.approx(0.065)
    assert curve.batch_ms(0, 50) == 0.0
    assert curve.batch_ms(2, 10) == curve.ms(2 * 10 * (A + 10))


def test_curve_rejects_bad_points():
    with pytest.raises(EvencutError):
        TimeCurve(A, ())
    with pytest.raises(EvencutError):
        TimeCurve(A, ((10.0, 5.0), (5.0, 6.0)))
    with pytest.raises(EvencutError):
        TimeCurve.fit(A, [(0.0, 1.0)])
    with pytest.raises(EvencutError):
        floor_curve().scaled(0)


def test_allreduce_interpolates_extrapolates_and_scales_with_gpus():
    ar = AllReduce(4, ((1.0, 0.1), (64.0, 1.0), (128.0, 2.0)))
    assert ar.ms(32e6) == pytest.approx(0.1 + 0.9 * 31 / 63)
    assert ar.ms(256e6) == pytest.approx(4.0)                       # bandwidth of the two largest points
    assert ar.ms(0.5e6) == pytest.approx(0.1)                      # latency floor
    assert ar.ms(128e6, gpus=8) == pytest.approx(2.0 * ring_factor(8) / ring_factor(4))
    assert ar.ms(128e6, gpus=1) == 0.0
    assert ar.bus_gbps() == pytest.approx(128 * 1.5 / 2.0)
    with pytest.raises(EvencutError):
        AllReduce(1, ()).ms(1e6, gpus=2)


def test_probe_file_round_trip_and_checks(tmp_path):
    p = make_probe(gpus=4, allreduce=AllReduce(4, ((1.0, 0.1), (120.0, 1.0))), watts=[300.0, None, 310.0, 0],
                   price_per_hour=4.28)
    path = tmp_path / "m.json"
    p.save(path)
    q = Probe.load(path)
    assert q.to_dict() == p.to_dict()
    assert q.source == "m.json" and q.label() == "4x NVIDIA GeForce RTX 4060 (m.json)"
    assert q.mean_watts() == pytest.approx(305.0)
    assert q.fixed_ms() == pytest.approx(1.0 + q.allreduce.ms(4 * MODEL["params"]))
    assert make_probe().allreduce_ms(1) == 0.0
    with pytest.raises(EvencutError, match="one GPU"):
        make_probe().allreduce_ms(8)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something else"}))
    with pytest.raises(EvencutError, match="not an Evencut probe"):
        Probe.load(bad)
    bad.write_text(json.dumps({"format": "evencut-probe/1", "machine": {}}))
    with pytest.raises(EvencutError, match="damaged"):
        Probe.load(bad)


# ---------------------------------------------------------------------------------------------
# evaluate with a measured curve


def test_linear_curve_gives_cost_times_rate():
    lens = long_tail(8 * 4 * 20, seed=3)
    order = list(range(len(lens)))
    plan = P.window(order, lens, 4, 16, 2, cost=P.AttentionCost(A))
    st_ = P.evaluate(plan, lens, 4, 2, cost=P.AttentionCost(A), timing=linear_curve(per=3e-6))
    assert st_.step_ms == pytest.approx(3e-6 * st_.step_cost)
    assert st_.busy_ms == pytest.approx(3e-6 * st_.busy_cost)
    assert st_.linear_ms == pytest.approx(3e-6 * st_.busy_cost)
    assert st_.useful_ms == pytest.approx(3e-6 * st_.useful_cost)


@PROPS
@given(st.integers(1, 5), st.integers(1, 4), st.integers(1, 3), st.integers(0, 9), st.sampled_from(["window", "microstep"]),
       st.floats(1.0, 60.0))
def test_time_parts_are_ordered(gpus, per_device, accum, seed, sync, floor):
    lens = long_tail(gpus * per_device * accum * 6, seed)
    plan = P.per_step(list(range(len(lens))), lens, gpus, gpus * per_device, "strided")
    st_ = P.evaluate(plan, lens, gpus, accum, sync=sync, timing=floor_curve(floor=floor))
    assert st_.step_ms >= st_.busy_ms - 1e-6
    assert st_.busy_ms >= st_.linear_ms - 1e-6
    assert st_.linear_ms >= st_.useful_ms - 1e-6 >= -1e-6


# ---------------------------------------------------------------------------------------------
# diagnose with a probe


SETUP = Setup(gpus=4, global_batch=16, accum=2, max_len=2048)


def test_diagnose_with_probe_is_absolute_and_shares_add_up():
    lens = long_tail(4000, seed=5)
    d = diagnose(lens, SETUP, probe=make_probe(gpus=4, allreduce=AllReduce(4, ((1.0, 0.2), (120.0, 2.0))),
                                                 price_per_hour=2.0, watts=[250.0] * 4), sample_steps=40)
    assert d.model == Model(MODEL["params"], 8, 512)
    for r in d.rows:
        if r.available:
            assert sum(r.shares.values()) == pytest.approx(1.0)
            assert r.step_ms > 0
    cur = d.row("default")
    assert cur.shares["small_batches"] > 0
    assert cur.shares["fixed"] == pytest.approx((1.0 + 2.0 * 4 * MODEL["params"] / 120e6) / cur.step_ms, rel=0.02)
    m = d.money
    assert m["predicted"] and m["run_hours"] == pytest.approx(cur.step_ms * len(lens) / 32 / 3.6e6)
    assert m["price_per_gpu_hour"] == pytest.approx(0.5) and m["watts"] == 250.0
    assert "predicted from the probe" in d.report() and "small" in d.report()


def test_linear_probe_without_fixed_time_matches_the_cost_model():
    """Oracle: a curve that is exactly proportional to cost must rank and scale like the cost model."""
    lens = long_tail(3000, seed=8)
    p = make_probe(curve=linear_curve(), optimizer_ms=0.0)
    a = diagnose(lens, SETUP, probe=p, sample_steps=30)
    b = diagnose(lens, SETUP, model=Model(MODEL["params"], 8, 512), sample_steps=30)
    for ra, rb in zip(a.rows, b.rows):
        assert ra.key == rb.key
        if ra.available:
            assert ra.speedup == pytest.approx(rb.speedup)
            assert ra.shares["small_batches"] == pytest.approx(0.0, abs=1e-9)


def test_measured_step_rescales_the_probe_and_says_so():
    lens = long_tail(3000, seed=9)
    p = make_probe(optimizer_ms=2.0)
    plain = diagnose(lens, Setup(gpus=1, global_batch=4, accum=2), probe=p, sample_steps=20)
    t = plain.row("default").step_ms
    d = diagnose(lens, Setup(gpus=1, global_batch=4, accum=2), probe=p, sample_steps=20,
                 calibration=Calibration(step_ms=2 * t))
    assert d.row("default").step_ms == pytest.approx(2 * t)
    assert any("scaled by" in n for n in d.notes)
    assert not d.money["predicted"]


def test_probe_mismatches_and_missing_sync_are_reported():
    lens = long_tail(3000, seed=2)
    with pytest.raises(EvencutError, match="different model"):
        diagnose(lens, SETUP, probe=make_probe(), model=Model(7_000_000_000, 32, 4096))
    d = diagnose(lens, SETUP, probe=make_probe(gpus=1), sample_steps=10)
    assert any("gradient sync of 4 GPUs is not included" in n for n in d.notes)
    d = diagnose(lens, SETUP, probe=make_probe(gpus=8, allreduce=AllReduce(8, ((120.0, 4.0),))), sample_steps=10)
    assert any("scaled from the probe's 8 GPUs to 4" in n for n in d.notes)


def test_memory_limit_from_the_probe_marks_big_batches():
    lens = long_tail(3000, seed=4)
    d = diagnose(lens, SETUP, probe=make_probe(max_tokens_fit=1000), sample_steps=20)
    assert any(r.over_memory for r in d.rows if r.available)
    assert "!" in d.report()
    scaled = make_probe(max_tokens_fit=1000, model={"probe_layers": 2})       # memory of 2 of 8 layers says nothing
    d = diagnose(lens, SETUP, probe=scaled, sample_steps=20)
    assert not any(r.over_memory for r in d.rows if r.available)


# ---------------------------------------------------------------------------------------------
# pick


def test_layout_keeps_the_step_and_fills_memory():
    assert PK.layout(8, 256, 16) == (16, 2)
    assert PK.layout(8, 256, 12) == (8, 4)          # largest divisor of 32 that fits
    assert PK.layout(4, 256, 64) == (64, 1)
    assert PK.layout(3, 256, 16) is None
    assert PK.layout(1, 7, 4) == (1, 7)


def test_gpu_spec_names():
    assert PK.gpu_spec("NVIDIA A100-SXM4-80GB") == (312, "nvlink")
    assert PK.gpu_spec("A100 PCIE") == (312, "pcie")
    assert PK.gpu_spec("NVIDIA GeForce RTX 4090") == (165, "host")
    assert PK.gpu_spec("NVIDIA H100 80GB HBM3") == (989, "nvlink")
    assert PK.gpu_spec("RTX 6000Ada") == (364, "pcie")
    assert PK.gpu_spec("Tesla V100") is None
    assert PK.gpu_spec("NVIDIA GeForce RTX 4060 Ti") is None              # not an RTX 4060
    assert PK.gpu_spec("NVIDIA GeForce RTX 4080 SUPER") is None
    assert PK.gpu_spec("NVIDIA GA100") is None                            # a chip name, not the A100 card
    assert PK.gpu_spec("NVIDIA L40S") == (362, "pcie") and PK.gpu_spec("L4") == (121, "pcie")
    assert PK.gpu_model("A100 SXM4") == PK.gpu_model("NVIDIA A100-SXM4-80GB") == "a100 sxm4"


def test_cards_outside_the_table_get_a_rough_figure_or_none():
    o = offer(gpu_name="RTX 5080", num_gpus=4, total_flops=212.6, compute_cap=1200)
    assert PK.offer_spec(o) == (pytest.approx(2.0 * 212.6 / 4), "host", True)
    pro = offer(gpu_name="RTX PRO 4000", num_gpus=2, total_flops=88.6, compute_cap=1200)
    assert PK.offer_spec(pro) == (pytest.approx(4.5 * 44.3), "pcie", True)
    assert PK.offer_spec(offer(gpu_name="RTX 2080 Ti", total_flops=59.7, compute_cap=750)) is None    # no bf16
    assert PK.offer_spec(offer(gpu_name="CMP 170HX", total_flops=48.6, compute_cap=800)) is None
    assert PK.offer_spec(offer()) == (165, "host", False)
    c = PK.from_offer(o, make_probe())
    assert "rough" in c.notes[0]


def offer(**kw):
    o = {"id": 1, "gpu_name": "RTX 4090", "num_gpus": 8, "dph_total": 2.9, "bw_nvlink": 0.0, "pcie_bw": 21.0,
         "gpu_max_power": 450, "geolocation": "Utah, US"}
    o.update(kw)
    return o


def test_offer_estimates_scale_compute_keep_the_floor_and_price_the_link():
    ref = make_probe(curve=floor_curve(floor=20.0))
    c = PK.from_offer(offer(), ref)
    assert not c.measured and "no peer-to-peer (GeForce)" in c.notes
    assert c.link_gbps == pytest.approx(21.0 * PK.LINK_EFFICIENCY["host"])
    assert c.curve.ms(1.0) == 20.0                                            # launch floor kept
    assert c.curve.ms(1e10) == pytest.approx(1e10 * 2e-6 * 30 / 165)          # compute scaled by TFLOPS
    fast_host = PK.from_offer(offer(), make_probe(curve=floor_curve(floor=20.0), launch_ms=12.0))
    assert fast_host.curve.ms(1.0) == 12.0                                    # the fastest tiny batch seen
    nv = PK.from_offer(offer(gpu_name="A100 SXM4", num_gpus=4, bw_nvlink=300.0), ref)
    assert nv.link_gbps == pytest.approx(300 * PK.LINK_EFFICIENCY["nvlink"])
    assert nv.sync_ms(120e6) < c.sync_ms(120e6)
    # an SXM card without NVLink between the rented GPUs (a real 2x A100 SXM4 offer: bw_nvlink 0, link SYS)
    no_nv = PK.from_offer(offer(gpu_name="A100 SXM4", num_gpus=2, bw_nvlink=0.0, pcie_bw=12.0), ref)
    assert no_nv.link_gbps == pytest.approx(12.0 * PK.LINK_EFFICIENCY["pcie"])
    unknown = {k: v for k, v in offer(gpu_name="A100 SXM4", num_gpus=2).items() if k != "bw_nvlink"}
    assert PK.from_offer(unknown, ref).link_gbps == pytest.approx(300 * PK.LINK_EFFICIENCY["nvlink"])  # not stated: as the card
    with pytest.raises(EvencutError):
        PK.from_offer(offer(gpu_name="Tesla V100"), ref)


def test_pick_ranks_by_cost_and_time_and_skips_what_cannot_run(tmp_path):
    lens = long_tail(4000, seed=6)
    job = PK.Job(lens, sequences_per_step=64, per_device_batch=8, max_len=2048, epochs=2, sample_steps=20)
    slow = make_probe(curve=floor_curve(floor=20.0, per=4e-6), gpus=4, allreduce=AllReduce(4, ((120.0, 1.0),)),
                      price_per_hour=1.0, source="slow.json")
    fast = make_probe(curve=floor_curve(floor=10.0, per=1e-6), gpus=4, allreduce=AllReduce(4, ((120.0, 1.0),)),
                      price_per_hour=1.5, source="fast.json")
    pk = PK.pick(job, [slow, fast], [offer(num_gpus=3)])
    names = [r.candidate.name for r in pk.results]
    assert names == [PK._short(fast.label()), PK._short(slow.label())] == ["4x RTX 4060 (fast.json)", "4x RTX 4060 (slow.json)"]
    assert pk.results[0].kwh == pytest.approx(pk.results[0].hours * 4 * 115 / 1000)      # board power fallback
    assert any("do not split evenly over 3 GPUs" in s for s in pk.skipped)
    for r in pk.results:
        assert r.per_device * r.accum * r.candidate.gpus == 64
        assert r.step_ms <= r.default_step_ms + 1e-9
        assert r.cost == pytest.approx(r.hours * r.candidate.price_per_hour)
    pricey = PK.pick(job, [slow, fast], prices={"fast.json": 100.0})
    assert pricey.results[0].candidate.name == PK._short(slow.label())
    by_time = PK.pick(job, [slow, fast], prices={"fast.json": 100.0}, by="time")
    assert by_time.results[0].candidate.name == PK._short(fast.label())
    other = make_probe(model={"params": 91_245_312, "layers": 12, "d_model": 768})
    with pytest.raises(EvencutError, match="another model"):
        PK.pick(job, [slow, other])
    with pytest.raises(EvencutError, match="at least one probe"):
        PK.pick(job, [])


def test_cli_probe_diagnose_and_pick(tmp_path, capsys):
    probe = make_probe(curve=floor_curve(), gpus=2, allreduce=AllReduce(2, ((120.0, 3.0),)), price_per_hour=0.8)
    probe.save(tmp_path / "box.json")
    lens = long_tail(3000, seed=7)
    (tmp_path / "l.txt").write_text("\n".join(map(str, lens)))
    (tmp_path / "offers.json").write_text(json.dumps([offer(num_gpus=2, id=7), offer(gpu_name="Tesla V100", id=8)]))
    assert cli.main(["diagnose", str(tmp_path / "l.txt"), "--gpus", "2", "--per-device-batch", "4", "--max-len", "2048",
                     "--probe", str(tmp_path / "box.json"), "--sample-steps", "20"]) == 0
    out = capsys.readouterr().out
    assert "measured on 2x" in out and "predicted from the probe" in out
    assert cli.main(["pick", str(tmp_path / "l.txt"), "--sequences-per-step", "32", "--per-device-batch", "8",
                     "--probe", str(tmp_path / "box.json"), "--offers", str(tmp_path / "offers.json"),
                     "--price", "box.json=0.5", "--json", str(tmp_path / "pick.json"), "--sample-steps", "10"]) == 0
    out = capsys.readouterr().out
    assert "Evencut pick" in out and "vast 7" in out and "skipped: vast 8" in out
    rows = json.loads((tmp_path / "pick.json").read_text())["rows"]
    assert {r["measured"] for r in rows} == {True, False}
    assert next(r for r in rows if r["measured"])["price_per_hour"] == 0.5
    assert cli.main(["pick", str(tmp_path / "l.txt"), "--sequences-per-step", "32", "--per-device-batch", "8",
                     "--probe", str(tmp_path / "box.json"), "--price", "nonsense"]) == 2


def test_pick_adds_gradient_sync_so_a_slow_link_costs_exactly_its_sync_time():
    lens = long_tail(3000, seed=11)
    job = PK.Job(lens, sequences_per_step=32, per_device_batch=8, max_len=2048, sample_steps=10)
    nv = make_probe(gpus=4, allreduce=AllReduce(4, ((120.0, 1.0),)), price_per_hour=1.0, source="nv.json")
    host = make_probe(gpus=4, allreduce=AllReduce(4, ((120.0, 60.0),)), price_per_hour=1.0, source="host.json")
    pk = PK.pick(job, [host, nv])
    assert [r.candidate.name for r in pk.results] == [PK._short(nv.label()), PK._short(host.label())]
    grad = 4 * MODEL["params"]
    gap = host.allreduce.ms(grad) - nv.allreduce.ms(grad)
    assert pk.results[1].step_ms - pk.results[0].step_ms == pytest.approx(gap)
    assert pk.results[1].default_step_ms - pk.results[0].default_step_ms == pytest.approx(gap)


def test_an_offer_of_a_probed_card_uses_that_probe():
    a100 = make_probe(curve=floor_curve(floor=18.0, per=3e-7), gpus=4, gpu="NVIDIA A100-SXM4-80GB",
                      allreduce=AllReduce(4, ((120.0, 1.0),)), source="a100.json")
    a100.machine["worst_link"] = "NV12"
    ref = make_probe(curve=floor_curve(floor=20.0))
    c = PK.from_offer(offer(gpu_name="A100 SXM4", num_gpus=8, bw_nvlink=300.0), ref, [ref, a100])
    assert c.curve == a100.curve and "same GPU (a100.json)" in c.notes[0]
    assert c.sync_ms(120e6) == pytest.approx(a100.allreduce.ms(120e6, 8))    # measured sync, 8 GPUs
    pcie = PK.from_offer(offer(gpu_name="A100 PCIE", num_gpus=4), ref, [ref, a100])
    assert pcie.curve != a100.curve                                          # another card: datasheet scaling


def test_rows_within_the_error_of_the_first_are_marked():
    lens = long_tail(3000, seed=12)
    job = PK.Job(lens, sequences_per_step=32, per_device_batch=8, max_len=2048, sample_steps=10)
    a = make_probe(gpus=4, allreduce=AllReduce(4, ((120.0, 1.0),)), price_per_hour=1.00, source="a.json")
    b = make_probe(gpus=4, allreduce=AllReduce(4, ((120.0, 1.0),)), price_per_hour=1.03, source="b.json")
    c = make_probe(gpus=4, allreduce=AllReduce(4, ((120.0, 1.0),)), price_per_hour=1.20, source="c.json")
    pk = PK.pick(job, [a, b, c])
    assert [PK.close_to_first(pk, r) for r in pk.results[1:]] == [True, False]    # 3% and 20% behind, measured
    report = pk.report()
    assert report.count("~#1") == 1
    b.price_per_hour = 1.15                     # 15% behind: a tie only when an estimate is involved
    pk = PK.pick(job, [a, b])
    assert not PK.close_to_first(pk, pk.results[1])
    pk.results[1].candidate.error = PK.ERROR_ESTIMATED      # an estimate: its own error band counts
    assert PK.close_to_first(pk, pk.results[1])


def test_estimates_across_architectures_or_vendors_are_flagged_and_widened():
    ref = make_probe(curve=floor_curve(floor=20.0))            # an RTX 4060 (Ada)
    same_family = PK.from_offer(offer(gpu_name="RTX 4090"), ref)
    other_family = PK.from_offer(offer(gpu_name="A100 SXM4", num_gpus=4, bw_nvlink=300.0), ref)
    other_vendor = PK.from_offer(offer(gpu_name="AMD Instinct MI300X", num_gpus=8, bw_nvlink=0.0), ref)
    assert same_family.error == PK.ERROR_ESTIMATED and not any("unreliable" in n for n in same_family.notes)
    assert other_family.error == PK.ERROR_CROSS_ARCH
    assert any("different architecture (ada -> ampere)" in n for n in other_family.notes)
    assert other_vendor.error == PK.ERROR_CROSS_ARCH and any("different vendor" in n for n in other_vendor.notes)
    assert PK.gpu_family("Q RTX 5000") == "turing" and PK.gpu_family("B200") == "blackwell"
    assert PK.gpu_family("RTX 6000Ada") == "ada" and PK.gpu_family("RTX A6000") == "ampere"
    assert PK.gpu_spec("AMD Instinct MI300X") == (1307, "nvlink") and PK.gpu_vendor("MI250X") == "amd"
    assert PK.gpu_family("AMD Instinct MI300X") == "amd cdna" and PK.gpu_family("MI250X") == "amd cdna"


@PROPS
@given(st.sampled_from([1, 2, 4, 8]), st.integers(1, 6), st.integers(0, 9), st.sampled_from([1, 8]))
def test_cached_plan_times_equal_a_full_evaluation(gpus, per_device, seed, multiple):
    """Oracle: pick's shortcut (plan shapes timed by the curve) equals evaluate(..., timing=curve)."""
    lens = long_tail(2000, seed=seed)
    job = PK.Job(lens, sequences_per_step=gpus * per_device * 4, per_device_batch=per_device, max_len=2048,
                 pad_multiple=multiple, sample_steps=6, seed=seed)
    cache = PK.PlanCache(job, A)
    curve = floor_curve(floor=7.0)
    sync = Sync(9.0, 0.4)
    for mode, (shapes, peak) in cache.plans(gpus, per_device, 4).items():
        g = gpus * per_device
        plan = {"default": P.per_step(cache.order, cache.lens, gpus, g, "strided"),
                "step": P.per_step(cache.order, cache.lens, gpus, g, "padded", grouped=True, cost=P.AttentionCost(A)),
                "window": P.window(cache.order, cache.lens, gpus, g, 4, cost=P.AttentionCost(A))}[mode]
        full = P.evaluate(plan, cache.lens, gpus, 4, cost=P.AttentionCost(A), pad_multiple=multiple, timing=curve,
                          exposed=sync)
        assert PK.plan_ms(shapes, curve) == pytest.approx(full.step_ms / full.steps)
        assert PK.plan_ms(shapes, curve, sync) == pytest.approx((full.step_ms + full.sync_ms) / full.steps)
        assert peak == full.peak_padded


def test_sync_hides_behind_the_last_micro_batch():
    s = Sync(10.0, 0.25)
    assert s(0.0) == 10.0 and s(20.0) == 5.0 and s(100.0) == 0.0 and Sync(10.0)(50.0) == 10.0
    # one optimizer step of 2 micro-steps on 2 GPUs (time = sequences x length^2 with a = 0)
    curve = linear_curve(per=1.0, a=0)
    lens = [2, 2, 1, 3]
    plan = [[[0], [1]], [[2], [3]]]           # micro 1: 4 / 4 ms, micro 2: 1 / 9 ms -> GPU 1 slowest, last 9 ms
    st_ = P.evaluate(plan, lens, 2, 2, cost=P.AttentionCost(0), timing=curve, exposed=Sync(10.0, 0.5))
    assert st_.step_ms == pytest.approx(13.0) and st_.sync_ms == pytest.approx(10.0 - 0.5 * 9.0)
    micro = P.evaluate(plan, lens, 2, 2, cost=P.AttentionCost(0), timing=curve, exposed=Sync(10.0, 0.5),
                       sync="microstep")      # a sync after every micro-step (FSDP-like)
    assert micro.sync_ms == pytest.approx((10.0 - 2.0) + (10.0 - 4.5))


def test_sync_points_interpolate_the_exposed_share():
    s = Sync(10.0, points=((20.0, 0.0), (40.0, 0.5), (60.0, 0.9)))
    assert s(5.0) == 0.0 and s(20.0) == 0.0                     # small batches: all hidden
    assert s(30.0) == pytest.approx(2.5) and s(50.0) == pytest.approx(7.0)
    assert s(500.0) == pytest.approx(9.0)                      # held beyond the last point
    assert Sync(10.0, overlap=0.9, points=((1.0, 1.0),))(100.0) == 10.0     # points win over the proportion
    assert Sync(0.0)(5.0) == 0.0 and Sync(10.0).share(3.0) == 1.0
    assert Sync(10.0, points=((1.0, 1.5),))(5.0) == 10.0 and Sync(10.0, points=((1.0, -0.2),))(5.0) == 0.0  # clamped
    shuffled = Sync(10.0, points=((60.0, 0.9), (20.0, 0.0), (40.0, 0.5)))      # e.g. a hand-edited probe file
    assert shuffled == s and shuffled(30.0) == pytest.approx(2.5)


def test_probe_sync_points_change_the_prediction_in_both_directions():
    lens = long_tail(3000, seed=14)
    ar = AllReduce(4, ((120.0, 30.0),))
    full = diagnose(lens, SETUP, probe=make_probe(gpus=4, allreduce=ar), sample_steps=20)
    hidden = diagnose(lens, SETUP, probe=make_probe(gpus=4, allreduce=ar, sync_points=[[1.0, 0.1], [1e6, 0.1]]),
                      sample_steps=20)
    for a, b in zip(full.rows, hidden.rows):
        if a.available and a.key != "ceiling":
            assert a.step_ms - b.step_ms == pytest.approx(0.9 * ar.ms(4 * MODEL["params"], 4))
    assert not any("did not measure" in n for n in hidden.notes)
    job = PK.Job(lens, sequences_per_step=32, per_device_batch=8, max_len=2048, sample_steps=10)
    p = make_probe(gpus=4, allreduce=ar, sync_points=[[1.0, 0.1], [1e6, 0.1]], source="p.json")
    q = make_probe(gpus=4, allreduce=ar, source="q.json")
    rp, rq = PK.pick(job, [p]).results[0], PK.pick(job, [q]).results[0]
    assert rq.step_ms - rp.step_ms == pytest.approx(0.9 * ar.ms(4 * MODEL["params"], 4))
    # an offer of the same card reuses its measured sync only over the same kind of link
    assert PK.from_offer(offer(gpu_name="RTX 4060", num_gpus=4), make_probe(), [p]).sync_points == ()
    p.machine.update(worst_link="PHB", p2p="none")          # host copies, like the GeForce offer
    twin = PK.from_offer(offer(gpu_name="RTX 4060", num_gpus=4), make_probe(), [p])
    assert twin.sync_points == ((1.0, 0.1), (1e6, 0.1)) and twin.allreduce is p.allreduce


def test_measured_overlap_shortens_predicted_steps():
    lens = long_tail(3000, seed=13)
    ar = AllReduce(4, ((120.0, 30.0),))
    full = diagnose(lens, SETUP, probe=make_probe(gpus=4, allreduce=ar), sample_steps=20)
    hid = diagnose(lens, SETUP, probe=make_probe(gpus=4, allreduce=ar, overlap=0.3), sample_steps=20)
    for a, b in zip(full.rows, hid.rows):
        if a.available and a.key != "ceiling":
            assert b.step_ms < a.step_ms
            assert sum(b.shares.values()) == pytest.approx(1.0)
    assert any("did not measure how much sync DDP hides" in n for n in full.notes)
    assert not any("did not measure" in n for n in hid.notes)
    job = PK.Job(lens, sequences_per_step=32, per_device_batch=8, max_len=2048, sample_steps=10)
    a = PK.pick(job, [make_probe(gpus=4, allreduce=ar, source="a.json")]).results[0]
    b = PK.pick(job, [make_probe(gpus=4, allreduce=ar, overlap=0.3, source="b.json")]).results[0]
    assert b.step_ms < a.step_ms and b.default_step_ms < a.default_step_ms


def test_an_empty_last_micro_batch_hides_no_sync():
    """The slowest GPU gets no batch in the last micro-step: nothing overlaps its all-reduce."""
    lens = [5, 5, 5, 5]
    plan = [[[0, 1, 2], [3]], [[], []]]                       # step of 2 micro-steps; the last one is empty
    curve, sync = linear_curve(per=1.0, a=0), Sync(10.0, 0.5)
    full = P.evaluate(plan, lens, 2, 2, cost=P.AttentionCost(0), timing=curve, exposed=sync)
    job = PK.Job(lens, sequences_per_step=4, per_device_batch=4, sample_steps=1)
    shapes, _ = PK.PlanCache(job, 0)._shapes(plan, 2, 2)
    assert full.sync_ms == pytest.approx(10.0)
    assert PK.plan_ms(shapes, curve, sync) == pytest.approx((full.step_ms + full.sync_ms) / full.steps)


def test_review_fixes_in_pick_and_power_table():
    # a same-GPU twin's link class comes from its measurement: an A100 SXM pair joined over PCIe
    twin = make_probe(gpus=2, gpu="NVIDIA A100-SXM4-40GB", allreduce=AllReduce(2, ((120.0, 11.6),)), source="sys.json")
    twin.machine.update(worst_link="SYS", p2p="all")
    assert PK.measured_link_class(twin) == "pcie"
    ref = make_probe()
    over_pcie = PK.from_offer(offer(gpu_name="A100 SXM4", num_gpus=2, bw_nvlink=0.0), ref, [twin])
    over_nvlink = PK.from_offer(offer(gpu_name="A100 SXM4", num_gpus=2, bw_nvlink=300.0), ref, [twin])
    assert over_pcie.allreduce is twin.allreduce            # same kind of link: the measured sync is reused
    assert over_nvlink.allreduce is None                     # a real NVLink box is not slowed to the PCIe one
    # the reference must have measured the same model
    lens = long_tail(2000, seed=21)
    job = PK.Job(lens, sequences_per_step=16, per_device_batch=8, max_len=2048, sample_steps=5)
    other = make_probe(model={"params": 91_245_312, "layers": 12, "d_model": 768}, source="base.json")
    with pytest.raises(EvencutError, match="reference base.json measured another model"):
        PK.pick(job, [make_probe()], reference=other)
    # whole-word power figures
    from evencut.train.diagnose import gpu_watts
    assert gpu_watts("NVIDIA RTX A4000") is None and gpu_watts("RTX A1000") is None
    assert gpu_watts("NVIDIA A40") == 300 and gpu_watts("NVIDIA GeForce RTX 4060 Ti") == 165   # its own row
    assert gpu_watts("NVIDIA GeForce RTX 4070 Ti") is None          # not the RTX 4070's figure
    assert gpu_watts("NVIDIA GeForce RTX 4060") == 115 and gpu_watts("A100 SXM4") == 400


def test_window_step_rule_is_shared():
    times = [[3.0, 4.0], [1.0, 9.0], [2.0, 2.0]]
    assert P.window_step_ms(times) == (10.0, 0.0)
    assert P.window_step_ms(times, Sync(8.0, 0.5)) == (10.0, pytest.approx(8.0 - 4.5))
    assert P.window_step_ms([[9.0]], Sync(8.0, 0.5), totals=[20.0]) == (20.0, pytest.approx(3.5))


def test_cli_pick_tells_probe_files_of_one_name_apart(tmp_path, capsys):
    lens = long_tail(3000, seed=31)
    (tmp_path / "l.txt").write_text("\n".join(map(str, lens)))
    for box, ms in (("a", 3.0), ("b", 30.0)):
        (tmp_path / box).mkdir()
        make_probe(curve=floor_curve(), gpus=2, allreduce=AllReduce(2, ((120.0, ms),))).save(tmp_path / box / "probe.json")
    base = ["pick", str(tmp_path / "l.txt"), "--sequences-per-step", "32", "--per-device-batch", "8", "--sample-steps", "5",
            "--probe", str(tmp_path / "a" / "probe.json"), str(tmp_path / "b" / "probe.json")]
    assert cli.main(base + ["--price", f"{tmp_path / 'a' / 'probe.json'}=1.0", "--price",
                            f"{tmp_path / 'b' / 'probe.json'}=2.0"]) == 0
    out = capsys.readouterr().out
    assert "(a/probe.json)" in out and "(b/probe.json)" in out and "1.00" in out and "2.00" in out
    assert cli.main(base + ["--price", "probe.json=1.0"]) == 2                  # ambiguous by name alone
    assert "several probe files" in capsys.readouterr().err


def test_pick_report_keeps_long_machine_names_whole(tmp_path, capsys):
    lens = long_tail(3000, seed=32)
    (tmp_path / "l.txt").write_text("\n".join(map(str, lens)))
    for box in ("vast_a100x2_long_folder_name_v3", "vast_rtx4090x2_long_folder_name_v3"):
        (tmp_path / box).mkdir()
        make_probe(curve=floor_curve(), gpus=2, allreduce=AllReduce(2, ((120.0, 3.0),))).save(
            tmp_path / box / "probe_small.json")
    assert cli.main(["pick", str(tmp_path / "l.txt"), "--sequences-per-step", "32", "--per-device-batch", "8",
                     "--sample-steps", "5", "--probe",
                     str(tmp_path / "vast_a100x2_long_folder_name_v3" / "probe_small.json"),
                     str(tmp_path / "vast_rtx4090x2_long_folder_name_v3" / "probe_small.json")]) == 0
    out = capsys.readouterr().out
    assert "(vast_a100x2_long_folder_name_v3/probe_small.json)" in out
    assert "(vast_rtx4090x2_long_folder_name_v3/probe_small.json)" in out
    rows = [ln for ln in out.splitlines() if "probe_small.json)" in ln]
    assert len(rows) == 2 and len({ln.index(") ") for ln in rows}) == 2      # names of two lengths, columns aligned
    assert len({len(ln.split(") ")[0]) + len(ln.split(") ")[1]) - len(ln.split(") ")[1].lstrip()) for ln in rows}) == 1
