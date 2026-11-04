"""evencut.train.diagnose and the evencut-train command line."""

import json
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evencut import EvencutError
from evencut.train import cli
from evencut.train.diagnose import Calibration, Model, Setup, diagnose, gpu_watts

PROPS = settings(max_examples=40, deadline=None)


def long_tail(n, seed=1, cap=2048):
    rng = random.Random(seed)
    return [max(1, min(cap, int(rng.paretovariate(1.1) * 40))) for _ in range(n)]


@PROPS
@given(st.integers(1, 6), st.integers(1, 4), st.integers(1, 3), st.integers(0, 5), st.booleans())
def test_shares_add_up_and_nothing_beats_the_ceiling(gpus, per_device, accum, seed, fsdp):
    lengths = long_tail(gpus * per_device * accum * 12, seed)
    d = diagnose(lengths, Setup(gpus=gpus, global_batch=gpus * per_device, accum=accum, seed=seed,
                                sync="microstep" if fsdp else "window"), sample_steps=10)
    ceiling = d.row("ceiling").speedup
    assert d.row("default").speedup == pytest.approx(1.0)
    for r in d.rows:
        if r.available:
            assert sum(r.shares.values()) == pytest.approx(1.0)
            assert r.speedup <= ceiling * (1 + 1e-9)


def test_equal_lengths_waste_nothing_and_the_advice_says_so():
    d = diagnose([128] * 4096, Setup(gpus=8, global_batch=64, accum=2))
    cur = d.row("default")
    assert cur.shares["useful"] == pytest.approx(1.0)
    assert "would not pay off" in d.recommendation


def test_long_tail_finds_waste_and_recommends_a_balanced_split():
    d = diagnose(long_tail(40_000), Setup(gpus=8, global_batch=64, accum=4, max_len=2048))
    cur = d.row("default")
    assert cur.shares["padding"] + cur.shares["waiting"] > 0.3
    best = max((r for r in d.rows if r.available and r.fixed_global_batch and r.key != "ceiling"),
               key=lambda r: r.speedup)
    assert best.speedup > 1.5
    assert d.recommendation.startswith("Switch to")
    # window balancing is offered only with accumulation, and is at least as good here
    assert d.row("evencut_window").speedup >= d.row("evencut_step").speedup


def test_money_and_energy_follow_the_shares():
    cal = Calibration(run_hours=10.0, price_per_gpu_hour=2.0, watts=400, watts_source="given")
    d = diagnose(long_tail(20_000), Setup(gpus=4, global_batch=32), calibration=cal)
    m, cur = d.money, d.row("default")
    assert m["run_hours"] == pytest.approx(10.0)
    assert m["gpu_hours"] == pytest.approx(40.0)
    wasted = 40.0 * (cur.shares["padding"] + cur.shares["waiting"])
    assert m["wasted_gpu_hours"] == pytest.approx(wasted)
    assert m["wasted_cost"] == pytest.approx(wasted * 2.0)
    assert m["wasted_kwh"] == pytest.approx(wasted * 0.4)
    best = d.row("evencut_step")
    assert m["saves"]["evencut_step"]["hours_saved"] == pytest.approx(10.0 * (1 - 1 / best.speedup))


def test_fixed_time_shrinks_every_gain():
    lengths = long_tail(20_000)
    setup = Setup(gpus=8, global_batch=64)
    free = diagnose(lengths, setup, calibration=Calibration(step_ms=100.0))
    slow = diagnose(lengths, setup, calibration=Calibration(step_ms=100.0, fixed_ms=60.0))
    for r in free.rows:
        if r.available and r.key != "default":
            assert 1.0 <= slow.row(r.key).speedup <= r.speedup
    assert slow.row("default").shares["fixed"] == pytest.approx(0.6)
    # without a time scale the communication cannot be combined, and that is said
    ignored = diagnose(lengths, setup, calibration=Calibration(fixed_ms=60.0))
    assert any("fixed_ms was ignored" in n for n in ignored.notes)


def test_max_length_padding_shows_the_dynamic_padding_fix():
    d = diagnose(long_tail(20_000), Setup(gpus=4, global_batch=32, max_len=2048, pad="max_length"))
    dyn = d.row("dynamic_padding")
    # dynamic padding helps, but the step still waits for the GPU holding the longest sequence;
    # balancing the padded batches gains far more on a long tail
    assert 1.2 < dyn.speedup < d.row("evencut_step").speedup
    assert "max_length" in d.recommendation


def test_attention_cost_changes_the_balance_target():
    lengths = long_tail(20_000, cap=8192)
    plain = diagnose(lengths, Setup(gpus=8, global_batch=32, max_len=8192))
    attn = diagnose(lengths, Setup(gpus=8, global_batch=32, max_len=8192), model=Model(125_000_000, 12, 768))
    assert "attention" in attn.cost_model and "attention" not in plain.cost_model.split(";")[0]
    assert attn.row("evencut_step").speedup != plain.row("evencut_step").speedup


def test_packed_runs_are_told_to_keep_packing():
    d = diagnose(long_tail(5_000), Setup(gpus=2, global_batch=16, current="packed"))
    assert "Keep packing" in d.recommendation


def test_bad_setups_are_refused():
    with pytest.raises(EvencutError):
        diagnose([1] * 100, Setup(gpus=3, global_batch=8))           # not per-device x gpus
    with pytest.raises(EvencutError):
        diagnose([1, -2, 3] * 100, Setup(gpus=1, global_batch=4))
    with pytest.raises(EvencutError):
        diagnose([1] * 10, Setup(gpus=2, global_batch=8, accum=2))   # less than one optimizer step
    with pytest.raises(EvencutError):
        diagnose([1] * 100, Setup(gpus=2, global_batch=8, pad="max_length"))
    with pytest.raises(EvencutError):
        diagnose([5] * 100, Setup(gpus=2, global_batch=8), calibration=Calibration(step_ms=10, fixed_ms=20))


def test_gpu_power_table():
    assert gpu_watts("NVIDIA A100-SXM4-80GB") == 400
    assert gpu_watts("RTX 4060 Ti") == 165 and gpu_watts("GeForce RTX 4060") == 115
    assert gpu_watts("H100 PCIe") == 350 and gpu_watts("H100") == 700
    assert gpu_watts("Voodoo 2") is None


# ---------------------------------------------------------------------------------------------
# command line


def test_cli_diagnose_writes_report_and_json(tmp_path, capsys):
    f = tmp_path / "lengths.txt"
    f.write_text("﻿" + "\n".join(map(str, long_tail(8_000))) + "\n", encoding="utf-8")  # PowerShell BOM
    out = tmp_path / "d.json"
    rc = cli.main(["diagnose", str(f), "--gpus", "4", "--per-device-batch", "8", "--accum", "2",
                   "--max-len", "1024", "--run-hours", "12", "--price-per-hour", "10", "--gpu", "A100 SXM",
                   "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "Where the GPU time goes now" in text and "Advice:" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["setup"]["global_batch"] == 32
    assert data["money"]["price_per_gpu_hour"] == pytest.approx(2.5)
    assert data["money"]["watts"] == 400


def test_cli_reads_json_jsonl_and_rejects_garbage(tmp_path, capsys):
    j = tmp_path / "l.json"
    j.write_text(json.dumps([3, 4, 5] * 20), encoding="utf-8")
    assert cli.read_lengths(str(j)) == [3, 4, 5] * 20
    jl = tmp_path / "l.jsonl"
    jl.write_text('{"n": 3}\n{"n": 7}\n\n{"ids": [1, 2]}\n', encoding="utf-8")
    with pytest.raises(EvencutError):
        cli.read_lengths(str(jl), "n")                      # third record lacks the field
    jl.write_text('{"ids": [1, 2, 3]}\n{"ids": [9]}\n', encoding="utf-8")
    assert cli.read_lengths(str(jl), "ids") == [3, 1]
    bad = tmp_path / "bad.txt"
    bad.write_text("12 apples\n", encoding="utf-8")
    assert cli.main(["diagnose", str(bad), "--gpus", "1", "--global-batch", "1"]) == 2
    assert "whole numbers" in capsys.readouterr().err
    assert cli.main(["diagnose", str(j), "--gpus", "2", "--global-batch", "4", "--params", "1M"]) == 2


@pytest.mark.parametrize("text,value", [("29.9M", 29_900_000), ("7B", 7_000_000_000), ("91245312", 91_245_312),
                                        ("1_000k", 1_000_000)])
def test_parse_count(text, value):
    assert cli.parse_count(text) == value


def test_report_ends_with_the_one_line_fix_only_when_it_pays():
    from evencut.train.diagnose import how_to
    d = diagnose(long_tail(20_000), Setup(gpus=4, global_batch=32, accum=2, max_len=2048))
    code = how_to(d)
    assert code.startswith("DataLoader(dataset, batch_sampler=BalancedBatchSampler(lengths, per_device_batch=8")
    assert "accum=2" in code and "max_len=2048" in code
    assert d.report().rstrip().endswith(code)
    flat = diagnose([64] * 4096, Setup(gpus=4, global_batch=32))
    assert how_to(flat) == "" and "BalancedBatchSampler" not in flat.report()
