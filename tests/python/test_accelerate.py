"""BalancedBatchSampler in a plain Accelerate loop, two processes (gloo on CPU): every rank gets its
planned batches through ``accelerate_prepare``, and ``loss_scale(i, accum_divided=True)`` with
``accelerator.backward`` yields the whole step's gradient."""

import os
import socket

import pytest

torch = pytest.importorskip("torch")
accelerate = pytest.importorskip("accelerate")
import torch.multiprocessing as mp  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from evencut.train import BalancedBatchSampler, accelerate_prepare  # noqa: E402
from test_sampler_torch import ACCUM, PER_DEVICE, Seqs, Tiny, make_lengths, pad_collate  # noqa: E402


def _worker(rank, port, q, plain):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE="2")
    from accelerate import Accelerator
    acc = Accelerator(cpu=True, gradient_accumulation_steps=ACCUM)
    lengths = make_lengths()
    sampler = BalancedBatchSampler(lengths, per_device_batch=PER_DEVICE, accum=ACCUM)
    loader = DataLoader(Seqs(lengths), batch_sampler=sampler, collate_fn=pad_collate)
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model, opt = acc.prepare(model, opt)
    loader = acc.prepare(loader) if plain else accelerate_prepare(acc, loader)
    seen, grads = [], None
    for i, (idx, tokens, labels) in enumerate(loader):
        seen.append(list(idx))
        with acc.accumulate(model):
            acc.backward(model(tokens, labels) * sampler.loss_scale(i, accum_divided=True))
            if acc.sync_gradients and grads is None:
                grads = [p.grad.detach().clone() for p in model.parameters()]
            opt.step()
            opt.zero_grad()
    ref = Tiny()
    step0 = [i for m in sampler.plan(0)[:ACCUM] for b in m for i in b]
    ds = Seqs(lengths)
    _, tokens, labels = pad_collate([ds[i] for i in step0])
    (ref(tokens, labels) / sampler.global_tokens(0)).backward()
    exact = all(torch.allclose(g, p.grad, atol=1e-6) for g, p in zip(grads, ref.parameters()))
    q.put((rank, seen, [m[rank] for m in sampler.plan(0)], exact))
    acc.end_training()


def run(plain):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    procs = [ctx.Process(target=_worker, args=(r, port, q, plain)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0, f"worker exit code {p.exitcode}"
    return sorted(q.get() for _ in range(2))


def test_accelerate_prepare_keeps_the_planned_batches_and_exact_gradient():
    for rank, seen, planned, exact in run(plain=False):
        assert seen == planned, f"rank {rank} got other batches than planned"
        assert exact, f"rank {rank}: first-step gradient differs from the whole-step gradient"


def test_plain_prepare_would_split_the_batches_again():
    """Why accelerate_prepare exists: plain accelerator.prepare re-splits the per-GPU batches."""
    results = run(plain=True)
    assert any(seen != planned for _, seen, planned, _ in results)
