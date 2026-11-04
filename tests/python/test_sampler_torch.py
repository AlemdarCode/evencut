"""BalancedBatchSampler with real PyTorch: DataLoader, and two-process DDP (gloo on CPU).

The DDP test checks the documented loss recipe: with per-GPU batches of different sizes,
``token_loss.sum() * world_size / sampler.global_tokens(step)`` makes DDP's averaged gradient
equal the gradient of the whole optimizer step computed in one process.
"""

import contextlib
import random

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from evencut.train import BalancedBatchSampler  # noqa: E402

VOCAB, WORLD, PER_DEVICE, ACCUM = 37, 2, 3, 2


def make_lengths(n=WORLD * PER_DEVICE * ACCUM * 6, seed=4):
    rng = random.Random(seed)
    return [rng.choice([2, 3, 5, 9, 20, 41]) for _ in range(n)]


class Seqs(Dataset):
    def __init__(self, lengths):
        g = torch.Generator().manual_seed(0)
        self.items = [torch.randint(0, VOCAB, (n,), generator=g) for n in lengths]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return i, self.items[i]


def pad_collate(batch):
    idx = [i for i, _ in batch]
    longest = max(len(x) for _, x in batch)
    tokens = torch.zeros(len(batch), longest, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for r, (_, x) in enumerate(batch):
        tokens[r, :len(x)] = x
        labels[r, :len(x)] = x
    return idx, tokens, labels


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.emb = torch.nn.Embedding(VOCAB, 8)
        self.out = torch.nn.Linear(8, VOCAB)

    def forward(self, tokens, labels):
        logits = self.out(torch.tanh(self.emb(tokens)))
        return torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1),
                                                 ignore_index=-100, reduction="sum")


def test_dataloader_yields_the_sampler_batches():
    lengths = make_lengths()
    s = BalancedBatchSampler(lengths, per_device_batch=PER_DEVICE, accum=ACCUM, rank=1, world_size=WORLD)
    loader = DataLoader(Seqs(lengths), batch_sampler=s, collate_fn=pad_collate)
    got = [idx for idx, _, _ in loader]
    assert got == list(s) and len(loader) == len(s)
    for idx, tokens, _labels in DataLoader(Seqs(lengths), batch_sampler=s, collate_fn=pad_collate):
        assert tokens.shape[1] == max(lengths[i] for i in idx)          # padded to its own longest


def _worker(rank, init, out):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=WORLD)
    try:
        lengths = make_lengths()
        ds = Seqs(lengths)
        sampler = BalancedBatchSampler(lengths, per_device_batch=PER_DEVICE, accum=ACCUM)   # rank/world from dist
        assert (sampler.rank, sampler.world_size) == (rank, WORLD)
        model = DDP(Tiny())
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=pad_collate)
        seen, micro = [], 0
        for epoch in range(2):
            sampler.set_epoch(epoch)
            for i, (idx, tokens, labels) in enumerate(loader):
                step = i // ACCUM
                last = (i + 1) % ACCUM == 0
                with contextlib.nullcontext() if last else model.no_sync():
                    loss = model(tokens, labels) * sampler.loss_scale(i)
                    loss.backward()
                micro += 1
                if epoch == 0:
                    seen += idx
                if epoch == 0 and step == 0 and last:
                    grads = [p.grad.clone() for p in model.parameters()]
                if last:
                    model.zero_grad(set_to_none=True)
        # reference: the whole first optimizer step in one process, mean over its real tokens
        ref = Tiny()
        step0 = [i for m in sampler.plan(0)[:ACCUM] for b in m for i in b]
        _, tokens, labels = pad_collate([ds[i] for i in step0])
        (ref(tokens, labels) / sampler.global_tokens(0, epoch=0)).backward()
        same = all(torch.allclose(g, p.grad, atol=1e-6) for g, p in zip(grads, ref.parameters()))
        out.put((rank, micro, sorted(seen), same))
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_same_steps_full_coverage_exact_gradient(tmp_path):
    init = "file:///" + str(tmp_path / "init").replace("\\", "/")
    ctx = mp.get_context("spawn")
    out = ctx.SimpleQueue()
    procs = [ctx.Process(target=_worker, args=(r, init, out)) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=240)
        assert p.exitcode == 0, f"worker exit code {p.exitcode}"
    results = sorted(out.get() for _ in range(WORLD))
    lengths = make_lengths()
    (_, micro0, seen0, same0), (_, micro1, seen1, same1) = results
    assert micro0 == micro1                                  # no rank ran an extra step (no hang)
    assert sorted(seen0 + seen1) == list(range(len(lengths)))   # the epoch, each sequence once
    assert same0 and same1                                   # DDP gradient == whole-step gradient
