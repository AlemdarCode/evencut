"""BalancedBatchSampler in PyTorch Lightning (2.6) on CPU, two DDP processes (gloo).

Checks the documented recipe: ``Trainer(use_distributed_sampler=False)``, the sampler as the
DataLoader's batch_sampler, and ``loss * sampler.loss_scale(batch_idx, accum_divided=True)``.
Every rank must train on exactly its planned batches, epochs must reshuffle (Lightning only calls
``set_epoch`` on ``batch_sampler.sampler``), and the first optimizer step's gradient must equal
the whole step's gradient computed in one process.
"""

import os
import socket

import pytest

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")
import torch.multiprocessing as mp  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from evencut.train import BalancedBatchSampler  # noqa: E402
from test_sampler_torch import ACCUM, PER_DEVICE, Seqs, Tiny, make_lengths, pad_collate  # noqa: E402

EPOCHS = 2


class Lit(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Tiny()
        self.lengths = make_lengths()
        self.recorded, self.first_grads = [], None

    def train_dataloader(self):
        self.sampler = BalancedBatchSampler(self.lengths, per_device_batch=PER_DEVICE, accum=ACCUM)
        return DataLoader(Seqs(self.lengths), batch_sampler=self.sampler, collate_fn=pad_collate)

    def training_step(self, batch, batch_idx):
        idx, tokens, labels = batch
        self.recorded.append((self.current_epoch, idx))
        return self.net(tokens, labels) * self.sampler.loss_scale(batch_idx, accum_divided=True)

    def on_before_optimizer_step(self, optimizer):
        if self.first_grads is None:
            self.first_grads = [p.grad.detach().clone() for p in self.net.parameters()]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _worker(rank, port, root, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
                      NODE_RANK="0", WORLD_SIZE="2")
    model = Lit()
    trainer = L.Trainer(accelerator="cpu", devices=2, strategy="ddp", use_distributed_sampler=False,
                        accumulate_grad_batches=ACCUM, max_epochs=EPOCHS, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, default_root_dir=root)
    trainer.fit(model)
    s = model.sampler
    planned = [(e, s.plan(e)[k][rank]) for e in range(EPOCHS) for k in range(len(s))]
    ref = Tiny()                                         # the first optimizer step in one process
    step0 = [i for m in s.plan(0)[:ACCUM] for b in m for i in b]
    ds = Seqs(model.lengths)
    _, tokens, labels = pad_collate([ds[i] for i in step0])
    (ref(tokens, labels) / s.global_tokens(0, epoch=0)).backward()
    exact = all(torch.allclose(g, p.grad, atol=1e-6) for g, p in zip(model.first_grads, ref.parameters()))
    q.put((rank, model.recorded, planned, exact, s.plan(0) != s.plan(1)))


def test_lightning_ddp_recipe(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    procs = [ctx.Process(target=_worker, args=(r, port, str(tmp_path), q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0, f"worker exit code {p.exitcode}"
    for rank, recorded, planned, exact, reshuffled in sorted(q.get() for _ in range(2)):
        assert recorded == planned, f"rank {rank}: trained batches differ from the plan (epochs not reshuffled?)"
        assert reshuffled
        assert exact, f"rank {rank}: first-step gradient differs from the whole-step gradient"
