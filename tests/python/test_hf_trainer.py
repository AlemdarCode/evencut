"""EvencutTrainer with a real (tiny) transformers model on CPU: one process, and two DDP processes
(gloo). Checks that every rank trains on exactly the batches the sampler planned for it, i.e.
Accelerate did not split them again, and that the epoch is covered once."""

import os
import random
import socket

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import torch.multiprocessing as mp  # noqa: E402

VOCAB, N, PER_DEVICE, ACCUM = 40, 96, 4, 2


def lengths():
    rng = random.Random(3)
    return [rng.choice([3, 4, 6, 10, 18, 30]) for _ in range(N)]


class Data(torch.utils.data.Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.rows = [{"idx": i, "input_ids": torch.randint(1, VOCAB, (n,), generator=g).tolist()}
                     for i, n in enumerate(lengths())]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Recorder:
    """Pads to the batch's longest and remembers which examples each batch held."""

    def __init__(self):
        self.batches = []

    def __call__(self, rows):
        self.batches.append([r["idx"] for r in rows])
        longest = max(len(r["input_ids"]) for r in rows)
        ids = torch.zeros(len(rows), longest, dtype=torch.long)
        mask = torch.zeros(len(rows), longest, dtype=torch.long)
        for k, r in enumerate(rows):
            ids[k, :len(r["input_ids"])] = torch.tensor(r["input_ids"])
            mask[k, :len(r["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


def build(tmp, **extra):
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

    from evencut.train.hf import EvencutTrainer
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=16, n_layer=1, n_head=2))
    args = TrainingArguments(output_dir=str(tmp), per_device_train_batch_size=PER_DEVICE,
                             gradient_accumulation_steps=ACCUM, num_train_epochs=1, use_cpu=True, report_to="none",
                             save_strategy="no", logging_strategy="no", remove_unused_columns=False,
                             dataloader_drop_last=True, seed=7, disable_tqdm=True, **extra)
    rec = Recorder()
    trainer = EvencutTrainer(model=model, args=args, train_dataset=Data(), data_collator=rec,
                             evencut_lengths=lengths())
    return trainer, rec


def test_single_process_trains_on_the_sampler_batches(tmp_path):
    trainer, rec = build(tmp_path)
    out = trainer.train()
    s = trainer.evencut_sampler
    assert s.world_size == 1
    assert rec.batches == [b for b in s.plan(0) for b in b]       # world 1: one batch per micro-step
    assert out.global_step == s.steps and torch.isfinite(torch.tensor(out.training_loss))


def _worker(rank, port, tmp, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE="2")
    trainer, rec = build(tmp, ddp_backend="gloo")
    out = trainer.train()
    s = trainer.evencut_sampler
    q.put((rank, rec.batches, [m[rank] for m in s.plan(0)], out.global_step, s.steps))


def test_two_process_ddp_each_rank_gets_its_own_batches(tmp_path):
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
    results = sorted(q.get() for _ in range(2))
    seen = []
    for rank, got, planned, steps_done, steps in results:
        assert got == planned, f"rank {rank} trained on other batches than planned"
        assert steps_done == steps
        seen += [i for b in got for i in b]
    assert sorted(seen) == sorted(set(seen)) and len(seen) == (N // (2 * PER_DEVICE * ACCUM)) * 2 * PER_DEVICE * ACCUM
