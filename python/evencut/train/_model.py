"""The synthetic decoder that ``evencut-train probe`` (and bench/gpu_batches.py) time: a
pre-norm transformer with RoPE, SwiGLU, causal scaled-dot-product attention, tied input and
output embeddings, trained in bf16 autocast. Token values do not change its speed, so random ids
stand in for real data. Needs torch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ffn_width(d: int) -> int:
    return 64 * math.ceil(8 * d / 3 / 64)


def weights(vocab: int, d: int, layers: int) -> int:
    """Weights of :class:`GPT` (norms left out, the tied embedding counted once)."""
    return layers * (4 * d * d + 3 * d * ffn_width(d)) + d * vocab


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        ffn = ffn_width(d)
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(d, eps=1e-6)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.w13, self.w2 = nn.Linear(d, 2 * ffn, bias=False), nn.Linear(ffn, d, bias=False)

    @staticmethod
    def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        t, half = x.shape[-2], x.shape[-1] // 2
        cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.n1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(self.rope(q, cos, sin), self.rope(k, cos, sin), v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        gate, up = self.w13(self.n2(x)).chunk(2, dim=-1)
        return x + self.w2(F.silu(gate) * up)


class GPT(nn.Module):
    def __init__(self, vocab: int = 8192, seq_len: int = 512, layers: int = 8, d: int = 512, heads: int = 8):
        super().__init__()
        if d % heads or (d // heads) % 2:
            raise ValueError("d_model must split into heads of even width")
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        hd = d // heads
        inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        ang = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        for p in self.parameters():
            nn.init.normal_(p, std=0.02) if p.dim() >= 2 else None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
