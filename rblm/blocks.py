"""Shared neural building blocks.

All three models (A/B/C) use the *same* embedding, a deliberately minimal
causal context encoder, and a shared per-iteration readout head. They differ
ONLY in the core (blocks.py provides the pieces; models.py wires the cores).

The context encoder is one causal attention layer with NO feed-forward, so it
can move information between positions but performs little nonlinear work. That
keeps real computation for the recursive core -- otherwise extra depth would
have nothing to do and Q2 (do iterations help?) would be untestable.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """Pre-norm MLP. Returns a *delta* (residual added by the caller).

    F_b(h) = W2 . sigma(W1 . LN(h))     (the plan's block operator)
    """
    def __init__(self, d, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        # small init on the output so stacked/iterated residuals stay stable
        nn.init.zeros_(self.fc2.bias)
        self.fc2.weight.data.mul_(0.5)

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(self.norm(h))))


class CausalAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.norm(x).chunk(3, dim=-1) if False else \
            self.qkv(self.norm(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return x + self.proj(out)


class ContextEncoder(nn.Module):
    def __init__(self, d, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalAttention(d, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
