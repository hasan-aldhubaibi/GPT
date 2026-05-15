import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        t = torch.arange(max_seq_len).float()

        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x):

        B, H, T, D = x.shape

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

        out = torch.stack(
            [
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ],
            dim=-1
        )

        return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        dropout,
        block_size
    ):
        super().__init__()

        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=False
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.dropout = dropout

        self.rope = RoPE(
            self.head_dim,
            max_seq_len=block_size
        )

        mask = torch.tril(
            torch.ones(block_size, block_size)
        ).view(1, 1, block_size, block_size)

        self.register_buffer("mask", mask)

    def forward(self, x, return_attn=False):

        B, T, C = x.shape

        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        # ============================================
        # APPLY ROPE
        # ============================================

        q = self.rope(q)
        k = self.rope(k)

        # ============================================
        # ATTENTION
        # ============================================

        attn_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attn = F.softmax(
            attn_scores,
            dim=-1
        )

        attn = F.dropout(
            attn,
            p=self.dropout,
            training=self.training
        )

        out = attn @ v

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        out = self.out_proj(out)

        if return_attn:
            return out, attn

        return out