import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()

        from model.attention import MultiHeadAttention

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout, block_size)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x