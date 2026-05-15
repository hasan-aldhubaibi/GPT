# tests/test_attention_basic.py
import torch
from model.attention import MultiHeadAttention

x = torch.randn(2, 8, 32)  # (batch, seq, d_model)

attn = MultiHeadAttention(
    d_model=32,
    n_head=4,
    dropout=0.1,
    block_size=16
)

out = attn(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)