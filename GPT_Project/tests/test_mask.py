import torch
from model.attention import MultiHeadAttention

torch.manual_seed(0)

x = torch.randn(1, 4, 8)

attn = MultiHeadAttention(
    d_model=8,
    n_head=2,
    dropout=0.0,
    block_size=4
)

with torch.no_grad():
    out, weights = attn(x, return_attn=True)

    print("Attention weights (head 0):")
    print(weights[0, 0])