# tests/test_block.py
import torch
from model.transformer_block import TransformerBlock

x = torch.randn(2, 8, 32)

block = TransformerBlock(
    d_model=32,
    n_head=4,
    dropout=0.1,
    block_size=16
)

out = block(x)

print("Input:", x.shape)
print("Output:", out.shape)