# tests/test_gpt.py
import torch
from model.gpt import GPT

config = {
    "vocab_size": 100,
    "block_size": 16,
    "n_layer": 2,
    "n_head": 4,
    "d_model": 32,
    "dropout": 0.1
}

model = GPT(config)

x = torch.randint(0, 100, (2, 8))
out = model(x)

print("Input:", x.shape)
print("Output:", out.shape)