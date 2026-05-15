import torch
import yaml
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

from model.gpt import GPT


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/small.yaml")
    device = config["device"]

    # model
    model = GPT(config)
    ckpt = torch.load("experiments/checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # tokenizer
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    # use REAL text (important)
    prompt = "The meaning of life is to"
    tokens = tokenizer.encode(prompt)

    idx = torch.tensor([tokens.ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, attn = model(idx, return_attn=True)

    # attn shape: (B, heads, T, T)
    attn = attn[0, 0].cpu()  # first batch, first head

    token_strings = tokens.tokens

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()

    plt.xticks(range(len(token_strings)), token_strings, rotation=45)
    plt.yticks(range(len(token_strings)), token_strings)

    plt.title("Attention Map (Head 0)")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    import os
    os.makedirs("experiments", exist_ok=True)

    plt.tight_layout()
    plt.savefig("experiments/attention_map.png")
    plt.show()


if __name__ == "__main__":
    main()