import os

import torch
import yaml

from tokenizers import Tokenizer

from model.gpt import GPT
from inference.sampler import sample


def load_config(path):

    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate(
    model,
    idx,
    tokenizer,
    max_new_tokens,
    block_size,
    temperature=0.8,
    top_k=40,
    top_p=0.92
):

    model.eval()

    eos_id = tokenizer.token_to_id("[EOS]")

    generated_tokens = idx[0].tolist()

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        with torch.no_grad():

            logits = model(idx_cond)

        logits = logits[:, -1, :]

        next_token = sample(
            logits,
            generated_tokens=generated_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        token_id = next_token.item()

        generated_tokens.append(token_id)

        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

        # ========================================
        # EOS STOPPING
        # ========================================

        if token_id == eos_id:
            break

    return idx


def main():

    config = load_config(
        "configs/small.yaml"
    )

    device = config["device"]

    # ============================================
    # MODEL
    # ============================================

    model = GPT(config)

    ckpt = torch.load(
        "experiments/checkpoints/best.pt",
        map_location=device
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.to(device)

    model.eval()

    # ============================================
    # TOKENIZER
    # ============================================

    tokenizer = Tokenizer.from_file(
        "tokenizer/tokenizer.json"
    )

    # ============================================
    # PROMPT
    # ============================================

    prompt = "Once upon a time"

    input_ids = tokenizer.encode(prompt).ids

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long
    ).to(device)

    # ============================================
    # GENERATE
    # ============================================

    out = generate(
        model=model,
        idx=idx,
        tokenizer=tokenizer,
        max_new_tokens=120,
        block_size=config["block_size"],
        temperature=0.8,
        top_k=40,
        top_p=0.92
    )

    # ============================================
    # DECODE
    # ============================================

    output_ids = out[0].tolist()

    text = tokenizer.decode(output_ids)

    print("\n==============================")
    print("GENERATED TEXT")
    print("==============================\n")

    print(text)

    os.makedirs(
        "experiments/generations",
        exist_ok=True
    )

    with open(
        "experiments/generations/sample.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)


if __name__ == "__main__":
    main()