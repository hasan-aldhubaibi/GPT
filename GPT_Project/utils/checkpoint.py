import torch
from pathlib import Path


def save_checkpoint(model, optimizer, step, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }, path)


def load_checkpoint(model, optimizer, path, device):
    import os

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"Checkpoint {path} is empty or missing. Starting fresh.")
        return 0

    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Starting from scratch.")
        return 0

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    print(f"Loaded checkpoint from step {ckpt['step']}")
    return ckpt["step"]