import torch
import yaml
from model.gpt import GPT
from training.dataset import GPTDataset
from torch.utils.data import DataLoader
from evaluation.metrics import compute_loss, compute_perplexity


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/small.yaml")
    device = config["device"]

    # model
    model = GPT(config)
    ckpt = torch.load("experiments/checkpoints/best_medium.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # dataset
    val_dataset = GPTDataset("data/processed/val_ids.npy", config["block_size"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    loss = compute_loss(model, val_loader, device)
    ppl = compute_perplexity(loss)

    print(f"Validation Loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()