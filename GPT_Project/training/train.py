from wandb import config
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
import os

from model.gpt import GPT
from training.dataset import GPTDataset
from training.trainer import Trainer
from utils.checkpoint import load_checkpoint


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # dataset
    train_dataset = GPTDataset("data/processed/train_ids.npy", config["block_size"])
    val_dataset = GPTDataset("data/processed/val_ids.npy", config["block_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"]
    )

    # model
    model = GPT(config)

    # trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # checkpoint loading
    start_step = 0
    ckpt_path = "experiments/checkpoints/last.pt"

    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")
        start_step = load_checkpoint(
            trainer.model,
            trainer.optimizer,
            ckpt_path,
            config["device"]
        )

    # train
    trainer.train(start_step)


if __name__ == "__main__":
    main()