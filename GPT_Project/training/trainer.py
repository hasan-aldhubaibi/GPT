import math
import os

import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler

from utils.checkpoint import save_checkpoint
from utils.logger import Logger


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config
    ):

        self.model = model.to(config["device"])

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config

        self.device = config["device"]

        self.logger = Logger(
            "experiments/logs/train_loss.csv"
        )

        self.best_val_loss = float("inf")

        # ============================================
        # OPTIMIZER
        # ============================================

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            betas=(
                config["beta1"],
                config["beta2"]
            ),
            weight_decay=config["weight_decay"]
        )

        self.criterion = nn.CrossEntropyLoss()

        self.scaler = GradScaler()

        os.makedirs(
            "experiments/checkpoints",
            exist_ok=True
        )

    # =====================================================
    # LEARNING RATE SCHEDULER
    # =====================================================

    def get_lr(self, step):

        warmup_steps = self.config["warmup_steps"]

        max_iters = self.config["max_iters"]

        base_lr = self.config["learning_rate"]

        min_lr = self.config["min_lr"]

        # ============================================
        # WARMUP
        # ============================================

        if step < warmup_steps:

            return base_lr * step / warmup_steps

        # ============================================
        # COSINE DECAY
        # ============================================

        progress = (
            (step - warmup_steps)
            /
            (max_iters - warmup_steps)
        )

        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * progress)
        )

        lr = min_lr + (
            cosine_decay * (base_lr - min_lr)
        )

        return lr

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, start_step=0):

        self.model.train()

        data_iter = iter(self.train_loader)

        for step in range(
            start_step,
            self.config["max_iters"]
        ):

            # ========================================
            # LR UPDATE
            # ========================================

            lr = self.get_lr(step)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            total_loss = 0.0

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ========================================
            # GRAD ACCUMULATION
            # ========================================

            for _ in range(
                self.config["accumulation_steps"]
            ):

                try:
                    x, y = next(data_iter)

                except StopIteration:

                    data_iter = iter(self.train_loader)

                    x, y = next(data_iter)

                x = x.to(self.device)
                y = y.to(self.device)

                with autocast(device_type="cuda"):

                    logits = self.model(x)

                    B, T, C = logits.shape

                    loss = self.criterion(
                        logits.view(B * T, C),
                        y.view(B * T)
                    )

                    loss = (
                        loss
                        /
                        self.config["accumulation_steps"]
                    )

                self.scaler.scale(loss).backward()

                total_loss += loss.item()

            # ========================================
            # GRAD CLIP
            # ========================================

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_clip"]
            )

            # ========================================
            # OPTIMIZER STEP
            # ========================================

            self.scaler.step(self.optimizer)

            self.scaler.update()

            # ========================================
            # EVALUATION
            # ========================================

            if step % self.config["eval_interval"] == 0:

                val_loss, ppl = self.evaluate()

                print(
                    f"\nStep {step}"
                    f" | LR {lr:.6f}"
                    f" | Train {total_loss:.4f}"
                    f" | Val {val_loss:.4f}"
                    f" | PPL {ppl:.2f}"
                )

                self.logger.log(
                    step,
                    total_loss,
                    val_loss,
                    ppl
                )

                # ====================================
                # SAVE LAST
                # ====================================

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    step,
                    "experiments/checkpoints/last.pt"
                )

                # ====================================
                # SAVE BEST
                # ====================================

                if val_loss < self.best_val_loss:

                    self.best_val_loss = val_loss

                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        step,
                        "experiments/checkpoints/best.pt"
                    )

    # =====================================================
    # EVALUATE
    # =====================================================

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0

        count = 0

        max_eval_iters = self.config["eval_iters"]

        for i, (x, y) in enumerate(self.val_loader):

            if i >= max_eval_iters:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            with autocast(device_type="cuda"):

                logits = self.model(x)

                B, T, C = logits.shape

                loss = self.criterion(
                    logits.view(B * T, C),
                    y.view(B * T)
                )

            total_loss += loss.item()

            count += 1

        avg_loss = total_loss / count

        ppl = torch.exp(
            torch.tensor(avg_loss)
        ).item()

        self.model.train()

        return avg_loss, ppl