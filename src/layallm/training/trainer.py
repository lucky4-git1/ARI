"""Training loop and engine for LAYA-LLM."""

import os
import time
import math
from typing import Optional, Dict, Any, Callable
import torch
from torch.utils.data import DataLoader

from layallm.model.transformer import LayaTransformer
from layallm.training.optimizer import create_optimizer, create_cosine_scheduler
from layallm.training.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """Production-grade training loop with gradient accumulation, clipping, and metric tracking."""

    def __init__(
        self,
        model: LayaTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 20,
        max_steps: int = 100,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: str = "artifacts/checkpoints",
        save_every_steps: int = 50,
        eval_every_steps: int = 50,
        experiment_id: str = "exp_0",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.save_every_steps = save_every_steps
        self.eval_every_steps = eval_every_steps
        self.experiment_id = experiment_id

        self.optimizer = create_optimizer(self.model, learning_rate=learning_rate, weight_decay=weight_decay)
        self.scheduler = create_cosine_scheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=max_steps)

        self.current_step = 0
        self.best_val_loss = float("inf")
        self.training_history = []

    def evaluate(self) -> float:
        """Computes average cross-entropy validation loss."""
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _, loss, _ = self.model(x, targets=y)
                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    total_batches += 1

        self.model.train()
        return total_loss / max(1, total_batches)

    def train(self) -> Dict[str, Any]:
        """Runs the main autoregressive training loop."""
        self.model.train()
        start_time = time.time()
        running_loss = 0.0
        total_tokens_processed = 0
        
        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0

        print(f"=== Starting Training Run: {self.experiment_id} ===")
        print(f"Device: {self.device} | Max Steps: {self.max_steps} | Accumulation: {self.grad_accum_steps}")

        while self.current_step < self.max_steps:
            self.optimizer.zero_grad()
            step_tokens = 0

            # Gradient accumulation micro-steps
            for micro_step in range(self.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device)
                y = y.to(self.device)
                step_tokens += x.numel()

                _, loss, _ = self.model(x, targets=y)
                loss_scaled = loss / self.grad_accum_steps
                loss_scaled.backward()
                accumulated_loss += loss.item() / self.grad_accum_steps

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer and LR schedule step
            self.optimizer.step()
            self.scheduler.step()

            self.current_step += 1
            total_tokens_processed += step_tokens
            running_loss += accumulated_loss

            # Periodic logging
            if self.current_step % 10 == 0 or self.current_step == 1:
                current_lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                tok_per_sec = total_tokens_processed / max(1.0, elapsed)
                print(
                    f"Step {self.current_step:4d}/{self.max_steps:4d} | "
                    f"Train Loss: {accumulated_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Throughput: {tok_per_sec:.1f} tok/s"
                )

            # Periodic evaluation & Checkpoint saving
            if self.current_step % self.eval_every_steps == 0 or self.current_step == self.max_steps:
                val_loss = self.evaluate()
                print(f"--> [Step {self.current_step}] Validation Loss: {val_loss:.4f}")
                
                # Save regular checkpoint
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.experiment_id}_step_{self.current_step}.pt")
                save_checkpoint(
                    save_path=ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.current_step,
                    loss=accumulated_loss,
                    config=self.model.config,
                    extra_metadata={"val_loss": val_loss, "tokens": total_tokens_processed},
                )

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.checkpoint_dir, f"{self.experiment_id}_best.pt")
                    save_checkpoint(
                        save_path=best_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.current_step,
                        loss=accumulated_loss,
                        config=self.model.config,
                        extra_metadata={"best_val_loss": val_loss, "tokens": total_tokens_processed},
                    )

            self.training_history.append({"step": self.current_step, "loss": accumulated_loss})
            accumulated_loss = 0.0

        total_time = time.time() - start_time
        avg_loss = running_loss / max(1, self.current_step)
        final_tok_per_sec = total_tokens_processed / max(1.0, total_time)

        print(f"=== Training Complete in {total_time:.2f}s ({final_tok_per_sec:.1f} tokens/sec) ===")
        return {
            "experiment_id": self.experiment_id,
            "total_steps": self.current_step,
            "final_loss": self.training_history[-1]["loss"] if self.training_history else 0.0,
            "initial_loss": self.training_history[0]["loss"] if self.training_history else 0.0,
            "total_tokens": total_tokens_processed,
            "elapsed_time": total_time,
            "tokens_per_second": final_tok_per_sec,
            "best_val_loss": self.best_val_loss,
        }
