"""Checkpoint save, atomic write, and restoration utilities."""

import os
import shutil
import tempfile
from typing import Dict, Any, Optional, Union
import torch

from arillm.model.config import ModelConfig
from arillm.model.transformer import AriTransformer


def save_checkpoint(
    save_path: str,
    model: AriTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    loss: float = 0.0,
    config: Optional[ModelConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Atomically saves full training checkpoint to avoid file corruption on interruption."""
    directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(directory, exist_ok=True)

    state: Dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "model_config": (config or model.config).to_dict(),
        "extra_metadata": extra_metadata or {},
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()

    # Atomic write pattern using tempfile
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".pt.tmp") as tmp_f:
        torch.save(state, tmp_f)
        tmp_name = tmp_f.name

    shutil.move(tmp_name, save_path)


def load_checkpoint(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Loads checkpoint and reconstitutes model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = AriTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if load_optimizer and scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "model": model,
        "config": config,
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "extra_metadata": checkpoint.get("extra_metadata", {}),
    }
