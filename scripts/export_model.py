"""Export LAYA-LLM to ONNX format."""

import os
import argparse
import torch

from layallm.training.checkpoint import load_checkpoint


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _, _ = self.model(input_ids, use_cache=False)
        return logits


def export_onnx(checkpoint_path: str, output_path: str):
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    model = loaded["model"].eval()
    
    wrapper = OnnxExportWrapper(model)
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.long)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Exporting model to ONNX: {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size", 1: "seq_len"}},
        opset_version=17,
    )
    print(f"ONNX export succeeded! File size: {os.path.getsize(output_path):,} bytes")


if __name__ == "__main__":
    export_onnx("artifacts/checkpoints/exp_pretrain_v1_best.pt", "artifacts/models/laya_model.onnx")
