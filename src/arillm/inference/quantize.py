"""INT8 Dynamic Quantization and CPU Acceleration for Ari-LLM.

Quantizes linear weight layers from FP32 to INT8 to:
1. Halve RAM memory consumption (~50% reduction).
2. Accelerate CPU matrix multiplications using AVX2/VNNI vector integer operations.
3. Preserve token generation accuracy and perplexity.
"""

import os
import time
from typing import Dict, Any
import torch
import torch.nn as nn

from arillm.training.checkpoint import load_checkpoint, save_checkpoint
from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.inference.generate import CPUInferenceEngine


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Applies dynamic INT8 quantization to linear projection layers."""
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def run_quantization(
    base_checkpoint_path: str = "artifacts/checkpoints/exp_kaggle_sft_best.pt",
    output_checkpoint_path: str = "artifacts/checkpoints/exp_kaggle_sft_int8.pt",
    tokenizer_path: str = "artifacts/tokenizers/ari_tokenizer_v2.json",
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"       Ari-LLM INT8 Dynamic CPU Quantization           ")
    print(f"=======================================================")
    print(f"Source Checkpoint: {base_checkpoint_path}")

    loaded = load_checkpoint(base_checkpoint_path, device="cpu")
    base_model = loaded["model"].eval()
    tokenizer = AriTokenizer.load(tokenizer_path)

    # 1. Benchmark FP32 Baseline
    print("\n[1/3] Benchmarking FP32 Baseline Speed...")
    fp32_engine = CPUInferenceEngine(base_model, tokenizer, num_threads=4)
    fp32_profile = fp32_engine.benchmark_throughput(prompt="What is gravity?", num_tokens_to_generate=40, num_runs=3)
    orig_size_bytes = os.path.getsize(base_checkpoint_path)

    # 2. Apply Dynamic INT8 Quantization
    print("\n[2/3] Applying Dynamic INT8 Quantization (torch.qint8)...")
    t0 = time.perf_counter()
    quantized_model = quantize_model_int8(base_model)
    quant_time = time.perf_counter() - t0
    print(f"Quantized in {quant_time:.2f}s.")

    # 3. Benchmark INT8 Quantized Model
    print("\n[3/3] Benchmarking INT8 Quantized Model Speed...")
    int8_engine = CPUInferenceEngine(quantized_model, tokenizer, num_threads=4)
    int8_profile = int8_engine.benchmark_throughput(prompt="What is gravity?", num_tokens_to_generate=40, num_runs=3)

    # Save quantized checkpoint
    save_checkpoint(
        save_path=output_checkpoint_path,
        model=quantized_model,
        config=loaded["config"],
        extra_metadata={"quantization": "dynamic_int8", "base": base_checkpoint_path},
    )
    quant_size_bytes = os.path.getsize(output_checkpoint_path)

    summary = {
        "original_checkpoint_size_mb": round(orig_size_bytes / (1024 * 1024), 2),
        "quantized_checkpoint_size_mb": round(quant_size_bytes / (1024 * 1024), 2),
        "size_reduction_percent": round((1.0 - quant_size_bytes / orig_size_bytes) * 100, 1),
        "fp32_tokens_per_second": round(fp32_profile["tokens_per_second"], 2),
        "int8_tokens_per_second": round(int8_profile["tokens_per_second"], 2),
        "fp32_ttft_ms": round(fp32_profile["ttft_ms"], 2),
        "int8_ttft_ms": round(int8_profile["ttft_ms"], 2),
    }

    print("\nQuantization Summary:")
    print(f"  • Original Size:     {summary['original_checkpoint_size_mb']} MB")
    print(f"  • Quantized Size:    {summary['quantized_checkpoint_size_mb']} MB (Saved {summary['size_reduction_percent']}%)")
    print(f"  • FP32 Throughput:   {summary['fp32_tokens_per_second']} tok/s (TTFT: {summary['fp32_ttft_ms']} ms)")
    print(f"  • INT8 Throughput:   {summary['int8_tokens_per_second']} tok/s (TTFT: {summary['int8_ttft_ms']} ms)")
    print(f"\nQuantized checkpoint saved to: {output_checkpoint_path}")
    return summary


if __name__ == "__main__":
    run_quantization()
