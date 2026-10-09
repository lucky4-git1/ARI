# CPU Inference Optimization & Benchmark Guide

## 1. Native CPU Optimizations
- **PyTorch Intra-op Threading**: Tuned with `torch.set_num_threads(N)`. For 4 physical cores, 4 threads minimizes context switching while maximizing SIMD throughput.
- **Key-Value (KV) Caching**: Fast incremental autoregressive decoding without re-evaluating past attention keys and values.
- **Grouped Query Attention (GQA)**: Cuts attention bandwidth by a factor of 2x or 4x during decoding.

## 2. Benchmark Results on Host CPU
- **Hardware**: Windows 11 64-bit, 4 Cores / Threads utilized, 16 GB Total RAM.
- **Model Checkpoint**: `artifacts/checkpoints/exp_pretrain_v1_best.pt` (909,824 parameters).
- **Prompt**: 5 tokens (`def quicksort(arr):`).
- **Generation**: 40 tokens.
- **Time to First Token (TTFT)**: **6.18 ms**.
- **Decode Throughput**: **140.92 tokens/second**.

## 3. Serving via Local HTTP API
Run:
```bash
python -c "from arillm.api.app import app, initialize_api; initialize_api('artifacts/checkpoints/exp_pretrain_v1_best.pt', 'artifacts/tokenizers/ari_tokenizer.json'); import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"
```
Endpoints:
- `GET /health`
- `GET /info`
- `POST /generate` with body `{"prompt": "def add(a, b):", "max_new_tokens": 30}`
