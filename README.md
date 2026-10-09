# Ari-LLM (LAYA-LLM)

An original, decoder-only Transformer Large Language Model built and trained completely from scratch with random parameter initialization and a CPU-optimized runtime.

## Quickstart

### 1. Run Tests
```bash
python -c "import tests.test_model as t; t.test_model_config_presets(); t.test_rmsnorm(); t.test_swiglu_mlp(); t.test_causal_self_attention(); t.test_transformer_forward_backward(); t.test_causality(); t.test_generation(); print('PASS')"
python -c "import tests.test_training as t; t.test_tokenizer_training_and_encoding(); t.test_checkpoint_atomic_save_load(); t.test_tiny_training_loop_overfitting(); print('PASS')"
```

### 2. Prepare Data & Train Tokenizer
```bash
python scripts/prepare_data.py
python scripts/train_tokenizer.py
```

### 3. Pretrain Model From Random Initialization
```bash
python scripts/train.py --max_steps 120 --batch_size 8 --seq_len 128 --hidden_dim 128 --num_layers 4 --experiment_id exp_pretrain_v1
```

### 4. Instruction Fine-Tuning (SFT)
```bash
python scripts/train_sft.py --max_steps 50 --lr 2e-4
```

### 5. Generate Text & Run CPU Benchmark
```bash
python scripts/generate.py --checkpoint artifacts/checkpoints/exp_pretrain_v1_best.pt --prompt "def binary_search(arr, target):"
python scripts/benchmark_cpu.py --threads 4 --tokens 40
```
