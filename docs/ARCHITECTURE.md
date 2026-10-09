# LAYA-LLM Architecture

## 1. Overview
LAYA-LLM (Ari-LLM) is an original, compact, decoder-only Transformer language model engineered from scratch. The model starts from randomly initialized parameters and employs modern architectural enhancements:
- **Rotary Position Embeddings (RoPE)**: Applied to query and key projections to provide relative position awareness and length extrapolation without learned absolute positional tables.
- **Root Mean Square Normalization (RMSNorm)**: Pre-layer normalization with learned scaling parameters, eliminating mean-centering for lower CPU computational overhead.
- **Grouped-Query Attention (GQA)**: Configurable multi-query or grouped-query attention to dramatically reduce KV-cache memory footprint and accelerate CPU decoding.
- **SwiGLU Feed-Forward Network**: Gated linear unit using SiLU non-linearity: `(SiLU(xW_gate) * (xW_up)) * W_down`.
- **Tied Word Embeddings**: Shares weights between input token embeddings and final language-model output projection for parameter efficiency.
- **Residual Projection Scaling**: Scales residual projection weights by `1 / sqrt(2 * num_layers)` to stabilize deep gradient backpropagation.

## 2. Configuration Presets

| Name | Layers | Hidden Dim | Heads | KV Heads | Intermediate Dim | Context | Parameters (Tied) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Smoke** | 4 | 128 | 4 | 2 | 384 | 256 | ~1.38M |
| **Tiny** | 8 | 256 | 8 | 4 | 682 | 512 | ~7.86M |
| **Small** | 12 | 512 | 8 | 4 | 1365 | 1024 | ~43.0M |
| **Medium**| 16 | 768 | 12| 4 | 2048 | 2048 | ~125.8M |

## 3. Mathematical Operations
### RoPE Precomputation
$$\text{freqs} = \frac{1}{\theta^{2i / d}}, \quad \text{freqs}_{\text{cis}} = e^{i \cdot t \cdot \text{freqs}}$$

### RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

### SwiGLU
$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$
