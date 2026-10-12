# Kaggle Scaled Training Guide: Ari-LLM 95M MoE (2048 Context)

This guide provides instructions to run the scaled **94.09M Sparse Mixture-of-Experts (MoE)** model on a free Kaggle GPU (T4 or P100).

---

## 1. Model Specifications

| Parameter | Original MoE | **Doubled Scaled MoE** |
| :--- | :--- | :--- |
| **Total Parameters** | 47.96M | **94.09M Parameters** |
| **Active / Token** | 27.32M | **52.80M Parameters** |
| **Context Window** | 1,024 tokens | **2,048 tokens** |
| **Model Depth** | 8 layers | **16 layers** |
| **Attention Architecture** | 8 Query, 4 KV Heads (GQA) | **8 Query, 4 KV Heads (GQA)** |
| **Experts per Block** | 4 experts (top-2 routing) | **4 experts (top-2 routing)** |
| **Feed-Forward Network**| SwiGLU (`dim: 448`, `intermediate: 960`) | **SwiGLU (`dim: 448`, `intermediate: 960`)** |
| **Vocabulary** | 4,096 Byte-level BPE | **4,096 Byte-level BPE** |

---

## 2. Training on Kaggle (Step-by-Step)

### Step A: Open Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and create a **New Notebook**.
2. Under **Notebook Options** (right sidebar):
   - Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
   - Turn **Internet** to **ON**.

---

### Step B: Clone & Setup Code
Paste into the first Kaggle cell:

```bash
!git clone https://github.com/lucky4-git1/ARI.git
%cd ARI
!pip install -q torch torchvision torchaudio
```

---

### Step C: Execute Scaled 95M Training
Run the training script in the second Kaggle cell:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Run the 95M MoE 2048-context training script
!python scripts/train_kaggle_moe_95m.py
```

---

### Step D: Download & Deploy Locally
Once training finishes, Kaggle outputs the checkpoint to `/kaggle/working/ARI/artifacts/checkpoints/exp_kaggle_95m_moe_best.pt`.
1. Download the checkpoint `.pt` or zip file.
2. Place it in `artifacts/checkpoints/` on your PC.
3. The server will automatically load the model for local CPU inference.
