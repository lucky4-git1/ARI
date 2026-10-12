# Kaggle Guide: Training the RewriteBot 95M Model (QuillBot Rival)

This guide provides instructions to run the **Ari-LLM RewriteBot** training on a free Kaggle GPU. All coding data has been removed so that 100% of the model capacity is dedicated to English language fluency, grammar, clause inversion, and anti-AI humanization.

---

## 1. Model & Dataset Specifications

- **Total Parameters:** **126.56M Sparse Mixture-of-Experts (MoE)**
- **Active Parameters / Token:** **39.85M Parameters** (GQA Attention + top-2 SwiGLU routing)
- **Vocabulary Size:** **8,192 Byte-level BPE** (Upgraded from 4,096 for richer multi-syllable word encoding)
- **Total Experts:** **8 specialized experts** per block (Doubled from 4)
- **Active Experts:** **2 active per token** (Saves GPU compute while offering 8-way specialization)
- **Context Window:** **2,048 tokens**
- **Depth:** **12 layers**
- **Data Domains Included:**
  1. **QuillBot Paraphrasing Modes:** Standard, Fluency, Formal, Academic, Simple, Creative, Shorten, Expand.
  2. **Grammar Polishing:** Subject-verb agreement, tense consistency, dangling modifiers, preposition choices.
  3. **Clause Inversion:** Flipping dependent and independent clauses while preserving logical meaning.
  4. **Anti-AI Humanizer:** High burstiness, varied sentence lengths, removing cliches (*"delve"*, *"testament"*, *"tapestry"*, *"in conclusion"*).
- **Data Domains Excluded:** All Python, JavaScript, HTML, CSS, and SQL data (reserved for future models).

---

## 2. Running on Kaggle (Step-by-Step)

### Step 1: Create a Kaggle Notebook
1. Open [kaggle.com](https://www.kaggle.com) and create a **New Notebook**.
2. In the right-hand sidebar under **Notebook Options**:
   - **Accelerator:** Select **GPU P100** or **GPU T4 x2**.
   - **Internet:** Set to **ON**.

---

### Step 2: Clone the Repository
In the **first cell**, run:

```bash
!git clone https://github.com/lucky4-git1/ARI.git
%cd ARI
!pip install -q torch torchvision torchaudio
```

---

### Step 3: Run the RewriteBot Training
In the **second cell**, run:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Run RewriteBot 95M MoE GPU training
!python scripts/train_kaggle_rewritebot_95m.py
```

---

### Step 4: Download Checkpoint
Once the run finishes, Kaggle outputs the trained checkpoint:
`/kaggle/working/ARI/artifacts/checkpoints/exp_kaggle_rewritebot_best.pt`

1. Download the `.pt` file.
2. Place it in `artifacts/checkpoints/` on your PC.
3. The server will automatically load the model for local CPU inference.
