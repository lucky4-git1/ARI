"""Ari-LLM RewriteBot 135M: Ultra-Fast DPO Reinforcement Alignment Pipeline.

- Instant stdout flushing (prints live updates immediately).
- Fast pre-tokenization (<3 seconds).
- 400 Steps of DPO with frozen reference model.
- Automatically generates live rewrites and direct Gofile download link.
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import zipfile
import requests
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset

print("=" * 80, flush=True)
print("     Ari-LLM RewriteBot 135M: Ultra-Fast DPO Alignment", flush=True)
print("=" * 80, flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

CANDIDATES = [
    "/kaggle/working/ari_rewritebot_135m",
    "artifacts/checkpoints/ari_rewritebot_135m"
]
MODEL_DIR = None
for c in CANDIDATES:
    if os.path.exists(c) and os.path.exists(os.path.join(c, "model.safetensors")):
        MODEL_DIR = c
        break

if MODEL_DIR is None:
    raise FileNotFoundError("Could not find SFT model checkpoint. Ensure ari_rewritebot_135m exists.")

print(f"Loading Base Model from: {MODEL_DIR}", flush=True)
OUTPUT_DIR = "/kaggle/working/ari_rewritebot_135m_dpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load Tokenizer & Policy Model & Frozen Reference Model
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n[1/4] Loading Policy Model & Frozen Reference Model...", flush=True)
policy_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, device_map="auto")
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, device_map="auto")

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

print("  -> Both Models loaded in GPU memory successfully!", flush=True)

# ----------------------------------------------------------------------
# 2. Curated DPO Preference Pairs (High-Impact 2,500 Pairs)
# ----------------------------------------------------------------------
print("\n[2/4] Assembling Curated DPO Preference Pairs...", flush=True)

preference_pairs = []

# (A) PAWS Pairs: Chosen = clause inversion; Rejected = verbatim copy
try:
    print("  -> Loading PAWS dataset...", flush=True)
    paws = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
    count = 0
    for row in paws:
        if row["label"] == 1 and count < 2000:
            s1, s2 = row["sentence1"].strip(), row["sentence2"].strip()
            if s1 and s2 and s1 != s2:
                preference_pairs.append({
                    "prompt": f"[Standard Paraphrase] {s1}",
                    "chosen": s2,
                    "rejected": s1
                })
                count += 1
    print(f"     Loaded {len(preference_pairs):,} PAWS DPO pairs", flush=True)
except Exception as e:
    print(f"     Notice on PAWS: {e}", flush=True)

# (B) Master QuillBot Mode Preferences (Reward Stylistic Brilliance, Punish Mediocrity)
QUILLBOT_DPO_TEMPLATES = [
    {
        "prompt": "[Formal Rewrite] Can you send me that document as soon as possible?",
        "chosen": "Could you kindly provide the requested document at your earliest convenience?",
        "rejected": "Can you send me that document as soon as possible please."
    },
    {
        "prompt": "[Academic Rewrite] This shows that the theory is true in almost every case.",
        "chosen": "These empirical findings substantiate the hypothesis across a statistically significant spectrum of test observations.",
        "rejected": "This shows that the theory is true in almost every case and is proven."
    },
    {
        "prompt": "[Simple Rewrite] Utilizing advanced computational methodologies facilitates enhanced operational efficiencies.",
        "chosen": "Using modern computers makes daily work faster and easier.",
        "rejected": "Utilizing advanced computing facilitates operational efficiency."
    },
    {
        "prompt": "[Creative Rewrite] The sun went down and the stars came out.",
        "chosen": "Twilight melted into twilight shadows as diamonds pierced the velvet expanse of night.",
        "rejected": "The sun went down and then the stars appeared in the night sky."
    },
    {
        "prompt": "[Anti-AI Humanize] In conclusion, it is important to remember that modern life presents myriad challenges.",
        "chosen": "Look, at the end of the day, navigating daily life gets messy—and that is totally normal.",
        "rejected": "In conclusion, it is crucial to recognize that contemporary existence embodies numerous difficulties."
    },
    {
        "prompt": "[Active Voice] The software bug was discovered by the quality assurance team.",
        "chosen": "The quality assurance team discovered the software bug.",
        "rejected": "The software bug got discovered by the quality team."
    },
    {
        "prompt": "[Grammar Polish] She don't know the answer to the question.",
        "chosen": "She does not know the answer to the question.",
        "rejected": "She don't know the question's answer."
    },
    {
        "prompt": "[Shorten Mode] Due to the unfortunate circumstance that weather conditions deteriorated, the scheduled flight was canceled.",
        "chosen": "The flight was canceled due to bad weather.",
        "rejected": "Because weather conditions deteriorated, the scheduled flight was canceled."
    }
]

for _ in range(100):
    preference_pairs.extend(QUILLBOT_DPO_TEMPLATES)

random.seed(42)
random.shuffle(preference_pairs)
print(f"  -> Total DPO Dataset: {len(preference_pairs):,} pairs", flush=True)

# ----------------------------------------------------------------------
# 3. Fast In-Memory Tokenization (< 2 seconds)
# ----------------------------------------------------------------------
def encode_pair(prompt, response, max_len=160):
    p_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    r_text = f"{response}<|im_end|>"
    
    p_ids = tokenizer.encode(p_text, add_special_tokens=False)
    r_ids = tokenizer.encode(r_text, add_special_tokens=False)
    
    max_p_len = max_len - min(len(r_ids), 40) - 1
    if len(p_ids) > max_p_len:
        p_ids = p_ids[:max_p_len]
        
    all_ids = p_ids + r_ids
    if len(all_ids) > max_len:
        all_ids = all_ids[:max_len]
        
    prompt_len = len(p_ids)
    pad_len = max_len - len(all_ids)
    
    input_ids = all_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(all_ids) + [0] * pad_len
    
    resp_len = min(len(r_ids), max_len - prompt_len)
    labels = [-100] * prompt_len + r_ids[:resp_len] + [-100] * pad_len
    
    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

class FastDPODataset(Dataset):
    def __init__(self, pairs, max_len=160):
        self.samples = []
        t0 = time.time()
        for p in pairs:
            c_x, c_m, c_y = encode_pair(p["prompt"], p["chosen"], max_len)
            r_x, r_m, r_y = encode_pair(p["prompt"], p["rejected"], max_len)
            if (c_y != -100).sum() > 1 and (r_y != -100).sum() > 1:
                self.samples.append((c_x, c_m, c_y, r_x, r_m, r_y))
        print(f"  -> Pre-tokenized {len(self.samples):,} pairs in {time.time()-t0:.2f}s!", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dpo_dataset = FastDPODataset(preference_pairs, max_len=160)
dpo_loader = DataLoader(dpo_dataset, batch_size=4, shuffle=True, drop_last=True)

def compute_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = (shift_labels != -100)
    
    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1),
        dim=2,
        index=shift_labels.unsqueeze(2).clamp(min=0)
    ).squeeze(2)
    
    return (per_token_logps * loss_mask).sum(-1)

# ----------------------------------------------------------------------
# 4. DPO Optimization Loop (400 Steps ~ 3 Minutes)
# ----------------------------------------------------------------------
BETA = 0.1
LR = 1.5e-5
TOTAL_STEPS = 400
GRAD_ACCUM = 2

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=30,
    num_training_steps=TOTAL_STEPS
)

print(f"\n[3/4] Launching DPO Post-Training ({TOTAL_STEPS} steps, Beta={BETA})...\n", flush=True)

policy_model.train()
step = 0
running_loss = 0.0
running_margin = 0.0
start_time = time.time()
data_iter = iter(dpo_loader)

while step < TOTAL_STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dpo_loader)
        batch = next(data_iter)

    c_x, c_m, c_y, r_x, r_m, r_y = [t.to(DEVICE) for t in batch]

    pi_c_logits = policy_model(c_x, attention_mask=c_m).logits
    pi_r_logits = policy_model(r_x, attention_mask=r_m).logits
    pi_c_logps = compute_logps(pi_c_logits, c_y)
    pi_r_logps = compute_logps(pi_r_logits, r_y)

    with torch.no_grad():
        ref_c_logits = ref_model(c_x, attention_mask=c_m).logits
        ref_r_logits = ref_model(r_x, attention_mask=r_m).logits
        ref_c_logps = compute_logps(ref_c_logits, c_y)
        ref_r_logps = compute_logps(ref_r_logits, r_y)

    pi_ratios = pi_c_logps - pi_r_logps
    ref_ratios = ref_c_logps - ref_r_logps
    logits_diff = BETA * (pi_ratios - ref_ratios)
    loss = -F.logsigmoid(logits_diff).mean()

    with torch.no_grad():
        chosen_rewards = BETA * (pi_c_logps - ref_c_logps)
        rejected_rewards = BETA * (pi_r_logps - ref_r_logps)
        margin = (chosen_rewards - rejected_rewards).mean().item()

    loss_scaled = loss / GRAD_ACCUM
    loss_scaled.backward()

    running_loss += loss.item()
    running_margin += margin

    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    step += 1

    if step % 25 == 0 or step == TOTAL_STEPS:
        avg_loss = running_loss / (25 if step % 25 == 0 else (step % 25))
        avg_margin = running_margin / (25 if step % 25 == 0 else (step % 25))
        elapsed = time.time() - start_time
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Step {step:3d}/{TOTAL_STEPS} | DPO Loss: {avg_loss:.4f} | Reward Margin: +{avg_margin:.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed/60:.1f}m", flush=True)
        running_loss = 0.0
        running_margin = 0.0

print("\nDPO Complete! Saving Aligned Checkpoint...", flush=True)
policy_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved aligned checkpoint to: {OUTPUT_DIR}", flush=True)

# ----------------------------------------------------------------------
# 5. Live Evaluation
# ----------------------------------------------------------------------
print("\n" + "=" * 80, flush=True)
print("     [4/4] LIVE EVALUATION: POST-DPO ALIGNED REWRITER", flush=True)
print("=" * 80, flush=True)
policy_model.eval()

eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

def rewrite_dpo(prompt, max_new_tokens=64):
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = policy_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0].strip()
    return text

eval_tests = [
    "[Standard Paraphrase] The flight was canceled because of severe weather conditions.",
    "[Formal Rewrite] Can you send me that project update as soon as possible?",
    "[Academic Rewrite] This shows that the theory is true in almost every case.",
    "[Simple Rewrite] Utilizing advanced computational methodologies facilitates enhanced operational efficiencies.",
    "[Creative Rewrite] The sun went down and the stars came out.",
    "[Anti-AI Humanize] In conclusion, it is important to remember that modern life presents myriad challenges.",
    "[Active Voice] The software bug was discovered by the quality assurance team.",
    "[Grammar Polish] She don't know the answer to the question."
]

for t in eval_tests:
    out = rewrite_dpo(t)
    print(f"\nPROMPT   : {t}", flush=True)
    print(f"REWRITE  : {out}", flush=True)

# ----------------------------------------------------------------------
# 6. Package & Upload to Gofile
# ----------------------------------------------------------------------
print("\n" + "=" * 80, flush=True)
print("     PACKAGING & UPLOADING ALIGNED ARTIFACTS", flush=True)
print("=" * 80, flush=True)

zip_file = "/kaggle/working/ari_rewritebot_135m_dpo.zip"
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, "/kaggle/working")
            zf.write(full_path, arcname=arcname)

zip_size_mb = os.path.getsize(zip_file) / (1024 * 1024)
print(f"Created Archive: {zip_file} ({zip_size_mb:.2f} MB)", flush=True)

try:
    print("Uploading to Gofile...", flush=True)
    server_res = requests.get("https://api.gofile.io/servers").json()
    server_name = server_res["data"]["servers"][0]["name"]
    upload_url = f"https://{server_name}.gofile.io/contents/uploadfile"
    with open(zip_file, "rb") as f:
        upload_res = requests.post(upload_url, files={"file": f}).json()
    download_link = upload_res["data"]["downloadPage"]
    print(f"\n=================================================================", flush=True)
    print(f" GOFILE DIRECT DOWNLOAD LINK: {download_link}", flush=True)
    print(f"=================================================================", flush=True)
except Exception as e:
    print(f"Notice on Gofile upload: {e}", flush=True)

print("Ari-LLM RewriteBot DPO Post-Training Finished Successfully!", flush=True)
