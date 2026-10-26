import subprocess
import time
import sys
import os

# Ensure clean UTF-8 output everywhere
sys.stdout.reconfigure(encoding='utf-8')

LLAMA_CLI = r"E:\Ari-llm-source\ari-chat-app\bin\llama-cli.exe"
MODEL_PATH = r"artifacts/checkpoints/ari_paraphraser_1.5b.gguf"

test_passage_1 = (
    "Modern clinical interventions designed for fertility preservation among adolescent and young adult "
    "cancer survivors have progressively incorporated multifaceted cryopreservation protocols, encompassing "
    "mature oocyte vitrification, cryobanking of ovarian cortical strips, and emergency semen cryopreservation "
    "prior to gonadotoxic chemotherapeutic regimens. Despite demonstrably heightened efficacy in longitudinal "
    "gamete survival indices, widespread global implementation remains severely constrained by socio-economic "
    "disparities, ambiguous clinical guidelines across community oncology networks, and inadequate insurance "
    "mandates covering reproductive endocrinology consultations. Furthermore, empirical findings highlight "
    "profound psychosocial morbidity among patients whose post-treatment reproductive potential was inadequately "
    "addressed during immediate post-diagnosis therapeutic planning."
)

test_passage_2 = (
    "Artificial intelligence models operating at the edge require sophisticated quantization methodologies "
    "to balance parameter efficiency with inference accuracy. While standard 4-bit integer quantization preserves "
    "weight representation, activation outliers frequently degrade perplexity across critical attention heads. "
    "Recent advancements in activation-aware weight quantization (AWQ) mitigate this limitation by selectively "
    "protecting salient weight channels based on their observed magnitude distributions during calibration."
)

tests = [
    ("Academic", test_passage_1),
    ("Standard", test_passage_1),
    ("Formal", test_passage_2),
]

md_lines = []
md_lines.append("# Ari Paraphraser 1.5B (v2 Editorial Engine) - Live Benchmark Report\n")

for mode, text in tests:
    print(f"\n[*] Running {mode} mode...", flush=True)
    prompt = (
        f"<|im_start|>system\n"
        f"You are Ari RewriteBot, an expert AI paraphrasing and writing assistant. You rewrite, polish, and rephrase text with exceptional fluency, precision, and stylistic versatility.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Mode: {mode}\nText: {text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "256",
        "-c", "2048",
        "--temp", "0.3",
        "--repeat-penalty", "1.15",
        "-r", "<|im_end|>",
        "-r", "<|endoftext|>",
        "-r", "<|im_start|>",
        "--single-turn",
        "--simple-io",
        "-t", "6"
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    elapsed = time.time() - t0
    
    raw = res.stdout
    # Isolate assistant output
    if "<|im_start|>assistant" in raw:
        gen = raw.split("<|im_start|>assistant")[-1]
    elif "> Mode:" in raw:
        gen = raw.split("> Mode:")[1]
        if "\n" in gen:
            gen = gen.split("\n", 1)[1]
    else:
        parts = raw.split("Exiting...")
        gen = parts[0]
        if "[ Prompt:" in gen:
            gen = gen.split("[ Prompt:")[0]
            
    for stop in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "[ Prompt:", "Exiting..."]:
        if stop in gen:
            gen = gen.split(stop)[0]
            
    lines = [
        l.strip() for l in gen.splitlines()
        if l.strip() and not l.startswith("> ")
        and not "available commands:" in l
        and not "Loading model" in l
        and not "build " in l
        and not "modalities" in l
        and not "ftype " in l
        and not "model " in l
    ]
    clean_output = " ".join(lines).strip()
    
    in_words = len(text.split())
    out_words = len(clean_output.split())
    ratio = out_words / in_words if in_words else 0
    
    md_lines.append(f"## Mode: {mode}")
    md_lines.append(f"- **Input Word Count:** {in_words} words")
    md_lines.append(f"- **Output Word Count:** {out_words} words (Length Ratio: **{ratio:.2f}**)")
    md_lines.append(f"- **Latency:** {elapsed:.1f}s\n")
    md_lines.append("### Original Passage")
    md_lines.append(f"> {text}\n")
    md_lines.append("### Paraphrased Output")
    md_lines.append(f"{clean_output}\n")
    md_lines.append("---\n")
    
    print(f"[OK] {mode} completed in {elapsed:.1f}s: {out_words} words (Ratio: {ratio:.2f})", flush=True)

with open("benchmark_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("\n[SUCCESS] Full benchmark report saved to benchmark_report.md!", flush=True)
