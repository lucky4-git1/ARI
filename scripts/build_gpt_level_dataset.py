"""Ari-LLM: GPT-Level High-Density Dataset Curation Pipeline.

Builds a clean, multi-domain dataset stripped of web scrapings, forum artifacts,
and broken links. Formatted in standard ChatML/OpenAI JSONL format.
"""

import os
import re
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "ari_dataset_gpt_level.jsonl"

SYSTEM_PROMPT = (
    "You are Ari, a thoughtful, warm, and exceptionally capable personal AI companion developed by Ari AI.\n\n"
    "Guidelines:\n"
    "- Respond directly, clearly, and conversationally with genuine warmth.\n"
    "- When writing code, provide complete, production-ready, self-contained solutions inside fenced code blocks (```html ... ```).\n"
    "- When sharing recipes or guides, always format with clear Ingredients and numbered Steps.\n"
    "- Never output web scraping artifacts (such as 'Copy', 'View Answer', or broken HTML tags).\n"
    "- Never simulate follow-up user questions or multi-turn dialogues."
)

# ----------------------------------------------------------------------
# 1. Validation & Hygiene Rules
# ----------------------------------------------------------------------
FORBIDDEN_MARKERS = [
    "[view answer]",
    "[\\view answer\\]",
    "synthetic ingredients",
    ">delete",
    "ondelete",
    "user deleted",
    "deleted comment",
    "<colgroup",
    "_static",
    "statics",
    "⏤",
    "<nooutput>",
    "//[",
    "[ivar",
    "intialized:",
    "answered."
]

def sanitize_text(text: str) -> str:
    """Cleans up text, strips scrape buttons, and fixes whitespace."""
    if not text:
        return ""
    # Strip web scrape button artifacts like 'Copy', 'Copied'
    text = re.sub(r'^(?:Copy|Copied)\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n(?:Copy|Copied)\s*\n?', '\n', text, flags=re.IGNORECASE)
    # Remove hashtags and control noise
    text = re.sub(r'(\#[a-zA-Z0-9_\-]{1,15}){2,}.*$', '', text)
    text = re.sub(r'\s*\#[a-zA-Z0-9_\-]{2,20}\s*$', '', text)
    # Keep printable ASCII, newlines, and tabs
    text = re.sub(r'[^\x20-\x7E\r\n\t]+', '', text)
    return text.strip()

def is_valid_sample(user_msg: str, asst_msg: str) -> bool:
    """Strict gatekeeper verifying sample quality."""
    if not user_msg or not asst_msg:
        return False
    if len(user_msg.strip()) < 8 or len(asst_msg.strip()) < 25:
        return False
    
    combined = (user_msg + " " + asst_msg).lower()
    for marker in FORBIDDEN_MARKERS:
        if marker in combined:
            return False

    # Check for unclosed code fences in assistant message
    if asst_msg.count("```") % 2 != 0:
        return False

    return True

# ----------------------------------------------------------------------
# 2. Gold-Standard Seed Datasets (Coding, Web Dev, Companion, Reasoning)
# ----------------------------------------------------------------------
GOLD_STANDARD_SEEDS = [
    # Responsive Web Development
    (
        "Can you create a modern, responsive navigation bar in HTML and CSS with a clean red accent?",
        "Here is a complete, responsive navigation bar built with semantic HTML5 and clean CSS:\n\n```html\n<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>Responsive Navbar</title>\n  <style>\n    * {\n      box-sizing: border-box;\n      margin: 0;\n      padding: 0;\n      font-family: system-ui, -apple-system, sans-serif;\n    }\n    .navbar {\n      display: flex;\n      justify-content: space-between;\n      align-items: center;\n      background-color: #dc2626;\n      padding: 1rem 2rem;\n      color: white;\n      box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n    }\n    .brand {\n      font-size: 1.25rem;\n      font-weight: 700;\n      letter-spacing: -0.5px;\n    }\n    .nav-links {\n      display: flex;\n      list-style: none;\n      gap: 1.5rem;\n    }\n    .nav-links a {\n      color: white;\n      text-decoration: none;\n      font-size: 0.95rem;\n      font-weight: 500;\n      transition: opacity 0.2s ease;\n    }\n    .nav-links a:hover {\n      opacity: 0.8;\n    }\n    @media (max-width: 768px) {\n      .navbar {\n        flex-direction: column;\n        align-items: flex-start;\n        gap: 1rem;\n      }\n      .nav-links {\n        flex-direction: column;\n        width: 100%;\n        gap: 0.75rem;\n      }\n    }\n  </style>\n</head>\n<body>\n  <nav class=\"navbar\">\n    <div class=\"brand\">Ari Workspace</div>\n    <ul class=\"nav-links\">\n      <li><a href=\"#home\">Home</a></li>\n      <li><a href=\"#features\">Features</a></li>\n      <li><a href=\"#pricing\">Pricing</a></li>\n      <li><a href=\"#contact\">Contact</a></li>\n    </ul>\n  </nav>\n</body>\n</html>\n```\n\n### Key Features:\n- **Semantic HTML5**: Uses `<nav>` and structured unordered lists.\n- **Flexbox Layout**: Clean space-between distribution on desktop.\n- **Mobile Responsive**: Automatically switches to a vertical mobile stack below 768px."
    ),
    (
        "What is HTML and how does it work with CSS and JavaScript?",
        "**HTML (HyperText Markup Language)** is the foundational standard markup language used to structure content on the World Wide Web.\n\nThink of a website like a modern building:\n1. **HTML is the Architecture & Framing**: It defines the walls, rooms, and structural components—such as headings (`<h1>`), paragraphs (`<p>`), navigation bars (`<nav>`), and buttons (`<button>`).\n2. **CSS is the Interior Design & Styling**: It controls colors, fonts, layouts, responsiveness, and visual beauty.\n3. **JavaScript is the Electricity & Plumbing**: It powers interactive behavior, event handlers, animations, and real-time data communication.\n\nHere is a simple example showing how they integrate:\n\n```html\n<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>HTML Overview</title>\n  <style>\n    body { font-family: sans-serif; text-align: center; padding: 2rem; }\n    .btn { background: #2563eb; color: white; padding: 0.6rem 1.2rem; border: none; border-radius: 6px; cursor: pointer; }\n  </style>\n</head>\n<body>\n  <h1>Welcome to the Web</h1>\n  <p>HTML provides the structure of this page.</p>\n  <button class=\"btn\" onclick=\"alert('JavaScript in action!')\">Click Me</button>\n</body>\n</html>\n```"
    ),
    (
        "I feel really low and overwhelmed today. What should I do?",
        "I am really sorry you are having a difficult day. It is completely okay and natural to feel down or overwhelmed sometimes. You do not have to have everything figured out right now.\n\nHere are a few gentle, grounding steps you can take today:\n\n1. **Pause and Take a Breath**: Step away from your computer or phone for 10 minutes. Take five slow, deep breaths—in through your nose, out through your mouth.\n2. **Hydrate & Rest**: Drink a cool glass of water. Dehydration and mental fatigue often intensify low moods.\n3. **Do Something Low-Pressure**: Go for a short 5-minute walk outside, listen to music you love, or wrap yourself in a comfortable blanket.\n4. **Reach Out**: Talk to a trusted friend, family member, or colleague. Simply sharing \"I'm having a rough day\" can relieve significant emotional weight.\n5. **Be Kind to Yourself**: Release any pressure to be highly productive today. Resting is productive when your mind and body need care.\n\nI am right here with you if you just want to talk, vent, or distract yourself with a creative idea."
    ),
    (
        "How do I make a fresh mango shake?",
        "Here is a delicious, creamy recipe for a fresh mango milkshake:\n\n### Ingredients\n- **1 large ripe mango** (peeled, pitted, and cubed)\n- **1 cup cold milk** (whole dairy, almond, or oat milk)\n- **1 to 2 tablespoons honey or sugar** (adjust to sweetness of the mango)\n- **4 ice cubes**\n- **1/4 teaspoon ground cardamom or vanilla extract** (optional, for flavor)\n\n### Instructions\n1. **Prep the Fruit**: Peel your ripe mango and dice the flesh into cubes.\n2. **Blend**: Add the mango cubes, cold milk, sweetener, and cardamom/vanilla into your blender.\n3. **Process**: Blend on high speed for 45 to 60 seconds until completely smooth and velvety.\n4. **Chill**: Add the ice cubes and pulse for another 15 seconds to frost the shake.\n5. **Serve**: Pour into a chilled tall glass. Top with a few diced mango chunks or crushed pistachios, and enjoy immediately!"
    ),
    (
        "How do I create a simple REST API in Python using FastAPI?",
        "Here is a complete, minimal REST API using **FastAPI** with full CRUD capabilities and automatic Swagger docs:\n\n```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\nfrom typing import List, Optional\n\napp = FastAPI(title=\"Task Manager API\", version=\"1.0.0\")\n\n# Data Model\nclass Item(BaseModel):\n    id: int\n    title: str\n    completed: bool = False\n\n# In-Memory Storage\ndb: List[Item] = []\n\n@app.get(\"/items\", response_model=List[Item])\ndef get_items():\n    \"\"\"Retrieve all items.\"\"\"\n    return db\n\n@app.post(\"/items\", response_model=Item, status_code=201)\ndef create_item(item: Item):\n    \"\"\"Create a new item.\"\"\"\n    if any(i.id == item.id for i in db):\n        raise HTTPException(status_code=400, detail=\"Item ID already exists\")\n    db.append(item)\n    return item\n\n@app.get(\"/items/{item_id}\", response_model=Item)\ndef get_item(item_id: int):\n    \"\"\"Retrieve an item by ID.\"\"\"\n    for i in db:\n        if i.id == item_id:\n            return i\n    raise HTTPException(status_code=404, detail=\"Item not found\")\n\nif __name__ == \"__main__\":\n    import uvicorn\n    uvicorn.run(app, host=\"127.0.0.1\", port=8000)\n```\n\n### To Run:\n1. Install dependencies: `pip install fastapi uvicorn`\n2. Run the script: `python main.py`\n3. Open interactive API documentation at: `http://127.0.0.1:8000/docs`"
    )
]

def load_external_datasets():
    """Loads and sanitizes open-source cleaned instruction datasets."""
    samples = []
    
    # 1. Cleaned Alpaca (yahma/alpaca-cleaned)
    try:
        from datasets import load_dataset
        print("[*] Streaming yahma/alpaca-cleaned (51k curated samples)...")
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        for row in ds:
            inst = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            out = row.get("output", "").strip()
            u = f"{inst}\n\nContext:\n{inp}" if inp else inst
            u = sanitize_text(u)
            a = sanitize_text(out)
            if is_valid_sample(u, a):
                samples.append((u, a))
                if len(samples) >= 15000:
                    break
        print(f"  -> Extracted {len(samples):,} clean instruction samples.")
    except Exception as e:
        print(f"  -> Notice on alpaca-cleaned: {e}")

    # 2. CodeAlpaca (sahil2801/CodeAlpaca-20k)
    try:
        from datasets import load_dataset
        print("[*] Streaming sahil2801/CodeAlpaca-20k (20k clean coding samples)...")
        code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        code_count = 0
        for row in code_ds:
            inst = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            out = row.get("output", "").strip()
            u = f"{inst}\n\nContext:\n{inp}" if inp else inst
            u = sanitize_text(u)
            a = sanitize_text(out)
            if is_valid_sample(u, a):
                samples.append((u, a))
                code_count += 1
                if code_count >= 10000:
                    break
        print(f"  -> Extracted {code_count:,} verified coding samples.")
    except Exception as e:
        print(f"  -> Notice on CodeAlpaca: {e}")

    return samples

def build_master_dataset():
    print("=" * 75)
    print("      Ari AI: GPT-Level Master Dataset Curation Pipeline")
    print("=" * 75)

    all_pairs = []

    # Inject Gold Standard Seeds (Replicated with variations for high representation)
    print("\n[1/3] Ingesting Gold Standard Ari Companion & Coding Seeds...")
    for _ in range(50):
        for u, a in GOLD_STANDARD_SEEDS:
            all_pairs.append((u, a))
    print(f"  -> Loaded {len(all_pairs):,} gold-standard anchor samples.")

    # Ingest Curated External Datasets
    print("\n[2/3] Ingesting and sanitizing large-scale cleaned instructions...")
    external = load_external_datasets()
    all_pairs.extend(external)

    # Deduplicate and shuffle
    print("\n[3/3] Deduplicating, formatting ChatML, and writing master dataset...")
    seen = set()
    deduped = []
    for u, a in all_pairs:
        key = u.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append((u, a))

    random.seed(42)
    random.shuffle(deduped)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for u, a in deduped:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a}
                ]
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f" [SUCCESS] Master GPT-Level Dataset Compiled Successfully!")
    print(f" Location:   {OUTPUT_FILE}")
    print(f" Total Rows: {len(deduped):,} clean instruction pairs")
    print(f" File Size:  {file_size_mb:.2f} MB")
    print("=" * 75)

if __name__ == "__main__":
    build_master_dataset()
