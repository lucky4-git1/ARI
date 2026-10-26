"""
Ari Chat — Comprehensive General Domains Evaluation Suite
Evaluates non-coding capability across:
1. Creative Writing & Storytelling
2. Business Strategy & Professional Communication
3. Philosophy & Deep Intellectual Inquiries
4. Empathy & Emotional Intelligence
5. Complex Everyday Decision Analysis
6. Health, Science & Nutrition
7. History & Geopolitics
8. Wit, Humor & Conversational Banter
"""

import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

GENERAL_PROMPTS = [
    {
        "domain": "Creative Writing",
        "title": "Atmospheric Narrative Scene",
        "prompt": "Write a short, evocative 2-paragraph narrative scene describing an old watchmaker in Prague closing his shop on a rainy autumn evening as the streetlamps flicker on. Focus on tactile textures and quiet atmosphere."
    },
    {
        "domain": "Business & Strategy",
        "title": "Executive Pitch Email",
        "prompt": "Draft a compelling, professional cold email to a VP of Engineering proposing an enterprise pilot for a privacy-first, on-premise AI assistant. Keep it concise, high-value, and action-oriented."
    },
    {
        "domain": "Philosophy & Thought",
        "title": "Ethics & Meaning of Time",
        "prompt": "Explain the philosophical difference between 'Chronos' (quantitative time) and 'Kairos' (qualitative opportune time) from ancient Greek thought, and why this distinction matters in modern life."
    },
    {
        "domain": "Empathy & Support",
        "title": "Career Burnout & Anxiety",
        "prompt": "I've been feeling completely overwhelmed and burnt out with my career lately. Every morning feels like an uphill battle and I'm losing confidence. What would you say to help me regain perspective?"
    },
    {
        "domain": "Everyday Decision Making",
        "title": "Renting vs Buying Decision Matrix",
        "prompt": "Create a clear, structured decision framework comparing renting vs buying a home for a 30-year-old professional. Include financial trade-offs, lifestyle flexibility, and key trigger questions."
    },
    {
        "domain": "Health & Science",
        "title": "Circadian Rhythm & Sleep Protocol",
        "prompt": "Provide a science-backed, practical daily protocol to optimize circadian rhythm and deep sleep architecture. Use clear time blocks (Morning, Afternoon, Evening)."
    },
    {
        "domain": "History & Culture",
        "title": "Fall of the Roman Republic",
        "prompt": "Analyze the three primary systemic factors that led to the collapse of the Roman Republic and the rise of the Roman Empire under Augustus."
    },
    {
        "domain": "Wit & Banter",
        "title": "Philosophical Banter",
        "prompt": "If a coffee cup, an alarm clock, and a smartphone had a support group meeting on Monday morning, what would their conversation sound like?"
    }
]

def run_suite():
    print("=" * 80)
    print("ARI CHAT — GENERAL NON-CODING DOMAINS COMPREHENSIVE BENCHMARK")
    print("=" * 80)

    # 1. Guest Login
    req = urllib.request.Request(f"{BASE_URL}/api/auth/guest", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        token = json.loads(resp.read().decode())["access_token"]
    print("[Auth] Guest token acquired successfully.\n")

    results = []

    for idx, item in enumerate(GENERAL_PROMPTS, 1):
        domain = item["domain"]
        title = item["title"]
        prompt = item["prompt"]

        print(f"\n[{idx}/8] Testing Domain: {domain} ('{title}')")
        print(f"Prompt: \"{prompt}\"")
        print("-" * 70)

        # Fresh conversation
        creq = urllib.request.Request(
            f"{BASE_URL}/api/chat/conversations",
            data=json.dumps({"title": f"Eval-{domain}"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(creq) as resp:
            conv_id = json.loads(resp.read().decode())["id"]

        # Send message
        mreq = urllib.request.Request(
            f"{BASE_URL}/api/chat/conversations/{conv_id}/messages",
            data=json.dumps({
                "content": prompt,
                "web_search": False,
                "reasoning_mode": False,
                "model": "ari-assistant-3b"
            }).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        )

        t0 = time.time()
        tokens = []
        final_clean = ""

        with urllib.request.urlopen(mreq, timeout=120) as stream_resp:
            for line in stream_resp:
                l = line.decode().strip()
                if l.startswith("data: "):
                    d = json.loads(l[6:])
                    if d.get("type") == "token":
                        tok = d.get("token", "")
                        tokens.append(tok)
                        print(tok, end="", flush=True)
                    elif d.get("type") == "done":
                        final_clean = d.get("clean_content", "")

        elapsed = time.time() - t0
        output = final_clean or "".join(tokens)
        tps = len(tokens) / elapsed if elapsed > 0 else 0

        print(f"\n\n[Stats] {len(tokens)} tokens in {elapsed:.1f}s ({tps:.1f} t/s)")
        print("=" * 70)

        results.append({
            "domain": domain,
            "title": title,
            "prompt": prompt,
            "tokens": len(tokens),
            "elapsed": round(elapsed, 1),
            "tps": round(tps, 1),
            "output": output
        })

    with open("general_domains_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nALL 8 GENERAL DOMAIN EVALUATIONS COMPLETED!")

if __name__ == "__main__":
    run_suite()
