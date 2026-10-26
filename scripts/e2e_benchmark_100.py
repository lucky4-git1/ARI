"""
Ari Chat — End-to-End Comprehensive 100-Question Benchmark & Visual Quality Evaluator
Tests all API endpoints, auth flows, model streaming, and evaluates 100 diverse prompts
across 10 core domains to measure presentation fidelity, formatting, and coherence.
"""

import urllib.request
import urllib.error
import json
import time
import os
import re
import sys
from typing import Dict, Any, List, Tuple

BASE_URL = "http://127.0.0.1:8000"

QUESTIONS = [
    # 1. Code Generation & Software Engineering
    {"id": 1, "cat": "Coding", "q": "Write a Python function for binary search with edge case handling."},
    {"id": 2, "cat": "Coding", "q": "Implement an LRU Cache in Python using OrderedDict with get and put methods."},
    {"id": 3, "cat": "Coding", "q": "Write a clean JavaScript function to deep clone a nested object without using JSON.parse."},
    {"id": 4, "cat": "Coding", "q": "Write a SQL query to find the 2nd highest salary from an Employee table with id and salary columns."},
    {"id": 5, "cat": "Coding", "q": "Create a responsive CSS Flexbox navbar with mobile hamburger toggle structure."},
    {"id": 6, "cat": "Coding", "q": "Write a Python script to download files with a progress bar using requests."},
    {"id": 7, "cat": "Coding", "q": "Implement Fibonacci sequence using dynamic programming memoization in Python."},
    {"id": 8, "cat": "Coding", "q": "Write a production Dockerfile for a FastAPI Python application with multi-stage build."},
    {"id": 9, "cat": "Coding", "q": "Write a debounce utility function in JavaScript with TypeScript typing."},
    {"id": 10, "cat": "Coding", "q": "Write a Python generator function to stream large log files line by line without memory overflow."},

    # 2. Mathematics & Formal Reasoning (LaTeX)
    {"id": 11, "cat": "Math", "q": "State and explain Bayes' Theorem with formula in standard LaTeX."},
    {"id": 12, "cat": "Math", "q": "Derive the quadratic formula step by step with LaTeX equations."},
    {"id": 13, "cat": "Math", "q": "Explain the formula for normal distribution probability density function in LaTeX."},
    {"id": 14, "cat": "Math", "q": "Calculate the derivative of f(x) = x^3 * sin(x) using the product rule."},
    {"id": 15, "cat": "Math", "q": "What is Euler's formula e^(i*pi) + 1 = 0 and why is it considered profound?"},
    {"id": 16, "cat": "Math", "q": "Explain Eigenvalues and Eigenvectors with mathematical definitions in LaTeX."},
    {"id": 17, "cat": "Math", "q": "What is the Pythagorean theorem and how is it extended to 3D Euclidean distance?"},
    {"id": 18, "cat": "Math", "q": "What is the Taylor series expansion of e^x around x = 0?"},
    {"id": 19, "cat": "Math", "q": "Calculate the compound interest formula and define each variable."},
    {"id": 20, "cat": "Math", "q": "What is Shannon Entropy in information theory? Provide the formula in LaTeX."},

    # 3. Comparative Analysis & Markdown Tables
    {"id": 21, "cat": "Tables", "q": "Compare PostgreSQL vs MongoDB in a comparison table with pros, cons, and best use cases."},
    {"id": 22, "cat": "Tables", "q": "Compare REST APIs vs GraphQL vs gRPC in a structured table."},
    {"id": 23, "cat": "Tables", "q": "Compare React vs Vue vs Svelte with a breakdown table."},
    {"id": 24, "cat": "Tables", "q": "Compare Docker vs Virtual Machines with a clear comparison table."},
    {"id": 25, "cat": "Tables", "q": "Compare TCP vs UDP in a side-by-side comparison table."},
    {"id": 26, "cat": "Tables", "q": "Compare Microservices vs Monolithic architecture with a trade-off table."},
    {"id": 27, "cat": "Tables", "q": "Compare SSD vs HDD in terms of speed, durability, cost, and architecture."},
    {"id": 28, "cat": "Tables", "q": "Compare synchronous vs asynchronous programming in Python with a table."},
    {"id": 29, "cat": "Tables", "q": "Compare Git merge vs Git rebase in a comparison table."},
    {"id": 30, "cat": "Tables", "q": "Compare symmetric vs asymmetric encryption in a structured table."},

    # 4. Step-by-Step Practical Guides & Recipes
    {"id": 31, "cat": "Guides", "q": "Give me a step-by-step authentic Italian carbonara recipe with ingredients and instructions."},
    {"id": 32, "cat": "Guides", "q": "Step-by-step guide to troubleshooting high CPU usage on a Linux server."},
    {"id": 33, "cat": "Guides", "q": "How to set up an SSL certificate with Let's Encrypt and Nginx step by step."},
    {"id": 34, "cat": "Guides", "q": "Step-by-step guide to making homemade pizza dough from scratch."},
    {"id": 35, "cat": "Guides", "q": "How to optimize a slow SQL database query step by step."},
    {"id": 36, "cat": "Guides", "q": "Step-by-step guide to preparing for a senior technical coding interview."},
    {"id": 37, "cat": "Guides", "q": "How to brew cold brew coffee at home step by step."},
    {"id": 38, "cat": "Guides", "q": "Step-by-step checklist for launching a production web app securely."},
    {"id": 39, "cat": "Guides", "q": "How to change a flat tire safely on the roadside step by step."},
    {"id": 40, "cat": "Guides", "q": "Step-by-step guide to setting up CI/CD pipeline with GitHub Actions."},

    # 5. Creative Writing & Tone Adaptation
    {"id": 41, "cat": "Creative", "q": "Write a compelling, concise cold outreach email to a venture capitalist for an AI startup."},
    {"id": 42, "cat": "Creative", "q": "Write a short sci-fi flash fiction story about the first self-aware quantum computer."},
    {"id": 43, "cat": "Creative", "q": "Draft a polite and professional resignation letter with two weeks' notice."},
    {"id": 44, "cat": "Creative", "q": "Write a witty, engaging tweet thread hook about the future of local AI on consumer laptops."},
    {"id": 45, "cat": "Creative", "q": "Write an evocative haiku about autumn leaves and rain."},
    {"id": 46, "cat": "Creative", "q": "Draft an executive memo explaining why our engineering team should adopt automated testing."},
    {"id": 47, "cat": "Creative", "q": "Write an encouraging motivational pep talk for a software engineer facing imposter syndrome."},
    {"id": 48, "cat": "Creative", "q": "Write a product launch announcement for a new minimalist mechanical keyboard."},
    {"id": 49, "cat": "Creative", "q": "Compose a short modern fable teaching the importance of persistence."},
    {"id": 50, "cat": "Creative", "q": "Rewrite this sentence in 3 distinct tones (Formal, Playful, Assertive): 'We cannot fulfill your request right now.'"},

    # 6. Logic Puzzles & Critical Thinking
    {"id": 51, "cat": "Logic", "q": "If you have 3 apples and you take away 2, how many apples do you have? Explain clearly."},
    {"id": 52, "cat": "Logic", "q": "Explain the Monty Hall problem and why switching doors increases probability of winning."},
    {"id": 53, "cat": "Logic", "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"},
    {"id": 54, "cat": "Logic", "q": "You have a 3-liter jug and a 5-liter jug and an unlimited water source. How do you measure exactly 4 liters?"},
    {"id": 55, "cat": "Logic", "q": "There are three boxes: apples, oranges, and both. All are mislabeled. You pick one fruit from one box. How do you correctly label all three?"},
    {"id": 56, "cat": "Logic", "q": "A farmer needs to cross a river with a wolf, a goat, and a cabbage. How does he cross safely without any item being eaten?"},
    {"id": 57, "cat": "Logic", "q": "If five machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?"},
    {"id": 58, "cat": "Logic", "q": "You have 8 identical balls, 1 is slightly heavier. What is the minimum number of balance scale weighings to find it?"},
    {"id": 59, "cat": "Logic", "q": "Explain the Ship of Theseus paradox and its philosophical implications."},
    {"id": 60, "cat": "Logic", "q": "If all roses are flowers and some flowers fade quickly, can we logically deduce that some roses fade quickly?"},

    # 7. Scientific & Historical Inquiries
    {"id": 61, "cat": "Science", "q": "Explain quantum superposition simply using Schrödinger's cat thought experiment."},
    {"id": 62, "cat": "Science", "q": "What caused the French Revolution in 1789? Highlight the 3 primary drivers."},
    {"id": 63, "cat": "Science", "q": "How does CRISPR gene editing technology work at a molecular level?"},
    {"id": 64, "cat": "Science", "q": "Explain Einstein's General Relativity and gravitational time dilation."},
    {"id": 65, "cat": "Science", "q": "How do vaccines train the human immune system to fight viruses?"},
    {"id": 66, "cat": "Science", "q": "What was the significance of the Rosetta Stone in deciphering ancient hieroglyphs?"},
    {"id": 67, "cat": "Science", "q": "Explain the Doppler Effect with everyday sound and astronomical light examples."},
    {"id": 68, "cat": "Science", "q": "How do solar panels convert sunlight into electricity via the photovoltaic effect?"},
    {"id": 69, "cat": "Science", "q": "Why is the sky blue during the day and red during sunset?"},
    {"id": 70, "cat": "Science", "q": "What are black holes and what happens at the event horizon?"},

    # 8. Executive Summaries & Bullet Syntheses
    {"id": 71, "cat": "Synthesis", "q": "Summarize the key benefits of edge computing versus cloud computing in 5 crisp bullet points."},
    {"id": 72, "cat": "Synthesis", "q": "What are the 4 fundamental pillars of Object-Oriented Programming (OOP)? Define each concisely."},
    {"id": 73, "cat": "Synthesis", "q": "Synthesize the core principles of Agile software development in 4 bullet points."},
    {"id": 74, "cat": "Synthesis", "q": "What are the key architectural differences between HTTP/1.1, HTTP/2, and HTTP/3?"},
    {"id": 75, "cat": "Synthesis", "q": "Summarize the ACID properties of database transactions in 4 clear bullets."},
    {"id": 76, "cat": "Synthesis", "q": "What are the primary OWASP Top 10 web application security risks?"},
    {"id": 77, "cat": "Synthesis", "q": "Summarize Moore's Law and explain why physical scaling limits are challenging it today."},
    {"id": 78, "cat": "Synthesis", "q": "What are the 3 main paradigms of machine learning: supervised, unsupervised, and reinforcement?"},
    {"id": 79, "cat": "Synthesis", "q": "Summarize the principles of the Twelve-Factor App methodology for cloud-native software."},
    {"id": 80, "cat": "Synthesis", "q": "Summarize three essential microservices design patterns: API Gateway, Circuit Breaker, and CQRS."},

    # 9. Conversational Nuance & Small Talk
    {"id": 81, "cat": "Conversation", "q": "Hello Ari! Who are you and what makes you unique?"},
    {"id": 82, "cat": "Conversation", "q": "What should I make for a quick healthy dinner under 20 minutes?"},
    {"id": 83, "cat": "Conversation", "q": "I'm feeling overwhelmed with my programming project today. Any practical advice?"},
    {"id": 84, "cat": "Conversation", "q": "What are some good book recommendations for someone interested in computer architecture and systems?"},
    {"id": 85, "cat": "Conversation", "q": "Can you help me brainstorm 5 creative names for a new local privacy-first AI assistant?"},
    {"id": 86, "cat": "Conversation", "q": "How can I improve my focus when coding for long stretches?"},
    {"id": 87, "cat": "Conversation", "q": "What is the fundamental difference between computer science and software engineering?"},
    {"id": 88, "cat": "Conversation", "q": "How does human biological memory consolidation compare to AI contextual memory?"},
    {"id": 89, "cat": "Conversation", "q": "What are three key habits of highly effective senior software engineers?"},
    {"id": 90, "cat": "Conversation", "q": "Give me a thought-provoking philosophical question to reflect on today."},

    # 10. Tough Edge Cases & Stress Tests
    {"id": 91, "cat": "EdgeCases", "q": "Write a markdown table comparing JSON, YAML, and TOML with features and syntax style."},
    {"id": 92, "cat": "EdgeCases", "q": "Output the numbers 1 to 10 in a numbered list with one short sentence for each number."},
    {"id": 93, "cat": "EdgeCases", "q": "What is the formula for quadratic roots? State it in LaTeX with an example."},
    {"id": 94, "cat": "EdgeCases", "q": "Write a Python snippet that prints 'Hello World' and explain it in 2 sentences."},
    {"id": 95, "cat": "EdgeCases", "q": "Can you repeat the word 'apple' exactly 3 times separated by commas?"},
    {"id": 96, "cat": "EdgeCases", "q": "What is the difference between let, const, and var in JavaScript?"},
    {"id": 97, "cat": "EdgeCases", "q": "Explain recursion with a short factorial function example in Python."},
    {"id": 98, "cat": "EdgeCases", "q": "How do cookies differ from localStorage and sessionStorage in web browsers?"},
    {"id": 99, "cat": "EdgeCases", "q": "Provide a comparison table of Python lists vs tuples vs sets vs dictionaries."},
    {"id": 100, "cat": "EdgeCases", "q": "Write a clean bash script to monitor disk usage and print an alert if it exceeds 90%."}
]

def make_request(path: str, method: str = "GET", data: Dict = None, token: str = None) -> Tuple[int, Any]:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(content)
            except Exception:
                return resp.status, content
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            return e.code, json.loads(err_content)
        except Exception:
            return e.code, err_content
    except Exception as e:
        return 0, str(e)

def evaluate_formatting(text: str) -> Dict[str, Any]:
    """Evaluates the visual and presentation quality of an output."""
    has_code_block = "```" in text
    unclosed_code = (text.count("```") % 2 != 0)
    # Strip out fenced code blocks before checking for raw HTML leaks in conversational prose
    prose_only = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    has_html_leak = bool(re.search(r'<(?:div|span|h[1-6]|ol|ul|li|p|table|colgroup|tr|td|th)\b', prose_only, re.IGNORECASE))
    has_markdown_table = bool(re.search(r'\|.+\|\s*\n\|[-:\|\s]+\|\s*\n\|.+\|', text))
    has_markdown_headers = bool(re.search(r'^#{1,4}\s+\S', text, re.MULTILINE))
    has_markdown_lists = bool(re.search(r'^\s*(?:-|\*|\d+\.)\s+\S', text, re.MULTILINE))
    has_latex = bool(re.search(r'\$.+?\$|\\\[.+?\\\]', text))
    has_repetition = False

    # Check for short looping patterns
    words = text.split()
    if len(words) > 20:
        for n in range(3, 10):
            for i in range(len(words) - 2 * n):
                chunk1 = words[i:i+n]
                chunk2 = words[i+n:i+2*n]
                chunk3 = words[i+2*n:i+3*n] if i+3*n <= len(words) else []
                if chunk1 == chunk2 == chunk3:
                    has_repetition = True
                    break

    return {
        "len": len(text),
        "words": len(words),
        "has_code": has_code_block,
        "unclosed_code": unclosed_code,
        "has_html_leak": has_html_leak,
        "has_table": has_markdown_table,
        "has_headers": has_markdown_headers,
        "has_lists": has_markdown_lists,
        "has_latex": has_latex,
        "has_repetition": has_repetition,
    }

def main():
    print("=" * 80)
    print("ARI CHAT — 100 QUESTION E2E BENCHMARK & VISUAL QUALITY EVALUATOR")
    print("=" * 80)

    # 1. Check Health
    status, health = make_request("/api/health")
    print(f"[Health Check] Status: {status}, Engine: {health.get('engine', {}).get('online')}")
    if status != 200:
        print("ERROR: App is not healthy. Exiting.")
        sys.exit(1)

    # 2. Test Guest Auth
    status, auth_data = make_request("/api/auth/guest", method="POST", data={})
    print(f"[Guest Auth Check] Status: {status}, User: {auth_data.get('user', {}).get('username')}")
    if status != 200 or not auth_data.get("access_token"):
        print("ERROR: Guest login failed.")
        sys.exit(1)
    token = auth_data["access_token"]

    results = []
    print("\nStarting execution of benchmark suite across 10 domains (clean conversation per question)...", flush=True)
    print("-" * 80, flush=True)

    start_all = time.time()

    for i, item in enumerate(QUESTIONS):
        q_id = item["id"]
        cat = item["cat"]
        prompt = item["q"]

        # Clean conversation per question to prevent compound prompt latency
        status, cdata = make_request("/api/chat/conversations", method="POST", data={"title": f"Bench-Q{q_id}"}, token=token)
        q_conv_id = cdata.get("id")

        t0 = time.time()
        url = f"{BASE_URL}/api/chat/conversations/{q_conv_id}/messages"
        payload = json.dumps({
            "content": prompt,
            "web_search": False,
            "reasoning_mode": False,
            "model": "ari-3b"
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

        tokens = []
        final_clean = ""
        circuit_break = None

        try:
            with urllib.request.urlopen(req, timeout=90) as stream_resp:
                for line in stream_resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        p = json.loads(line_str[6:])
                        if p.get("type") == "token":
                            tokens.append(p.get("token", ""))
                        elif p.get("type") == "done":
                            final_clean = p.get("clean_content", "")
                        if p.get("circuit_break"):
                            circuit_break = p.get("reason")
        except Exception as e:
            print(f"[{q_id}/100] ERROR on question: {e}", flush=True)
            final_clean = f"ERROR: {e}"

        elapsed = time.time() - t0
        output_text = final_clean or "".join(tokens)
        eval_metrics = evaluate_formatting(output_text)
        tps = len(tokens) / elapsed if elapsed > 0 else 0

        res_entry = {
            "id": q_id,
            "cat": cat,
            "prompt": prompt,
            "elapsed": round(elapsed, 2),
            "tokens": len(tokens),
            "tps": round(tps, 1),
            "circuit_break": circuit_break,
            "metrics": eval_metrics,
            "sample_output": output_text[:200] + ("..." if len(output_text) > 200 else "")
        }
        results.append(res_entry)

        issues = []
        if eval_metrics["unclosed_code"]:
            issues.append("UnclosedCode")
        if eval_metrics["has_html_leak"]:
            issues.append("HTMLLeak")
        if eval_metrics["has_repetition"]:
            issues.append("Loop")
        if circuit_break:
            issues.append(f"Break:{circuit_break}")

        flag_str = " | ".join(issues) if issues else "CLEAN"
        print(f"[{q_id:3d}/100] [{cat:12s}] {elapsed:4.1f}s | {len(tokens):3d} toks ({tps:4.1f} t/s) | {flag_str}", flush=True)

        # Save progress after every single question
        with open("benchmark_100_progress.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - start_all
    print("=" * 80)
    print(f"Completed 100 questions in {total_time/60:.2f} minutes.")
    
    # Summary calculations
    total_tokens = sum(r["tokens"] for r in results)
    avg_tps = sum(r["tps"] for r in results) / len(results) if results else 0
    clean_count = sum(1 for r in results if not r["metrics"]["unclosed_code"] and not r["metrics"]["has_html_leak"] and not r["metrics"]["has_repetition"])
    table_count = sum(1 for r in results if r["metrics"]["has_table"])
    latex_count = sum(1 for r in results if r["metrics"]["has_latex"])
    code_count = sum(1 for r in results if r["metrics"]["has_code"])

    summary_md = f"""# Ari Chat 100-Question End-to-End Evaluation Report

- **Total Questions Evaluated:** {len(results)}
- **Total Tokens Generated:** {total_tokens}
- **Average Decoding Speed:** {avg_tps:.1f} tokens/second
- **Total Runtime:** {total_time/60:.2f} minutes
- **Presentation & Formatting Quality Score:** {clean_count / len(results) * 100:.1f}% ({clean_count}/{len(results)} flawlessly formatted)
- **Tables Successfully Rendered:** {table_count}
- **LaTeX Math Formulations:** {latex_count}
- **Syntax-Highlighted Code Blocks:** {code_count}

## Breakdown by Category
"""
    cats = sorted(list(set(r["cat"] for r in results)))
    for c in cats:
        c_items = [r for r in results if r["cat"] == c]
        c_clean = sum(1 for r in c_items if not r["metrics"]["unclosed_code"] and not r["metrics"]["has_html_leak"] and not r["metrics"]["has_repetition"])
        c_avg_tps = sum(r["tps"] for r in c_items) / len(c_items)
        summary_md += f"- **{c:15s}**: {c_clean}/{len(c_items)} clean ({c_clean/len(c_items)*100:.0f}%) | Avg Speed: {c_avg_tps:.1f} t/s\n"

    with open("benchmark_100_report.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(summary_md)

if __name__ == "__main__":
    main()
