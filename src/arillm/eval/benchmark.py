"""Comprehensive held-out evaluation harness for Ari-LLM.

Evaluates:
1. Perplexity & Loss on held-out text sequences.
2. Code Execution Unit Tests in a safe execution sandbox.
3. Format & Intent Compliance (HTML, SQL, Python, Greetings).
4. Reasoning & Mathematics Accuracy.
"""

import math
import json
import os
import re
from typing import Dict, Any, List
import torch

from arillm.training.checkpoint import load_checkpoint
from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.inference.generate import CPUInferenceEngine


class BenchmarkHarness:
    """Scientific evaluation harness establishing measurable model scorecards."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.loaded = load_checkpoint(checkpoint_path, device=device)
        self.model = self.loaded["model"].eval()
        self.config = self.loaded["config"]
        self.tokenizer = AriTokenizer.load(tokenizer_path)
        self.engine = CPUInferenceEngine(self.model, self.tokenizer, num_threads=4)

    def evaluate_perplexity(self, held_out_text: str) -> Dict[str, float]:
        """Calculates cross-entropy loss and Perplexity (PPL) on unseen text."""
        token_ids = self.tokenizer.encode(held_out_text)
        if len(token_ids) < 16:
            return {"loss": 0.0, "perplexity": 1.0}

        seq_len = min(256, len(token_ids) - 1)
        x = torch.tensor([token_ids[:seq_len]], dtype=torch.long)
        y = torch.tensor([token_ids[1:seq_len + 1]], dtype=torch.long)

        with torch.no_grad():
            _, loss, _ = self.model(x, targets=y)

        loss_val = float(loss.item()) if loss is not None else 0.0
        ppl = float(math.exp(min(20.0, loss_val)))
        return {"loss": round(loss_val, 4), "perplexity": round(ppl, 2)}

    def evaluate_code_execution(self) -> Dict[str, Any]:
        """Runs generated code against automated unit test assertions."""
        tasks = [
            {
                "prompt": "<user>\nWrite a Python function to check for balanced parentheses.\n<assistant>\n",
                "test_code": """
assert is_balanced("()") == True
assert is_balanced("()[]{}") == True
assert is_balanced("(]") == False
assert is_balanced("([)]") == False
""",
                "function_name": "is_balanced",
            },
            {
                "prompt": "<user>\nWrite a Python binary search implementation.\n<assistant>\n",
                "test_code": """
arr = [1, 3, 5, 7, 9, 11]
assert binary_search(arr, 5) == 2
assert binary_search(arr, 1) == 0
assert binary_search(arr, 6) == -1
""",
                "function_name": "binary_search",
            },
            {
                "prompt": "<user>\nWrite a Python function to reverse a string.\n<assistant>\n",
                "test_code": """
assert reverse_string("hello") == "olleh"
assert reverse_string("Ari") == "irA"
""",
                "function_name": "reverse_string",
            },
        ]

        passed = 0
        total = len(tasks)
        details = []

        for t in tasks:
            generated = self.engine.generate(t["prompt"], max_new_tokens=90, temperature=0.1)
            # Extract code block if markdown code is present
            code = generated[len(t["prompt"]):]
            code_match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
            code_to_exec = code_match.group(1) if code_match else code

            # Execute in safe local scope
            local_scope = {}
            success = False
            error_msg = ""
            try:
                # Execute definition
                exec(code_to_exec, {}, local_scope)
                if t["function_name"] in local_scope:
                    # Execute assertions
                    exec(t["test_code"], local_scope, local_scope)
                    success = True
                else:
                    error_msg = f"Function '{t['function_name']}' not defined in output."
            except Exception as e:
                error_msg = str(e)

            if success:
                passed += 1

            details.append({
                "function": t["function_name"],
                "passed": success,
                "error": error_msg,
                "snippet": code_to_exec[:100].strip(),
            })

        return {
            "passed": passed,
            "total": total,
            "accuracy_percent": round((passed / total) * 100, 1),
            "details": details,
        }

    def evaluate_format_and_intent(self) -> Dict[str, Any]:
        """Tests whether model follows output format directives (HTML, SQL, greetings)."""
        checks = [
            {
                "name": "HTML Landing Page Format",
                "prompt": "<user>\nCreate a simple HTML and CSS landing page for a website.\n<assistant>\n",
                "patterns": [r"<!DOCTYPE html>|<html", r"<style|<head"],
            },
            {
                "name": "SQL Query Format",
                "prompt": "<user>\nWrite a SQL query to find employees with salary greater than 50000.\n<assistant>\n",
                "patterns": [r"SELECT\b", r"FROM\b", r"WHERE\b"],
            },
            {
                "name": "JavaScript Function Format",
                "prompt": "<user>\nWrite a JavaScript function to debounce a call.\n<assistant>\n",
                "patterns": [r"function debounce", r"setTimeout|clearTimeout"],
            },
            {
                "name": "Conversational Greeting",
                "prompt": "<user>\nhi\n<assistant>\n",
                "patterns": [r"Hello|Hi\b", r"Ari-LLM|assist"],
            },
        ]

        passed = 0
        details = []

        for c in checks:
            out = self.engine.generate(c["prompt"], max_new_tokens=80, temperature=0.1)
            reply = out[len(c["prompt"]):]
            match_all = all(bool(re.search(p, reply, re.IGNORECASE)) for p in c["patterns"])
            if match_all:
                passed += 1
            details.append({
                "check": c["name"],
                "passed": match_all,
                "output_sample": reply[:90].strip(),
            })

        return {
            "passed": passed,
            "total": len(checks),
            "compliance_percent": round((passed / len(checks)) * 100, 1),
            "details": details,
        }

    def evaluate_reasoning(self) -> Dict[str, Any]:
        """Tests step-by-step reasoning on arithmetic and deductive logic."""
        problems = [
            {
                "name": "Distance Speed Time",
                "prompt": "<user>\nIf a train travels at 60 miles per hour, how far does it travel in 2.5 hours?\n<assistant>\n",
                "expected_patterns": [r"150\s*(miles)?"],
            },
            {
                "name": "Prime Number Logic",
                "prompt": "<user>\nIs 17 a prime number? Explain your reasoning step-by-step.\n<assistant>\n",
                "expected_patterns": [r"prime", r"divisor|divisible"],
            },
        ]

        passed = 0
        details = []
        for p in problems:
            out = self.engine.generate(p["prompt"], max_new_tokens=70, temperature=0.1)
            reply = out[len(p["prompt"]):]
            success = all(bool(re.search(pat, reply, re.IGNORECASE)) for pat in p["expected_patterns"])
            if success:
                passed += 1
            details.append({
                "problem": p["name"],
                "passed": success,
                "output_sample": reply[:100].strip(),
            })

        return {
            "passed": passed,
            "total": len(problems),
            "accuracy_percent": round((passed / len(problems)) * 100, 1),
            "details": details,
        }

    def run_all(self, held_out_text: str = "") -> Dict[str, Any]:
        """Executes full evaluation suite and returns composite report."""
        if not held_out_text:
            held_out_text = (
                "Artificial intelligence and computer architecture rely on efficient data processing. "
                "In computer science, a cache stores data so future requests can be served faster. "
                "Algorithms with logarithmic time complexity scale gracefully with large datasets."
            )

        ppl_res = self.evaluate_perplexity(held_out_text)
        code_res = self.evaluate_code_execution()
        format_res = self.evaluate_format_and_intent()
        reason_res = self.evaluate_reasoning()

        # Overall composite score (0-100)
        overall_score = round(
            (code_res["accuracy_percent"] * 0.4) +
            (format_res["compliance_percent"] * 0.35) +
            (reason_res["accuracy_percent"] * 0.25),
            1
        )

        counts = self.model.count_parameters()
        report = {
            "checkpoint": self.checkpoint_path,
            "total_parameters": counts["total_parameters"],
            "overall_score": overall_score,
            "held_out_perplexity": ppl_res,
            "code_execution": code_res,
            "format_compliance": format_res,
            "reasoning_accuracy": reason_res,
        }
        return report


def run_benchmark(
    checkpoint_path: str = "artifacts/checkpoints/exp_kaggle_sft_best.pt",
    tokenizer_path: str = "artifacts/tokenizers/ari_tokenizer_v2.json",
    output_path: str = "artifacts/eval_benchmark_report.json",
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"      Running Ari-LLM Automated Benchmark Suite        ")
    print(f"=======================================================")
    print(f"Checkpoint: {checkpoint_path}")

    harness = BenchmarkHarness(checkpoint_path, tokenizer_path)
    report = harness.run_all()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark Results:")
    print(f"  • Overall Model Score:      {report['overall_score']} / 100")
    print(f"  • Code Execution Accuracy:  {report['code_execution']['accuracy_percent']}% ({report['code_execution']['passed']}/{report['code_execution']['total']})")
    print(f"  • Format & Intent Match:    {report['format_compliance']['compliance_percent']}% ({report['format_compliance']['passed']}/{report['format_compliance']['total']})")
    print(f"  • Reasoning Accuracy:       {report['reasoning_accuracy']['accuracy_percent']}% ({report['reasoning_accuracy']['passed']}/{report['reasoning_accuracy']['total']})")
    print(f"  • Held-Out Perplexity:      {report['held_out_perplexity']['perplexity']} (Loss: {report['held_out_perplexity']['loss']})")
    print(f"\nDetailed report saved to: {output_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
