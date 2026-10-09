"""CPU inference engine, token generation, and latency profiling."""

import time
from typing import Optional, Dict, Any, Generator
import torch
import torch.nn.functional as F

from arillm.model.transformer import AriTransformer
from arillm.tokenizer.tokenizer import AriTokenizer


class CPUInferenceEngine:
    """Optimized CPU inference runner supporting KV caching and greedy/top-p sampling."""

    def __init__(
        self,
        model: AriTransformer,
        tokenizer: AriTokenizer,
        num_threads: int = 4,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:
        """Generates completed text from string prompt."""
        token_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        
        generated_list = output_ids[0].tolist()
        return self.tokenizer.decode(generated_list)

    def benchmark_throughput(
        self,
        prompt: str = "def binary_search(arr, target):",
        num_tokens_to_generate: int = 30,
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """Profiles latency, time-to-first-token (TTFT), and generation tokens-per-second."""
        token_ids = self.tokenizer.encode(prompt)
        prompt_len = len(token_ids)
        input_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Warmup
        self.model.generate(input_tensor, max_new_tokens=5, use_cache=True)

        ttft_times = []
        tok_speeds = []

        for _ in range(num_runs):
            t0 = time.perf_counter()
            # Measure TTFT (prompt prefill pass)
            with torch.no_grad():
                logits, _, kv_caches = self.model(input_tensor, use_cache=True, start_pos=0)
            t_first = time.perf_counter() - t0
            ttft_times.append(t_first)

            # Measure next token generation decode loop
            t_gen_start = time.perf_counter()
            curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            for step in range(1, num_tokens_to_generate):
                start_pos = prompt_len + step
                with torch.no_grad():
                    logits, _, kv_caches = self.model(
                        curr_token,
                        kv_caches=kv_caches,
                        use_cache=True,
                        start_pos=start_pos,
                    )
                curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            t_gen_end = time.perf_counter() - t_gen_start
            tok_speeds.append((num_tokens_to_generate - 1) / max(1e-5, t_gen_end))

        avg_ttft_ms = (sum(ttft_times) / len(ttft_times)) * 1000.0
        avg_tok_sec = sum(tok_speeds) / len(tok_speeds)

        return {
            "num_threads": self.num_threads,
            "prompt_length": prompt_len,
            "tokens_generated": num_tokens_to_generate,
            "ttft_ms": avg_ttft_ms,
            "tokens_per_second": avg_tok_sec,
        }
