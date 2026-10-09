"""Prepares synthetic and permissively licensed seed dataset for LAYA-LLM Stage A/B."""

import os
import json

GENERAL_PROSE = [
    "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations automatically.",
    "Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks.",
    "A Transformer is a deep learning architecture developed by researchers at Google and based on the multi-head attention mechanism.",
    "In natural language processing, language models assign probabilities to sequences of words or tokens. Autoregressive language models predict the next token given preceding context.",
    "Rotary Positional Embedding, or RoPE, encodes positional information by multiplying key and query representations with rotation matrices.",
    "Grouped Query Attention reduces the memory bandwidth requirement during autoregressive generation by sharing key and value heads across multiple query heads.",
    "Root Mean Square Layer Normalization simplifies LayerNorm by scaling inputs by their root mean square, skipping the mean centering step.",
    "SwiGLU is an activation function and MLP variant that combines the Swish function with gated linear units, providing superior convergence characteristics in large transformers.",
    "Loss functions in generative language modeling measure the cross-entropy difference between the predicted next token probability distribution and the true target token.",
    "Optimization algorithms such as AdamW decouple weight decay from gradient updates, stabilizing training dynamics and preventing overfitting.",
    "The Central Processing Unit, or CPU, executes machine code instructions through fetch, decode, and execute stages, utilizing hardware caches for speed.",
    "Quantization in neural networks reduces numerical precision of weights and activations from 32-bit floating point down to 8-bit or 4-bit integers, reducing memory consumption.",
    "Checkpoints capture model weights, optimizer states, learning rate schedules, and step counts so that training can be resumed safely after interruption.",
    "A tokenizer converts raw character strings into discrete integer identifiers called tokens using algorithms such as Byte-Pair Encoding.",
    "Deduplication prevents language models from memorizing repeated training text, improving generalization and downstream performance.",
]

ALGORITHMIC_CODE = [
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n",
    "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
    "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    "def reverse_string(s):\n    return s[::-1]\n",
    "def count_vowels(text):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for char in text if char in vowels)\n",
    "def merge_sorted_lists(l1, l2):\n    res = []\n    i = j = 0\n    while i < len(l1) and j < len(l2):\n        if l1[i] < l2[j]:\n            res.append(l1[i])\n            i += 1\n        else:\n            res.append(l2[j])\n            j += 1\n    res.extend(l1[i:])\n    res.extend(l2[j:])\n    return res\n",
    "function linearSearch(arr, target) {\n    for (let i = 0; i < arr.length; i++) {\n        if (arr[i] === target) return i;\n    }\n    return -1;\n}\n",
    "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}\n",
]

INSTRUCTION_PAIRS = [
    {
        "prompt": "What is a Transformer in machine learning?",
        "response": "A Transformer is a neural network architecture introduced in 2017 that relies on self-attention mechanisms to process sequence data in parallel without recurrent connections."
    },
    {
        "prompt": "Write a Python function to compute the factorial of a number.",
        "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
    {
        "prompt": "Explain what RMSNorm does.",
        "response": "RMSNorm normalizes activation vectors by their root mean square rather than subtracting the mean, reducing computational overhead while stabilizing layer activations."
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "response": "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"
    },
    {
        "prompt": "What is the purpose of Rotary Positional Embeddings (RoPE)?",
        "response": "RoPE injects relative positional information into attention keys and queries by multiplying them with orthogonal 2D rotation matrices, enabling extrapolation across sequence lengths."
    },
]

def build_datasets():
    # 1. Pretraining corpus (repeating clean text & code to form 200+ KB training corpus)
    raw_pretrain_path = "data/raw/pretrain_corpus.txt"
    with open(raw_pretrain_path, "w", encoding="utf-8") as f:
        for _ in range(80):
            for prose in GENERAL_PROSE:
                f.write(prose + "\n\n")
            for code in ALGORITHMIC_CODE:
                f.write(code + "\n\n")

    print(f"Created pretrain corpus at {raw_pretrain_path} ({os.path.getsize(raw_pretrain_path)} bytes)")

    # 2. SFT instruction dataset
    sft_path = "data/processed/sft_data.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for _ in range(25):
            for item in INSTRUCTION_PAIRS:
                f.write(json.dumps(item) + "\n")

    print(f"Created SFT dataset at {sft_path} ({os.path.getsize(sft_path)} bytes)")

if __name__ == "__main__":
    build_datasets()
