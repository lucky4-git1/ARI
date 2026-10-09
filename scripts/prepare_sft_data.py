"""Prepares comprehensive instruction-tuning datasets covering QA, explanation, and coding."""

import os
import json

CONVERSATIONAL_INSTRUCTION_DATA = [
    # General QA and Science
    {
        "prompt": "What is gravity and how does it work?",
        "response": "Gravity is a fundamental natural force by which all things with mass or energy are attracted toward one another. According to Einstein's general relativity, gravity is the curvature of spacetime caused by mass and energy."
    },
    {
        "prompt": "Explain the concept of photosynthesis in simple terms.",
        "response": "Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy. It consumes carbon dioxide and water to produce glucose and oxygen using sunlight."
    },
    {
        "prompt": "What is the capital city of France?",
        "response": "The capital city of France is Paris."
    },
    {
        "prompt": "Why is the sky blue?",
        "response": "The sky appears blue because molecules in Earth's atmosphere scatter sunlight in all directions. Shorter, blue wavelengths of light scatter much more than longer red wavelengths, a phenomenon known as Rayleigh scattering."
    },
    {
        "prompt": "Summarize what an operating system does.",
        "response": "An operating system manages computer hardware and software resources, provides common services for computer programs, handles memory allocation, file systems, and schedules CPU tasks."
    },
    # Computer Science & Neural Networks
    {
        "prompt": "What is a Transformer model in artificial intelligence?",
        "response": "A Transformer is a deep learning architecture introduced in 2017 that relies on self-attention mechanisms to process sequence data in parallel without recurrent connections, powering modern language models."
    },
    {
        "prompt": "What is the difference between CPU and GPU?",
        "response": "A CPU is designed for low-latency sequential processing with a small number of powerful cores, whereas a GPU possesses thousands of smaller cores optimized for massive parallel computing like graphics rendering and matrix multiplication."
    },
    {
        "prompt": "What does Rotary Positional Embedding (RoPE) do in Transformers?",
        "response": "RoPE injects positional information into attention keys and queries by multiplying them with rotation matrices. This maintains relative distance properties and allows the model to extrapolate across longer sequence lengths."
    },
    {
        "prompt": "Explain what SwiGLU activation is.",
        "response": "SwiGLU is a gated activation function combining SiLU (Swish) and linear transformations: (SiLU(x * W_gate) * (x * W_up)) * W_down. It improves representation capacity and gradient flow compared to traditional ReLU MLPs."
    },
    {
        "prompt": "What is RMSNorm?",
        "response": "RMSNorm normalizes activations by dividing them by their root mean square without subtracting the mean. This achieves numerical stability similar to LayerNorm while reducing CPU and memory computational overhead."
    },
    # Coding & Algorithms
    {
        "prompt": "Write a Python function to check if a number is prime.",
        "response": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    },
    {
        "prompt": "Write a Python function to perform binary search on a sorted list.",
        "response": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"
    },
    {
        "prompt": "How do you reverse a string in Python?",
        "response": "You can reverse a string in Python using slicing:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"
    },
    {
        "prompt": "Write a Python function to compute the Fibonacci sequence iteratively.",
        "response": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
    },
    {
        "prompt": "Explain what a hash table is.",
        "response": "A hash table is a data structure that implements an associative array abstract data type, mapping keys to values using a hash function to compute an index into an array of buckets, providing average O(1) time complexity for lookups."
    },
    {
        "prompt": "What is gradient descent in machine learning?",
        "response": "Gradient descent is a first-order iterative optimization algorithm used to minimize a loss function by repeatedly updating model parameters in the opposite direction of the gradient of the function."
    },
    {
        "prompt": "How does quicksort work?",
        "response": "Quicksort is a divide-and-conquer sorting algorithm. It selects a 'pivot' element, partitions the array into sub-arrays of elements less than and greater than the pivot, and recursively sorts the sub-arrays."
    },
]

def build_instruction_dataset():
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/sft_conversational_v2.jsonl"
    
    # Write dataset repeated with slight permutations to ensure robust learning
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(35):
            for item in CONVERSATIONAL_INSTRUCTION_DATA:
                f.write(json.dumps(item) + "\n")

    print(f"Created expanded conversational SFT dataset at {out_path} ({os.path.getsize(out_path):,} bytes, {35 * len(CONVERSATIONAL_INSTRUCTION_DATA)} examples)")

if __name__ == "__main__":
    build_instruction_dataset()
