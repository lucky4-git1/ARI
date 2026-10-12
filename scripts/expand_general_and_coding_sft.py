"""Generates and augments SFT conversational dataset with real-world knowledge and coding exercises.

Covers:
1. Real-world encyclopedic knowledge (fruits, plants, physics, geography, biology).
2. Advanced coding patterns (Python, JavaScript, SQL, HTML/CSS).
3. Chain-of-thought step-by-step reasoning and debugging.
"""

import json
import os

OUTPUT_FILE = "data/processed/sft_conversational_v3.jsonl"
BASE_FILE = "data/processed/sft_conversational_v2.jsonl"

ADDITIONAL_QA = [
    # --- Real-World Encyclopedic Knowledge ---
    {
        "prompt": "<user>\nwhat is a mango\n<assistant>\n",
        "completion": "A mango is an edible stone fruit (drupe) produced by the tropical tree Mangifera indica. Originating from South Asia, mangoes are known for their sweet, juicy yellow-orange flesh surrounding a fibrous pit. They are rich in vitamin C, vitamin A, and dietary antioxidants.<eos>\n"
    },
    {
        "prompt": "<user>\nwhat is an apple\n<assistant>\n",
        "completion": "An apple is a round, edible fruit produced by an apple tree (Malus domestica). Grown worldwide in thousands of cultivars, apples are crisp, sweet or tart, and rich in fiber (pectin) and vitamin C.<eos>\n"
    },
    {
        "prompt": "<user>\nwhat is photosynthesis\n<assistant>\n",
        "completion": "Photosynthesis is the biological process by which green plants and algae convert sunlight, water, and carbon dioxide into oxygen and chemical energy in the form of glucose using chlorophyll.<eos>\n"
    },
    {
        "prompt": "<user>\nwhat is gravity\n<assistant>\n",
        "completion": "Gravity is the fundamental natural force of attraction between all objects that possess mass or energy. In Newtonian physics it is an attractive force proportional to mass, while in Einstein's general relativity it is described as the curvature of spacetime.<eos>\n"
    },
    {
        "prompt": "<user>\nwhat is the capital of France\n<assistant>\n",
        "completion": "The capital of France is Paris. It is the country's largest city and a major global center for commerce, culture, diplomacy, and science.<eos>\n"
    },
    {
        "prompt": "<user>\nwhat is the capital of Japan\n<assistant>\n",
        "completion": "The capital of Japan is Tokyo. Located on the eastern coast of Honshu, Tokyo is one of the most populous metropolitan areas in the world and Japan's financial and political center.<eos>\n"
    },
    {
        "prompt": "<user>\nwhy is the sky blue\n<assistant>\n",
        "completion": "The sky appears blue due to a phenomenon called Rayleigh scattering. Sunlight consists of all colors of the visible spectrum. As sunlight passes through Earth's atmosphere, shorter wavelengths of light (blue and violet) are scattered in all directions by gas molecules much more than longer wavelengths (red and yellow).<eos>\n"
    },
    {
        "prompt": "<user>\nwhat are mitochondria\n<assistant>\n",
        "completion": "Mitochondria are membrane-bound organelles found in eukaryotic cells responsible for producing energy in the form of ATP (adenosine triphosphate) through cellular respiration. They are commonly known as the powerhouses of the cell.<eos>\n"
    },
    
    # --- Advanced Frontend, Backend & Database Coding ---
    {
        "prompt": "<user>\nwrite a python function to check if a string is a palindrome\n<assistant>\n",
        "completion": "```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Checks if a string reads the same forwards and backwards ignoring non-alphanumeric chars.\"\"\"\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n```<eos>\n"
    },
    {
        "prompt": "<user>\nwrite a python function to compute factorial recursively and iteratively\n<assistant>\n",
        "completion": "```python\ndef factorial_iterative(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"Factorial not defined for negative numbers.\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```<eos>\n"
    },
    {
        "prompt": "<user>\nwrite a sql query to find the second highest salary\n<assistant>\n",
        "completion": "```sql\nSELECT MAX(salary) AS SecondHighestSalary\nFROM employees\nWHERE salary < (\n    SELECT MAX(salary) FROM employees\n);\n```<eos>\n"
    },
    {
        "prompt": "<user>\nwrite a javascript debounce function\n<assistant>\n",
        "completion": "```javascript\nfunction debounce(func, delay = 300) {\n  let timeoutId;\n  return function(...args) {\n    clearTimeout(timeoutId);\n    timeoutId = setTimeout(() => {\n      func.apply(this, args);\n    }, delay);\n  };\n}\n```<eos>\n"
    },
    {
        "prompt": "<user>\nwrite a javascript throttle function\n<assistant>\n",
        "completion": "```javascript\nfunction throttle(func, limit = 300) {\n  let inThrottle = false;\n  return function(...args) {\n    if (!inThrottle) {\n      func.apply(this, args);\n      inThrottle = true;\n      setTimeout(() => inThrottle = false, limit);\n    }\n  };\n}\n```<eos>\n"
    },
    {
        "prompt": "<user>\nwrite python code for depth-first search on a graph\n<assistant>\n",
        "completion": "```python\ndef dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    print(start, end=' ')\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited\n```<eos>\n"
    },

    # --- Reasoning, Math & Logical Deduction ---
    {
        "prompt": "<user>\nIf an item costs $80 after a 20% discount, what was the original price?\n<assistant>\n",
        "completion": "Let the original price be P.\n1. A 20% discount means the customer pays 80% of the original price: 0.80 * P = 80.\n2. Divide both sides by 0.80: P = 80 / 0.80 = 100.\nTherefore, the original price was $100.<eos>\n"
    },
    {
        "prompt": "<user>\nA car travels at 45 mph for 4 hours. How far does it travel?\n<assistant>\n",
        "completion": "Distance is calculated as speed multiplied by time:\n1. Speed = 45 mph\n2. Time = 4 hours\n3. Calculation: Distance = 45 * 4 = 180 miles.\nThe car travels 180 miles.<eos>\n"
    }
]

def main():
    os.makedirs("data/processed", exist_ok=True)
    existing_examples = []
    if os.path.exists(BASE_FILE):
        with open(BASE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_examples.append(json.loads(line))
        print(f"Loaded {len(existing_examples)} existing examples from {BASE_FILE}.")

    # Combine with new QA examples
    combined = existing_examples + ADDITIONAL_QA
    
    # Save v3
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    print(f"Successfully generated {len(combined)} examples into {OUTPUT_FILE}!")

if __name__ == "__main__":
    main()
