"""Expanded conversational dataset adding paraphrasing, rewriting, summarization, and reasoning."""

import os
import json

PARAPHRASE_AND_REASONING_PAIRS = [
    # Paraphrasing & Rewriting Examples
    {
        "prompt": "Paraphrase this sentence: The weather today is exceptionally sunny and warm.",
        "response": "Today's weather is remarkably bright, clear, and pleasantly hot."
    },
    {
        "prompt": "Rewrite the following in a more professional tone: I can't come to work because I am sick.",
        "response": "I am unable to attend work today due to illness and will keep you updated on my recovery."
    },
    {
        "prompt": "Paraphrase this statement: Artificial intelligence is transforming how businesses operate worldwide.",
        "response": "Machine intelligence is fundamentally revolutionizing global business practices and workflows."
    },
    {
        "prompt": "Paraphrase this sentence: The quick brown fox jumps over the lazy dog.",
        "response": "A swift, brown fox leaps across a sluggish canine."
    },
    {
        "prompt": "Rewrite this simply: Utilizing advanced computational methodologies facilitates enhanced operational efficiencies.",
        "response": "Using better computer tools helps get work done faster."
    },
    {
        "prompt": "Paraphrase this: She decided to postpone the meeting until next week.",
        "response": "She chose to reschedule the gathering for the following week."
    },
    {
        "prompt": "Paraphrase this sentence: Eating a balanced diet and exercising regularly promotes good health.",
        "response": "Maintaining nutritious eating habits and routine physical activity fosters overall wellness."
    },
    {
        "prompt": "Rewrite this sentence in active voice: The novel was written by Mary Shelley.",
        "response": "Mary Shelley wrote the novel."
    },

    # Summarization Examples
    {
        "prompt": "Summarize the theory of relativity in one sentence.",
        "response": "Einstein's theory of relativity states that the laws of physics are the same for all observers and that gravity is the curvature of spacetime caused by mass and energy."
    },
    {
        "prompt": "Summarize what photosynthesis does in one sentence.",
        "response": "Photosynthesis is the biological process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen."
    },
    {
        "prompt": "Summarize the role of a computer operating system.",
        "response": "An operating system manages computer hardware and system resources while providing an environment for running software applications."
    },

    # Logic & Question Answering
    {
        "prompt": "If all roses are flowers and all flowers are plants, are all roses plants?",
        "response": "Yes, by transitive logic, because all roses are flowers and all flowers are plants, all roses must be plants."
    },
    {
        "prompt": "What is the primary difference between synchronous and asynchronous programming?",
        "response": "In synchronous programming, tasks execute sequentially and block further execution until completed, while in asynchronous programming, tasks execute concurrently without blocking the main program thread."
    },
    {
        "prompt": "What does API stand for and what is its purpose?",
        "response": "API stands for Application Programming Interface. It defines a set of rules and protocols allowing different software applications to communicate and exchange data with each other."
    },

    # Coding & Algorithms
    {
        "prompt": "Write a Python function to count the frequency of words in a string.",
        "response": "def word_count(text):\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"
    },
    {
        "prompt": "Write a Python function to check if a year is a leap year.",
        "response": "def is_leap_year(year):\n    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)"
    },
    {
        "prompt": "Write a Python function to flatten a list of lists into a single list.",
        "response": "def flatten_list(nested_list):\n    return [item for sublist in nested_list for item in sublist]"
    },
]

def append_paraphrase_data():
    sft_path = "data/processed/sft_conversational_v2.jsonl"
    with open(sft_path, "a", encoding="utf-8") as f:
        for _ in range(40):
            for item in PARAPHRASE_AND_REASONING_PAIRS:
                f.write(json.dumps(item) + "\n")

    print(f"Updated SFT dataset with paraphrasing and reasoning: {os.path.getsize(sft_path):,} bytes")

if __name__ == "__main__":
    append_paraphrase_data()
