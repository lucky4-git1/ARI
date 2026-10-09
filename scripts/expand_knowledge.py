"""Generates comprehensive multi-domain knowledge corpus across history, science, coding, and culture."""

import os
import json

COMPREHENSIVE_KNOWLEDGE = [
    # Historical Figures & World Events
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics alongside quantum mechanics. He is best known for his mass-energy equivalence formula E = mc^2 and received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
    "Isaac Newton was an English mathematician, physicist, and astronomer who formulated the laws of motion and universal gravitation, laying the foundations of classical mechanics in his work Philosophiæ Naturalis Principia Mathematica in 1687.",
    "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win Nobel Prizes in two scientific fields (Physics and Chemistry), and the first woman to become a professor at the University of Paris.",
    "Alan Turing was an English mathematician, computer scientist, and cryptanalyst. He is widely considered the father of theoretical computer science and artificial intelligence, having conceptualized the Turing machine and played a pivotal role in cracking intercepted coded messages during World War II at Bletchley Park.",
    "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States that occurred during the period from around 1760 to 1840, transforming agrarian societies into mechanized urban economies.",
    "The Renaissance was a fervent period of European cultural, artistic, political, and economic rebirth following the Middle Ages, spanning roughly from the 14th century to the 17th century, marked by humanism and celebrated through figures like Leonardo da Vinci and Michelangelo.",

    # Natural Sciences & Geography
    "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. The eight planets in order from the Sun are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune, with Jupiter being the largest gas giant.",
    "Photosynthesis is the process by which green plants and certain organisms transform light energy into chemical energy. During photosynthesis, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.",
    "DNA, or deoxyribonucleic acid, is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. It carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms and many viruses.",
    "Plate tectonics is the scientific theory explaining the movement of Earth's lithosphere, which is divided into several major and minor tectonic plates whose interactions along boundaries generate earthquakes, volcanic activity, and mountain building.",
    "Water has the chemical formula H2O, meaning each molecule consists of two hydrogen atoms bonded to a single oxygen atom. It is known as the universal solvent due to its ability to dissolve many substances, and it exists in solid, liquid, and gaseous states at common planetary temperatures.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south, covering an area larger than all of Earth's landmass combined.",

    # Computer Science & Software Engineering
    "A computer operating system is system software that manages computer hardware, software resources, and provides common services for computer programs. Examples include Linux, macOS, and Microsoft Windows.",
    "Data structures organize, manage, and store data in formats that enable efficient access and modification. Fundamental structures include arrays, linked lists, stacks, queues, hash tables, trees, and graphs.",
    "Object-Oriented Programming (OOP) is a programming paradigm based on the concept of objects, which can contain data in the form of fields (attributes) and code in the form of procedures (methods). Core pillars of OOP include encapsulation, abstraction, inheritance, and polymorphism.",
    "Relational databases store data in tables consisting of rows and columns, enforcing relational integrity through primary and foreign keys and accessed using Structured Query Language (SQL).",
    "Computer networks allow computing devices to exchange data and share resources over wired or wireless connections using standardized communication protocols such as TCP/IP, HTTP, and DNS.",
    "Artificial intelligence is the field of computer science dedicated to developing systems capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and translation.",
]

EXPANDED_INSTRUCTION_DATASET = [
    {
        "prompt": "Who was Albert Einstein and what did he discover?",
        "response": "Albert Einstein was a theoretical physicist who developed the theory of relativity, which revolutionized our understanding of space, time, and gravity. He also formulated the mass-energy equivalence equation E = mc^2 and won the Nobel Prize in Physics for explaining the photoelectric effect."
    },
    {
        "prompt": "Who was Alan Turing?",
        "response": "Alan Turing was an English mathematician and computer scientist widely regarded as the father of modern computer science. He formalized the concepts of algorithm and computation with the Turing machine and played a crucial role in cracking the Enigma cipher during World War II."
    },
    {
        "prompt": "List the planets of the solar system in order from the Sun.",
        "response": "The eight planets in order from the Sun are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
    },
    {
        "prompt": "What is the Industrial Revolution?",
        "response": "The Industrial Revolution was a period of major technological and socioeconomic change starting in the mid-18th century, transitioning production from manual craftsmanship to powered machinery and factories."
    },
    {
        "prompt": "Explain what a hash table is and its time complexity.",
        "response": "A hash table is a data structure that maps keys to values using a hash function. It offers an average time complexity of O(1) for lookups, insertions, and deletions, making it highly efficient for key-value retrieval."
    },
    {
        "prompt": "Write a Python function to check if a word is an anagram of another.",
        "response": "def is_anagram(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())"
    },
    {
        "prompt": "Write a Python function to find the maximum element in a list without using the built-in max function.",
        "response": "def find_max(arr):\n    if not arr:\n        return None\n    largest = arr[0]\n    for x in arr[1:]:\n        if x > largest:\n            largest = x\n    return largest"
    },
    {
        "prompt": "What are the four pillars of Object-Oriented Programming?",
        "response": "The four fundamental pillars of Object-Oriented Programming (OOP) are Encapsulation, Abstraction, Inheritance, and Polymorphism."
    },
]

def build_scale_data():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    # 1. Append to pretrain corpus
    pretrain_extended = "data/processed/combined_language_code_corpus.txt"
    with open(pretrain_extended, "a", encoding="utf-8") as f:
        for _ in range(150):
            for passage in COMPREHENSIVE_KNOWLEDGE:
                f.write(passage + "\n\n")

    print(f"Updated unified pretrain corpus: {os.path.getsize(pretrain_extended):,} bytes")

    # 2. Update SFT conversational dataset
    sft_path = "data/processed/sft_conversational_v2.jsonl"
    with open(sft_path, "a", encoding="utf-8") as f:
        for _ in range(40):
            for item in EXPANDED_INSTRUCTION_DATASET:
                f.write(json.dumps(item) + "\n")

    print(f"Updated SFT conversational dataset: {os.path.getsize(sft_path):,} bytes")

if __name__ == "__main__":
    build_scale_data()
