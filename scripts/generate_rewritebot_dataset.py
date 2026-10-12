"""High-Density RewriteBot / QuillBot Professional Training Dataset Generator.

Purposely EXCLUDES all programming/coding data to dedicate 100% of network capacity
to natural human language, deep paraphrasing, grammar, clause inversion, and style transfer.

Modes:
1. Standard Paraphrasing (fluent, balanced restructuring)
2. Grammar Correction & Polish (fixing tense, subject-verb agreement, dangling modifiers)
3. Clause Inversion (flipping dependent & independent clauses, active/passive shifts)
4. Formal / Corporate (email and professional communication)
5. Academic (scholarly literature, research paper tone)
6. Simple / Plain English (eliminating complex jargon)
7. Creative / Expressive (poetic and vivid descriptive vocabulary)
8. Shorten / Condense (removing wordiness and redundancy)
9. Expand / Elaborate (adding depth, clarity, and precision)
10. Anti-AI Humanizer (varying sentence length and burstiness to sound authentically human)
"""

import os
import json
import random
import hashlib

OUTPUT_FILE = "data/processed/rewritebot_master_dataset.jsonl"

# ----------------------------------------------------------------------
# 1. GRAMMAR CORRECTION & POLISHING PAIRS (Error -> Clean Grammar)
# ----------------------------------------------------------------------
GRAMMAR_PAIRS = [
    ("She don't know the answer to the question.", "She does not know the answer to the question."),
    ("Me and him went to the grocery store yesterday.", "He and I went to the grocery store yesterday."),
    ("The team are playing their best match today.", "The team is playing its best match today."),
    ("Neither of the boys were present at the meeting.", "Neither of the boys was present at the meeting."),
    ("He should of told us before leaving the office.", "He should have told us before leaving the office."),
    ("Its a well known fact that smoking damages lungs.", "It is a well-known fact that smoking damages the lungs."),
    ("The data shows that sales has decreased by ten percent.", "The data show that sales have decreased by ten percent."),
    ("Every student must bring their own laptop to class.", "All students must bring their own laptops to class."),
    ("Walking down the street, the trees were beautiful.", "As I walked down the street, I noticed the beautiful trees."),
    ("She was more smarter than anyone else in the room.", "She was smarter than anyone else in the room."),
    ("Between you and I, this decision makes no sense.", "Between you and me, this decision makes no sense."),
    ("The amount of people attending the event was huge.", "The number of people attending the event was very large."),
    ("He ran quick to catch the departing morning train.", "He ran quickly to catch the departing morning train."),
    ("The reason is because the server crashed overnight.", "The reason is that the server crashed overnight."),
    ("Supposably, the contract will be signed tomorrow afternoon.", "Supposedly, the contract will be signed tomorrow afternoon.")
]

# ----------------------------------------------------------------------
# 2. CLAUSE INVERSION & SYNTACTIC SHIFTS (A because B <-> Because B, A)
# ----------------------------------------------------------------------
CLAUSE_INVERSION_PAIRS = [
    ("The flight was canceled because the storm was severe.", "Because the storm was severe, the flight was canceled."),
    ("She succeeded in her exams although she had very little time to study.", "Although she had very little time to study, she succeeded in her exams."),
    ("We decided to launch the product early since user demand was extraordinarily high.", "Since user demand was extraordinarily high, we decided to launch the product early."),
    ("The team reached the summit after they climbed for six grueling hours.", "After climbing for six grueling hours, the team reached the summit."),
    ("You will never master a musical instrument unless you practice every single day.", "Unless you practice every single day, you will never master a musical instrument."),
    ("He was promoted to director while the company was undergoing a massive restructuring.", "While the company was undergoing a massive restructuring, he was promoted to director."),
    ("The economy began to recover as soon as the central bank lowered interest rates.", "As soon as the central bank lowered interest rates, the economy began to recover."),
    ("Scientists confirmed the hypothesis once the secondary trials concluded successfully.", "Once the secondary trials concluded successfully, scientists confirmed the hypothesis."),
    ("We stayed indoors throughout the afternoon until the heavy rainfall finally subsided.", "Until the heavy rainfall finally subsided, we stayed indoors throughout the afternoon.")
]

# ----------------------------------------------------------------------
# 3. MULTI-MODE REWRITE SEEDS (Standard, Formal, Academic, Simple, Creative)
# ----------------------------------------------------------------------
REWRITE_MODES_DATA = [
    {
        "input": "We need to fix this error immediately or customers will get angry.",
        "modes": {
            "Formal": "We must prioritize resolving this issue promptly to maintain client satisfaction.",
            "Academic": "Expedited rectification of this discrepancy is essential to prevent adverse stakeholder sentiment.",
            "Simple": "We need to fix this bug quickly so customers don't get upset.",
            "Shorten": "We must fix this immediately to prevent customer frustration.",
            "Expand": "It is critical that our engineering team identifies and patches this error without delay, as failing to do so will directly jeopardize customer trust and retention."
        }
    },
    {
        "input": "The weather today is exceptionally sunny and warm.",
        "modes": {
            "Standard": "Today's weather is remarkably bright, clear, and pleasantly warm.",
            "Creative": "A brilliant golden sun bathed the landscape in soothing, radiant warmth.",
            "Formal": "The meteorological conditions today are uncommonly clear and elevated in temperature.",
            "Simple": "It is very sunny and warm outside today.",
            "Shorten": "Today is exceptionally sunny and warm."
        }
    },
    {
        "input": "Artificial intelligence is transforming how businesses operate worldwide.",
        "modes": {
            "Standard": "Machine intelligence is fundamentally reshaping corporate operations and global business practices.",
            "Formal": "Modern computational intelligence is currently restructuring organizational workflows across international markets.",
            "Academic": "The widespread integration of artificial intelligence represents an epochal paradigm shift in global operational methodologies.",
            "Simple": "AI is changing the way companies work all over the world.",
            "Shorten": "AI is revolutionizing global business operations."
        }
    },
    {
        "input": "She decided to postpone the meeting until next week.",
        "modes": {
            "Standard": "She chose to reschedule the gathering for the following week.",
            "Formal": "She opted to defer the scheduled conference to the ensuing week.",
            "Simple": "She moved the meeting to next week.",
            "Shorten": "She rescheduled the meeting for next week."
        }
    },
    {
        "input": "Eating a balanced diet and exercising regularly promotes good health.",
        "modes": {
            "Standard": "Maintaining nutritious eating habits and routine physical activity fosters overall wellness.",
            "Academic": "Sustained adherence to dietary equilibrium in conjunction with systematic physical exertion yields quantifiable physiological benefits.",
            "Simple": "Eating healthy food and working out regularly keeps you in good shape.",
            "Creative": "Nourishing the body with wholesome meals and daily movement kindles lasting vitality and inner strength."
        }
    },
    {
        "input": "Prices of residential properties have increased substantially over the past decade.",
        "modes": {
            "Standard": "Housing and home values have risen dramatically over the last ten years.",
            "Formal": "Residential property valuations have experienced significant appreciation across the previous decennial period.",
            "Academic": "Longitudinal empirical assessments reveal an escalating inflationary trajectory in domestic real estate valuation over the past decade.",
            "Simple": "Houses have become a lot more expensive over the last ten years."
        }
    }
]

# ----------------------------------------------------------------------
# 4. ANTI-AI DETECTOR HUMANIZER (High Burstiness & Natural Cadence)
# ----------------------------------------------------------------------
HUMANIZER_PAIRS = [
    (
        "In conclusion, it is important to remember that the multifaceted tapestry of modern life presents myriad challenges. Furthermore, one must delve deeply into the nuances.",
        "Look, modern life is pretty chaotic right now. There's a lot thrown at us all at once. Instead of dressing it up with complicated jargon, it just makes sense to look at the facts and take things one step at a time."
    ),
    (
        "This serves as a testament to the fact that technological advancements foster seamless communication paradigms across diverse demographics worldwide.",
        "Honestly, this just shows how much easier it is to talk to people nowadays. Tech has made staying in touch feel natural, whether you're chatting with someone down the street or halfway across the globe."
    ),
    (
        "Undoubtedly, education acts as a cornerstone for societal progression, encapsulating both pedagogical methodologies and experiential insights.",
        "A good education really does change everything for a community. It isn't just about cramming textbook facts into your head; it's about learning how to solve problems and think for yourself."
    ),
    (
        "Moreover, it is crucial to recognize that an individual must meticulously navigate their professional trajectory to ensure optimal outcomes in their career.",
        "At the end of the day, you have to steer your own career. Nobody is going to plan out your future for you, so being proactive and figuring out what works is what actually gets you ahead."
    ),
    (
        "In summation, the implementation of these strategic imperatives will inevitably catalyze a paradigm shift in organizational efficacy.",
        "Bottom line: if we actually follow through on these changes, the whole team is going to work a lot smoother and get things done without the usual headaches."
    )
]

def generate_rewritebot_dataset():
    records = []
    seen = set()

    def add(prompt, response):
        p = prompt.strip()
        r = response.strip()
        h = hashlib.md5(f"{p}|{r}".encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            records.append({"prompt": p, "response": r})

    print("Generating comprehensive RewriteBot training corpus...")

    # A. Grammar Correction Templates
    grammar_templates = [
        "Fix the grammar in this sentence: {text}",
        "Correct all grammatical errors in the following: \"{text}\"",
        "Grammar Polish: {text}",
        "Proofread and correct this sentence: {text}",
        "Rewrite this with perfect grammar: {text}"
    ]
    for error, clean in GRAMMAR_PAIRS:
        for tmpl in grammar_templates:
            add(tmpl.format(text=error), clean)

    # B. Clause Inversion Templates
    clause_templates = [
        "Invert the clauses in this sentence while preserving meaning: {text}",
        "Flip the dependent and independent clauses: {text}",
        "Rewrite this sentence starting with the secondary clause: {text}",
        "Syntactic clause inversion: \"{text}\""
    ]
    for a, b in CLAUSE_INVERSION_PAIRS:
        for tmpl in clause_templates:
            add(tmpl.format(text=a), b)
            add(tmpl.format(text=b), a)  # Bidirectional

    # C. Multi-Mode Paraphrasing Templates
    for item in REWRITE_MODES_DATA:
        inp = item["input"]
        modes = item["modes"]
        for mode, target in modes.items():
            templates = [
                f"Rewrite this in {mode.lower()} mode: {{text}}",
                f"QuillBot {mode.lower()} mode: {{text}}",
                f"Rephrase the following in a {mode.lower()} tone: \"{{text}}\"",
                f"Rewrite this: {{text}}" if mode == "Standard" else f"Convert to {mode.lower()}: {{text}}"
            ]
            for tmpl in templates:
                add(tmpl.format(text=inp), target)

    # D. Anti-AI Humanizer Templates
    humanizer_templates = [
        "Humanize this text to bypass AI detectors: {text}",
        "Make this sound like a real human wrote it: {text}",
        "Remove robotic AI cliches and rewrite naturally: {text}",
        "Rewrite with high burstiness and varied sentence cadence: {text}",
        "Rewrite this text to sound completely authentic and human: \"{text}\""
    ]
    for robotic, human in HUMANIZER_PAIRS:
        for tmpl in humanizer_templates:
            add(tmpl.format(text=robotic), human)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records):,} high-quality RewriteBot language samples in {OUTPUT_FILE}!")

if __name__ == "__main__":
    generate_rewritebot_dataset()
