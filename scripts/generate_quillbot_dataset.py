"""Generates a rich, multi-mode QuillBot-style Paraphrasing & Rewriting SFT Dataset.

Modes covered:
1. Standard Paraphrasing (fluent, natural synonym and clause restructuring)
2. Professional / Formal Rewriting (corporate emails, polite declines, official reports)
3. Creative / Expressive Rewriting (vivid vocabulary, engaging tone)
4. Simplicity / Plain English (demystifying jargon into straightforward sentences)
5. Shorten / Condense (trimming fluff and redundancy)
6. Expand / Elaborate (adding clarity and detail)
7. Active Voice Conversion
"""

import json
import os

OUTPUT_FILE = "data/processed/sft_paraphrase_quillbot.jsonl"
BASE_FILE = "data/processed/sft_conversational_v3.jsonl"

PARAPHRASE_CORE_DATA = [
    # --- Standard Paraphrasing ---
    {"mode": "Standard", "input": "The weather today is exceptionally sunny and warm.", "output": "Today's weather is remarkably bright, clear, and pleasantly warm."},
    {"mode": "Standard", "input": "Artificial intelligence is transforming how businesses operate worldwide.", "output": "Machine intelligence is fundamentally reshaping corporate operations and global business practices."},
    {"mode": "Standard", "input": "She decided to postpone the meeting until next week.", "output": "She chose to reschedule the gathering for the following week."},
    {"mode": "Standard", "input": "Eating a balanced diet and exercising regularly promotes good health.", "output": "Maintaining nutritious eating habits and routine physical activity fosters overall wellness."},
    {"mode": "Standard", "input": "The team worked around the clock to finish the project on time.", "output": "The team labored tirelessly day and night to meet the project deadline."},
    {"mode": "Standard", "input": "Prices of residential properties have increased substantially over the past decade.", "output": "Housing and home values have risen dramatically over the last ten years."},
    {"mode": "Standard", "input": "Learning a new foreign language requires consistent dedication and regular practice.", "output": "Acquiring another language demands persistent effort and everyday study."},
    {"mode": "Standard", "input": "He was overwhelmed by the vast amount of information presented in the seminar.", "output": "He felt inundated by the immense volume of data shared during the conference."},
    {"mode": "Standard", "input": "Regular software updates are necessary to protect your system from cyber threats.", "output": "Frequent software patches are essential for defending your machine against security vulnerabilities."},
    {"mode": "Standard", "input": "The experiment yielded results that exceeded all initial expectations.", "output": "The test produced outcomes far superior to what researchers initially anticipated."},

    # --- Professional & Formal Rewriting (Email & Business) ---
    {"mode": "Formal", "input": "I can't come to work because I am sick.", "output": "I am writing to inform you that I will be unable to attend work today due to illness. I will keep you posted on my recovery."},
    {"mode": "Formal", "input": "Can you send me that document as soon as possible?", "output": "Could you kindly provide the requested document at your earliest convenience?"},
    {"mode": "Formal", "input": "Sorry for answering you so late.", "output": "Thank you for your patience; please accept my apologies for the delayed response."},
    {"mode": "Formal", "input": "I don't agree with your idea at all.", "output": "I have several reservations regarding this proposal and would like to suggest an alternative perspective."},
    {"mode": "Formal", "input": "Let me know what you think about this plan.", "output": "I would appreciate your feedback and insights regarding the proposed strategy."},
    {"mode": "Formal", "input": "We need to fix this error right now or clients will get mad.", "output": "We must prioritize resolving this critical issue immediately to ensure continued client satisfaction."},
    {"mode": "Formal", "input": "Thanks a lot for helping me out yesterday.", "output": "I sincerely appreciate your valuable assistance and guidance yesterday."},
    {"mode": "Formal", "input": "I want to quit my job next month.", "output": "Please accept this formal notification of my resignation, effective at the end of next month."},
    {"mode": "Formal", "input": "Where is the update you promised yesterday?", "output": "I am following up on the status update discussed in yesterday's meeting."},
    {"mode": "Formal", "input": "Can we talk about my pay raise?", "output": "I would like to schedule a brief meeting to review my recent contributions and discuss compensation adjustments."},

    # --- Simplicity / Plain English (Jargon to Plain) ---
    {"mode": "Simple", "input": "Utilizing advanced computational methodologies facilitates enhanced operational efficiencies.", "output": "Using better computer tools helps get work done faster and cheaper."},
    {"mode": "Simple", "input": "The patient presented with acute cephalalgia and manifestations of pyrexia.", "output": "The patient had a severe headache and a fever."},
    {"mode": "Simple", "input": "In accordance with aforementioned stipulations, discontinuation of service is imminent.", "output": "As stated earlier, we will cancel your service shortly."},
    {"mode": "Simple", "input": "This innovative paradigm shift synergizes interdepartmental cross-collaboration.", "output": "This new method helps different teams work together more easily."},
    {"mode": "Simple", "input": "The implementation of stringent fiscal austerity measures precipitated widespread dissatisfaction.", "output": "Strict spending cuts caused widespread unhappiness among the public."},
    {"mode": "Simple", "input": "We must optimize our throughput to ameliorate current workflow bottlenecks.", "output": "We need to speed up our output to fix delays in our workflow."},

    # --- Shorten / Conciseness ---
    {"mode": "Shorten", "input": "Due to the fact that the weather was extremely rainy and stormy, we had no choice but to cancel the event.", "output": "Because of heavy rain and storms, we canceled the event."},
    {"mode": "Shorten", "input": "At this point in time, we are currently in the process of evaluating each and every single proposal.", "output": "We are currently evaluating every proposal."},
    {"mode": "Shorten", "input": "In the event that you encounter any problems or difficulties, please do not hesitate to reach out to us.", "output": "If you experience any issues, please contact us."},
    {"mode": "Shorten", "input": "It is absolutely essential and critically necessary that all team members attend this mandatory meeting.", "output": "All team members must attend this meeting."},
    {"mode": "Shorten", "input": "The reason why he failed to achieve his goals was on account of his lack of adequate preparation.", "output": "He failed because he did not prepare adequately."},

    # --- Creative / Expressive ---
    {"mode": "Creative", "input": "The sun went down and the stars came out.", "output": "The crimson sun dipped beneath the horizon, giving way to a velvet sky dusted with luminous stars."},
    {"mode": "Creative", "input": "She walked through the quiet forest.", "output": "She wandered softly through the whispering forest, beneath an emerald canopy alive with gentle shadows."},
    {"mode": "Creative", "input": "The city was very noisy and full of people.", "output": "The restless metropolis buzzed with electric energy, crowded avenues, and a symphony of blaring horns."},
    {"mode": "Creative", "input": "He was very sad after hearing the bad news.", "output": "A heavy wave of sorrow engulfed him the moment the tragic news broke his silence."},

    # --- Active Voice ---
    {"mode": "Active Voice", "input": "The novel was written by Mary Shelley.", "output": "Mary Shelley wrote the novel."},
    {"mode": "Active Voice", "input": "The software bug was discovered by the quality assurance team.", "output": "The quality assurance team discovered the software bug."},
    {"mode": "Active Voice", "input": "A delicious banquet was prepared by the head chef.", "output": "The head chef prepared a delicious banquet."},
    {"mode": "Active Voice", "input": "The championship game was won by the underdog team.", "output": "The underdog team won the championship game."}
]

def generate_multi_template_examples():
    records = []
    
    # Prompt template variations
    templates = {
        "Standard": [
            "Paraphrase this: {text}",
            "Paraphrase this sentence: {text}",
            "Can you paraphrase the following: \"{text}\"",
            "Rewrite this with different words while keeping the exact meaning: {text}",
            "QuillBot paraphrase: {text}",
        ],
        "Formal": [
            "Rewrite the following in a formal, professional tone: {text}",
            "Make this sound professional for an email: {text}",
            "Rewrite professionally: \"{text}\"",
            "Convert this into professional corporate tone: {text}",
        ],
        "Simple": [
            "Rewrite this simply in plain English: {text}",
            "Simplify this sentence: {text}",
            "Rewrite this without jargon so anyone can understand: {text}",
        ],
        "Shorten": [
            "Shorten this sentence and remove fluff: {text}",
            "Make this sentence more concise: {text}",
            "Condense the following: {text}",
        ],
        "Creative": [
            "Rewrite this creatively with vivid vocabulary: {text}",
            "Make this sentence poetic and descriptive: {text}",
        ],
        "Active Voice": [
            "Rewrite this sentence in active voice: {text}",
            "Convert to active voice: {text}",
        ]
    }

    for item in PARAPHRASE_CORE_DATA:
        mode = item["mode"]
        inp = item["input"]
        out = item["output"]
        
        # Expand across multiple variations
        for tmpl in templates.get(mode, ["Paraphrase: {text}"]):
            prompt = tmpl.format(text=inp)
            records.append({
                "prompt": prompt,
                "response": out
            })

    return records

def main():
    os.makedirs("data/processed", exist_ok=True)
    paraphrase_records = generate_multi_template_examples()
    print(f"Generated {len(paraphrase_records)} high-density QuillBot paraphrase examples.")

    # Load existing base conversational data
    base_records = []
    if os.path.exists(BASE_FILE):
        with open(BASE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    base_records.append(json.loads(line))
        print(f"Loaded {len(base_records)} base conversational/knowledge examples.")

    # Merge: Paraphrasing data + base conversational dataset (oversampling paraphrasing for strong style mastery)
    # Replicate paraphrase records 4x to establish dominant weights without forgetting base knowledge
    all_combined = base_records + (paraphrase_records * 4)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in all_combined:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully wrote {len(all_combined)} total training examples to {OUTPUT_FILE}!")

if __name__ == "__main__":
    main()
