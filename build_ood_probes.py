#!/usr/bin/env python
"""Build fixed out-of-domain probes for measuring capability drift.

- ood_prose: paragraphs from unrelated Wikipedia articles (LM likelihood)
- ood_qa:   general-knowledge QA in the same "Question:/Answer:" format as
            the domain qa_probe, unambiguous answers
"""
import json, time, requests
from pathlib import Path

DATA = Path("/home/catid/specialist/data")

TOPICS = [
    "Photosynthesis", "Roman Empire", "Quicksort", "Monetary policy",
    "Plate tectonics", "Ludwig van Beethoven", "Immune system", "HTTP",
    "Mount Everest", "Chess", "Coffee", "Semiconductor",
    "French Revolution", "Convolutional neural network", "Tide", "DNA",
]

GENERAL_QA = [
    ("What is the capital of Australia?", "Canberra"),
    ("What is the chemical symbol for gold?", "Au"),
    ("In what year did World War II end?", "1945"),
    ("What is the speed of light in a vacuum, approximately?", "About 300,000 kilometers per second (299,792 km/s)"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("What is the largest planet in the Solar System?", "Jupiter"),
    ("What data structure uses first-in, first-out ordering?", "A queue"),
    ("What is the powerhouse of the cell?", "The mitochondrion"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the smallest prime number?", "2"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What gas do plants primarily absorb during photosynthesis?", "Carbon dioxide"),
    ("What is the longest river in the world?", "The Nile (or the Amazon by some measures)"),
    ("Who developed the theory of general relativity?", "Albert Einstein"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("How many chromosomes do humans typically have?", "46"),
    ("What is the currency of Japan?", "The yen"),
    ("Which element has atomic number 1?", "Hydrogen"),
    ("What year did the Apollo 11 mission land humans on the Moon?", "1969"),
    ("What is the boiling point of water at sea level in Celsius?", "100 degrees Celsius"),
    ("Who wrote the novel 1984?", "George Orwell"),
    ("What is the square root of 144?", "12"),
    ("Which ocean is the largest?", "The Pacific Ocean"),
    ("What programming language is named after a British comedy group?", "Python (after Monty Python)"),
]

def main():
    prose = []
    for t in TOPICS:
        for attempt in range(3):
            try:
                r = requests.get("https://en.wikipedia.org/w/api.php", params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "redirects": 1, "format": "json", "titles": t},
                    headers={"User-Agent": "specialist-ood-probe/0.1 (kuang2@kuang2.ai)"},
                    timeout=30)
                for p in r.json()["query"]["pages"].values():
                    if "extract" in p and len(p["extract"]) > 1000:
                        prose.append({"title": p["title"], "text": p["extract"][:3200]})
                break
            except (requests.exceptions.JSONDecodeError, requests.RequestException):
                time.sleep(3 * (attempt + 1))
        time.sleep(1.0)
    with open(DATA / "ood_prose.jsonl", "w") as f:
        for x in prose:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(DATA / "ood_qa.jsonl", "w") as f:
        for q, a in GENERAL_QA:
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
    print(f"ood_prose: {len(prose)}, ood_qa: {len(GENERAL_QA)}")

if __name__ == "__main__":
    main()
