#!/usr/bin/env python
"""Eval-v2: ~300 verified, self-contained QA items (n=100 was +-4pt noise).

Every candidate question passes two verifier gates (grounding in its exact
source chunk, and self-containment) before acceptance. Excerpt-referential
phrasings are also regex-rejected outright.
"""
import json, random, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/home/catid/specialist/data")
PORTS = [int(x) for x in sys.argv[1:]] or [30001, 30004]
TARGET = {"train": 180, "heldout": 120}

GEN = """From the excerpt below, write 3 DIFFERENT factual quiz questions about specific facts stated in the excerpt (names, techniques, anatomy, safety guidance, materials, measurements, history). Rules:
- Each question must be fully self-contained and understandable by someone who has never seen this excerpt. NEVER use phrases like "the excerpt", "the text", "the author", "the book project", "this article".
- Each answer must be a short phrase stated in the excerpt.
Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}, ...]

Excerpt:
{excerpt}"""

VERIFY = """Excerpt:
{excerpt}

Question: {q}
Proposed answer: {a}

Checks:
1. Is the answer clearly supported by the excerpt?
2. Is the question self-contained (no reference to "the excerpt/text/author/article/book/project", understandable standalone)?
3. Is the question specific to this domain (rope bondage / shibari / kinbaku / safety / rope craft) rather than generic?

Reply exactly "YES" only if ALL three hold, otherwise "NO"."""

BAD = re.compile(r"the (excerpt|text|author|article|passage|book|project|post|interview|chapter)\b", re.I)

def chat(port, msg, max_tokens=600, temperature=0.4):
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def norm(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def main():
    train_chunks = [json.loads(l) for l in open(DATA / "train_chunks.jsonl")]
    held_docs = [json.loads(l) for l in open(DATA / "heldout_docs.jsonl")]
    rng = random.Random(2024)
    rng.shuffle(train_chunks); rng.shuffle(held_docs)
    pools = {"train": train_chunks, "heldout": held_docs}
    out, seen_q = [], []

    def make(args):
        i, split, doc = args
        port = PORTS[i % len(PORTS)]
        ex = doc["text"][:2800]
        got = []
        try:
            m = re.search(r"\[.*\]", chat(port, GEN.format(excerpt=ex)), re.S)
            for j in json.loads(m.group(0))[:3]:
                q, a = str(j["question"]).strip(), str(j["answer"]).strip()
                if not (15 < len(q) < 300 and 0 < len(a) < 200) or BAD.search(q):
                    continue
                v = chat(port, VERIFY.format(excerpt=ex, q=q, a=a), max_tokens=5,
                         temperature=0.0)
                if not re.match(r"\s*YES", v, re.I):
                    continue
                got.append({"split": split, "question": q, "answer": a,
                            "excerpt": ex, "url": doc.get("url", "")})
        except Exception:
            pass
        return got

    pool = ThreadPoolExecutor(max_workers=16)
    for split in ("train", "heldout"):
        docs = pools[split]
        idx = 0
        while sum(1 for o in out if o["split"] == split) < TARGET[split] and idx < len(docs):
            batch = [(i, split, d) for i, d in enumerate(docs[idx:idx + 40])]
            idx += 40
            for got in pool.map(make, batch):
                for item in got:
                    qn = norm(item["question"])
                    if any(len(qn & s) / max(len(qn | s), 1) > 0.65 for s in seen_q):
                        continue
                    seen_q.append(qn)
                    out.append(item)
        print(f"{split}: {sum(1 for o in out if o['split']==split)} items", flush=True)

    with open(DATA / "eval_qa_v2.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"total: {len(out)}")

if __name__ == "__main__":
    main()
