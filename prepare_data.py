#!/usr/bin/env python
"""Clean the raw corpus, chunk it, and split train/held-out by document."""
import json, glob, re, random, hashlib
from pathlib import Path

RAW = Path("/home/catid/specialist/data/raw")
OUT = Path("/home/catid/specialist/data")
CHUNK_CHARS = 3200          # ~800 tokens
MIN_DOC_CHARS = 600
HELDOUT_FRAC = 0.10

BOILER = re.compile(
    r"^(share this|leave a comment|posted (on|in)|tags?:|categor(y|ies):|"
    r"previous post|next post|related posts|subscribe|sign up|log ?in|"
    r"copyright|all rights reserved|cookie|privacy policy|read more).*",
    re.I)

def clean(text):
    lines = [l.strip() for l in text.split("\n")]
    keep = [l for l in lines if l and not BOILER.match(l) and len(l) > 2]
    return "\n".join(keep)

def chunks(text):
    paras = text.split("\n")
    cur, out = [], []
    n = 0
    for p in paras:
        cur.append(p)
        n += len(p) + 1
        if n >= CHUNK_CHARS:
            out.append("\n".join(cur))
            cur, n = [], 0
    if n > 800:
        out.append("\n".join(cur))
    return out

def main():
    docs = []
    seen = set()
    for f in sorted(glob.glob(str(RAW / "*.json"))):
        d = json.load(open(f))
        t = clean(d["text"])
        h = hashlib.sha1(t[:2000].encode()).hexdigest()
        if len(t) < MIN_DOC_CHARS or h in seen:
            continue
        seen.add(h)
        d["text"] = t
        docs.append(d)
    rng = random.Random(1234)
    rng.shuffle(docs)
    n_hold = max(int(len(docs) * HELDOUT_FRAC), 20)
    heldout, train = docs[:n_hold], docs[n_hold:]

    with open(OUT / "train_chunks.jsonl", "w") as f:
        n = 0
        for d in train:
            for c in chunks(d["text"]):
                f.write(json.dumps({"source": d["source"], "url": d["url"],
                                    "title": d["title"], "text": c},
                                   ensure_ascii=False) + "\n")
                n += 1
    with open(OUT / "heldout_docs.jsonl", "w") as f:
        for d in heldout:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"docs: {len(docs)} (train {len(train)}, heldout {len(heldout)}); "
          f"train chunks: {n}")

if __name__ == "__main__":
    main()
