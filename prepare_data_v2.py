#!/usr/bin/env python
"""Rebuild train chunks with the enlarged corpus, FREEZING the original
heldout split (eval-v2's heldout questions come from those docs; new docs
must never enter heldout or the eval goes stale)."""
import json, glob, hashlib
from pathlib import Path

import sys
sys.path.insert(0, "/home/catid/specialist")
from prepare_data import clean, chunks, MIN_DOC_CHARS

RAW = Path("/home/catid/specialist/data/raw")
OUT = Path("/home/catid/specialist/data")

def main():
    frozen_heldout_urls = {json.loads(l)["url"] for l in open(OUT / "heldout_docs.jsonl")}
    docs, seen = [], set()
    for f in sorted(glob.glob(str(RAW / "*.json"))):
        d = json.load(open(f))
        t = clean(d["text"])
        h = hashlib.sha1(t[:2000].encode()).hexdigest()
        if len(t) < MIN_DOC_CHARS or h in seen:
            continue
        seen.add(h)
        d["text"] = t
        docs.append(d)
    train = [d for d in docs if d["url"] not in frozen_heldout_urls]
    n_chunks = 0
    with open(OUT / "train_chunks_v2.jsonl", "w") as f:
        for d in train:
            for c in chunks(d["text"]):
                f.write(json.dumps({"source": d["source"], "url": d["url"],
                                    "title": d.get("title", ""), "text": c},
                                   ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"docs total {len(docs)}, train {len(train)} "
          f"(heldout frozen: {len(frozen_heldout_urls)} urls), "
          f"train chunks: {n_chunks}")

if __name__ == "__main__":
    main()
