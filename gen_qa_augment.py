#!/usr/bin/env python
"""Generate QA-formatted training texts from train chunks (knowledge-injection
augmentation). Output mixes several phrasings; near-duplicates of eval
questions are dropped to keep the benchmark honest."""
import json, random, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/home/catid/specialist/data")
PORTS = [int(x) for x in sys.argv[1:]] or [30003, 30004]
PER_CHUNK = 3

PROMPT = """From the excerpt below, write {n} DIFFERENT factual quiz questions, each about a distinct specific fact stated in the excerpt (names, techniques, anatomy, safety numbers, materials, history). Each question must be self-contained and answerable in one short sentence.

Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}, ...]

Excerpt:
{excerpt}"""

TEMPLATES = [
    "Question: {q}\nAnswer: {a}",
    "Q: {q}\nA: {a}",
    "Answer this question about rope bondage briefly and factually (one sentence):\n\n{q}\n\n{a}",
]

def norm_tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def main():
    chunks = [json.loads(l) for l in open(DATA / "train_chunks.jsonl")]
    eval_qs = [norm_tokens(json.loads(l)["question"]) for l in open(DATA / "eval_qa.jsonl")]
    rng = random.Random(4242)

    def gen(args):
        i, chunk = args
        port = PORTS[i % len(PORTS)]
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
                "model": "x",
                "messages": [{"role": "user", "content": PROMPT.format(n=PER_CHUNK, excerpt=chunk["text"][:2800])}],
                "max_tokens": 700, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
            m = re.search(r"\[.*\]", r.json()["choices"][0]["message"]["content"], re.S)
            out = []
            for j in json.loads(m.group(0))[:PER_CHUNK + 1]:
                q, a = str(j["question"]).strip(), str(j["answer"]).strip()
                if not (10 < len(q) < 300 and 0 < len(a) < 300):
                    continue
                qt = norm_tokens(q)
                if any(len(qt & e) / max(len(qt | e), 1) > 0.6 for e in eval_qs):
                    continue  # too close to an eval question
                tpl = rng.choice(TEMPLATES)
                out.append({"source": chunk["source"], "url": chunk["url"],
                            "kind": "qa", "text": tpl.format(q=q, a=a)})
            return out
        except Exception:
            return []

    pool = ThreadPoolExecutor(max_workers=24)
    results = list(pool.map(gen, enumerate(chunks)))
    qa_items = [x for r in results for x in r]
    with open(DATA / "train_mix.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps({**{k: c[k] for k in ("source", "url", "text")},
                                "kind": "raw"}, ensure_ascii=False) + "\n")
        for x in qa_items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"raw chunks: {len(chunks)}, qa texts: {qa_items and len(qa_items)}; "
          f"total {len(chunks) + len(qa_items)} -> train_mix.jsonl")

if __name__ == "__main__":
    main()
