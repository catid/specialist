#!/usr/bin/env python
"""Generate VERIFIED QA training data: low-temperature generation + per-chunk
grounding verification. Only pairs the verifier confirms are (a) answerable
from the exact source chunk and (b) self-contained survive.

Usage: python gen_qa_verified.py <gen_port> [verify_port]
"""
import json, random, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/catid/specialist")
from qa_quality import EvalFact, LOW_VALUE, leakage_reason

DATA = Path("/home/catid/specialist/data")
GEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30004
VER_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else GEN_PORT

GEN_PROMPT = """From the excerpt below, write 3 DIFFERENT factual quiz questions, each about a distinct specific fact stated verbatim in the excerpt (names, techniques, anatomy, safety numbers, materials, history). Rules:
- Each question must be fully self-contained: never refer to "the excerpt/text/author/article".
- The answer must be a short phrase copied or minimally paraphrased from the excerpt.
Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}, ...]

Excerpt:
{excerpt}"""

VERIFY_PROMPT = """Excerpt:
{excerpt}

Question: {q}
Proposed answer: {a}

Two checks:
1. Is the proposed answer clearly supported by the excerpt above?
2. Is the question understandable on its own, without seeing the excerpt (no references to "the excerpt/text/author")?

Reply exactly "YES" only if BOTH hold, otherwise "NO"."""

TEMPLATES = [
    "Question: {q}\nAnswer: {a}",
    "Q: {q}\nA: {a}",
    "Answer this question about rope bondage briefly and factually (one sentence):\n\n{q}\n\n{a}",
]

def chat(port, msg, max_tokens=600, temperature=0.0):
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def norm_tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def main():
    chunks = [json.loads(l) for l in open(DATA / "train_chunks.jsonl")]
    eval_items = []
    for eval_name in ("eval_qa.jsonl", "eval_qa_v2.jsonl"):
        eval_items.extend(json.loads(line) for line in open(DATA / eval_name))
    eval_facts = [EvalFact(item["question"], item["answer"],
                           split=item.get("split", ""))
                  for item in eval_items]
    rng = random.Random(777)

    def process(chunk):
        ex = chunk["text"][:2800]
        out = []
        try:
            m = re.search(r"\[.*\]", chat(GEN_PORT, GEN_PROMPT.format(excerpt=ex),
                                          temperature=0.2), re.S)
            pairs = json.loads(m.group(0))[:4]
        except Exception:
            return out
        for j in pairs:
            try:
                q, a = str(j["question"]).strip(), str(j["answer"]).strip()
            except Exception:
                continue
            if not (10 < len(q) < 300 and 0 < len(a) < 250):
                continue
            if LOW_VALUE.search(q) or leakage_reason(q, a, eval_facts):
                continue
            try:
                v = chat(VER_PORT, VERIFY_PROMPT.format(excerpt=ex, q=q, a=a),
                         max_tokens=5)
            except Exception:
                continue
            if not re.match(r"\s*YES", v, re.I):
                continue
            tpl = rng.choice(TEMPLATES)
            out.append({"source": chunk["source"], "url": chunk["url"],
                        "kind": "qa_verified", "text": tpl.format(q=q, a=a)})
        return out

    pool = ThreadPoolExecutor(max_workers=16)
    results = list(pool.map(process, chunks))
    items = [x for r in results for x in r]
    with open(DATA / "train_qa_verified.jsonl", "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"verified QA items: {len(items)} (from {len(chunks)} chunks)")

if __name__ == "__main__":
    main()
