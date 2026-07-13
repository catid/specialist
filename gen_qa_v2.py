#!/usr/bin/env python
"""Scaled verified QA training data: 3 themed passes over every chunk,
per-pair verification, dedupe vs eval-v2 and intra-set. Also emits a
chat-format twin for every accepted pair (so gains land in the deployed
chat pathway too).
"""
import json, random, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/home/catid/specialist/data")
PORTS = [int(x) for x in sys.argv[1:]] or [30001, 30004]

PASSES = [
    "names of people, techniques, ties, styles, or organizations",
    "safety guidance, anatomy, nerves, risks, and numbers or measurements",
    "history, culture, materials, rope craft, and terminology",
]

GEN = """From the excerpt below, write 3 DIFFERENT factual quiz questions focused on {theme}, each about a distinct specific fact stated in the excerpt. Rules:
- Fully self-contained questions (never reference "the excerpt/text/author").
- Answers are short phrases stated in the excerpt.
Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}, ...]

Excerpt:
{excerpt}"""

VERIFY = """Excerpt:
{excerpt}

Question: {q}
Proposed answer: {a}

Is the answer clearly supported by the excerpt AND is the question self-contained (no reference to the excerpt/text/author)? Reply exactly "YES" or "NO"."""

RAW_TPLS = ["Question: {q}\nAnswer: {a}", "Q: {q}\nA: {a}"]
CHAT_TPL = ("<|im_start|>user\nAnswer this question about rope bondage briefly "
            "and factually (one sentence):\n\n{q}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n{a}<|im_end|>")

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
    chunks = [json.loads(l) for l in open(DATA / "train_chunks.jsonl")]
    eval_qs = [norm(json.loads(l)["question"]) for l in open(DATA / "eval_qa_v2.jsonl")]
    eval_qs += [norm(json.loads(l)["question"]) for l in open(DATA / "eval_qa.jsonl")]
    rng = random.Random(31337)
    seen = []
    items = []

    def process(args):
        i, chunk, theme = args
        port = PORTS[i % len(PORTS)]
        ex = chunk["text"][:2800]
        got = []
        try:
            m = re.search(r"\[.*\]", chat(port, GEN.format(theme=theme, excerpt=ex)), re.S)
            for j in json.loads(m.group(0))[:3]:
                q, a = str(j["question"]).strip(), str(j["answer"]).strip()
                if not (15 < len(q) < 300 and 0 < len(a) < 250):
                    continue
                v = chat(port, VERIFY.format(excerpt=ex, q=q, a=a), max_tokens=5,
                         temperature=0.0)
                if re.match(r"\s*YES", v, re.I):
                    got.append((q, a, chunk))
        except Exception:
            pass
        return got

    pool = ThreadPoolExecutor(max_workers=24)
    for pi, theme in enumerate(PASSES):
        jobs = [(i, c, theme) for i, c in enumerate(chunks)]
        n0 = len(items)
        for got in pool.map(process, jobs):
            for q, a, chunk in got:
                qn = norm(q)
                if any(len(qn & e) / max(len(qn | e), 1) > 0.6 for e in eval_qs):
                    continue
                if any(len(qn & s) / max(len(qn | s), 1) > 0.75 for s in seen):
                    continue
                seen.append(qn)
                items.append({"source": chunk["source"], "url": chunk["url"],
                              "kind": "qa_raw",
                              "text": rng.choice(RAW_TPLS).format(q=q, a=a)})
                items.append({"source": chunk["source"], "url": chunk["url"],
                              "kind": "qa_chat",
                              "text": CHAT_TPL.format(q=q, a=a)})
        print(f"pass {pi+1}/{len(PASSES)} ({theme[:30]}...): +{(len(items)-n0)//2} pairs",
              flush=True)

    with open(DATA / "train_qa_v2.jsonl", "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"total texts: {len(items)} ({len(items)//2} pairs x raw+chat)")

if __name__ == "__main__":
    main()
