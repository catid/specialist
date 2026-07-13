#!/usr/bin/env python
"""Generate a knowledge-retrieval QA eval set from the corpus using the base model.

Questions come from train docs (did training inject these facts?) and
held-out docs (domain generalization control). Each item: question, reference
answer, the source excerpt, and split tag.
"""
import json, random, re, sys, requests
from pathlib import Path

DATA = Path("/home/catid/specialist/data")
PORT = 30001
N_TRAIN_Q, N_HELD_Q = 60, 40

PROMPT = """Below is an excerpt from an educational text about rope bondage (shibari/kinbaku), safety, technique, history, or equipment.

Write ONE factual quiz question about a specific, verifiable piece of information stated in the excerpt. The question must be self-contained (understandable without seeing the excerpt — never refer to "the text/author/article") and answerable in at most a short sentence. Prefer domain-specific facts (names, techniques, anatomy, materials, measurements, history) over generic ones.

Return ONLY JSON: {{"question": "...", "answer": "..."}}

Excerpt:
{excerpt}"""


def chat(msg, max_tokens=400):
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def make_qa(excerpt):
    txt = chat(PROMPT.format(excerpt=excerpt[:2800]))
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        q, a = j["question"].strip(), str(j["answer"]).strip()
        if len(q) > 15 and 0 < len(a) < 200:
            return q, a
    except Exception:
        return None
    return None


def main():
    rng = random.Random(99)
    train = [json.loads(l) for l in open(DATA / "train_chunks.jsonl")]
    held = [json.loads(l) for l in open(DATA / "heldout_docs.jsonl")]
    out = []
    for split, pool, n in (("train", train, N_TRAIN_Q), ("heldout", held, N_HELD_Q)):
        rng.shuffle(pool)
        i = 0
        for d in pool:
            if i >= n:
                break
            qa = make_qa(d["text"])
            if qa:
                out.append({"split": split, "question": qa[0], "answer": qa[1],
                            "excerpt": d["text"][:2800], "url": d.get("url", "")})
                i += 1
                print(f"{split} {i}/{n}: {qa[0][:90]}", flush=True)
    with open(DATA / "eval_qa.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} QA items")


if __name__ == "__main__":
    main()
