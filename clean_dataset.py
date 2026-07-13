#!/usr/bin/env python
"""Dataset cleanup pass over train_qa_v3:

1. Reconstruct facts (canonical question + paraphrases + answer + url) from
   the generated texts.
2. Rule filters: leftover excerpt-references, degenerate answers, boilerplate
   markers, over-long answers.
3. Full LLM judge pass (one call per fact): clarity / specificity /
   single-answerability / domain relevance / answer plausibility.
4. Conflict pass: near-identical questions with incompatible answers -> keep
   only the judged-KEEP one; drop whole cluster if still ambiguous.
5. Rebuild raw+chat texts for surviving facts -> train_qa_v3_clean.jsonl
"""
import json, random, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/home/catid/specialist/data")
PORTS = [int(x) for x in sys.argv[1:]] or [30001, 30004]

JUDGE = """You are curating a high-quality quiz deck about rope bondage (shibari/kinbaku technique, safety, anatomy, history, materials).

Question: {q}
Answer: {a}

Keep this item only if ALL hold:
1. The question is clear, specific, and has essentially one correct answer.
2. The question is self-contained (no "the excerpt/text/author/article", no ambiguous "this/that").
3. The answer is concise and plausibly correct for the question.
4. The item teaches something about the domain (not website boilerplate, pricing, navigation, or a generic fact unrelated to rope).

Reply exactly KEEP or DROP."""

RAW_TPLS = ["Question: {q}\nAnswer: {a}", "Q: {q}\nA: {a}"]
CHAT_TPL = ("<|im_start|>user\nAnswer this question about rope bondage briefly "
            "and factually (one sentence):\n\n{q}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n{a}<|im_end|>")

BAD_Q = re.compile(r"the (excerpt|text|author|article|passage|book|project|post|"
                   r"interview|chapter|website|page)\b|according to (the|this)", re.I)
BAD_A = re.compile(r"^(yes|no|unknown|not (stated|mentioned|specified)|n/?a)\.?$|"
                   r"add to cart|read more|sold out|click here", re.I)


def parse_qa(text):
    m = re.match(r"(?:Question|Q): (.+?)\n(?:Answer|A): (.+)", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"<\|im_start\|>user\n.*?:\n\n(.+?)<\|im_end\|>.*?</think>\n(.+?)<\|im_end\|>",
                  text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def norm(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def main():
    items = [json.loads(l) for l in open(DATA / "train_qa_v3.jsonl")]
    # Reconstruct facts: consecutive raw texts sharing (url, answer) = one fact
    facts = []  # {url, source, answer, questions: [...]}
    for it in items:
        if it["kind"] != "qa_raw":
            continue
        p = parse_qa(it["text"])
        if not p:
            continue
        q, a = p
        if facts and facts[-1]["url"] == it["url"] and facts[-1]["answer"] == a:
            facts[-1]["questions"].append(q)
        else:
            facts.append({"url": it["url"], "source": it["source"],
                          "answer": a, "questions": [q]})
    print(f"reconstructed facts: {len(facts)}", flush=True)

    # Rule filters
    kept = []
    for f in facts:
        f["questions"] = [q for q in f["questions"]
                          if not BAD_Q.search(q) and 15 < len(q) < 300]
        if not f["questions"]:
            continue
        a = f["answer"]
        if BAD_A.match(a) or not (0 < len(a) < 220):
            continue
        kept.append(f)
    print(f"after rule filters: {len(kept)}", flush=True)

    # LLM judge pass (canonical question)
    def judge(args):
        i, f = args
        port = PORTS[i % len(PORTS)]
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
                "model": "x", "messages": [{"role": "user", "content":
                JUDGE.format(q=f["questions"][0], a=f["answer"])}],
                "max_tokens": 5, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
            return bool(re.match(r"\s*KEEP", r.json()["choices"][0]["message"]["content"], re.I))
        except Exception:
            return False

    pool = ThreadPoolExecutor(max_workers=32)
    verdicts = list(pool.map(judge, enumerate(kept)))
    kept = [f for f, v in zip(kept, verdicts) if v]
    print(f"after judge pass: {len(kept)}", flush=True)

    # Conflict pass: near-identical canonical questions, incompatible answers
    by_key = {}
    for f in kept:
        qn = frozenset(norm(f["questions"][0]))
        placed = False
        for key in list(by_key):
            if len(qn & key) / max(len(qn | key), 1) > 0.8:
                by_key[key].append(f)
                placed = True
                break
        if not placed:
            by_key[qn] = [f]
    final = []
    n_conflict = 0
    for key, group in by_key.items():
        if len(group) == 1:
            final.append(group[0])
            continue
        answers = [norm(g["answer"]) for g in group]
        compat = all(len(answers[0] & a) / max(len(answers[0] | a), 1) > 0.3
                     for a in answers[1:])
        if compat:
            final.append(group[0])  # duplicates: keep one
        else:
            n_conflict += 1  # incompatible answers to the same question: drop all
    print(f"after conflict pass: {len(final)} (dropped {n_conflict} conflicting clusters)",
          flush=True)

    rng = random.Random(555)
    with open(DATA / "train_qa_v3_clean.jsonl", "w") as out:
        n = 0
        for f in final:
            for q in f["questions"]:
                out.write(json.dumps({"source": f["source"], "url": f["url"],
                                      "kind": "qa_raw",
                                      "text": rng.choice(RAW_TPLS).format(q=q, a=f["answer"])},
                                     ensure_ascii=False) + "\n")
                out.write(json.dumps({"source": f["source"], "url": f["url"],
                                      "kind": "qa_chat",
                                      "text": CHAT_TPL.format(q=q, a=f["answer"])},
                                     ensure_ascii=False) + "\n")
                n += 2
    print(f"clean texts: {n}")

if __name__ == "__main__":
    main()
