#!/usr/bin/env python
"""QA training data v3: themed passes + terminology pass + cross-chunk
windows + paraphrase expansion, with double verification.

Improvements over v2:
- 4th pass focused on definitions/terminology.
- Cross-chunk windows (staggered 2-chunk concatenations per URL) so facts
  spanning chunk boundaries are covered.
- Every accepted pair gets 2 question paraphrases (same answer), so each
  fact is seen under multiple phrasings.
- Double verification: two differently-phrased verifier prompts must both
  say YES.
Output: raw + chat format twins, deduped vs both evals and intra-set.
"""
import json, random, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/catid/specialist")
from qa_quality import EvalFact, LOW_VALUE, leakage_reason

DATA = Path("/home/catid/specialist/data")
PORTS = [int(x) for x in sys.argv[1:]] or [30001, 30002, 30004]

PASSES = [
    "names of people, techniques, ties, styles, or organizations",
    "safety guidance, anatomy, nerves, risks, and numbers or measurements",
    "history, culture, materials, rope craft, and terminology",
    "definitions of domain terms and what distinguishes similar concepts",
]

GEN = """From the excerpt below, write 3 DIFFERENT factual quiz questions focused on {theme}, each about a distinct specific fact stated in the excerpt. Rules:
- Fully self-contained questions (never reference "the excerpt/text/author").
- Answers are short phrases stated in the excerpt.
Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}, ...]

Excerpt:
{excerpt}"""

VERIFY_1 = """Excerpt:
{excerpt}

Question: {q}
Proposed answer: {a}

Is the answer clearly supported by the excerpt AND is the question self-contained (no reference to the excerpt/text/author)? Reply exactly "YES" or "NO"."""

VERIFY_2 = """You are fact-checking a quiz item against its source.

SOURCE:
{excerpt}

ITEM: someone claims the answer to "{q}" is "{a}".

Would a careful reader of the source agree the claim is accurate? Answer strictly YES or NO."""

PARA = """Rewrite this quiz question in 2 different ways that keep EXACTLY the same meaning and the same answer. Return ONLY a JSON array of 2 strings.

Question: {q}"""

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


def build_windows(chunks):
    """Per-URL staggered 2-chunk windows to cover cross-chunk facts."""
    by_url = {}
    for c in chunks:
        by_url.setdefault(c["url"], []).append(c)
    wins = []
    for url, cs in by_url.items():
        for i in range(0, len(cs) - 1, 2):
            wins.append({"source": cs[i]["source"], "url": url,
                         "text": cs[i]["text"][-1600:] + "\n" + cs[i + 1]["text"][:1600]})
    return wins


def main():
    chunks = [json.loads(l) for l in open(DATA / "train_chunks_v2.jsonl")]
    windows = build_windows(chunks)
    print(f"chunks: {len(chunks)}, cross-chunk windows: {len(windows)}", flush=True)
    eval_items = []
    for eval_name in ("eval_qa_v2.jsonl", "eval_qa.jsonl"):
        eval_items.extend(json.loads(line) for line in open(DATA / eval_name))
    eval_facts = [EvalFact(item["question"], item["answer"],
                           split=item.get("split", ""))
                  for item in eval_items]
    rng = random.Random(90210)
    seen, items = [], []

    def process(args):
        i, unit, theme = args
        port = PORTS[i % len(PORTS)]
        ex = unit["text"][:3000]
        got = []
        try:
            m = re.search(r"\[.*\]", chat(port, GEN.format(theme=theme, excerpt=ex)), re.S)
            for j in json.loads(m.group(0))[:3]:
                q, a = str(j["question"]).strip(), str(j["answer"]).strip()
                if not (15 < len(q) < 300 and 0 < len(a) < 250):
                    continue
                v1 = chat(port, VERIFY_1.format(excerpt=ex, q=q, a=a), max_tokens=5,
                          temperature=0.0)
                if not re.match(r"\s*YES", v1, re.I):
                    continue
                v2 = chat(port, VERIFY_2.format(excerpt=ex, q=q, a=a), max_tokens=5,
                          temperature=0.0)
                if not re.match(r"\s*YES", v2, re.I):
                    continue
                paras = []
                try:
                    pm = re.search(r"\[.*\]", chat(port, PARA.format(q=q),
                                                   max_tokens=300, temperature=0.5), re.S)
                    paras = [str(x).strip() for x in json.loads(pm.group(0))[:2]
                             if 15 < len(str(x)) < 300]
                except Exception:
                    pass
                got.append((q, a, paras, unit))
        except Exception:
            pass
        return got

    pool = ThreadPoolExecutor(max_workers=24)
    # themed passes over single chunks + one names/safety pass over windows
    jobs_sets = [(chunks, t) for t in PASSES] + [(windows, PASSES[0]), (windows, PASSES[1])]
    for si, (units, theme) in enumerate(jobs_sets):
        jobs = [(i, u, theme) for i, u in enumerate(units)]
        n0 = len(items)
        for got in pool.map(process, jobs):
            for q, a, paras, unit in got:
                # Every paraphrase is a distinct training example and must
                # independently pass leakage and duplicate checks.
                for qq in [q] + paras:
                    qqn = norm(qq)
                    if LOW_VALUE.search(qq) or leakage_reason(qq, a, eval_facts):
                        continue
                    if any(len(qqn & prior) / max(len(qqn | prior), 1) > 0.75
                           for prior in seen):
                        continue
                    seen.append(qqn)
                    items.append({"source": unit["source"], "url": unit["url"],
                                  "kind": "qa_raw",
                                  "text": rng.choice(RAW_TPLS).format(q=qq, a=a)})
                    items.append({"source": unit["source"], "url": unit["url"],
                                  "kind": "qa_chat",
                                  "text": CHAT_TPL.format(q=qq, a=a)})
        print(f"set {si+1}/{len(jobs_sets)}: total texts {len(items)} "
              f"(+{len(items)-n0})", flush=True)
        with open(DATA / "train_qa_v3.jsonl", "w") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"final: {len(items)} texts, {len(seen)} unique facts")

if __name__ == "__main__":
    main()
