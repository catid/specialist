#!/usr/bin/env python
"""Knowledge-retrieval benchmark: quiz a model server on eval_qa.jsonl.

Usage:
  python run_eval.py --answer-port 30001 --judge-port 30002 --tag base
Answers are generated closed-book (no excerpt). An LLM judge on judge-port
grades each answer against the reference (and source excerpt) YES/NO.
"""
import argparse, json, re, sys, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/home/catid/specialist/data")

JUDGE = """You are grading a quiz answer about rope bondage education (technique, safety, history, equipment).

Question: {q}
Reference answer: {ref}
Source excerpt the question was drawn from:
{excerpt}

Candidate answer: {cand}

Does the candidate answer state the same essential fact as the reference answer? Minor wording differences, extra detail, or partial-credit hedging are fine as long as the key fact is correct. Reply with exactly one word: YES or NO."""


def chat(port, msg, max_tokens=250, temperature=0.0):
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-port", type=int, required=True)
    ap.add_argument("--judge-port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qas = [json.loads(l) for l in open(DATA / "eval_qa.jsonl")]
    pool = ThreadPoolExecutor(max_workers=8)

    def answer(qa):
        prompt = (f"Answer this question about rope bondage briefly and factually "
                  f"(one sentence):\n\n{qa['question']}")
        return chat(args.answer_port, prompt)

    cands = list(pool.map(answer, qas))

    def judge(pair):
        qa, cand = pair
        v = chat(args.judge_port, JUDGE.format(
            q=qa["question"], ref=qa["answer"],
            excerpt=qa["excerpt"][:2000], cand=cand[:600]), max_tokens=8)
        return bool(re.match(r"\s*YES", v, re.I))

    verdicts = list(pool.map(judge, zip(qas, cands)))

    out = args.out or str(DATA / f"eval_results_{args.tag}.jsonl")
    with open(out, "w") as f:
        for qa, cand, ok in zip(qas, cands, verdicts):
            f.write(json.dumps({"split": qa["split"], "question": qa["question"],
                                "reference": qa["answer"], "candidate": cand,
                                "correct": ok}, ensure_ascii=False) + "\n")
    for split in ("train", "heldout", None):
        sel = [ok for qa, ok in zip(qas, verdicts) if split is None or qa["split"] == split]
        name = split or "overall"
        print(f"{args.tag} {name}: {sum(sel)}/{len(sel)} = {sum(sel)/max(len(sel),1):.1%}")


if __name__ == "__main__":
    main()
