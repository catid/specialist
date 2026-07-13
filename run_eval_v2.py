#!/usr/bin/env python
"""Eval-v2 runner with reproducible candidates, grader provenance, and CIs.

The benchmark is development-only: earlier training sets leaked some eval-v2
facts. Use ``--reuse-candidates-from`` to send identical saved candidates to
additional judges instead of regenerating answers.

Usage: python run_eval_v2.py --answer-port P --judge-port J --tag NAME [--modes raw chat]
"""
import argparse, hashlib, json, math, re, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/home/catid/specialist/data")
ANSWER_PROTOCOL = "eval-v2-think-bypass-v1"
JUDGE_PROMPT_VERSION = "essential-fact-yes-no-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_id(qa):
    if qa.get("item_id"):
        return qa["item_id"]
    material = "\0".join((qa.get("split", ""), qa["question"], qa["answer"]))
    return "eval-" + hashlib.sha256(material.encode()).hexdigest()[:20]


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius

def raw_answer(q, port):
    r = requests.post(f"http://127.0.0.1:{port}/generate", json={
        "text": f"Question: {q}\n<think>\n\n</think>\nAnswer:",
        "sampling_params": {"max_new_tokens": 80, "temperature": 0.0}}, timeout=300)
    txt = re.sub(r"<think>.*?(</think>|$)", " ", r.json()["text"], flags=re.S)
    for line in txt.split("\n"):
        line = line.strip()
        if line and not line.startswith(("Question:", "Q:")):
            return line
    return ""

def chat_answer(q, port):
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "user", "content":
        f"Answer this question about rope bondage briefly and factually (one sentence):\n\n{q}"}],
        "max_tokens": 250, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    return r.json()["choices"][0]["message"]["content"].strip()

def judge(qa, cand, port):
    if not cand.strip():
        return False, "EMPTY_CANDIDATE"
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "user", "content":
        f"Question: {qa['question']}\nReference answer: {qa['answer']}\n"
        f"Source excerpt:\n{qa['excerpt'][:1500]}\n\nCandidate answer: {cand[:400]}\n\n"
        "Does the candidate state the same essential fact as the reference? "
        "Reply exactly YES or NO."}],
        "max_tokens": 5, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    raw = r.json()["choices"][0]["message"]["content"]
    return bool(re.match(r"\s*YES", raw, re.I)), raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-port", type=int, required=True)
    ap.add_argument("--judge-port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--answer-model", default="unspecified")
    ap.add_argument("--judge-model", default="unspecified")
    ap.add_argument("--reuse-candidates-from",
                    help="tag whose per-mode output supplies frozen candidates")
    ap.add_argument("--modes", nargs="+", default=["raw", "chat"])
    ap.add_argument("--eval-file", default=str(DATA / "eval_qa_v2.jsonl"))
    args = ap.parse_args()

    qas = [json.loads(l) for l in open(args.eval_file)]
    eval_sha256 = sha256_file(args.eval_file)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for mode in args.modes:
            if args.reuse_candidates_from:
                frozen_path = DATA / f"evalv2_{args.reuse_candidates_from}_{mode}.jsonl"
                frozen = [json.loads(line) for line in open(frozen_path)]
                if len(frozen) != len(qas):
                    raise ValueError(f"candidate count mismatch in {frozen_path}")
                if any(row.get("q") != qa["question"] for row, qa in zip(frozen, qas)):
                    raise ValueError(f"candidate questions do not match {args.eval_file}")
                cands = [row["cand"] for row in frozen]
            else:
                fn = raw_answer if mode == "raw" else chat_answer
                cands = list(pool.map(
                    lambda qa: fn(qa["question"], args.answer_port), qas))
            judged = list(pool.map(
                lambda pair: judge(pair[0], pair[1], args.judge_port),
                zip(qas, cands)))
            verdicts = [result[0] for result in judged]
            raw_verdicts = [result[1] for result in judged]
            for split in ("train", "heldout", None):
                selected = [verdict for qa, verdict in zip(qas, verdicts)
                            if split is None or qa["split"] == split]
                low, high = wilson_interval(sum(selected), len(selected))
                print(f"{args.tag} {mode} {split or 'overall'}: "
                      f"{sum(selected)}/{len(selected)} = "
                      f"{sum(selected)/len(selected):.1%} "
                      f"(95% CI {low:.1%}–{high:.1%})", flush=True)
            with open(DATA / f"evalv2_{args.tag}_{mode}.jsonl", "w") as output:
                for qa, cand, verdict, raw_verdict in zip(
                        qas, cands, verdicts, raw_verdicts):
                    output.write(json.dumps({
                        "item_id": item_id(qa),
                        "split": qa["split"],
                        "q": qa["question"],
                        "reference": qa["answer"],
                        "cand": cand,
                        "ok": verdict,
                        "raw_judge_output": raw_verdict,
                        "answer_model": args.answer_model,
                        "answer_port": args.answer_port,
                        "judge_model": args.judge_model,
                        "judge_port": args.judge_port,
                        "answer_protocol": ANSWER_PROTOCOL,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "eval_file": str(Path(args.eval_file).resolve()),
                        "eval_sha256": eval_sha256,
                        "candidates_reused_from": args.reuse_candidates_from,
                    }, ensure_ascii=False, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
