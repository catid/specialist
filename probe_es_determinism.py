#!/usr/bin/env python3
"""Verify answer-level determinism across repeated four-GPU ES probes."""

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import requests

from es_train_acc import answer_score, gen_answers


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(predictions):
    payload = json.dumps(
        predictions, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_eval(path):
    items = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not all(key in item for key in ("question", "answer", "split")):
                raise ValueError(f"{path}:{line_number}: malformed evaluation row")
            items.append(item)
    if not items:
        raise ValueError("evaluation file is empty")
    return items


def fetch_server_mode(port, timeout=30):
    response = requests.get(f"http://127.0.0.1:{port}/server_info", timeout=timeout)
    response.raise_for_status()
    info = response.json()
    return {
        "port": port,
        "model_path": info.get("model_path"),
        "version": info.get("version"),
        "enable_deterministic_inference": info.get(
            "enable_deterministic_inference"
        ),
        "sampling_backend": info.get("sampling_backend"),
        "attention_backend": info.get("attention_backend"),
        "disable_radix_cache": info.get("disable_radix_cache"),
    }


def verify_server_modes(modes):
    if not modes:
        raise ValueError("no server modes supplied")
    for mode in modes:
        if mode["enable_deterministic_inference"] is not True:
            raise RuntimeError(
                f"port {mode['port']} is not in deterministic inference mode"
            )
        if mode["sampling_backend"] != "pytorch":
            raise RuntimeError(
                f"port {mode['port']} uses unexpected sampling backend "
                f"{mode['sampling_backend']!r}"
            )
    identities = {
        (mode["model_path"], mode["version"], mode["attention_backend"])
        for mode in modes
    }
    if len(identities) != 1:
        raise RuntimeError("servers do not report the same model/runtime identity")


def sharded_predictions(ports, questions, pool):
    shards = [[] for _ in ports]
    for index, question in enumerate(questions):
        shards[index % len(ports)].append((index, question))

    def run(port, shard):
        answers = gen_answers(port, [question for _, question in shard])
        return list(zip((index for index, _ in shard), answers))

    futures = [
        pool.submit(run, port, shard) for port, shard in zip(ports, shards)
    ]
    indexed = [item for future in futures for item in future.result()]
    return [answer for _, answer in sorted(indexed)]


def score_splits(items, predictions):
    totals = {"train": [0.0, 0], "heldout": [0.0, 0]}
    for item, prediction in zip(items, predictions):
        split = item["split"]
        if split not in totals:
            totals[split] = [0.0, 0]
        totals[split][0] += answer_score(prediction, item["answer"])
        totals[split][1] += 1
    return {
        split: total / count if count else None
        for split, (total, count) in totals.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", type=int,
                        default=[30001, 30002, 30003, 30004])
    parser.add_argument("--eval-file", type=Path,
                        default=Path("data/eval_qa_v2.jsonl"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--canary-items", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/es_determinism/probe_v1.json"))
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    if args.canary_items < 1:
        parser.error("--canary-items must be positive")

    items = load_eval(args.eval_file)
    questions = [item["question"] for item in items]
    modes = [fetch_server_mode(port) for port in args.ports]
    verify_server_modes(modes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.ports)) as pool:
        canary = questions[:min(args.canary_items, len(questions))]
        replica_futures = [
            pool.submit(gen_answers, port, canary) for port in args.ports
        ]
        replica_predictions = [future.result() for future in replica_futures]
        replica_hashes = [prediction_sha256(values)
                          for values in replica_predictions]

        trials = []
        for trial_index in range(args.trials):
            started = time.monotonic()
            predictions = sharded_predictions(args.ports, questions, pool)
            trials.append({
                "trial": trial_index,
                "prediction_sha256": prediction_sha256(predictions),
                "scores": score_splits(items, predictions),
                "seconds": time.monotonic() - started,
            })

    repeated_hashes = {trial["prediction_sha256"] for trial in trials}
    report = {
        "schema": "es-determinism-probe-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ports": args.ports,
        "server_modes": modes,
        "evaluation": {
            "path": str(args.eval_file.resolve()),
            "sha256": file_sha256(args.eval_file),
            "items": len(items),
        },
        "replica_canary": {
            "items": len(canary),
            "prediction_sha256_by_port": dict(zip(map(str, args.ports),
                                                   replica_hashes)),
            "identical": len(set(replica_hashes)) == 1,
        },
        "trials": trials,
        "repeated_trials_identical": len(repeated_hashes) == 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["replica_canary"]["identical"]:
        raise SystemExit("replicas produced different canary answers")
    if not report["repeated_trials_identical"]:
        raise SystemExit("repeated sharded probes produced different answers")


if __name__ == "__main__":
    main()
