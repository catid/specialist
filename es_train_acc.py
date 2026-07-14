#!/usr/bin/env python
"""ES fine-tuning with ACCURACY fitness: each member is scored by token-F1 of
its greedy raw-format answers on a QA minibatch — the actual task metric,
not answer-string likelihood (which run 1-3 showed rewards format blur).
"""
import argparse, hashlib, json, os, random, re, time, sys, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, "/home/catid/specialist")
from es_common import centered_ranks, perturb, verify_replica_state, zscore
from es_layer_plan import DEFAULT_MODEL, PLANS, GROUPS, plan_manifest
from qa_quality import qa_pair_from_record

DATA = Path("/home/catid/specialist/data")
JOURNAL_SCHEMA_VERSION = 2
RNG_SCHEME = "sha256-generation-v1/python-random-v2"

# Prompting notes (hard-won): the model is a thinking-mode chat model.
# - Raw "Answer:" prompts open with <think> blocks; a prefilled empty
#   think block forces direct answers.
# - Stop strings containing newlines instant-match leading blank lines and
#   yield empty answers; generate freely and parse instead.
PROMPT_TPL = "Question: {q}\n<think>\n\n</think>\nAnswer:"


def generation_seed(global_seed, generation):
    """Derive an independent, stable RNG seed for one ES generation."""
    if generation < 0:
        raise ValueError("generation must be non-negative")
    material = f"specialist-es-generation-v1\0{int(global_seed)}\0{generation}".encode()
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def generation_rng(global_seed, generation):
    """Return the generation-local RNG used for both batches and ES seeds."""
    return random.Random(generation_seed(global_seed, generation))


def file_identity(path, items):
    """Content identity recorded in every journal row for reproducibility."""
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size, "items": items}


def load_journal(path):
    """Load a committed journal and reject gaps/duplicates before resuming."""
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open() as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid journal JSON on line {line_number}: {exc}") from exc
            expected = len(rows)
            if row.get("gen") != expected:
                raise ValueError(
                    f"journal line {line_number} has generation {row.get('gen')!r}; "
                    f"expected {expected}")
            rows.append(row)
    return rows


def validate_resume(rows, expected):
    """Reject journals produced by a different run before touching a server."""
    for index, row in enumerate(rows):
        if row.get("schema_version") != JOURNAL_SCHEMA_VERSION:
            raise ValueError(
                f"journal generation {index} predates the reproducible schema; "
                "start a new journal or replay it explicitly as a legacy run")
        for key in ("trainer", "include_regex", "shaping", "global_seed",
                    "rng_scheme", "hparams"):
            if row.get(key) != expected[key]:
                raise ValueError(
                    f"journal generation {index} has incompatible {key}: "
                    f"{row.get(key)!r} != {expected[key]!r}")
        for key in ("data", "probe_data"):
            if row.get(key, {}).get("sha256") != expected[key]["sha256"]:
                raise ValueError(
                    f"journal generation {index} has incompatible {key} digest")


def normalize_answer(s):
    """Unicode-stable normalization used by both exact and token-F1 reward."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(re.findall(r"[^\W_]+", s, flags=re.UNICODE))


def toks(s):
    normalized = normalize_answer(s)
    return normalized.split() if normalized else []


def f1(pred, ref):
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def answer_score(pred, ref, exact_weight=0.7):
    """Exact correctness plus bounded dense token-overlap credit."""
    exact = float(bool(normalize_answer(ref)) and
                  normalize_answer(pred) == normalize_answer(ref))
    return exact_weight * exact + (1.0 - exact_weight) * f1(pred, ref)


def shape_fitness(values, shaping):
    """Transform rewards without silently amplifying sparse raw signals."""
    if shaping == "raw":
        return list(values)
    if shaping == "zscore":
        return zscore(values)
    if shaping == "ranks":
        return centered_ranks(values)
    raise ValueError(f"unknown shaping mode: {shaping}")


def batch_indices_without_replacement(population, batch_size, global_seed,
                                      generation):
    """Deterministic shuffled stream that covers every fact before reuse."""
    if population <= 0:
        raise ValueError("training dataset has no parseable QA records")
    batch_size = min(batch_size, population)
    absolute = generation * batch_size
    result = []
    while len(result) < batch_size:
        epoch, offset = divmod(absolute, population)
        rng = random.Random(generation_seed(global_seed ^ 0xD47A5E7, epoch))
        order = list(range(population))
        rng.shuffle(order)
        take = min(batch_size - len(result), population - offset)
        result.extend(order[offset:offset + take])
        absolute += take
    return result


def gen_answers(port, questions, timeout=600, max_new_tokens=64):
    r = requests.post(f"http://127.0.0.1:{port}/generate", json={
        "text": [PROMPT_TPL.format(q=q) for q in questions],
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0}},
        timeout=timeout)
    r.raise_for_status()
    out = []
    for item in r.json():
        txt = re.sub(r"<think>.*?(</think>|$)", " ", item["text"], flags=re.S)
        ans = ""
        for line in txt.split("\n"):
            line = line.strip()
            if line and not line.startswith(("Question:", "Q:")):
                ans = line
                break
        out.append(ans)
    return out


def warmup(port):
    """Exercise the post-perturbation cold path without decoding 64 tokens."""
    gen_answers(port, ["warmup"], max_new_tokens=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", type=int, nargs="+", default=[30001, 30002, 30003, 30004])
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--qa-per-eval", type=int, default=48)
    ap.add_argument("--exact-weight", type=float, default=0.7,
                    help="reward weight on normalized exact match; remainder is token-F1")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--verify-every", type=int, default=1,
                    help="compare FP32-master and serving digests every N commits; 0 disables")
    ap.add_argument("--journal", default="/home/catid/specialist/es_journal4.jsonl")
    ap.add_argument(
        "--data", default=str(DATA / "train_qa_verified_leakfree_v2.jsonl"),
        help="leakage-gated QA JSONL (defaults to the deduplicated v2 build)",
    )
    ap.add_argument(
        "--eval-file", default=str(DATA / "eval_qa_v2.jsonl"),
        help="development probe JSONL; do not treat it as a final test set",
    )
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--include-regex", default=None,
                        help="restrict perturbed units with an explicit regex")
    target.add_argument("--layer-plan", choices=sorted(PLANS),
                        help="validated whole-motif location plan")
    ap.add_argument("--unit-groups", nargs="+", choices=sorted(GROUPS),
                    default=["dense"], help="unit groups used with --layer-plan")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL,
                    help="checkpoint config used to validate --layer-plan")
    ap.add_argument(
        "--shaping", choices=["ranks", "raw", "zscore"], default="raw",
        help=("fitness transform; raw is the safe default for sparse accuracy "
              "rewards because ranks/zscore can amplify tiny differences"),
    )
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.include_regex == "":
        ap.error("--include-regex must be non-empty when provided")
    if not 0.0 <= args.exact_weight <= 1.0:
        ap.error("--exact-weight must be in [0, 1]")
    layer_manifest = None
    if args.layer_plan:
        layer_manifest = plan_manifest(
            args.model_path, args.layer_plan, args.unit_groups)
        IR = layer_manifest["include_regex"]
        print(f"layer plan {args.layer_plan}: {layer_manifest['layers']} "
              f"({layer_manifest['num_units']} units, "
              f"sha256 {layer_manifest['plan_sha256'][:12]})", flush=True)
    else:
        IR = args.include_regex

    qa_pairs = []
    seen_pairs = set()
    for line_number, line in enumerate(open(args.data), 1):
        item = json.loads(line)
        try:
            p = qa_pair_from_record(item)
        except ValueError as exc:
            raise ValueError(f"{args.data}:{line_number}: {exc}") from exc
        if p is None:
            raise ValueError(f"{args.data}:{line_number}: unsupported QA serialization")
        key = ((normalize_answer(p[0]), normalize_answer(p[1])) if p else None)
        if p and key not in seen_pairs:
            qa_pairs.append(p)
            seen_pairs.add(key)
    eval_items = [json.loads(l) for l in open(args.eval_file)]
    probe_train = [(x["question"], x["answer"]) for x in eval_items if x["split"] == "train"]
    probe_held = [(x["question"], x["answer"]) for x in eval_items if x["split"] == "heldout"]
    print(f"train QA pairs: {len(qa_pairs)}; probes: {len(probe_train)}/{len(probe_held)}",
          flush=True)

    train_data_identity = file_identity(args.data, len(qa_pairs))
    eval_data_identity = file_identity(args.eval_file, len(eval_items))

    expected_run = {
        "trainer": "es_train_acc",
        "include_regex": IR,
        "shaping": args.shaping,
        "global_seed": args.seed,
        "rng_scheme": RNG_SCHEME,
        "hparams": {
            "pairs": args.pairs,
            "learning_rate": args.lr,
            "noise_sigma": args.sigma,
            "noise_rank": args.rank,
            "qa_per_eval": args.qa_per_eval,
            "exact_weight": args.exact_weight,
            "batch_scheme": "shuffled-without-replacement-v1",
            "layer_plan_sha256": (layer_manifest or {}).get("plan_sha256"),
            "verify_every": args.verify_every,
        },
        "data": train_data_identity,
        "probe_data": eval_data_identity,
    }

    committed = load_journal(args.journal)
    validate_resume(committed, expected_run)
    start_gen = len(committed)
    if committed:
        print(f"resuming after {start_gen} committed generations", flush=True)

    pool = ThreadPoolExecutor(max_workers=len(args.ports))

    def fitness(port, member, batch):
        ops = [member] if member else []
        perturb(port, ops, args.sigma, rank=args.rank, mode="set", include_regex=IR)
        warmup(port)
        preds = gen_answers(port, [q for q, _ in batch])
        return sum(answer_score(p, a, args.exact_weight)
                   for p, (_, a) in zip(preds, batch)) / len(batch)

    def probe_part(port, indexed_items):
        perturb(port, [], args.sigma, rank=args.rank, mode="set", include_regex=IR)
        warmup(port)
        if not indexed_items:
            return []
        preds = gen_answers(port, [item[1][0] for item in indexed_items])
        return [(index, answer_score(pred, item[1], args.exact_weight))
                for (index, item), pred in zip(indexed_items, preds)]

    def probe_all(pset):
        shards = [[] for _ in args.ports]
        for index, item in enumerate(pset):
            shards[index % len(args.ports)].append((index, item))
        futures = [pool.submit(probe_part, port, shard)
                   for port, shard in zip(args.ports, shards)]
        scores = [score for future in futures for _, score in future.result()]
        return sum(scores) / len(scores)

    pt, ph = probe_all(probe_train), probe_all(probe_held)
    initial_state = verify_replica_state(
        args.ports, include_regex=IR, verify_serving=True)
    expected_units = layer_manifest["num_units"] if layer_manifest else None
    if expected_units is not None and initial_state["num_units"] != expected_units:
        raise RuntimeError(
            f"layer plan expected {expected_units} units but servers registered "
            f"{initial_state['num_units']}")
    print(f"verified {initial_state['num_replicas']} replicas: "
          f"{initial_state['num_units']} units, manifest "
          f"{initial_state['target_manifest_hash'][:12]}", flush=True)
    print(
        f"[gen {start_gen}] probe reward train {pt:.4f} heldout {ph:.4f}",
        flush=True,
    )

    for g in range(start_gen, args.gens):
        t0 = time.time()
        rng = generation_rng(args.seed, g)
        batch_indices = batch_indices_without_replacement(
            len(qa_pairs), args.qa_per_eval, args.seed, g)
        batch = [qa_pairs[index] for index in batch_indices]
        seeds = [rng.randrange(1, 2**60) for _ in range(args.pairs)]
        members = [(s, +1.0) for s in seeds] + [(s, -1.0) for s in seeds]
        per_port = {p: [] for p in args.ports}
        # Keep each antithetic (+/-) pair consecutive on the same replica.
        for pair_index in range(args.pairs):
            port = args.ports[pair_index % len(args.ports)]
            per_port[port].append((pair_index, members[pair_index]))
            negative = pair_index + args.pairs
            per_port[port].append((negative, members[negative]))
        results = [None] * len(members)

        def run_port(port):
            return [(i, fitness(port, m, batch)) for i, m in per_port[port]]

        for fut in [pool.submit(run_port, p) for p in args.ports]:
            for i, f in fut.result():
                results[i] = f

        f_tilde = shape_fitness(results, args.shaping)
        coeffs = [(s, args.lr * (f_tilde[j] - f_tilde[j + args.pairs]) /
                   (args.pairs * args.sigma)) for j, s in enumerate(seeds)]
        informative_pairs = sum(
            results[j] != results[j + args.pairs]
            for j in range(args.pairs)
        )
        max_abs_coefficient = max((abs(coeff) for _, coeff in coeffs),
                                  default=0.0)
        list(pool.map(lambda p: perturb(p, coeffs, args.sigma, rank=args.rank,
                                        mode="commit", include_regex=IR), args.ports))
        replica_state = None
        if args.verify_every and (g + 1) % args.verify_every == 0:
            replica_state = verify_replica_state(
                args.ports, include_regex=IR, verify_serving=True)
        with open(args.journal, "a") as jf:
            jf.write(json.dumps({
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "trainer": "es_train_acc",
                "gen": g,
                "ops": [[s, c] for s, c in coeffs],
                "sigma": args.sigma,
                "rank": args.rank,
                "include_regex": IR,
                "layer_plan": layer_manifest,
                "shaping": args.shaping,
                "global_seed": args.seed,
                "rng_scheme": RNG_SCHEME,
                "rng_generation_seed": f"{generation_seed(args.seed, g):064x}",
                "batch_indices": batch_indices,
                "hparams": expected_run["hparams"],
                "data": train_data_identity,
                "probe_data": eval_data_identity,
                "fitness": results,
                "fitness_diagnostics": {
                    "informative_pairs": informative_pairs,
                    "unique_member_fitness": len(set(results)),
                    "max_abs_commit_coefficient": max_abs_coefficient,
                },
                "replica_state": replica_state,
                "mean_fit": sum(results) / len(results),
                "max_fit": max(results),
                "min_fit": min(results),
            }, sort_keys=True) + "\n")
            jf.flush()
            os.fsync(jf.fileno())
        msg = (f"[gen {g}] reward mean {sum(results)/len(results):.4f} "
               f"max {max(results):.4f} min {min(results):.4f} "
               f"informative_pairs {informative_pairs}/{args.pairs} "
               f"max_abs_coeff {max_abs_coefficient:.6g} "
               f"({time.time()-t0:.0f}s)")
        if (g + 1) % args.eval_every == 0:
            pt, ph = probe_all(probe_train), probe_all(probe_held)
            msg += (
                f" | probe_train_reward {pt:.4f} "
                f"probe_held_reward {ph:.4f}"
            )
        print(msg, flush=True)


if __name__ == "__main__":
    main()
