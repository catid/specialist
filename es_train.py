#!/usr/bin/env python
"""EGGROLL-style ES fine-tuning of Qwen3.6-35B-A3B via modified SGLang.

Algorithm (OpenES with antithetic sampling + centered-rank shaping, low-rank
seed-regenerated noise; after HyperscaleES/EGGROLL):
  each generation:
    minibatch B of train chunks (common random numbers across members)
    for j in 1..P pairs: fitness of theta +/- sigma*eps(seed_j) on B
    f_tilde = centered ranks over the 2P scores
    commit: theta += (lr/(P*sigma)) * sum_j (f~+j - f~-j)/1 * sigma*eps_j
            (as perturb coeffs c_j = lr*(f~+j - f~-j)/(P*sigma))
Population members are distributed across N single-GPU servers; every server
applies the identical commit, so all replicas stay bit-identical.

Journal (es_journal.jsonl): one line per committed generation, including the
target manifest and data/run identities.  Export the live FP32 masters and use
es_masters_to_checkpoint.py for deployment; BF16-per-generation journal replay
is not an exact checkpoint reconstruction path.
"""
import argparse, hashlib, json, os, random, time, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/catid/specialist")
from es_common import centered_ranks, perturb, score_texts, verify_replica_state
from es_layer_plan import DEFAULT_MODEL, GROUPS, PLANS, plan_manifest

DATA = Path("/home/catid/specialist/data")
JOURNAL_SCHEMA_VERSION = 2
RNG_SCHEME = "sha256-generation-v1/python-random-v2"


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
        if row.get("data", {}).get("sha256") != expected["data"]["sha256"]:
            raise ValueError(
                f"journal generation {index} has incompatible data digest")
        recorded_probes = row.get("probe_data", {})
        for name, identity in expected["probe_data"].items():
            if recorded_probes.get(name, {}).get("sha256") != identity["sha256"]:
                raise ValueError(
                    f"journal generation {index} has incompatible {name} digest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", type=int, nargs="+", default=[30001, 30002, 30003, 30004])
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=None, help="default: sigma")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--chunks-per-eval", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--verify-every", type=int, default=1,
                    help="compare FP32-master and serving digests every N commits; 0 disables")
    ap.add_argument("--journal", default="/home/catid/specialist/es_journal.jsonl")
    ap.add_argument("--data", default=str(DATA / "train_chunks.jsonl"))
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--include-regex", default=None,
                        help="restrict perturbed units with an explicit regex")
    target.add_argument("--layer-plan", choices=sorted(PLANS),
                        help="validated whole-motif location plan")
    ap.add_argument("--unit-groups", nargs="+", choices=sorted(GROUPS),
                    default=["dense"], help="unit groups used with --layer-plan")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL,
                    help="checkpoint config used to validate --layer-plan")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.include_regex == "":
        ap.error("--include-regex must be non-empty when provided")
    lr = args.lr if args.lr is not None else args.sigma
    layer_manifest = None
    if args.layer_plan:
        layer_manifest = plan_manifest(
            args.model_path, args.layer_plan, args.unit_groups)
        args.include_regex = layer_manifest["include_regex"]
        print(f"layer plan {args.layer_plan}: {layer_manifest['layers']} "
              f"({layer_manifest['num_units']} units, "
              f"sha256 {layer_manifest['plan_sha256'][:12]})", flush=True)

    heldout_path = DATA / "heldout_docs.jsonl"
    eval_qa_path = DATA / "eval_qa.jsonl"
    ood_prose_path = DATA / "ood_prose.jsonl"
    ood_qa_path = DATA / "ood_qa.jsonl"
    chunks = [json.loads(l)["text"] for l in open(args.data)]
    held = [json.loads(l)["text"][:3200] for l in open(heldout_path)]
    held = held[:24]
    # QA-format probes: eval questions as LM text (scored, never trained).
    # heldout split = generalization; train split = fact memorization.
    qa_probe = [f"Question: {j['question']}\nAnswer: {j['answer']}"
                for l in open(eval_qa_path)
                if (j := json.loads(l))["split"] == "heldout"]
    qa_train_probe = [f"Question: {j['question']}\nAnswer: {j['answer']}"
                      for l in open(eval_qa_path)
                      if (j := json.loads(l))["split"] == "train"]
    # Out-of-domain drift probes (general capability preservation)
    ood_prose = [json.loads(l)["text"] for l in open(ood_prose_path)]
    ood_qa = [f"Question: {j['question']}\nAnswer: {j['answer']}"
              for l in open(ood_qa_path) if (j := json.loads(l))]
    print(f"train chunks: {len(chunks)}, heldout probes: {len(held)}", flush=True)

    train_data_identity = file_identity(args.data, len(chunks))
    probe_data_identity = {
        "heldout_docs": file_identity(heldout_path, len(held)),
        "eval_qa": file_identity(eval_qa_path, len(qa_probe) + len(qa_train_probe)),
        "ood_prose": file_identity(ood_prose_path, len(ood_prose)),
        "ood_qa": file_identity(ood_qa_path, len(ood_qa)),
    }
    expected_run = {
        "trainer": "es_train",
        "include_regex": args.include_regex,
        "shaping": "ranks",
        "global_seed": args.seed,
        "rng_scheme": RNG_SCHEME,
        "hparams": {
            "pairs": args.pairs,
            "learning_rate": lr,
            "noise_sigma": args.sigma,
            "noise_rank": args.rank,
            "chunks_per_eval": args.chunks_per_eval,
            "layer_plan_sha256": (layer_manifest or {}).get("plan_sha256"),
            "verify_every": args.verify_every,
        },
        "data": train_data_identity,
        "probe_data": probe_data_identity,
    }

    # Resume support: skip past generations already in the journal.
    committed = load_journal(args.journal)
    validate_resume(committed, expected_run)
    start_gen = len(committed)
    if committed:
        print(f"resuming after {start_gen} committed generations", flush=True)
        # Journal ops must be re-applied if servers restarted fresh: caller's
        # responsibility (run es_replay_to_servers.py) - here we just continue.

    pool = ThreadPoolExecutor(max_workers=len(args.ports))

    def eval_member(port, seed_coeff, batch):
        if seed_coeff is None:
            perturb(port, [], args.sigma, rank=args.rank, mode="set",
                    include_regex=args.include_regex)
        else:
            perturb(port, [seed_coeff], args.sigma, rank=args.rank, mode="set",
                    include_regex=args.include_regex)
        scores = score_texts(port, batch)
        return sum(scores) / len(scores)

    def mean_score_all(texts):
        shards = [texts[index::len(args.ports)] for index in range(len(args.ports))]
        futures = [pool.submit(eval_member, port, None, shard)
                   for port, shard in zip(args.ports, shards) if shard]
        weighted = [(future.result(), len(shard))
                    for future, shard in zip(futures, [s for s in shards if s])]
        return sum(mean * count for mean, count in weighted) / len(texts)

    def heldout_loss():
        # Shard each probe over all replicas so periodic evaluation stays busy.
        h = mean_score_all(held)
        q = mean_score_all(qa_probe)
        qt = mean_score_all(qa_train_probe)
        op = mean_score_all(ood_prose)
        oq = mean_score_all(ood_qa)
        return h, q, qt, op, oq

    base_h, base_q, base_qt, base_op, base_oq = heldout_loss()
    initial_state = verify_replica_state(
        args.ports, include_regex=args.include_regex, verify_serving=True)
    expected_units = layer_manifest["num_units"] if layer_manifest else None
    if expected_units is not None and initial_state["num_units"] != expected_units:
        raise RuntimeError(
            f"layer plan expected {expected_units} units but servers registered "
            f"{initial_state['num_units']}")
    print(f"verified {initial_state['num_replicas']} replicas: "
          f"{initial_state['num_units']} units, manifest "
          f"{initial_state['target_manifest_hash'][:12]}", flush=True)
    print(f"[gen {start_gen}] heldout mean logprob (base): {base_h:.5f} "
          f"qa_probe: {base_q:.5f} qa_train: {base_qt:.5f} "
          f"ood_prose: {base_op:.5f} ood_qa: {base_oq:.5f}",
          flush=True)

    for g in range(start_gen, args.gens):
        t0 = time.time()
        rng = generation_rng(args.seed, g)
        batch_indices = rng.sample(range(len(chunks)),
                                   min(args.chunks_per_eval, len(chunks)))
        batch = [chunks[index] for index in batch_indices]
        seeds = [rng.randrange(1, 2**60) for _ in range(args.pairs)]
        members = [(s, +1.0) for s in seeds] + [(s, -1.0) for s in seeds]

        # Round-robin members over servers; serial within a server.
        per_port = {p: [] for p in args.ports}
        for pair_index in range(args.pairs):
            port = args.ports[pair_index % len(args.ports)]
            per_port[port].append((pair_index, members[pair_index]))
            negative = pair_index + args.pairs
            per_port[port].append((negative, members[negative]))

        results = [None] * len(members)

        def run_port(port):
            out = []
            for i, m in per_port[port]:
                out.append((i, eval_member(port, m, batch)))
            return out

        for fut in [pool.submit(run_port, p) for p in args.ports]:
            for i, f in fut.result():
                results[i] = f

        f_tilde = centered_ranks(results)
        coeffs = []
        for j, s in enumerate(seeds):
            d = f_tilde[j] - f_tilde[j + args.pairs]
            c = lr * d / (args.pairs * args.sigma)
            coeffs.append((s, c))

        # Commit identical update on every server.
        list(pool.map(lambda p: perturb(p, coeffs, args.sigma, rank=args.rank,
                                        mode="commit", include_regex=args.include_regex),
                      args.ports))
        replica_state = None
        if args.verify_every and (g + 1) % args.verify_every == 0:
            replica_state = verify_replica_state(
                args.ports, include_regex=args.include_regex, verify_serving=True)

        with open(args.journal, "a") as jf:
            jf.write(json.dumps({
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "trainer": "es_train",
                "gen": g,
                "ops": [[s, c] for s, c in coeffs],
                "sigma": args.sigma,
                "rank": args.rank,
                "include_regex": args.include_regex,
                "layer_plan": layer_manifest,
                "shaping": "ranks",
                "global_seed": args.seed,
                "rng_scheme": RNG_SCHEME,
                "rng_generation_seed": f"{generation_seed(args.seed, g):064x}",
                "batch_indices": batch_indices,
                "hparams": expected_run["hparams"],
                "data": train_data_identity,
                "probe_data": probe_data_identity,
                "fitness": results,
                "replica_state": replica_state,
                "mean_fit": sum(results) / len(results),
                "max_fit": max(results), "min_fit": min(results),
            }, sort_keys=True) + "\n")
            jf.flush()
            os.fsync(jf.fileno())

        msg = (f"[gen {g}] fit mean {sum(results)/len(results):.5f} "
               f"max {max(results):.5f} min {min(results):.5f} "
               f"({time.time()-t0:.0f}s)")
        if (g + 1) % args.eval_every == 0:
            h, q, qt, op, oq = heldout_loss()
            msg += (f" | heldout {h:.5f} (start {base_h:.5f})"
                    f" qa_probe {q:.5f} (start {base_q:.5f})"
                    f" qa_train {qt:.5f} (start {base_qt:.5f})"
                    f" ood_prose {op:.5f} (start {base_op:.5f})"
                    f" ood_qa {oq:.5f} (start {base_oq:.5f})")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
