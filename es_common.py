"""Shared helpers for the EGGROLL-on-SGLang ES trainer."""
import json
from concurrent.futures import ThreadPoolExecutor
import requests


def centered_ranks(values):
    """Map fitness values to [-0.5, 0.5], assigning ties their mean rank.

    Giving equal rewards different ranks creates an order-dependent ES update.
    This matters for sparse and discrete rewards, where whole populations can
    tie.  In particular, an all-tie generation must have exactly zero utility.
    """
    count = len(values)
    if count <= 1:
        return [0.0] * count

    order = sorted(range(count), key=values.__getitem__)
    ranks = [0.0] * count
    start = 0
    while start < count:
        stop = start + 1
        value = values[order[start]]
        while stop < count and values[order[stop]] == value:
            stop += 1
        mean_ordinal_rank = (start + stop - 1) / 2
        utility = mean_ordinal_rank / (count - 1) - 0.5
        for position in range(start, stop):
            ranks[order[position]] = utility
        start = stop
    return ranks


def zscore(values):
    """Population z-scores; a constant population maps exactly to zero."""
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0:
        return [0.0] * len(values)
    stddev = variance ** 0.5
    return [(value - mean) / stddev for value in values]


def perturb(port, ops, sigma, rank=4, mode="set", include_regex=None,
            flush_cache=True, timeout=600):
    """ops: list of (seed, coeff). mode='set' perturbs from base (empty ops =
    reset to base); mode='commit' permanently updates the base. rank<=0 =
    full-rank i.i.d. Gaussian noise. include_regex restricts perturbed units
    (and, on the first call of a run, which units get fp32 masters)."""
    body = {"ops": [[int(s), float(c)] for s, c in ops],
            "sigma": sigma, "rank": rank, "mode": mode,
            "flush_cache": flush_cache}
    if include_regex is not None:
        body["include_regex"] = include_regex
    r = requests.post(
        f"http://127.0.0.1:{port}/perturb_weights", json=body, timeout=timeout)
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(f"perturb failed on :{port}: {j}")
    return j["num_units"]


def perturb_info(port, mode="state", include_regex=None, timeout=1200):
    """Read a server's locked target manifest and/or authoritative digest."""
    if mode not in {"manifest", "master_digest", "serving_digest", "state"}:
        raise ValueError(f"unsupported introspection mode: {mode}")
    body = {"ops": [], "sigma": 0.0, "rank": 1, "mode": mode,
            "flush_cache": False}
    if include_regex is not None:
        body["include_regex"] = include_regex
    response = requests.post(
        f"http://127.0.0.1:{port}/perturb_weights", json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"perturb {mode} failed on :{port}: {payload}")
    try:
        return json.loads(payload["message"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"perturb {mode} returned malformed payload on :{port}: {payload}") from exc


def verify_replica_state(ports, include_regex=None, verify_serving=False):
    """Require identical targets and FP32 masters on every population replica."""
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        states = list(pool.map(
            lambda port: perturb_info(port, "state", include_regex), ports))
    expected_manifest = states[0]["manifest"]["target_manifest_hash"]
    expected_master = states[0]["master_digest"]["digest"]
    expected_serving = states[0]["serving_digest"]["digest"]
    for port, state in zip(ports, states):
        manifest = state["manifest"]
        if manifest["target_manifest_hash"] != expected_manifest:
            raise RuntimeError(
                f"replica :{port} target manifest differs from :{ports[0]}")
        if state["master_digest"]["digest"] != expected_master:
            raise RuntimeError(
                f"replica :{port} FP32 master differs from :{ports[0]}")
        if verify_serving and state["serving_digest"]["digest"] != expected_serving:
            raise RuntimeError(
                f"replica :{port} serving weights differ from :{ports[0]}")
    return {
        "num_replicas": len(states),
        "num_units": states[0]["manifest"]["num_units"],
        "total_parameters": states[0]["manifest"]["total_parameters"],
        "target_manifest_hash": expected_manifest,
        "master_digest": expected_master,
        "serving_digest": expected_serving if verify_serving else None,
    }


def score_texts(port, texts, timeout=600, warmup=True):
    """Mean per-token logprob for each text (prefill-only scoring).

    The first request after a weight-perturbation control op takes a "cold"
    scheduler path with slightly different numerics; a throwaway warmup
    request first keeps all real scores on the deterministic warm path.
    """
    if warmup:
        score_texts(port, ["warmup"], timeout=timeout, warmup=False)
    # Input-logprob scoring materializes [tokens, vocab] logits; with a 248k
    # vocab that is ~0.4 GB per 800-token text. Keep batches small.
    BATCH = 4
    if len(texts) > BATCH:
        out = []
        for i in range(0, len(texts), BATCH):
            out += score_texts(port, texts[i:i + BATCH], timeout=timeout, warmup=False)
        return out
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={"text": texts,
              "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
              "return_logprob": True, "logprob_start_len": 0},
        timeout=timeout)
    r.raise_for_status()
    out = []
    for item in r.json():
        lps = [t[0] for t in item["meta_info"]["input_token_logprobs"] if t[0] is not None]
        out.append(sum(lps) / max(len(lps), 1))
    return out


def flush_cache(port):
    requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=60)


def save_masters(port, out_path, timeout=1200):
    """Dump the server's authoritative FP32 ES masters to safetensors."""
    r = requests.post(
        f"http://127.0.0.1:{port}/perturb_weights",
        json={"ops": [], "sigma": 0.0, "rank": 1, "mode": "save",
              "include_regex": out_path, "flush_cache": False},
        timeout=timeout)
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(f"save_masters failed: {j}")
    return j["num_units"]
