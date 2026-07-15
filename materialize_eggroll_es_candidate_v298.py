#!/usr/bin/env python3
"""Materialize the sealed v298 train candidate without opening eval data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import build_curated_qa


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR_V298 = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v298"
).resolve()
OUTPUT_PATH_V298 = OUTPUT_DIR_V298 / "train_qa_context_merit_v298.jsonl"
MANIFEST_PATH_V298 = OUTPUT_DIR_V298 / "manifest.json"
V283_PATH = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v283/"
    "train_qa_context_merit_v283.jsonl"
).resolve()
V283_MANIFEST_PATH = V283_PATH.parent / "manifest.json"
BUILD_CURATED_PATH = (ROOT / "build_curated_qa.py").resolve()

V298_SEAL_COMMIT = "37cb23bfc9db64c30b05972337ee4747425cadd8"
V283_ROWS = 492
V283_SHA256 = (
    "83d14d9d42740c836b49a8ec9e4237766e9d751c827c21d4d2c79500ee4bc3b9"
)
V283_MANIFEST_SHA256 = (
    "014f37177073d5a433b2da2b01298463cc87856f0278a60d66e53a0dce55bbfb"
)
V298_ROWS = 519
V298_SHA256 = (
    "20530df56ea5eca3d0e775f455a1448b91a997f5f0f4c2492868f7ee492b01ff"
)

CURATION_VERSIONS = tuple(range(284, 290))
ADDITION_PATHS = {
    290: "data/manual_reviews/technique_equipment_additions_v1/"
    "pending_additions_technique_equipment_tranche_01_v1.jsonl",
    291: "data/manual_reviews/technique_additions_v2/"
    "pending_additions_technique_tranche_02_v1.jsonl",
    292: "data/manual_reviews/resource_additions_v3/"
    "pending_additions_resource_tranche_03_v1.jsonl",
    293: "data/manual_reviews/technique_equipment_additions_v4/"
    "pending_additions_technique_equipment_tranche_04_v1.jsonl",
    294: "data/manual_reviews/technique_additions_v5/"
    "pending_additions_technique_tranche_05_v1.jsonl",
    295: "data/manual_reviews/resource_additions_v6/"
    "pending_additions_resource_tranche_06_v1.jsonl",
    296: "data/manual_reviews/equipment_additions_v7/"
    "pending_additions_equipment_tranche_07_v1.jsonl",
    297: "data/manual_reviews/technique_additions_v8/"
    "pending_additions_technique_tranche_08_v1.jsonl",
    298: "data/manual_reviews/community_additions_v9/"
    "pending_additions_community_tranche_09_v1.jsonl",
}
CHECKPOINTS = {
    284: (492, "107a43403edbc51a6b2b72f5c964648e865d384fe2156ce12c84b7cbcb63c614"),
    285: (492, "172340fec38e97a8dcf2a70b15157af3a9ce627d359dd780d6b7043324e819ae"),
    286: (492, "338514d23d367ae3a8240ef686b47227e8443e23942120540d9a3085a40c69c2"),
    287: (492, "527ef93fac6f53b1dc6a433523f62244cad9978bf36103bbb56a2f61d9c0a1b5"),
    288: (492, "cdaf70a32714f3a4e248618688fcfc840d10eade0cafca0e1e9c9b15e65ab2c3"),
    289: (492, "17948a72bd383f9445f269567c1fe4964468cecc5892259cb88cfcaf217a5cdb"),
    290: (495, "1afd8517320e5465ad3f52d915bc9391b19ca56a32a2db3ffcd713f88442acf1"),
    291: (498, "ed516dffd88a6300945ead3b83062ca667d1b18977a6c96a8bcb6724880830fa"),
    292: (501, "3a1b3ca8e9ada0be0a8758fd9b9cc4ce01d17164746d587dcb028a0e916a7a17"),
    293: (504, "1913fbab4e3b0638683b44f02682cd9316548750046f13c70bd8abfe7327d960"),
    294: (507, "b9768507b55aa6650f86b1cc4cf849d268f4d0c619f53384b3b74eca2af61c5b"),
    295: (510, "53dfec56416923431838a74914ba3900553aae8cfc23c95d42b8169792c61b1f"),
    296: (513, "9038c7b518e2962083f6579dddd257b9e19f7e00b8119b9f5fd8bc0a1421ae4c"),
    297: (516, "93097a170a71269f46b538fdfdf40a22e9d228795abbf0262deb52cc8aa4535c"),
    298: (V298_ROWS, V298_SHA256),
}

# These are every file hash directly named by the sealed v298 report, plus the
# sealed report, builder, and focused test used to certify that report.
SEAL_REPORTED_COMPONENTS = {
    "data/manual_reviews/context_merit_audit_v298/report_context_merit_v298.json":
    "efe7527ab4a77385368941d2107a059633d9b33d34846393a90d766dfcc4b741",
    "data/manual_reviews/context_merit_audit_v298/context_merit_audit_v298.jsonl":
    "c884261bdfff1fd93b5b0dde53118ff2f0344484664a926ebed20297e9d078ce",
    "data/manual_reviews/context_merit_audit_v298/pending_curation_context_merit_v298.jsonl":
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "data/manual_reviews/context_merit_audit_v298/build_context_merit_audit_v298.py":
    "cc6acef57645451f35ba1690b531483b433433d615ca94319deeac7170543fa1",
    "data/manual_reviews/context_merit_audit_v298/test_context_merit_audit_v298.py":
    "a0a884c1d3426121b3c7e247f0b6d11148df5e6562aecc927312d552d7466595",
    ADDITION_PATHS[298]:
    "9b96a61054a7cc36d1a7ef34b3d5576d476c8c9f9663142ceab14ed40ad2b16e",
    "data/manual_reviews/context_merit_audit_v297/context_merit_audit_v297.jsonl":
    "8512245e221f6cac12265139022c16d357f67df5e229993e0b85d459993d4bd7",
    "data/manual_reviews/context_merit_audit_v297/pending_curation_context_merit_v297.jsonl":
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "data/manual_reviews/context_merit_audit_v297/report_context_merit_v297.json":
    "6994445fb08e026c52b50e4aa908b996ce7e2307a8c07561ab6b556f9b311956",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _relative(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _git_blob(path: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{V298_SEAL_COMMIT}:{_relative(path)}"], cwd=ROOT,
    )


def _verify_sealed_path(path: Path, expected_sha256: str | None = None) -> str:
    current = file_sha256(path)
    committed = hashlib.sha256(_git_blob(path)).hexdigest()
    if current != committed or (
        expected_sha256 is not None and current != expected_sha256
    ):
        raise RuntimeError(f"sealed v298 replay input changed: {_relative(path)}")
    return current


def replay_input_paths_v298() -> tuple[Path, ...]:
    curations = tuple(
        ROOT / f"data/manual_reviews/context_merit_audit_v{version}/"
        f"pending_curation_context_merit_v{version}.jsonl"
        for version in CURATION_VERSIONS
    )
    additions = tuple(ROOT / ADDITION_PATHS[version] for version in range(290, 299))
    return (V283_PATH, V283_MANIFEST_PATH, BUILD_CURATED_PATH, *curations, *additions)


def verify_sealed_inputs_v298() -> dict[str, str]:
    expected = {
        V283_PATH: V283_SHA256,
        V283_MANIFEST_PATH: V283_MANIFEST_SHA256,
    }
    inventory = {
        _relative(path): _verify_sealed_path(path, expected.get(path))
        for path in replay_input_paths_v298()
    }
    for relative, digest in SEAL_REPORTED_COMPONENTS.items():
        path = ROOT / relative
        if _verify_sealed_path(path, digest) != digest:
            raise RuntimeError("unreachable sealed component mismatch")
    return dict(sorted(inventory.items()))


def _row_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for line in source if line.strip())


def build_candidate_bytes_v298() -> tuple[bytes, list[dict]]:
    verify_sealed_inputs_v298()
    checkpoints = []
    OUTPUT_DIR_V298.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v298-train-only-replay-", dir=OUTPUT_DIR_V298.parent,
    ) as temporary:
        directory = Path(temporary)
        current = V283_PATH
        for version in range(284, 299):
            output = directory / f"v{version}.jsonl"
            report = directory / f"v{version}.report.json"
            if version in CURATION_VERSIONS:
                inputs = [current]
                curations = [
                    ROOT / f"data/manual_reviews/context_merit_audit_v{version}/"
                    f"pending_curation_context_merit_v{version}.jsonl"
                ]
            else:
                inputs = [current, ROOT / ADDITION_PATHS[version]]
                curations = []
            # Empty facts is deliberate: all accepted inputs were already sealed
            # against eval. Exact intermediate hashes prove this train-only replay
            # is byte-identical without opening any evaluation file.
            build_curated_qa.merge(inputs, output, report, (), curations)
            observed = (_row_count(output), file_sha256(output))
            if observed != CHECKPOINTS[version]:
                raise RuntimeError(f"train-only v{version} replay checkpoint changed")
            checkpoints.append({
                "version": version,
                "rows": observed[0],
                "sha256": observed[1],
            })
            current = output
        candidate = current.read_bytes()
    if (
        sum(1 for line in candidate.splitlines() if line.strip()) != V298_ROWS
        or hashlib.sha256(candidate).hexdigest() != V298_SHA256
    ):
        raise RuntimeError("v298 final candidate identity changed")
    return candidate, checkpoints


def build_manifest_v298(checkpoints: list[dict]) -> dict:
    replay_inventory = verify_sealed_inputs_v298()
    value = {
        "schema": "eggroll-es-immutable-train-candidate-manifest-v298",
        "status": "materialized_train_only_no_eval_or_runtime_authorization",
        "candidate": {
            "path": str(OUTPUT_PATH_V298),
            "rows": V298_ROWS,
            "file_sha256": V298_SHA256,
        },
        "provenance": {
            "seal_commit": V298_SEAL_COMMIT,
            "starting_candidate": {
                "path": str(V283_PATH),
                "rows": V283_ROWS,
                "file_sha256": V283_SHA256,
                "manifest_path": str(V283_MANIFEST_PATH),
                "manifest_file_sha256": V283_MANIFEST_SHA256,
            },
            "method": (
                "direct_build_curated_qa_merge_train_manual_only_with_empty_"
                "eval_fact_set_and_exact_checkpoint_abort"
            ),
            "curation_versions_in_order": list(CURATION_VERSIONS),
            "addition_versions_in_order": list(range(290, 299)),
            "replay_input_file_sha256": replay_inventory,
            "replay_input_inventory_root_sha256": canonical_sha256(
                replay_inventory
            ),
            "sealed_reported_component_file_sha256": dict(
                sorted(SEAL_REPORTED_COMPONENTS.items())
            ),
            "sealed_reported_component_root_sha256": canonical_sha256(
                dict(sorted(SEAL_REPORTED_COMPONENTS.items()))
            ),
            "checkpoint_chain": checkpoints,
            "checkpoint_chain_root_sha256": canonical_sha256(checkpoints),
        },
        "firewall": {
            "evaluation_paths_passed_to_merge": [],
            "evaluation_facts": "empty_tuple",
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "row_content_in_manifest": False,
            "runtime_launch_authorized": False,
            "model_update_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_manifest_v298(value)
    return value


def validate_manifest_v298(value: dict) -> dict:
    without_self = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("schema")
        != "eggroll-es-immutable-train-candidate-manifest-v298"
        or value.get("candidate", {}).get("rows") != V298_ROWS
        or value.get("candidate", {}).get("file_sha256") != V298_SHA256
        or value.get("provenance", {}).get("seal_commit") != V298_SEAL_COMMIT
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(without_self)
        or value.get("firewall", {}).get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or value.get("firewall", {}).get("runtime_launch_authorized") is not False
        or value.get("firewall", {}).get("model_update_authorized") is not False
    ):
        raise RuntimeError("v298 train-only candidate manifest changed")
    return value


def _exclusive_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"immutable output already exists: {path}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def materialize_v298() -> dict:
    candidate, checkpoints = build_candidate_bytes_v298()
    manifest = build_manifest_v298(checkpoints)
    _exclusive_write_bytes(OUTPUT_PATH_V298, candidate)
    _exclusive_write_bytes(
        MANIFEST_PATH_V298,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V298))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH_V298))
    args = parser.parse_args(argv)
    if (
        Path(args.output).resolve() != OUTPUT_PATH_V298
        or Path(args.manifest).resolve() != MANIFEST_PATH_V298
    ):
        raise ValueError("v298 immutable output path changed")
    manifest = materialize_v298()
    result = {
        "schema": "eggroll-es-v298-train-only-materialization",
        "candidate_path": str(OUTPUT_PATH_V298),
        "candidate_rows": V298_ROWS,
        "candidate_file_sha256": file_sha256(OUTPUT_PATH_V298),
        "manifest_path": str(MANIFEST_PATH_V298),
        "manifest_file_sha256": file_sha256(MANIFEST_PATH_V298),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "evaluation_content_opened": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
