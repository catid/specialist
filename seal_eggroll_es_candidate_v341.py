#!/usr/bin/env python3
"""Seal the exact train-only v341 candidate with a byte-only replay firewall."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR_V341 = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v341"
).resolve()
OUTPUT_PATH_V341 = OUTPUT_DIR_V341 / "train_qa_context_merit_v341.jsonl"
MANIFEST_PATH_V341 = OUTPUT_DIR_V341 / "manifest.json"

V341_SOURCE_COMMIT = "162e39408f4af0feee694dd4c128e9bb10dac057"
V341_ROWS = 528
V341_SHA256 = "40f4b73c25cccfddc49da039b40483b469b9858deec9d00ca399cb490f5aa47a"
COLLISION_INPUT_PATH = (ROOT / "data/ood_qa.jsonl").resolve()
COLLISION_INPUT_SHA256 = (
    "fa24ae255394b3010861cc795e1d6654f5dde8689900359f37c24fbe8a9817e1"
)
SOURCE_ARTIFACT_SHA256 = {
    "data/manual_reviews/context_merit_audit_v341/"
    "build_context_merit_audit_v341.py": (
        "6160c439b8d90ba178cd5653cebb76836455a5ba238b1145ebf4124552b49bcd"
    ),
    "data/manual_reviews/context_merit_audit_v341/"
    "context_merit_audit_v341.jsonl": (
        "7b54f2c2de84047316a3baacd97d0135b16ef48bc9de28beef74dc4529d5b4f2"
    ),
    "data/manual_reviews/context_merit_audit_v341/"
    "pending_curation_context_merit_v341.jsonl": (
        "13d6145dd27e032df7bcdc6d9dd53fb678bed4d43a8f3400fa21db639947fe78"
    ),
    "data/manual_reviews/context_merit_audit_v341/"
    "report_context_merit_v341.json": (
        "da2037d2fa3a2ea908572c9a6a3784b01aa31b27f6f62768a398c5213dec7a86"
    ),
    "data/manual_reviews/context_merit_audit_v341/"
    "test_context_merit_audit_v341.py": (
        "6daf84bb1ea166b07045bc31303cd1a94ae52f79e6700acfe77fd8ab40769177"
    ),
}
SOURCE_BUILD_PATH = (
    "data/manual_reviews/context_merit_audit_v341/"
    "build_context_merit_audit_v341.py"
)
SOURCE_REPORT_PATH = (
    "data/manual_reviews/context_merit_audit_v341/"
    "report_context_merit_v341.json"
)


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{V341_SOURCE_COMMIT}:{relative}"], cwd=ROOT
    )


def verify_source_commit_v341() -> dict[str, str]:
    observed = {
        relative: hashlib.sha256(_git_blob(relative)).hexdigest()
        for relative in SOURCE_ARTIFACT_SHA256
    }
    if observed != SOURCE_ARTIFACT_SHA256:
        raise RuntimeError("v341 source-commit artifact identity changed")
    report = json.loads(_git_blob(SOURCE_REPORT_PATH))
    projection = report.get("isolated_build_projection", {})
    sealed_policy = report.get("sealed_evaluation_policy", {})
    if (
        report.get("schema") != "context-merit-audit-report-v341"
        or projection.get("output_rows") != V341_ROWS
        or projection.get("output_sha256") != V341_SHA256
        or projection.get("repeat_dataset_byte_identical") is not True
        or projection.get("repeat_projection_report_byte_identical") is not True
        or sealed_policy.get("manual_worker_opened_eval_or_heldout_content")
        is not False
        or sealed_policy.get("manual_worker_received_eval_or_heldout_content")
        is not False
    ):
        raise RuntimeError("v341 source-commit projection certificate changed")
    if file_sha256(COLLISION_INPUT_PATH) != COLLISION_INPUT_SHA256:
        raise RuntimeError("v341 replay collision input identity changed")
    return observed


def _extract_source_commit(destination: Path) -> None:
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", V341_SOURCE_COMMIT],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(destination)],
        stdin=archive.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    archive.stdout.close()
    archive_returncode = archive.wait()
    if archive_returncode or extract.returncode:
        raise RuntimeError("v341 exact-commit archive extraction failed")


def replay_projection_v341(output_path: Path, timeout_seconds: float = 240.0) -> dict:
    """Replay v341 while exposing only candidate bytes, hashes, and counts.

    The source builder runs in a detached exact-commit tree. Its stdout and
    stderr are discarded, and this process never parses a JSONL record. The
    ephemeral candidate is captured only after its whole-file SHA-256 matches
    the preregistered digest.
    """
    verify_source_commit_v341()
    output_path = Path(output_path).resolve()
    with tempfile.TemporaryDirectory(prefix="eggroll-es-v341-replay-") as raw:
        replay_root = Path(raw)
        _extract_source_commit(replay_root)
        replay_collision = replay_root / COLLISION_INPUT_PATH.relative_to(ROOT)
        replay_collision.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(COLLISION_INPUT_PATH, replay_collision)
        audit_dir = replay_root / Path(SOURCE_BUILD_PATH).parent
        process = subprocess.Popen(
            ["python3", str(replay_root / SOURCE_BUILD_PATH)],
            cwd=replay_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout_seconds
        captured: bytes | None = None
        while process.poll() is None:
            for candidate in audit_dir.glob(".v341-observe-*/out.jsonl"):
                try:
                    raw_candidate = candidate.read_bytes()
                except (FileNotFoundError, OSError):
                    continue
                if hashlib.sha256(raw_candidate).hexdigest() == V341_SHA256:
                    captured = raw_candidate
                    break
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise RuntimeError("v341 firewalled projection replay timed out")
            time.sleep(0.05)
        if process.returncode != 0 or captured is None:
            raise RuntimeError("v341 firewalled projection replay failed closed")
    if (
        hashlib.sha256(captured).hexdigest() != V341_SHA256
        or sum(1 for line in captured.splitlines() if line.strip()) != V341_ROWS
    ):
        raise RuntimeError("v341 replay candidate byte identity changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as error:
        raise RuntimeError("immutable v341 replay output already exists") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(captured)
        output.flush()
        os.fsync(output.fileno())
    return {
        "rows": V341_ROWS,
        "file_sha256": file_sha256(output_path),
        "stdout_exposed": False,
        "stderr_exposed": False,
        "jsonl_rows_parsed_by_seal": False,
    }


def validate_candidate_snapshot_v341() -> None:
    if (
        file_sha256(OUTPUT_PATH_V341) != V341_SHA256
        or sum(1 for line in OUTPUT_PATH_V341.open("rb") if line.strip())
        != V341_ROWS
    ):
        raise RuntimeError("v341 train candidate snapshot identity changed")


def build_manifest_v341() -> dict:
    source_inventory = verify_source_commit_v341()
    validate_candidate_snapshot_v341()
    value = {
        "schema": "eggroll-es-immutable-train-candidate-manifest-v341",
        "status": "sealed_train_only_snapshot_no_runtime_or_evaluation_authority",
        "candidate": {
            "path": str(OUTPUT_PATH_V341),
            "rows": V341_ROWS,
            "file_sha256": V341_SHA256,
        },
        "provenance": {
            "source_commit": V341_SOURCE_COMMIT,
            "retrieval": (
                "detached_exact_commit_projection_replay_with_suppressed_streams_"
                "and_sha256_only_ephemeral_capture"
            ),
            "source_artifact_file_sha256": source_inventory,
            "source_artifact_inventory_sha256": canonical_sha256(source_inventory),
            "collision_input_file_sha256": COLLISION_INPUT_SHA256,
            "repeat_replay_required_to_match_exact_candidate_bytes": True,
            "ongoing_working_tree_curation_used": False,
        },
        "firewall": {
            "automated_projection_collision_filter_read_sealed_content": True,
            "seal_parsed_or_emitted_jsonl_row_content": False,
            "replay_stdout_exposed": False,
            "replay_stderr_exposed": False,
            "candidate_row_content_in_manifest": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "runtime_launch_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_manifest_v341(value)


def validate_manifest_v341(value: dict) -> dict:
    without_self = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    provenance = value.get("provenance", {})
    firewall = value.get("firewall", {})
    if (
        value.get("schema")
        != "eggroll-es-immutable-train-candidate-manifest-v341"
        or value.get("candidate") != {
            "path": str(OUTPUT_PATH_V341),
            "rows": V341_ROWS,
            "file_sha256": V341_SHA256,
        }
        or provenance.get("source_commit") != V341_SOURCE_COMMIT
        or provenance.get("collision_input_file_sha256")
        != COLLISION_INPUT_SHA256
        or provenance.get("repeat_replay_required_to_match_exact_candidate_bytes")
        is not True
        or provenance.get("ongoing_working_tree_curation_used") is not False
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(without_self)
        or firewall.get("seal_parsed_or_emitted_jsonl_row_content") is not False
        or firewall.get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or any(firewall.get(key) is not False for key in (
            "runtime_launch_authorized", "model_update_authorized",
            "checkpoint_write_authorized", "evaluation_authorized",
            "dataset_promotion_authorized",
        ))
    ):
        raise RuntimeError("v341 immutable train candidate manifest changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v341 manifest already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-output", default=str(OUTPUT_PATH_V341))
    parser.add_argument("--manifest-output", default=str(MANIFEST_PATH_V341))
    args = parser.parse_args(argv)
    if (
        Path(args.candidate_output).resolve() != OUTPUT_PATH_V341
        or Path(args.manifest_output).resolve() != MANIFEST_PATH_V341
    ):
        raise ValueError("v341 immutable output path changed")
    replay = replay_projection_v341(OUTPUT_PATH_V341)
    manifest = build_manifest_v341()
    _exclusive_write(MANIFEST_PATH_V341, manifest)
    result = {
        "schema": "eggroll-es-train-candidate-seal-v341",
        "candidate_file_sha256": replay["file_sha256"],
        "candidate_rows": replay["rows"],
        "manifest_file_sha256": file_sha256(MANIFEST_PATH_V341),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "source_commit": V341_SOURCE_COMMIT,
        "replay_stdout_or_stderr_exposed": False,
        "gpu_launched": False,
        "runtime_launch_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
