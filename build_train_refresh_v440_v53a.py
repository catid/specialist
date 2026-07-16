#!/usr/bin/env python3
"""Build the train-only v440 fold-3 refresh without protected-data access.

The accepted v431-v440 edits are replayed from the already sealed v430
projection.  Fold membership is inherited from the immutable v412 fold-3
training root identities; edited content is never used to assign a new fold.
Only the requested 531-row train projection and its 448-row training subset
are materialized.  No shadow, OOD, validation, holdout, or benchmark artifact
is opened or written.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import build_curated_qa as curated
import build_train_refresh_v430_v47a as replay
import build_train_shadow_folds_v37a as frozen
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_v37a as objective
from qa_quality import stable_fact_id


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = (
    ROOT / "experiments/sft_controls/v53a_train_refresh_v440_fold3"
).resolve()
PROJECTION = (ARTIFACT_DIR / "train_projection_v440.jsonl").resolve()
TRAIN = (ARTIFACT_DIR / "fold_3_train_v440.jsonl").resolve()
MANIFEST = (ARTIFACT_DIR / "manifest_v53a.json").resolve()
FROZEN_TRAIN = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/fold_3_train.jsonl"
).resolve()
FROZEN_TRAIN_FILE_SHA256 = (
    "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a"
)
REPLAY_V431_V440 = (
    (431, "56d40784197f9c124acc21abcd6313ca55e648a921e95287da0821ca5fade824"),
    (432, "6bb26f0a0ef094168bed42826eadd1dc39dbd891d1ff62798032f16250c8becd"),
    (433, "5d21081e044c3f5a2110212dae19e22dfdc08e4b59d483a134871cbeaa9c268c"),
    (434, "f86f0618b0ac87ffd58b863763fd8d6609179c13dce2b945ddf0b96d75f3c099"),
    (435, "25bc23bf80ed9cd027ead487e41ab8c9364b0b7db9c6de236f54b5128d6b87c8"),
    (436, "9f1b6c4cf21bef4ecbefb33cf58063dcec1ce44f52efa9cfb501a3aa2e9eb7f8"),
    (437, "9183e7422b64678bd5a618b02a5d3ca648bd605d40064f999da2321d6a094385"),
    (438, "4fdfc367a79f238d057116c0995bcc90ba6dd76e4a56ad25b6aadd5beb61d7d4"),
    (439, "8fd31994bf89dfa01dd3f1baae3b93ffa273cba9c325417ff11275eac5e0f8ad"),
    (440, "8988b14443dd3ab51be98be24b1aec6bc82ec4233241f3422cb0887c842af078"),
)
EXPECTED = {
    "projection_rows": 531,
    "projection_sha256": REPLAY_V431_V440[-1][1],
    "accepted_edit_decisions_v413_v440": 81,
    "train_rows": 448,
    "train_conflict_units": 208,
    "train_sha256": "5d3de8f7bf3cfa802837cb65597f6d3bcd5906090fc54729fd1a1ef153c98021",
    "root_membership_sha256": "e0e5eca72438aebc40d915eb824e03c42441c180d6ab585e1b375f83ddec3275",
    "weight_identity_sha256": "2ac1bbf18fcfbb7bedd2fa24b6cd23c25807348f41bfd34f82b12d37eafdf50c",
    "manifest_content_sha256": "2184aac5c49402550d7a2ba9e713c2fb16e5de1421a99305f03eabbb09c51aca",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _curation_path(version: int) -> Path:
    return (
        ROOT / "data/manual_reviews" / f"context_merit_audit_v{version}"
        / f"pending_curation_context_merit_v{version}.jsonl"
    ).resolve()


def replay_projection_v440() -> tuple[bytes, list[dict]]:
    """Replay accepted train curations with the evaluation input list empty."""
    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix=".v53a-v440-train-only-", dir=ROOT) as name:
        directory = Path(name)
        current = replay.PROJECTION
        for version, expected_sha256 in REPLAY_V431_V440:
            output = directory / f"train_projection_v{version}.jsonl"
            report = directory / f"train_projection_v{version}.report.json"
            decision = _curation_path(version)
            curated.merge([current], output, report, [], [decision])
            observed = engine.file_sha256(output)
            if observed != expected_sha256:
                raise RuntimeError(f"V53A v440 train replay drifted at v{version}")
            records.append({
                "version": version,
                "decision_file_sha256": engine.file_sha256(decision),
                "projection_sha256": observed,
            })
            current = output
        payload = current.read_bytes()
    if (
        payload.count(b"\n") != EXPECTED["projection_rows"]
        or hashlib.sha256(payload).hexdigest() != EXPECTED["projection_sha256"]
    ):
        raise RuntimeError("V53A final v440 train projection changed")
    return payload, records


def root_lineage_v440() -> tuple[dict[str, str], int]:
    roots = {row["fact_id"]: row["fact_id"] for row in _read_jsonl(replay.SOURCE)}
    decisions = 0
    for version in [item[0] for item in replay.REPLAY] + [
        item[0] for item in REPLAY_V431_V440
    ]:
        for decision in _read_jsonl(_curation_path(version)):
            if decision.get("action") != "edit":
                raise RuntimeError("V53A accepted lineage unexpectedly adds or drops")
            root = roots.pop(decision["fact_id"])
            successor = stable_fact_id(decision["question"], decision["answer"])
            if successor in roots:
                raise RuntimeError("V53A edited fact lineage collided")
            roots[successor] = root
            decisions += 1
    return roots, decisions


def construct() -> tuple[dict, dict[Path, bytes]]:
    if engine.file_sha256(FROZEN_TRAIN) != FROZEN_TRAIN_FILE_SHA256:
        raise RuntimeError("V53A frozen fold-3 training membership changed")
    first, first_records = replay_projection_v440()
    second, second_records = replay_projection_v440()
    if first != second or first_records != second_records:
        raise RuntimeError("V53A v440 train-only replay is nondeterministic")

    projection = [json.loads(line) for line in first.splitlines() if line]
    roots, decision_count = root_lineage_v440()
    if (
        set(roots) != {row["fact_id"] for row in projection}
        or decision_count != EXPECTED["accepted_edit_decisions_v413_v440"]
    ):
        raise RuntimeError("V53A v440 root lineage coverage changed")

    frozen_train_roots = {row["fact_id"] for row in _read_jsonl(FROZEN_TRAIN)}
    train = [
        row for row in projection
        if roots[row["fact_id"]] in frozen_train_roots
    ]
    selected_roots = {roots[row["fact_id"]] for row in train}
    train_payload = frozen.jsonl_bytes(train)
    units = frozen.build_conflict_units(train)
    _weights, weighting = objective.assign_equal_unit_weights(train)
    observed = {
        "train_rows": len(train),
        "train_conflict_units": len(units),
        "train_sha256": hashlib.sha256(train_payload).hexdigest(),
        "root_membership_sha256": engine.canonical_sha256(sorted(selected_roots)),
        "weight_identity_sha256": weighting["identity_sha256"],
    }
    if (
        selected_roots != frozen_train_roots
        or any(observed[key] != EXPECTED[key] for key in observed)
    ):
        raise RuntimeError("V53A v440 frozen fold-3 training projection changed")

    result = {
        "schema": "specialist-v440-train-only-fold3-refresh-v53a",
        "status": "sealed_v440_projection_fold3_train_unlaunched",
        "projection": {
            "path": str(PROJECTION),
            "rows": EXPECTED["projection_rows"],
            "sha256": EXPECTED["projection_sha256"],
            "repeat_replay_byte_identical": True,
            "replay_v431_v440": first_records,
        },
        "fold_3_train": {
            "path": str(TRAIN),
            "rows": observed["train_rows"],
            "conflict_units": observed["train_conflict_units"],
            "sha256": observed["train_sha256"],
            "root_membership_sha256": observed["root_membership_sha256"],
            "membership_exactly_frozen_v412_fold3_train": True,
            "edited_content_rehashed_for_fold_assignment": False,
            "unique_documents": len({row["document_sha256"] for row in train}),
        },
        "equal_conflict_unit_weighting": weighting,
        "lineage_stability": {
            "accepted_edit_decisions_v413_v440": decision_count,
            "added_rows": 0,
            "dropped_rows": 0,
            "root_membership_exactly_preserved": True,
            "fold_assignment_changes": 0,
        },
        "step_schedule": {
            "world_size": 4,
            "per_device_batch_size": 7,
            "effective_global_batch_size": 28,
            "dataloader_drop_last": True,
            "optimizer_steps_per_epoch": 16,
            "epochs_argument": 3.0,
            "explicit_max_steps": 48,
            "expected_optimizer_steps": 48,
            "examples_emitted_per_epoch": 448,
            "total_examples_emitted": 1344,
            "complete_row_passes": 3.0,
        },
        "access_firewall": {
            "accepted_train_projection_and_curations_opened": True,
            "frozen_fold3_train_membership_opened": True,
            "shadow_artifact_opened": False,
            "shadow_artifact_written": False,
            "eval_ood_holdout_or_benchmark_opened": False,
            "external_metrics_used": False,
            "gpu_accessed": False,
            "training_launched": False,
        },
        "selection_firewall": {
            "training_input": "exact v440 rows in frozen v412 fold-3 train roots",
            "shadow_ood_holdout_feedback_authorized": False,
            "post_training_evaluation_authorized": False,
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    expected_content = EXPECTED["manifest_content_sha256"]
    if expected_content != "PENDING" and result[
        "content_sha256_before_self_field"
    ] != expected_content:
        raise RuntimeError("V53A refresh manifest identity changed")
    manifest_payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return result, {
        PROJECTION: first,
        TRAIN: train_payload,
        MANIFEST: manifest_payload,
    }


def main() -> None:
    result, payloads = construct()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in payloads if path.exists()]
    if existing:
        raise RuntimeError(f"V53A refuses to overwrite artifacts: {existing}")
    for path, payload in payloads.items():
        replay._atomic_exclusive_write(path, payload)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_file_sha256": engine.file_sha256(MANIFEST),
        "manifest_content_sha256": result["content_sha256_before_self_field"],
        "projection_sha256": result["projection"]["sha256"],
        "train_sha256": result["fold_3_train"]["sha256"],
        "gpu_accessed": False,
        "protected_inputs_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
