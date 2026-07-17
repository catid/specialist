import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_curated_qa import (
    ROOT,
    OPAQUE_COLLISION_SCHEMA,
    OPAQUE_IDENTITY_SCHEME,
    OpaqueCollisionAuthorization,
    canonical_json_bytes,
    load_opaque_collision_authorization,
    main,
    merge,
    opaque_candidate_binding,
    opaque_file_bindings,
    preflight_path_role_disjointness,
    require_synthetic_fixture_paths,
    require_training_input_path_firewall,
    validate_opaque_collision_authorization,
)
from qa_quality import EvalFact, stable_fact_id


def record(question, answer, kind="qa_test"):
    return {
        "answer": answer,
        "fact_id": stable_fact_id(question, answer),
        "kind": kind,
        "question": question,
        "source": "fixture",
        "text": f"Question: {question}\nAnswer: {answer}",
        "url": "https://example.test/",
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def opaque_authorization_payload(
        inputs, curations, rows, *, collision_count=0):
    return {
        "candidate": opaque_candidate_binding(rows),
        "collision": {"count": collision_count},
        "curation_inputs": opaque_file_bindings(curations),
        "evaluation": {
            "count": 2,
            "identity_sha256": hashlib.sha256(
                b"synthetic-evaluation-identities"
            ).hexdigest(),
        },
        "identity_scheme": OPAQUE_IDENTITY_SCHEME,
        "schema": OPAQUE_COLLISION_SCHEMA,
        "training_inputs": opaque_file_bindings(inputs),
    }


def write_opaque_authorization(path, payload):
    raw = canonical_json_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class CuratedQABuildTests(unittest.TestCase):
    def test_cli_no_argument_mode_is_rejected(self):
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main([])
        self.assertEqual(error.exception.code, 2)

    def test_cli_authorization_requires_independently_pinned_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "authorization.json"
            with mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--collision-authorization", str(manifest)])
            self.assertEqual(error.exception.code, 2)

    def test_cli_synthetic_empty_mode_uses_only_explicit_temp_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [
                record("Which synthetic material is named?", "hemp")
            ])
            output = root / "output.jsonl"
            report = root / "report.json"
            with mock.patch("sys.stdout", new=io.StringIO()):
                main([
                    "--synthetic-empty-eval",
                    "--inputs", str(source),
                    "--output", str(output),
                    "--report", str(report),
                    "--curation",
                ])
            self.assertTrue(output.is_file())
            built_report = json.loads(report.read_text())
            self.assertEqual(
                built_report["evaluation_boundary"],
                {
                    "evaluation_fact_count": 0,
                    "mode": "synthetic_fact_fixture",
                },
            )

    def test_cli_synthetic_empty_mode_rejects_canonical_defaults(self):
        with self.assertRaisesRegex(RuntimeError, "outside the repository"):
            main(["--synthetic-empty-eval"])

    def test_opaque_authorization_rebuild_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            manifest_sha256 = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            authorization = load_opaque_collision_authorization(
                manifest, manifest_sha256
            )
            output = root / "output.jsonl"
            report_path = root / "report.json"
            first = merge(
                [source], output, report_path, None, [], authorization
            )
            first_output = output.read_bytes()
            first_report = report_path.read_bytes()
            second = merge(
                [source], output, report_path, None, [], authorization
            )
            self.assertEqual(first, second)
            self.assertEqual(output.read_bytes(), first_output)
            self.assertEqual(report_path.read_bytes(), first_report)
            self.assertEqual(
                first["evaluation_boundary"]["mode"],
                "opaque_collision_authorization",
            )
            self.assertEqual(
                first["evaluation_boundary"]["authorization_sha256"],
                manifest_sha256,
            )
            self.assertEqual(first["eval_fact_count"], 2)

    def test_cli_production_mode_consumes_only_pinned_opaque_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            output = root / "output.jsonl"
            report = root / "report.json"
            with mock.patch(
                    "build_curated_qa.eval_facts",
                    side_effect=AssertionError(
                        "production command opened evaluation semantics"
                    )):
                with mock.patch("sys.stdout", new=io.StringIO()):
                    main([
                        "--collision-authorization", str(manifest),
                        "--collision-authorization-sha256", digest,
                        "--inputs", str(source),
                        "--output", str(output),
                        "--report", str(report),
                        "--curation",
                    ])
            self.assertTrue(output.is_file())
            built_boundary = json.loads(
                report.read_text()
            )["evaluation_boundary"]
            self.assertEqual(
                built_boundary["authorization_sha256"],
                digest,
            )

    def test_loaded_authorization_rejects_post_load_payload_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            authorization = load_opaque_collision_authorization(
                manifest, digest
            )
            authorization.payload["evaluation"]["count"] = 3
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                validate_opaque_collision_authorization(
                    authorization, [source], [], [item]
                )

    def test_direct_authorization_forgery_rejects_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            payload = opaque_authorization_payload([source], [], [item])
            forged = OpaqueCollisionAuthorization(
                payload,
                hashlib.sha256(b"forged authorization").hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                validate_opaque_collision_authorization(
                    forged, [source], [], [item]
                )

    def test_synthetic_input_symlink_cannot_alias_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alias = root / "repository-file-alias"
            target = Path(__file__).resolve()
            alias.symlink_to(target)
            real_lstat = os.lstat

            def guarded_lstat(path):
                lexical = Path(os.path.abspath(os.fspath(path)))
                if lexical == target:
                    raise AssertionError("repository symlink target was statted")
                return real_lstat(path)

            with mock.patch(
                    "build_curated_qa.os.lstat",
                    side_effect=guarded_lstat):
                with self.assertRaisesRegex(
                        RuntimeError, "outside the repository"):
                    require_synthetic_fixture_paths(
                        [alias], root / "output.jsonl",
                        root / "report.json", [],
                    )

    def test_synthetic_nonexistent_output_resolves_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [
                record("Which synthetic material is named?", "hemp")
            ])
            repository_alias = root / "repository-alias"
            repository_alias.symlink_to(ROOT, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "outside the repository"):
                require_synthetic_fixture_paths(
                    [source], repository_alias / "never-created.jsonl",
                    root / "report.json", [],
                )

    def test_training_firewall_rejects_repo_eval_before_target_stat(self):
        protected_path = ROOT / "data" / "eval_v3_never_touch.jsonl"
        with mock.patch(
                "build_curated_qa.os.lstat",
                side_effect=AssertionError("repo eval path was statted")):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                require_training_input_path_firewall([protected_path], [])

    def test_training_firewall_rejects_parent_symlink_before_target_stat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected_parent = ROOT / "data" / "eval_v3_never_touch"
            alias_parent = root / "ordinary-parent-alias"
            alias_parent.symlink_to(
                protected_parent, target_is_directory=True
            )
            protected_path = alias_parent / "training-input.jsonl"
            real_lstat = os.lstat

            def guarded_lstat(path):
                lexical = Path(os.path.abspath(os.fspath(path)))
                if lexical == protected_parent:
                    raise AssertionError("protected symlink target was statted")
                return real_lstat(path)

            with mock.patch(
                    "build_curated_qa.os.lstat",
                    side_effect=guarded_lstat):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    require_training_input_path_firewall(
                        [protected_path], []
                    )

    def test_training_firewall_rejects_external_input_symlink_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write_jsonl(source, [
                record("Which synthetic material is named?", "hemp")
            ])
            alias = root / "input-alias.jsonl"
            alias.symlink_to(source)
            with self.assertRaisesRegex(RuntimeError, "symlink aliases"):
                require_training_input_path_firewall([alias], [])

    def test_production_merge_firewall_runs_before_any_dataset_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            protected_curation = (
                ROOT / "data" / "manual_reviews" / "never-touch.jsonl"
            )
            payload = opaque_authorization_payload([source], [], [item])
            payload["curation_inputs"] = [{
                "index": 0,
                "sha256": "0" * 64,
            }]
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(manifest, payload)
            authorization = load_opaque_collision_authorization(
                manifest, digest
            )
            with mock.patch(
                    "build_curated_qa.load_curation",
                    side_effect=AssertionError("curation loader was called")):
                with mock.patch.object(
                        Path, "open",
                        side_effect=AssertionError("dataset content was opened")):
                    with self.assertRaisesRegex(RuntimeError, "forbidden"):
                        merge(
                            [source], root / "output.jsonl",
                            root / "report.json", None,
                            [protected_curation], authorization,
                        )

    def test_authorization_leaf_symlink_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            alias = root / "authorization-alias.json"
            alias.symlink_to(manifest)
            with mock.patch(
                    "build_curated_qa._read_regular_file_without_symlinks",
                    side_effect=AssertionError("authorization alias was read")):
                with self.assertRaisesRegex(RuntimeError, "symlinks"):
                    load_opaque_collision_authorization(alias, digest)

    def test_authorization_parent_alias_to_quarantine_fails_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            quarantine = root / "synthetic-quarantine"
            quarantine.mkdir()
            manifest = quarantine / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            alias_parent = root / "quarantine-alias"
            alias_parent.symlink_to(quarantine, target_is_directory=True)
            alias = alias_parent / manifest.name
            with mock.patch(
                    "build_curated_qa.QUARANTINED_LEGACY_EVAL",
                    frozenset({manifest.resolve()})):
                with mock.patch(
                        "build_curated_qa._read_regular_file_without_symlinks",
                        side_effect=AssertionError(
                            "quarantine alias was read"
                        )):
                    with self.assertRaisesRegex(
                            RuntimeError, "quarantined evaluation"):
                        load_opaque_collision_authorization(alias, digest)

    def test_authorization_hard_link_alias_is_rejected_before_content_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            alias = root / "authorization-hard-link.json"
            os.link(manifest, alias)
            with self.assertRaisesRegex(RuntimeError, "hard-link aliases"):
                load_opaque_collision_authorization(alias, digest)

    def test_preflight_rejects_output_report_alias_through_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [
                record("Which synthetic material is named?", "hemp")
            ])
            real_parent = root / "real-parent"
            real_parent.mkdir()
            alias_parent = root / "parent-alias"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                preflight_path_role_disjointness(
                    [source], [], real_parent / "same.json",
                    alias_parent / "same.json",
                )

    def test_preflight_rejects_output_symlink_to_input_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            original = source.read_bytes()
            output = root / "output.jsonl"
            output.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "aliases training input"):
                preflight_path_role_disjointness(
                    [source], [], output, root / "report.json"
                )
            self.assertEqual(source.read_bytes(), original)

    def test_preflight_rejects_report_hard_link_to_curation_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [
                record("Which synthetic material is named?", "hemp")
            ])
            curation = root / "curation.jsonl"
            curation.write_text("")
            report = root / "report.json"
            os.link(curation, report)
            with self.assertRaisesRegex(ValueError, "aliases curation input"):
                preflight_path_role_disjointness(
                    [source], [curation], root / "output.jsonl", report
                )
            self.assertEqual(curation.read_text(), "")

    def test_preflight_rejects_output_alias_to_authorization_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            authorization = load_opaque_collision_authorization(
                manifest, digest
            )
            original = manifest.read_bytes()
            output = root / "output-alias.json"
            output.symlink_to(manifest)
            report = root / "report.json"
            with self.assertRaisesRegex(
                    ValueError, "aliases collision authorization"):
                merge(
                    [source], output, report, None, [], authorization
                )
            self.assertEqual(manifest.read_bytes(), original)
            self.assertFalse(report.exists())

    def test_opaque_authorization_rejects_stale_input_hash_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            original = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [original])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [original]),
            )
            authorization = load_opaque_collision_authorization(
                manifest, digest
            )
            changed = record("Which synthetic material is named?", "jute")
            write_jsonl(source, [changed])
            output = root / "output.jsonl"
            with self.assertRaisesRegex(ValueError, "stale training input"):
                merge(
                    [source], output, root / "report.json", None, [],
                    authorization,
                )
            self.assertFalse(output.exists())

    def test_opaque_authorization_rejects_stale_candidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            payload = opaque_authorization_payload([source], [], [item])
            payload["candidate"]["identity_sha256"] = hashlib.sha256(
                b"stale candidate identity"
            ).hexdigest()
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(manifest, payload)
            authorization = load_opaque_collision_authorization(
                manifest, digest
            )
            output = root / "output.jsonl"
            with self.assertRaisesRegex(ValueError, "stale candidate"):
                merge(
                    [source], output, root / "report.json", None, [],
                    authorization,
                )
            self.assertFalse(output.exists())

    def test_opaque_authorization_rejects_stale_curation_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            curation = root / "curation.jsonl"
            curation.write_text("")
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload(
                    [source], [curation], [item]
                ),
            )
            authorization = load_opaque_collision_authorization(
                manifest, digest
            )
            curation.write_text("\n")
            output = root / "output.jsonl"
            with self.assertRaisesRegex(ValueError, "stale curation input"):
                merge(
                    [source], output, root / "report.json", None,
                    [curation], authorization,
                )
            self.assertFalse(output.exists())

    def test_opaque_authorization_rejects_stale_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            write_opaque_authorization(
                manifest,
                opaque_authorization_payload([source], [], [item]),
            )
            stale_digest = hashlib.sha256(b"stale manifest").hexdigest()
            with self.assertRaisesRegex(ValueError, "pinned digest"):
                load_opaque_collision_authorization(manifest, stale_digest)

    def test_opaque_authorization_rejects_reported_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(
                manifest,
                opaque_authorization_payload(
                    [source], [], [item], collision_count=1
                ),
            )
            with self.assertRaisesRegex(ValueError, "one or more collisions"):
                load_opaque_collision_authorization(manifest, digest)

    def test_opaque_authorization_rejects_semantic_or_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which synthetic material is named?", "hemp")
            write_jsonl(source, [item])
            payload = opaque_authorization_payload([source], [], [item])
            payload["notes"] = "semantic text is forbidden"
            manifest = root / "collision-authorization.json"
            digest = write_opaque_authorization(manifest, payload)
            with self.assertRaisesRegex(ValueError, "expected fields"):
                load_opaque_collision_authorization(manifest, digest)

    def test_merge_requires_exactly_one_evaluation_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [
                record("Which synthetic material is named?", "hemp")
            ])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                merge(
                    [source], root / "output.jsonl", root / "report.json",
                    None, [],
                )

    def test_merge_is_deterministic_and_counts_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(
                first, [
                    record(
                        "Where is resource B?", "https://b.test/")])
            write_jsonl(
                second, [
                    record(
                        "Where is resource A?", "https://a.test/")])
            output = root / "output.jsonl"
            report = merge([first, second], output, root / "report.json", [])
            rows = [json.loads(line)
                    for line in output.read_text().splitlines()]
            self.assertEqual(report["counts"]["output"], 2)
            self.assertEqual(rows[0]["question"], "Where is resource A?")

    def test_merge_canonicalizes_legacy_prompt_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("What material is described?", "jute")
            item["text"] = (
                "Answer this question briefly and factually:\n\n"
                "What material is described?\n\njute"
            )
            write_jsonl(source, [item])
            output = root / "out.jsonl"
            merge([source], output, root / "report.json", [])
            row = json.loads(output.read_text())
            self.assertEqual(
                row["text"],
                "Question: What material is described?\nAnswer: jute",
            )

    def test_duplicate_question_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(first, [record("What is a bight?", "a loop")])
            write_jsonl(second, [record("What is a bight?", "a curve")])
            with self.assertRaisesRegex(ValueError, "duplicate question"):
                merge([first, second], root / "out.jsonl",
                      root / "report.json", [])

    def test_evaluation_leakage_is_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [record("What is a bight?", "a loop")])
            facts = [
                EvalFact(
                    "What is a bight?",
                    "a loop",
                    "eval-1",
                    "heldout")]
            output = root / "out.jsonl"
            report = merge([source], output, root / "report.json", facts)
            self.assertEqual(output.read_text(), "")
            self.assertEqual(report["counts"]["excluded"], 1)
            self.assertEqual(
                report["counts"]["exclusion_reasons"],
                {"exact_question": 1},
            )

    def test_manual_curation_edits_and_drops_are_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            edit = record("What material is described?", "jute")
            drop = record("What was the old sale price?", "$10")
            write_jsonl(source, [edit, drop])
            curation = root / "curation.jsonl"
            write_jsonl(curation, [
                {
                    "action": "edit",
                    "answer": "jute rope",
                    "evidence": "The material described is jute rope.",
                    "evidence_url": "https://example.test/",
                    "expected_answer": "jute",
                    "expected_question": "What material is described?",
                    "fact_id": edit["fact_id"],
                    "question": "What rope material is described?",
                    "reason": (
                        "Names the subject and uses the complete phrase."
                    ),
                    "reason_code": "contextless_question",
                    "reviewed_at": "2026-07-14",
                    "reviewer": "fixture-reviewer",
                },
                {
                    "action": "drop",
                    "expected_answer": "$10",
                    "expected_question": "What was the old sale price?",
                    "fact_id": drop["fact_id"],
                    "reason": "Historical sale trivia is volatile.",
                    "reason_code": "volatile_commerce",
                    "reviewed_at": "2026-07-14",
                    "reviewer": "fixture-reviewer",
                },
            ])
            output = root / "out.jsonl"
            report = merge(
                [source],
                output,
                root /
                "report.json",
                [],
                curation)
            rows = [json.loads(line)
                    for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["answer"], "jute rope")
            self.assertEqual(rows[0]["quality_schema"], "curated-qa-v1")
            self.assertEqual(
                rows[0]["curation"]["original_fact_id"], edit["fact_id"])
            self.assertEqual(report["curation"]["by_action"],
                             {"drop": 1, "edit": 1})
            self.assertEqual(
                report["counts"]["exclusion_reasons"],
                {"manual_curation:volatile_commerce": 1},
            )

    def test_stale_manual_curation_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("What material is described?", "jute")
            write_jsonl(source, [item])
            curation = root / "curation.jsonl"
            write_jsonl(curation, [{
                "action": "drop",
                "expected_answer": "hemp",
                "expected_question": item["question"],
                "fact_id": item["fact_id"],
                "reason": "Fixture stale decision.",
                "reason_code": "fixture",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            with self.assertRaisesRegex(ValueError, "stale curation decision"):
                merge([source], root / "out.jsonl", root / "report.json", [],
                      curation)

    def test_manual_paraphrase_requires_explicit_support_and_rationale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("What shape is described?", "X")
            write_jsonl(source, [item])
            base_decision = {
                "action": "edit",
                "answer": "an X shape",
                "evidence": "The pattern creates a X in the back.",
                "evidence_url": item["url"],
                "expected_answer": item["answer"],
                "expected_question": item["question"],
                "fact_id": item["fact_id"],
                "question": item["question"],
                "reason": "Repairs a bare answer and source grammar.",
                "reason_code": "answer_too_bare",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }
            curation = root / "curation.jsonl"
            write_jsonl(curation, [base_decision])
            with self.assertRaisesRegex(ValueError, "must be extractive"):
                merge([source], root / "out.jsonl", root / "report.json", [],
                      curation)

            paraphrase = dict(base_decision)
            paraphrase["support_type"] = "manual_paraphrase"
            write_jsonl(curation, [paraphrase])
            with self.assertRaisesRegex(ValueError, "paraphrase_rationale"):
                merge([source], root / "out.jsonl", root / "report.json", [],
                      curation)

            paraphrase["paraphrase_rationale"] = (
                "Changes only the article and noun while preserving X."
            )
            write_jsonl(curation, [paraphrase])
            output = root / "out.jsonl"
            report = merge(
                [source], output, root / "report.json", [], curation)
            row = json.loads(output.read_text())
            self.assertEqual(row["answer"], "an X shape")
            self.assertEqual(
                row["curation"]["support_type"], "manual_paraphrase")
            self.assertEqual(
                report["curation"]["edit_support_types"],
                {"manual_paraphrase": 1},
            )

    def test_multiple_curation_ledgers_are_applied_and_attributed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            first = record("Which material is named?", "jute")
            second = record("Which material is softer?", "hemp")
            write_jsonl(source, [first, second])
            ledgers = [root / "first.jsonl", root / "second.jsonl"]
            write_jsonl(ledgers[0], [{
                "action": "drop",
                "expected_answer": first["answer"],
                "expected_question": first["question"],
                "fact_id": first["fact_id"],
                "reason": "Fixture drop.",
                "reason_code": "fixture_drop",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            write_jsonl(ledgers[1], [{
                "action": "edit",
                "answer": "soft hemp",
                "evidence": "The material is soft hemp.",
                "evidence_url": second["url"],
                "expected_answer": second["answer"],
                "expected_question": second["question"],
                "fact_id": second["fact_id"],
                "question": "Which soft material is named?",
                "reason": "Makes the fixture self-contained.",
                "reason_code": "fixture_edit",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            output = root / "out.jsonl"
            report = merge(
                [source], output, root / "report.json", [], ledgers)
            row = json.loads(output.read_text())
            self.assertEqual(
                row["curation"]["decision_file"], str(ledgers[1].resolve()))
            self.assertEqual(report["curation"]["decisions"], 2)
            self.assertEqual(
                [item["path"] for item in report["curation"]["artifacts"]],
                [str(path.resolve()) for path in ledgers],
            )

    def test_curation_accepts_recorded_url_evidence_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which URL is listed?", "https://old.test/")
            item["url_evidence_url"] = "https://example.test/sitemap.xml"
            write_jsonl(source, [item])
            curation = root / "curation.jsonl"
            write_jsonl(curation, [{
                "action": "edit",
                "answer": "https://new.test/",
                "evidence": "<loc>https://new.test/</loc>",
                "evidence_url": item["url_evidence_url"],
                "expected_answer": item["answer"],
                "expected_question": item["question"],
                "fact_id": item["fact_id"],
                "question": "Which URL does the sitemap list?",
                "reason": (
                    "Uses the source record's explicit sitemap evidence."
                ),
                "reason_code": "broken_canonicalization",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            output = root / "out.jsonl"
            merge([source], output, root / "report.json", [], curation)
            row = json.loads(output.read_text())
            self.assertEqual(row["answer"], "https://new.test/")
            self.assertEqual(
                row["evidence"], "<loc>https://new.test/</loc>")
            self.assertEqual(
                row["evidence_url"],
                "https://example.test/sitemap.xml",
            )
            self.assertEqual(row["url"], "https://example.test/")


if __name__ == "__main__":
    unittest.main()
