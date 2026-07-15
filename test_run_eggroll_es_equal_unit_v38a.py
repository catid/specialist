#!/usr/bin/env python3

import unittest

import run_eggroll_es_equal_unit_v38a as subject


class RuntimeV38ATest(unittest.TestCase):
    def test_worker_identity_validation(self):
        rows = [{
            "schema": "eggroll-es-worker-runtime-identity-v38a",
            "pid": 100 + index,
            "inter_engine_rank": index,
            "cuda_visible_devices": str(index),
            "cuda_current_device": 0,
        } for index in range(4)]
        self.assertEqual(subject.validate_worker_identities(rows), {
            0: 100, 1: 101, 2: 102, 3: 103,
        })

    def test_fixed_artifact_paths(self):
        self.assertNotIn("eval", subject.EXPERIMENT)
        self.assertNotIn("holdout", subject.EXPERIMENT)
        self.assertEqual(subject.EXPECTED_GPU_IDS, (0, 1, 2, 3))

    def test_sealed_update_gate_rejects_unchanged_state(self):
        with self.assertRaises(RuntimeError):
            subject.validate_sealed_update({
                "status": "one_nonzero_update_sealed_train_only",
                "alpha": 0.00015,
                "sigma": 0.0003,
                "coefficients": [1.0] * 32,
                "standardization": {"zero_spread": False},
                "application": {
                    "target_alpha": 0.00015,
                    "update_sequence": 1,
                    "manifest": {"expected_base_sha256": "same"},
                    "final_identity": {
                        "sha256": "same", "unselected": {"sha256": "u"},
                    },
                    "commits": [{"committed": True}] * 4,
                    "post_commit_states": [{}] * 4,
                },
                "snapshot_reports": [{
                    "written": True, "rank": 0, "tensor_count": 23,
                    "tensor_elements": 142_999_552,
                }],
            })

    def test_sealed_update_requires_reopened_selected_identity(self):
        selected = {"sha256": "selected-final"}
        final = {
            "sha256": "final",
            "layer_plan_sha256": "plan",
            "selected": selected,
            "unselected": {"sha256": "unselected"},
        }
        metadata = {
            "schema": "eggroll-es-selected-runtime-snapshot-v38a",
            "final_identity_sha256": "final",
            "accepted_alpha": repr(0.00015),
            "layer_plan_sha256": "plan",
        }
        update = {
            "status": "one_nonzero_update_sealed_train_only",
            "alpha": 0.00015,
            "sigma": 0.0003,
            "coefficients": [1.0] * 32,
            "standardization": {"zero_spread": False},
            "snapshot_file_sha256": "snapshot",
            "application": {
                "target_alpha": 0.00015,
                "update_sequence": 1,
                "manifest": {
                    "expected_base_sha256": "base",
                    "unselected_origin_sha256": "unselected",
                },
                "final_identity": final,
                "commits": [{"committed": True}] * 4,
                "post_commit_states": [{}] * 4,
            },
            "snapshot_reports": [{
                "written": True,
                "rank": 0,
                "tensor_count": 23,
                "tensor_elements": 142_999_552,
                "readback_verified": True,
                "reopened_selected_identity": selected,
                "file_sha256": "snapshot",
                "readback": {
                    "verified": True,
                    "selected_identity": selected,
                    "file_sha256": "snapshot",
                    "metadata": metadata,
                },
            }],
        }
        gate = subject.validate_sealed_update(update)
        self.assertTrue(gate["selected_snapshot_reopened_identity_exact"])
        update["snapshot_reports"][0]["readback"]["selected_identity"] = {
            "sha256": "different"
        }
        with self.assertRaises(RuntimeError):
            subject.validate_sealed_update(update)

    def test_gcs_removal_wait_requires_all_four_exact_groups(self):
        class Id:
            def __init__(self, value):
                self.value = value

            def hex(self):
                return self.value

        class Pg:
            def __init__(self, value):
                self.id = Id(value)

        groups = [Pg(f"pg-{index}") for index in range(4)]
        calls = 0

        def table(pg):
            nonlocal calls
            state = "CREATED" if calls < 4 else "REMOVED"
            calls += 1
            return {
                "placement_group_id": pg.id.hex(),
                "strategy": "PACK",
                "state": state,
                "bundles": {0: {"GPU": 1.0}},
                "bundles_to_node_id": {0: "node"},
            }

        rows = subject.wait_for_placement_groups_removed_v38a(
            groups, table, timeout_seconds=1.0, poll_seconds=0.0,
        )
        self.assertEqual([row["state"] for row in rows], ["REMOVED"] * 4)


if __name__ == "__main__":
    unittest.main()
