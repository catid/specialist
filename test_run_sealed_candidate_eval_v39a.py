#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import run_sealed_candidate_eval_v39a as subject


class RuntimeV39ATest(unittest.TestCase):
    def test_single_access_firewall_and_holdout_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.jsonl"
            raw = b'{"x":1}\n'
            path.write_bytes(raw)
            firewall = subject.SingleAccessFirewall({
                "shadow": {"path": str(path), "file_sha256": hashlib.sha256(raw).hexdigest()}
            })
            self.assertEqual(firewall.jsonl("shadow"), [{"x": 1}])
            with self.assertRaisesRegex(RuntimeError, "single-access"):
                firewall.jsonl("shadow")
            with self.assertRaisesRegex(RuntimeError, "holdout"):
                subject.SingleAccessFirewall({
                    "bad": {"path": str(Path(directory) / "holdout.jsonl"), "file_sha256": "x"}
                })

    def test_selection_is_predefined_and_strict(self):
        base = {
            "generated_equal_unit_mean_reward": 0.5,
            "generated_exact_count": 10, "generated_nonzero_count": 12,
            "teacher_forced_equal_unit_mean_answer_logprob": -1.0,
            "protocol_leak_counters": {"x": 0},
        }
        sft = dict(base); sft["generated_equal_unit_mean_reward"] = 0.6
        es = dict(base); es["generated_equal_unit_mean_reward"] = 0.55
        value = subject.select_candidate({"base_a": base, "sft_v37a": sft, "es_v38a": es})
        self.assertEqual(value["selected_arm"], "sft_v37a")
        self.assertTrue(value["shadow_improvement_gate_passed"])

    def test_ood_gates_require_literal_non_degradation(self):
        base = {"generated_row_mean_reward": 0.5, "generated_exact_count": 4}
        self.assertTrue(subject.qa_ood_gate(base, dict(base))["passed"])
        worse = dict(base); worse["generated_row_mean_reward"] = 0.49
        self.assertFalse(subject.qa_ood_gate(base, worse)["passed"])


if __name__ == "__main__":
    unittest.main()
