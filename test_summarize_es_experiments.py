import json
import tempfile
import unittest
from pathlib import Path

from summarize_es_experiments import (aggregate_layer_plans,
                                      aggregate_zero_lr_controls, parse_probes,
                                      summarize)


class SummarizeExperimentsTests(unittest.TestCase):
    def test_plan_aggregation_keeps_comparable_base_screens(self):
        runs = [
            {"name": "front_seed1", "layer_plan": "front",
             "generations": 3, "heldout_delta": 0.1},
            {"name": "front_seed2", "layer_plan": "front",
             "generations": 3, "heldout_delta": -0.05},
            {"name": "insert_front_seed1", "layer_plan": "front",
             "generations": 3, "heldout_delta": 9.0},
        ]
        aggregate = aggregate_layer_plans(runs)["front"]
        self.assertEqual(aggregate["num_runs"], 2)
        self.assertAlmostEqual(aggregate["mean_heldout_delta"], 0.025)
        self.assertEqual(aggregate["positive_runs"], 1)

    def test_probe_commit_accounting_across_resumes(self):
        log = """\
[gen 0] probe F1 train 0.1000 heldout 0.2000
[gen 2] F1 mean 0.1 max 0.2 min 0.0 (1s) | probe_train_F1 0.1100 probe_held_F1 0.2100
[gen 10] probe F1 train 0.1200 heldout 0.2200
[gen 17] F1 mean 0.1 max 0.2 min 0.0 (1s) | probe_train_F1 0.1300 probe_held_F1 0.2300
[gen 20] probe reward train 0.1400 heldout 0.2400
"""
        probes = parse_probes(log)
        self.assertEqual(
            [probe["committed_generations"] for probe in probes],
            [0, 3, 10, 18, 20])

    def test_zero_lr_control_aggregation_is_separate(self):
        runs = [
            {"name": "control_seed1", "generations": 3,
             "learning_rate": 0.0, "heldout_delta": 0.01},
            {"name": "control_seed2", "generations": 3,
             "learning_rate": 0.0, "heldout_delta": -0.01},
            {"name": "control_long", "generations": 6,
             "learning_rate": 0.0, "heldout_delta": 0.0},
            {"name": "front_seed1", "generations": 3,
             "learning_rate": 0.04, "heldout_delta": 1.0},
        ]
        control = aggregate_zero_lr_controls(runs)
        self.assertEqual(control["num_runs"], 3)
        self.assertAlmostEqual(control["mean_heldout_delta"], 0.0)
        self.assertEqual(control["positive_runs"], 1)
        self.assertEqual(control["negative_runs"], 1)

    def test_summary_uses_true_initial_and_latest_committed_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "trial.jsonl"
            rows = []
            for generation in range(20):
                rows.append({
                    "gen": generation,
                    "mean_fit": 0.1,
                    "max_fit": 0.2,
                    "replica_state": None,
                    "data": {"sha256": "training-digest", "items": 123},
                    "probe_data": {"sha256": "probe-digest", "items": 45},
                    "layer_plan": {"plan": "front", "layers": [0, 1, 2, 3],
                                   "num_units": 35},
                })
            journal.write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            journal.with_suffix(".log").write_text("""\
[gen 0] probe F1 train 0.1000 heldout 0.2000
[gen 19] reward mean 0.1 max 0.2 min 0.0 (1s) | probe_train_reward 0.1300 probe_held_reward 0.2300
[gen 20] probe F1 train 0.1400 heldout 0.2400
""")
            result = summarize(journal)
        self.assertEqual(result["start_probe_after_generations"], 0)
        self.assertEqual(result["end_probe_after_generations"], 20)
        self.assertAlmostEqual(result["heldout_delta"], 0.04)
        self.assertEqual(result["data"]["sha256"], "training-digest")
        self.assertEqual(result["probe_data"]["sha256"], "probe-digest")


if __name__ == "__main__":
    unittest.main()
