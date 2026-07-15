#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import run_sft_train_only_control_v36a as runtime


class SFTTrainOnlyControlV36ATest(unittest.TestCase):
    def test_dataset_binding_and_forbidden_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "train_candidate.jsonl"
            row = {
                "fact_id": "fact-a",
                "document_sha256": "a" * 64,
                "question": "What is checked?",
                "answer": "The train-only binding is checked.",
                "text": (
                    "Question: What is checked?\n"
                    "Answer: The train-only binding is checked."
                ),
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            observed = runtime.validate_train_dataset(path, sha256, 1)
            self.assertEqual((observed["rows"], observed["unique_documents"]), (1, 1))
            forbidden = root / "holdout_rows.jsonl"
            forbidden.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "forbidden"):
                runtime.validate_train_dataset(forbidden, sha256, 1)

    def test_command_is_four_gpu_train_only(self):
        args = argparse.Namespace(
            dataset="train.jsonl",
            dataset_sha256="a" * 64,
            dataset_rows=531,
            output_dir="out",
            epochs=3.0,
            rank=32,
            lora_dropout=0.0,
            grad_accum=2,
            per_device_batch_size=4,
            learning_rate=1e-4,
            seed=17,
            prompt_mode="es_exact",
            loss_mode="example_mean",
            target_layers="20,21,22,23",
            expected_trainable_elements=4_528_128,
            expected_trainable_tensors=70,
            max_length=1024,
            save_steps=100,
            attn_implementation="sdpa",
        )
        command = runtime.build_train_command(args)
        joined = " ".join(command)
        self.assertIn("--nproc-per-node=4", joined)
        self.assertIn("--data", command)
        self.assertNotIn("--eval-data", command)

    def test_gpu_summary_requires_all_four(self):
        samples = []
        for gpu in runtime.EXPECTED_GPU_IDS:
            samples.append({
                "gpu": gpu,
                "utilization_percent": 90,
                "memory_used_mib": 70_000,
                "power_watts": 250.0,
                "temperature_c": 75,
                "attributed_compute_pids": [100 + gpu],
            })
        summary = runtime.summarize_gpu_samples(samples)
        self.assertTrue(summary["all_four_positive_activity"])
        self.assertTrue(summary["all_four_model_resident"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runtime.summarize_gpu_samples(samples[:-1])

    def test_metric_parser(self):
        log = "noise\n{'train_runtime': '81.5', 'train_loss': '1.2', 'epoch': '3'}\n"
        self.assertEqual(
            runtime.extract_train_metrics(log),
            {"train_runtime": 81.5, "train_loss": 1.2, "epoch": 3.0},
        )

    def test_child_json_event_parser(self):
        payload = {"encoding_audit": runtime.EXPECTED_ENCODING_AUDIT}
        text = "prefix " + json.dumps(payload, sort_keys=True) + "\n"
        text += json.dumps(payload, sort_keys=True) + "\n"
        observed = runtime.extract_json_event(text, "encoding_audit")
        self.assertEqual(observed["value"], runtime.EXPECTED_ENCODING_AUDIT)
        self.assertEqual(observed["emission_count"], 2)


if __name__ == "__main__":
    unittest.main()
