import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIRECTORY = (
    ROOT
    / "experiments"
    / "vllm_moe_tuning"
    / "v026_rtx_pro_6000_bf16_tp1_pruned"
)
CONFIG = DIRECTORY / (
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition.json"
)
EVIDENCE = DIRECTORY / "benchmark_evidence.json"


class PrunedMoeConfigContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())
        self.evidence = json.loads(EVIDENCE.read_text())

    def test_exact_measured_batch_keys(self):
        self.assertEqual(
            [int(key) for key in self.config],
            [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        )

    def test_every_entry_is_a_complete_bounded_triton_tile(self):
        required = {
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        }
        for batch, entry in self.config.items():
            with self.subTest(batch=batch):
                self.assertEqual(set(entry), required)
                self.assertIn(entry["BLOCK_SIZE_M"], {16, 32, 64, 128})
                self.assertIn(entry["BLOCK_SIZE_N"], {64, 128, 256})
                self.assertIn(entry["BLOCK_SIZE_K"], {64, 128})
                self.assertIn(entry["GROUP_SIZE_M"], {1, 16})
                self.assertIn(entry["num_warps"], {4, 8})
                self.assertIn(entry["num_stages"], {2, 3, 4, 5})

    def test_evidence_covers_every_config_key_without_claiming_activation(self):
        self.assertEqual(
            set(self.evidence["paired_results"]), set(self.config)
        )
        self.assertEqual(
            self.evidence["status"],
            "end_to_end_verified_not_training_enabled",
        )
        self.assertEqual(
            self.evidence["activation_policy"]["default"], "unset"
        )
        self.assertEqual(
            self.evidence["activation_policy"]["future_training"],
            "requires separate recipe preregistration and task-level timing checks",
        )

    def test_paired_results_are_nonregressive_and_equivalence_is_exact(self):
        for batch, result in self.evidence["paired_results"].items():
            with self.subTest(batch=batch):
                self.assertGreaterEqual(result["speedup"], 1.0)
                self.assertLessEqual(result["selected_us"], result["default_us"])
        equivalence = self.evidence["numeric_equivalence"]
        self.assertTrue(equivalence["finite_default_and_selected"])
        self.assertEqual(equivalence["exact_element_fraction"], 1.0)
        self.assertEqual(equivalence["max_absolute_difference"], 0.0)

    def test_loader_selected_exact_file_entries(self):
        loader = self.evidence["loader_verification"]
        self.assertEqual(loader["worker_count"], 4)
        self.assertTrue(loader["exact_device_specific_file_selected"])
        self.assertTrue(loader["selected_configs_equal_file_entries"])
        self.assertEqual(
            set(loader["kernel_time_us"]), {"128", "256", "512", "1024"}
        )

    def test_end_to_end_generation_is_exact_and_faster_on_every_gpu(self):
        result = self.evidence["end_to_end_triton_generation"]
        self.assertEqual(result["backend"], "triton")
        self.assertEqual(result["prompt_source"], "256 fixed synthetic non-dataset prompts")
        self.assertTrue(result["exact_generated_text_and_token_identity"])
        self.assertEqual(len(result["per_gpu"]), 4)
        for gpu, row in result["per_gpu"].items():
            with self.subTest(gpu=gpu):
                self.assertGreater(row["speedup"], 1.0)
                self.assertLess(row["tuned_seconds"], row["default_seconds"])
                self.assertGreater(
                    row["tuned_tokens_per_second"],
                    row["default_tokens_per_second"],
                )
        self.assertGreater(
            result["aggregate"]["worst_per_gpu_speedup"], 1.16
        )


if __name__ == "__main__":
    unittest.main()
