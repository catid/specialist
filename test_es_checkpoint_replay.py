import sys
import unittest

import torch

sys.path.insert(0, "/home/catid/specialist/sglang/python")
from sglang.srt.managers.scheduler_components.es_perturb import (
    _noised_from_base,
    apply_commits_to_hf_tensor,
    unit_name_for,
)


class CheckpointReplayTests(unittest.TestCase):
    NAME = "model.language_model.layers.0.linear_attn.out_proj.weight"
    EXPERT_NAME = "model.language_model.layers.0.mlp.experts.down_proj"

    def test_replay_keeps_fp32_master_across_generations(self):
        original = torch.tensor([[0.125, -0.25], [0.5, -1.0]],
                                dtype=torch.bfloat16)
        generations = [
            {"ops": [[101, 0.5]], "sigma": 1e-3, "rank": 1,
             "include_regex": r"layers\.0\."},
            {"ops": [[102, -0.25]], "sigma": 1e-3, "rank": 1,
             "include_regex": r"layers\.0\."},
        ]
        unit = unit_name_for(0, "linear_attn.out_proj.weight")
        master = original.float()
        for generation in generations:
            master = _noised_from_base(
                master, unit,
                [(int(seed), float(coeff)) for seed, coeff in generation["ops"]],
                generation["sigma"], generation["rank"], torch.device("cpu"))
        replayed = original.clone()
        self.assertTrue(apply_commits_to_hf_tensor(
            self.NAME, replayed, generations))
        self.assertTrue(torch.equal(replayed, master.to(torch.bfloat16)))

    def test_target_regex_skips_unselected_tensor(self):
        tensor = torch.ones((2, 2), dtype=torch.bfloat16)
        before = tensor.clone()
        changed = apply_commits_to_hf_tensor(self.NAME, tensor, [{
            "ops": [[7, 1.0]], "sigma": 0.1, "rank": 1,
            "include_regex": r"layers\.39\.",
        }])
        self.assertFalse(changed)
        self.assertTrue(torch.equal(tensor, before))

    def test_stacked_expert_replay_keeps_fp32_master(self):
        # Seventeen experts cross the replay chunk boundary (16), exercising
        # the path used by routed MoE weights rather than only dense matrices.
        original = torch.linspace(
            -0.5, 0.5, 17 * 3 * 2, dtype=torch.float32
        ).reshape(17, 3, 2).to(torch.bfloat16)
        generations = [
            {"ops": [[201, 0.4]], "sigma": 2e-3, "rank": 2,
             "include_regex": r"layers\.0\.mlp\.experts"},
            {"ops": [[202, -0.3]], "sigma": 2e-3, "rank": 2,
             "include_regex": r"layers\.0\.mlp\.experts"},
        ]
        unit = unit_name_for(0, "mlp.experts.down_proj")
        master = original.float()
        for generation in generations:
            master = _noised_from_base(
                master, unit,
                [(int(seed), float(coeff))
                 for seed, coeff in generation["ops"]],
                generation["sigma"], generation["rank"],
                torch.device("cpu"))
        replayed = original.clone()
        self.assertTrue(apply_commits_to_hf_tensor(
            self.EXPERT_NAME, replayed, generations))
        self.assertTrue(torch.equal(replayed, master.to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
