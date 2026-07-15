import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import qwen_motif_surgery as surgery


class MotifMapTests(unittest.TestCase):
    def test_front_middle_and_back_maps(self):
        front, front_inserted = surgery.source_map(40, "front")
        self.assertEqual(front[:9], [0, 1, 2, 3, 0, 1, 2, 3, 4])
        self.assertEqual(front_inserted, {4, 5, 6, 7})
        middle, _ = surgery.source_map(40, "middle")
        self.assertEqual(middle[16:25], [16, 17, 18, 19, 16, 17, 18, 19, 20])
        back, back_inserted = surgery.source_map(40, "back")
        self.assertEqual(back[-8:], [36, 37, 38, 39, 36, 37, 38, 39])
        self.assertEqual(back_inserted, {40, 41, 42, 43})


class SyntheticCheckpointTests(unittest.TestCase):
    def test_front_insertion_renumbers_and_damps_residual_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, dst = root / "src", root / "dst"
            src.mkdir()
            layer_types = ["full_attention" if (index + 1) % 4 == 0
                           else "linear_attention" for index in range(40)]
            (src / "config.json").write_text(json.dumps({
                "text_config": {"num_hidden_layers": 40,
                                "layer_types": layer_types}}))
            tensors = {"model.embed_tokens.weight": torch.ones(1, 1)}
            for layer in range(40):
                tensors[f"model.language_model.layers.{layer}.input_layernorm.weight"] = (
                    torch.full((1,), float(layer + 1)))
                suffix = ("self_attn.o_proj.weight" if layer % 4 == 3
                          else "linear_attn.out_proj.weight")
                tensors[f"model.language_model.layers.{layer}.{suffix}"] = (
                    torch.full((1, 1), float(layer + 1)))
                tensors[
                    f"model.language_model.layers.{layer}.mlp.experts.down_proj"
                ] = torch.full((1, 1), float(layer + 1))
                tensors[
                    "model.language_model.layers."
                    f"{layer}.mlp.shared_expert.down_proj.weight"
                ] = torch.full((1, 1), float(layer + 1))
            shard = "model-00001-of-00001.safetensors"
            save_file(tensors, src / shard)
            (src / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values())},
                "weight_map": {name: shard for name in tensors},
            }))
            (src / "tokenizer.json").write_text("{}")

            result = surgery.build_checkpoint(src, dst, "front", 0.05)
            self.assertEqual(result["num_hidden_layers"], 44)
            self.assertEqual(result["scaled_inserted_tensors"], 12)
            self.assertEqual(
                set(result["scaled_suffixes_by_inserted_destination"]),
                {"4", "5", "6", "7"})
            config = json.loads((dst / "config.json").read_text())
            self.assertEqual(config["text_config"]["num_hidden_layers"], 44)
            with safe_open(dst / shard, framework="pt") as output:
                # Destination 4 is the damped copy of source layer 0.
                self.assertTrue(torch.equal(
                    output.get_tensor(
                        "model.language_model.layers.4.linear_attn.out_proj.weight"),
                    torch.tensor([[0.05]])))
                # Destination 8 is the renumbered original layer 4.
                self.assertTrue(torch.equal(
                    output.get_tensor(
                        "model.language_model.layers.8.linear_attn.out_proj.weight"),
                    torch.tensor([[5.0]])))
                # Norms in the inserted motif are copied but not damped.
                self.assertTrue(torch.equal(
                    output.get_tensor(
                        "model.language_model.layers.4.input_layernorm.weight"),
                    torch.tensor([1.0])))

    def test_insertion_fails_if_a_residual_output_is_not_covered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, dst = root / "src", root / "dst"
            src.mkdir()
            layer_types = ["full_attention" if (index + 1) % 4 == 0
                           else "linear_attention" for index in range(40)]
            (src / "config.json").write_text(json.dumps({
                "text_config": {"num_hidden_layers": 40,
                                "layer_types": layer_types}}))
            tensors = {"model.embed_tokens.weight": torch.ones(1, 1)}
            for layer in range(40):
                attention = ("self_attn.o_proj.weight" if layer % 4 == 3
                             else "linear_attn.out_proj.weight")
                tensors[
                    f"model.language_model.layers.{layer}.{attention}"
                ] = torch.ones(1, 1)
                tensors[
                    f"model.language_model.layers.{layer}.mlp.experts.down_proj"
                ] = torch.ones(1, 1)
                # Deliberately omit the shared-expert residual output.
            shard = "model-00001-of-00001.safetensors"
            save_file(tensors, src / shard)
            (src / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values())},
                "weight_map": {name: shard for name in tensors},
            }))

            with self.assertRaisesRegex(
                    ValueError, "residual-output damping coverage changed"):
                surgery.build_checkpoint(src, dst, "front", 0.05)
            self.assertFalse(dst.exists())


if __name__ == "__main__":
    unittest.main()
