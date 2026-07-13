import re
import unittest

import es_layer_plan


class LayerPlanTests(unittest.TestCase):
    def test_front_back_dense_has_expected_70_units(self):
        manifest = es_layer_plan.plan_manifest(
            es_layer_plan.DEFAULT_MODEL, "front_back", ["dense"])
        self.assertEqual(manifest["layers"], [0, 1, 2, 3, 36, 37, 38, 39])
        self.assertEqual(manifest["num_units"], 70)
        matcher = re.compile(manifest["include_regex"])
        self.assertTrue(matcher.search(
            "model.language_model.layers.39.self_attn.o_proj.weight"))
        self.assertFalse(matcher.search(
            "model.language_model.layers.39.mlp.experts.down_proj"))

    def test_front_back_all_has_expected_86_units(self):
        manifest = es_layer_plan.plan_manifest(
            es_layer_plan.DEFAULT_MODEL, "front_back", ["all"])
        self.assertEqual(manifest["num_units"], 86)

    def test_middle_control_matches_layer_type_composition(self):
        middle = es_layer_plan.plan_manifest(
            es_layer_plan.DEFAULT_MODEL, "middle_matched", ["dense"])
        edge = es_layer_plan.plan_manifest(
            es_layer_plan.DEFAULT_MODEL, "front_back", ["dense"])
        self.assertEqual(middle["num_units"], edge["num_units"])

    def test_expanded_checkpoint_back_plan_tracks_new_output_edge(self):
        manifest = es_layer_plan.plan_manifest(
            es_layer_plan.DEFAULT_MODEL.parent /
            "Qwen3.6-35B-A3B-depth-back-e005", "back", ["dense"])
        self.assertEqual(manifest["layers"], [40, 41, 42, 43])

    def test_insertion_plan_reads_checkpoint_provenance(self):
        parent = es_layer_plan.DEFAULT_MODEL.parent
        expected = {
            "front": [4, 5, 6, 7],
            "middle": [20, 21, 22, 23],
            "back": [40, 41, 42, 43],
        }
        for location, layers in expected.items():
            with self.subTest(location=location):
                manifest = es_layer_plan.plan_manifest(
                    parent / f"Qwen3.6-35B-A3B-depth-{location}-e005",
                    "inserted", ["dense"])
                self.assertEqual(manifest["layers"], layers)
                self.assertEqual(manifest["num_units"], 35)

    def test_arbitrary_non_motif_config_is_rejected(self):
        with self.assertRaises(ValueError):
            es_layer_plan.plan_manifest(
                es_layer_plan.DEFAULT_MODEL, "front", ["dense"], [100])


if __name__ == "__main__":
    unittest.main()
