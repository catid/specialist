import unittest

from train_eggroll_es_specialist import (
    extract_answer,
    specialist_collate,
    specialist_reward,
    specialist_template,
)


class EggrollSpecialistAdapterTests(unittest.TestCase):
    def test_collate_matches_upstream_prompt_target_contract(self):
        prompts, targets = specialist_collate([
            {"question": "What rope?", "answer": "hemp"},
            {"question": "What knot?", "answer": "bowline"},
        ])
        self.assertEqual(prompts, ["What rope?", "What knot?"])
        self.assertEqual(targets, ["hemp", "bowline"])

    def test_template_disables_thinking_and_opens_assistant_role(self):
        prompt = specialist_template("What is a bight?")
        self.assertIn("<|im_start|>user\nWhat is a bight?<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_reward_ignores_protocol_and_thinking_text(self):
        response = "<think>wrong draft</think>\nhemp<|im_end|>"
        self.assertEqual(extract_answer(response), "hemp")
        self.assertEqual(specialist_reward(response, "hemp"), ("exact", 1.0))


if __name__ == "__main__":
    unittest.main()
