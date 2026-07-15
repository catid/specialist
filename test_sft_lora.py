#!/usr/bin/env python3
from __future__ import annotations

import unittest
import argparse
import inspect
from types import SimpleNamespace

import torch

import sft_lora
from transformers import TrainingArguments


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("ES encoding must disable special-token insertion")
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return {"input_ids": [1, 2, 3]}


class SFTLoRATest(unittest.TestCase):
    def test_exact_es_encoding_masks_prompt_and_does_not_append_eos(self):
        tokenizer = FakeTokenizer()
        pair = ("Question?", "Answer.")
        encoded = sft_lora.encode_pair(tokenizer, pair, 1024, "es_exact")
        prompt = sft_lora.specialist_template(pair[0])
        expected = [ord(character) for character in prompt + pair[1]]
        self.assertEqual(encoded["input_ids"], expected)
        self.assertEqual(
            encoded["labels"],
            [-100] * len(prompt) + [ord(character) for character in pair[1]],
        )
        self.assertEqual(encoded["answer_token_count"], len(pair[1]))

    def test_batch_encoding_normalization(self):
        self.assertEqual(sft_lora.token_ids({"input_ids": [4, 5]}), [4, 5])
        self.assertEqual(sft_lora.token_ids(torch.tensor([6, 7])), [6, 7])
        with self.assertRaises(ValueError):
            sft_lora.token_ids({"attention_mask": [1]})

    def test_target_layers(self):
        self.assertIsNone(sft_lora.parse_target_layers("all"))
        self.assertEqual(sft_lora.parse_target_layers("23,20,21,20"), [20, 21, 23])
        with self.assertRaises(ValueError):
            sft_lora.parse_target_layers("40")

    def test_example_mean_loss_reduces_tokens_within_each_example(self):
        logits = torch.tensor([
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])

        class Model:
            def __call__(self, **inputs):
                del inputs
                return SimpleNamespace(logits=logits)

        labels = torch.tensor([
            [-100, 0, 1, -100],
            [-100, 2, -100, -100],
        ])
        shifted = labels[..., 1:]
        token_losses = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].reshape(-1, 3),
            shifted.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted)
        expected = torch.stack([
            token_losses[0, :2].mean(),
            token_losses[1, :1].mean(),
        ]).mean()
        observed = sft_lora.ExampleMeanTrainer.compute_loss(
            None,
            Model(),
            {
                "input_ids": torch.ones((2, 4), dtype=torch.long),
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
                "labels": labels,
            },
        )
        self.assertTrue(torch.allclose(observed, expected))

    def test_training_arguments_match_live_transformers_signature(self):
        arguments = argparse.Namespace(
            epochs=3.0,
            per_device_batch_size=7,
            grad_accum=1,
            learning_rate=1e-4,
            save_steps=19,
            gradient_checkpointing=True,
            seed=17,
        )
        kwargs = sft_lora.training_argument_kwargs(arguments, "out", False)
        supported = set(inspect.signature(TrainingArguments).parameters)
        self.assertLessEqual(set(kwargs), supported)
        self.assertNotIn("group_by_length", kwargs)
        instantiated = TrainingArguments(**kwargs)
        self.assertEqual(instantiated.per_device_train_batch_size, 7)


if __name__ == "__main__":
    unittest.main()
