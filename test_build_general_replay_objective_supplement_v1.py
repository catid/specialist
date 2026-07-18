from __future__ import annotations

from collections import Counter

import build_general_replay_objective_supplement_v1 as supplement
import general_replay_v1 as replay


class SyntheticQwenTokenizer:
    """Synthetic official-template double; no repository data is opened."""

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 6}[token]

    def encode(self, value, add_special_tokens=False):
        assert value == "\n" and add_special_tokens is False
        return [5]

    def decode(self, ids, skip_special_tokens=False):
        return {2: "system", 3: "user", 4: "assistant"}.get(
            ids[0], "synthetic"
        )

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["enable_thinking"] is False
        input_ids = []
        role_ids = {"system": 2, "user": 3, "assistant": 4}
        for index, message in enumerate(messages):
            input_ids.extend([1, role_ids[message["role"]], 5])
            if message["role"] == "assistant":
                if index == len(messages) - 1:
                    input_ids.extend([8, 9])
                input_ids.extend([30] * 64)
            else:
                input_ids.extend([20] * 8)
            input_ids.extend([6, 7])
        if kwargs["add_generation_prompt"]:
            input_ids.extend([1, 4, 5, 8, 9])
        return input_ids


def test_candidate_pool_is_deterministic_objective_and_identity_disjoint():
    first = supplement.build_candidate_pool()
    second = supplement.build_candidate_pool()
    assert first == second
    assert Counter(item["category"] for item in first) == Counter(
        supplement.POOL_COUNTS
    )
    assert {item["category"] for item in first} == set(
        supplement.SCOPED_CATEGORIES
    )
    assert len({item["spec_id"] for item in first}) == len(first)
    assert len({item["source_group_id"] for item in first}) == len(first)
    assert len({item["prompt_identity_sha256"] for item in first}) == len(first)
    assert all(
        item["lineage"] == {
            "source_kind": "deterministic_synthetic_v1",
            "direct_benchmark_prompt": False,
            "protected_source_access": False,
            "parent_artifacts": [],
        }
        for item in first
    )
    assert all(
        item["verifier"]["type"] != "approval_required_v1"
        for item in first
    )
    replay.validate_prompt_specs(first)


def test_multilingual_pool_uses_only_hand_authored_exact_templates():
    specs = [
        supplement._multilingual_spec(index)
        for index in range(len(supplement.TRANSLATION_TEMPLATES) * 2)
    ]
    replay.validate_prompt_specs(specs)
    assert all(item["verifier"]["type"] == "exact_text_v1" for item in specs)
    assert all(item["verifier"]["status"] == "ready" for item in specs)
    assert all(
        "{" not in item["messages"][-1]["content"]
        and "{" not in item["verifier"]["config"]["expected"]
        for item in specs
    )
    for item in specs:
        reference = replay.compile_objective_reference_message(item)
        assert replay.verify_candidate(item, reference)["passed"] is True


def test_selected_references_pass_shared_verifier_and_synthetic_official_mask():
    tokenizer = SyntheticQwenTokenizer()
    pool = [
        supplement.POOL_BUILDERS[category](0)
        for category in supplement.SCOPED_CATEGORIES
    ]
    required = {category: 65 for category in supplement.SCOPED_CATEGORIES}
    specs, rows, audit = supplement.select_exact_supplement(
        pool, tokenizer, required
    )
    assert len(specs) == len(rows) == len(supplement.SCOPED_CATEGORIES)
    assert {row["category"] for row in rows} == set(
        supplement.SCOPED_CATEGORIES
    )
    assert all(row["assistant_token_count"] == 65 for row in rows)
    assert all(row["verifier"]["status"] == "passed" for row in rows)
    assert all(
        item["exact_unique_source_group_subset_reached"] is True
        and item["selected_assistant_tokens"] == 65
        for item in audit["categories"].values()
    )
    for row in rows:
        assert sum(replay.assistant_token_mask(
            tokenizer, row["messages"], row["tools"]
        )) == row["assistant_token_count"]
