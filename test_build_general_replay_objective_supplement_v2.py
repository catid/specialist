from __future__ import annotations

from collections import Counter

import build_general_replay_objective_supplement_v2 as supplement
import general_replay_v1 as replay


class SyntheticQwenTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 6}[token]

    def encode(self, value, add_special_tokens=False):
        assert value == "\n" and add_special_tokens is False
        return [5]

    def decode(self, ids, skip_special_tokens=False):
        return {2: "system", 3: "user", 4: "assistant"}.get(ids[0], "x")

    def apply_chat_template(self, messages, **kwargs):
        result = []
        roles = {"system": 2, "user": 3, "assistant": 4}
        for index, message in enumerate(messages):
            result.extend([1, roles[message["role"]], 5])
            if message["role"] == "assistant":
                if index == len(messages) - 1:
                    result.extend([8, 9])
                result.extend([30] * 64)
            else:
                result.extend([20] * 8)
            result.extend([6, 7])
        if kwargs["add_generation_prompt"]:
            result.extend([1, 4, 5, 8, 9])
        return result


def test_diverse_pool_is_unique_objective_and_family_complete():
    specs, families = supplement.build_candidate_pool()
    assert Counter(item["category"] for item in specs) == Counter(
        supplement.POOL_COUNTS
    )
    assert len({item["spec_id"] for item in specs}) == len(specs)
    assert len({item["source_group_id"] for item in specs}) == len(specs)
    assert len({item["prompt_identity_sha256"] for item in specs}) == len(specs)
    assert all(
        item["verifier"]["type"] != "approval_required_v1"
        for item in specs
    )
    represented = {
        category: {
            families[item["spec_id"]]
            for item in specs if item["category"] == category
        }
        for category in supplement.SCOPED_CATEGORIES
    }
    assert represented == {
        category: set(supplement.FAMILIES[category])
        for category in supplement.SCOPED_CATEGORIES
    }


def test_requested_math_and_coding_families_compile_and_verify():
    specs = [
        supplement._math_spec(index)
        for index in range(len(supplement.MATH_FAMILIES))
    ] + [
        supplement._coding_spec(index)
        for index in range(len(supplement.CODING_FAMILIES))
    ]
    rows, audit = replay.build_reference_compiler_rows(
        specs, SyntheticQwenTokenizer()
    )
    assert len(rows) == len(specs)
    assert audit["compiled_rows"] == len(specs)
    assert all(row["verifier"]["status"] == "passed" for row in rows)
    assert {
        "fraction_sum", "modular_arithmetic", "rectangle_geometry",
        "arithmetic_sequence", "finite_probability", "two_variable_system",
    }.issubset(set(supplement.MATH_FAMILIES))
    assert {
        "inclusive_range_debug", "accumulator_debug",
    }.issubset(set(supplement.CODING_FAMILIES))


def test_multilingual_pairs_are_broad_exact_and_slot_sealed():
    templates = supplement.TRANSLATION_TEMPLATES
    languages = Counter(language for language, _, _ in templates)
    assert len(templates) == 96
    assert len(languages) == 8
    assert set(languages.values()) == {12}
    assert len(set(templates)) == len(templates)
    for index in range(len(templates)):
        spec = supplement._multilingual_spec(index)
        message = replay.compile_objective_reference_message(spec)
        assert replay.verify_candidate(spec, message)["passed"] is True
        assert "{" not in spec["messages"][-1]["content"]
        assert "{" not in message["content"]


def test_diversity_gate_rejects_concentrated_selection():
    specs = [supplement._math_spec(index) for index in range(8)]
    rows, _ = replay.build_reference_compiler_rows(
        specs, SyntheticQwenTokenizer()
    )
    family_by_spec = {
        row["lineage"]["spec_id"]: "one_family" for row in rows
    }
    try:
        supplement.diversity_audit(rows, family_by_spec)
    except RuntimeError as exc:
        assert "diversity gates" in str(exc)
    else:
        raise AssertionError("concentrated selection passed diversity gate")
