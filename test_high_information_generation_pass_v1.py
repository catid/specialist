from __future__ import annotations

import pytest

import verify_high_information_generation_pass_v1 as pass_verifier


def generator(pass_id: str, request_id: str) -> dict:
    spec = pass_verifier.PASS_SPECS[pass_id]
    value = {
        "model": "Qwen3.6-35B-A3B",
        "checkpoint": "sealed_local_base",
        "engine": "vllm-0.25.0",
        "dtype": "bfloat16",
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "seed": pass_verifier.expected_seed(pass_id, request_id),
        "enable_thinking": False,
    }
    if spec["generator_generation_pass"] is not None:
        value["generation_pass"] = spec["generator_generation_pass"]
    return value


def test_primary_and_fill_paths_and_seeds_are_distinct():
    request_id = "request-synthetic"
    assert pass_verifier.expected_seed(
        pass_verifier.PRIMARY_PASS_ID, request_id
    ) != pass_verifier.expected_seed(pass_verifier.FILL_PASS_ID, request_id)
    assert pass_verifier.generation_paths(
        pass_verifier.PRIMARY_PASS_ID, 0
    ) != pass_verifier.generation_paths(pass_verifier.FILL_PASS_ID, 0)
    assert pass_verifier.structural_paths(
        pass_verifier.PRIMARY_PASS_ID, 0
    ) != pass_verifier.structural_paths(pass_verifier.FILL_PASS_ID, 0)
    assert len(
        pass_verifier.PASS_SPECS[pass_verifier.PRIMARY_PASS_ID]["runtime_workers"]
    ) == 1
    assert len(
        pass_verifier.PASS_SPECS[pass_verifier.FILL_PASS_ID]["runtime_workers"]
    ) == 2


def test_fill_generator_requires_exact_pass_sampling_and_identity():
    request_id = "request-synthetic"
    expected = generator(pass_verifier.FILL_PASS_ID, request_id)
    pass_verifier.validate_generator(
        expected, pass_verifier.FILL_PASS_ID, request_id
    )
    for key, changed in (
        ("generation_pass", pass_verifier.PRIMARY_PASS_ID),
        ("temperature", 0.3),
        ("seed", expected["seed"] + 1),
    ):
        mutated = dict(expected)
        mutated[key] = changed
        with pytest.raises(RuntimeError, match="does not match"):
            pass_verifier.validate_generator(
                mutated, pass_verifier.FILL_PASS_ID, request_id
            )


def test_primary_candidate_cannot_be_mislabeled_as_fill():
    request_id = "request-synthetic"
    with pytest.raises(RuntimeError, match="does not match"):
        pass_verifier.validate_generator(
            generator(pass_verifier.PRIMARY_PASS_ID, request_id),
            pass_verifier.FILL_PASS_ID,
            request_id,
        )
