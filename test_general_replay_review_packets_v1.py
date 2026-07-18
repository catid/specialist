from __future__ import annotations

import build_general_replay_approval_packet_v1 as packet
import build_general_replay_priority_review_v1 as priority


def _item(spec: str, category: str, counts: list[int]) -> dict:
    return {
        "spec_id": spec,
        "source_group_id": "group-" + spec,
        "category": category,
        "messages": [{"role": "user", "content": "Synthetic prompt U-0001."}],
        "candidates": [
            {
                "response_sha256": str(index) * 64,
                "assistant_token_count": count,
                "assistant_message": {
                    "role": "assistant",
                    "content": (
                        "Cannot determine U-0001 because evidence is missing; "
                        "please provide the record."
                    ),
                },
            }
            for index, count in enumerate(counts, 1)
        ],
    }


def test_priority_selector_is_exact_minimum_and_group_disjoint():
    records = [
        _item("a", "uncertainty_hallucination_resistance", [7, 5]),
        _item("b", "uncertainty_hallucination_resistance", [4]),
        _item("c", "uncertainty_hallucination_resistance", [3]),
    ]
    selected = priority.minimum_exact_group_choices(records, 10)
    assert len(selected) == 2
    assert sum(candidate["assistant_token_count"] for _, candidate in selected) == 10
    assert len({item["source_group_id"] for item, _ in selected}) == 2


def test_subjective_rubrics_are_explicit_and_never_autoapprove():
    for rubric_id in (
        "ordinary-conversation-warm-practical-v1",
        "bilingual-translation-spanish-v1",
        "safe-boundary-constructive-alternative-v1",
        "uncertainty-no-fabrication-v1",
    ):
        rubric = packet.rubric_definition(rubric_id)
        assert rubric["rubric_id"] == rubric_id
        assert rubric["decision"] == "approve only when every criterion passes"
        assert len(rubric["criteria"]) == 4
