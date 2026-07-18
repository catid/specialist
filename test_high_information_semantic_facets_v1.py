from __future__ import annotations

import high_information_semantic_facets_v1 as facets


def test_when_and_where_extracts_exact_two_facets_and_catches_missing_location():
    question = "When and where did Dan Carabas establish Shibari Studio Berlin?"
    extracted = facets.extract_requested_facets(question)
    assert [item["kind"] for item in extracted] == ["temporal", "location"]
    signals = facets.deterministic_facet_signals(
        question, "Dan Carabas established it in 2024."
    )
    assert [item["deterministic_status"] for item in signals] == [
        "present",
        "missing",
    ]
    assert facets.missing_high_confidence_facets(
        question, "Dan Carabas established it in 2024."
    ) == ["facet_01_location"]


def test_when_and_where_complete_answer_has_both_signals():
    question = "When and where did Dan Carabas establish Shibari Studio Berlin?"
    signals = facets.deterministic_facet_signals(
        question, "Dan Carabas established Shibari Studio in Berlin in 2024."
    )
    assert [item["deterministic_status"] for item in signals] == [
        "present",
        "present",
    ]


def test_how_many_is_quantity_not_method():
    extracted = facets.extract_requested_facets("How many wraps are described?")
    assert [item["kind"] for item in extracted] == ["quantity"]


def test_generic_question_is_unresolved_not_automatically_accepted():
    signals = facets.deterministic_facet_signals(
        "What material is described?", "The material is hemp."
    )
    assert signals[0]["kind"] == "requested_content"
    assert signals[0]["deterministic_status"] == "unresolved"


def test_decade_one_word_person_and_bare_location_surface_forms_are_seen():
    temporal = facets.deterministic_facet_signals(
        "When and where did it operate?", "It operated in Berlin during the 2020s."
    )
    assert [item["deterministic_status"] for item in temporal] == [
        "present",
        "present",
    ]
    person = facets.deterministic_facet_signals(
        "Who established it?", "It was established by Carabas."
    )
    assert person[0]["deterministic_status"] == "present"
    location = facets.deterministic_facet_signals(
        "Where was it established?", "Berlin."
    )
    assert location[0]["deterministic_status"] == "present"
