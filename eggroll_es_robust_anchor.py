"""Deterministic document-robust anchor fitness for EGGROLL-ES.

This module is deliberately independent of trainers, vLLM, and evaluation
datasets.  It accepts only per-document selected-token sums and counts from a
train-only anchor.  Every perturbed population row is compared with one exact
alpha-zero reference, and the fitness is the lower endpoint of a fixed paired
document bootstrap.

Raw document identifiers are used only to align caller-provided rows.  The
returned provenance contains their SHA-256 identities and numeric summaries,
never the identifiers or document text.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from typing import Iterable, Mapping, Sequence


DOCUMENT_LCB_SCHEMA = "eggroll-es-document-lcb-anchor-v1"
DOCUMENT_SUMMARY_SCHEMA = "eggroll-es-anchor-document-summary-v1"
DOCUMENT_BOOTSTRAP_PLAN_SCHEMA = (
    "eggroll-es-common-document-bootstrap-plan-v1"
)
DOCUMENT_LCB_RESULT_SCHEMA = "eggroll-es-document-lcb-result-v1"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_715
LOWER_PERCENTILE = 0.025
STANDARDIZATION_EPSILON = 1e-8


def canonical_sha256(value) -> str:
    """Hash one JSON-compatible value with strict canonical serialization."""
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def document_lcb_config() -> dict:
    """Return a fresh copy of the immutable robust-objective semantics."""
    return {
        "schema": DOCUMENT_LCB_SCHEMA,
        "source_split": "anchor_prose",
        "reference": "exact_alpha_zero_selected_plan_weights",
        "document_unit": "document_id",
        "within_document": (
            "sum_selected_token_logprob_and_scored_token_count"
        ),
        "document_identity_in_provenance": "sha256_utf8",
        "bootstrap_document_sampling": "uniform_with_replacement",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_prng": "python_random_mt19937_randrange_v1",
        "common_resamples_across_population": True,
        "percentile": LOWER_PERCENTILE,
        "percentile_interpolation": "linear",
        "fitness": "paired_document_bootstrap_lower_bound",
        "higher_is_better": True,
        "population_standardization": {
            "method": "population_zscore",
            "epsilon": STANDARDIZATION_EPSILON,
            "zero_spread": "return_all_zero_coefficients",
        },
    }


DOCUMENT_LCB_CONFIG_SHA256 = canonical_sha256(document_lcb_config())


def linear_percentile(values: Sequence[float], probability: float) -> float:
    """Use the same linear interpolation as the strict prose gate."""
    if not values:
        raise ValueError("cannot take a percentile of no values")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    if ordered[lower] == ordered[upper]:
        return ordered[lower]
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _document_id_sha256(document_id: str) -> str:
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()


def _validated_documents(
    rows: Iterable[Mapping], *, label: str,
) -> list[dict]:
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError(f"{label} documents must be a sequence of rows")
    try:
        rows = list(rows)
    except TypeError as error:
        raise ValueError(
            f"{label} documents must be a sequence of rows"
        ) from error
    if not rows:
        raise ValueError(f"{label} documents are empty")
    seen_ids = set()
    seen_hashes = set()
    validated = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} document {position} is not an object")
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"{label} document {position} has no document_id"
            )
        if document_id in seen_ids:
            raise ValueError(f"{label} has duplicate document_id")
        identity = _document_id_sha256(document_id)
        if identity in seen_hashes:
            raise ValueError(f"{label} has duplicate document identity hash")
        token_count = row.get("scored_token_count")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            raise ValueError(
                f"{label} document {document_id!r} has an invalid token count"
            )
        token_sum = row.get("sum_token_logprob")
        if isinstance(token_sum, bool) or not isinstance(
            token_sum, (int, float),
        ):
            raise ValueError(
                f"{label} document {document_id!r} has no numeric token sum"
            )
        token_sum = float(token_sum)
        if not math.isfinite(token_sum):
            raise ValueError(
                f"{label} document {document_id!r} has a non-finite token sum"
            )
        seen_ids.add(document_id)
        seen_hashes.add(identity)
        validated.append({
            "document_id_sha256": identity,
            "scored_token_count": token_count,
            "sum_token_logprob": token_sum,
        })
    validated.sort(key=lambda row: row["document_id_sha256"])
    return validated


def _validated_population_rows(rows: Iterable[Mapping]) -> list[dict]:
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("population rows must be a sequence")
    try:
        rows = list(rows)
    except TypeError as error:
        raise ValueError("population rows must be a sequence") from error
    if len(rows) < 2:
        raise ValueError("population requires at least two rows")
    seen_seeds = set()
    validated = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"population row {position} is not an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"population row {position} has an invalid seed")
        if seed in seen_seeds:
            raise ValueError("population has duplicate seeds")
        documents = _validated_documents(
            row.get("documents"), label=f"population seed {seed}",
        )
        seen_seeds.add(seed)
        validated.append({"seed": seed, "documents": documents})
    validated.sort(key=lambda row: row["seed"])
    return validated


def _validated_persisted_documents(
    rows: Iterable[Mapping], *, label: str,
) -> list[dict]:
    """Validate already-hashed numeric summaries without rehashing IDs."""
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError(f"{label} documents must be a sequence of rows")
    try:
        rows = list(rows)
    except TypeError as error:
        raise ValueError(
            f"{label} documents must be a sequence of rows"
        ) from error
    if not rows:
        raise ValueError(f"{label} documents are empty")
    validated = []
    seen_hashes = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "document_id_sha256", "scored_token_count", "sum_token_logprob",
        }:
            raise ValueError(
                f"{label} document {position} has invalid persisted fields"
            )
        identity = row["document_id_sha256"]
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
            or identity in seen_hashes
        ):
            raise ValueError(
                f"{label} document {position} has an invalid identity hash"
            )
        token_count = row["scored_token_count"]
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            raise ValueError(
                f"{label} document {position} has an invalid token count"
            )
        token_sum = row["sum_token_logprob"]
        if (
            isinstance(token_sum, bool)
            or not isinstance(token_sum, (int, float))
            or not math.isfinite(float(token_sum))
        ):
            raise ValueError(
                f"{label} document {position} has a non-finite token sum"
            )
        seen_hashes.add(identity)
        validated.append({
            "document_id_sha256": identity,
            "scored_token_count": token_count,
            "sum_token_logprob": float(token_sum),
        })
    if validated != sorted(
        validated, key=lambda row: row["document_id_sha256"],
    ):
        raise ValueError(f"{label} persisted document order is not canonical")
    return validated


def _validated_persisted_population(rows: Iterable[Mapping]) -> list[dict]:
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("persisted population must be a sequence")
    try:
        rows = list(rows)
    except TypeError as error:
        raise ValueError("persisted population must be a sequence") from error
    if len(rows) < 2:
        raise ValueError("persisted population requires at least two rows")
    population = []
    seen_seeds = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"persisted population row {position} is not an object"
            )
        seed = row.get("seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed in seen_seeds
        ):
            raise ValueError(
                f"persisted population row {position} has an invalid seed"
            )
        seen_seeds.add(seed)
        population.append({
            "seed": seed,
            "documents": _validated_persisted_documents(
                row.get("documents"), label=f"persisted population seed {seed}",
            ),
        })
    if [row["seed"] for row in population] != sorted(seen_seeds):
        raise ValueError("persisted population seed order is not canonical")
    return population


def _require_alignment(reference: Sequence[dict], population: Sequence[dict]):
    reference_ids = [row["document_id_sha256"] for row in reference]
    population_ids = [row["document_id_sha256"] for row in population]
    if population_ids != reference_ids:
        raise ValueError("reference and population document IDs are misaligned")
    for reference_row, population_row in zip(reference, population):
        if (
            population_row["scored_token_count"]
            != reference_row["scored_token_count"]
        ):
            raise ValueError("reference and population token counts drifted")


def reference_document_summary_identity(
    reference_documents: Iterable[Mapping],
) -> dict:
    """Return the persisted identity of an exact-reference numeric summary."""
    reference = _validated_documents(
        reference_documents, label="reference",
    )
    manifest = [
        {
            "document_id_sha256": row["document_id_sha256"],
            "scored_token_count": row["scored_token_count"],
        }
        for row in reference
    ]
    return {
        "document_count": len(reference),
        "scored_token_count": sum(
            row["scored_token_count"] for row in reference
        ),
        "document_manifest_sha256": canonical_sha256(manifest),
        "reference_numeric_summary_sha256": canonical_sha256({
            "schema": DOCUMENT_SUMMARY_SCHEMA,
            "documents": reference,
        }),
    }


def _bootstrap_plan_identity(document_count: int) -> dict:
    return {
        "schema": DOCUMENT_BOOTSTRAP_PLAN_SCHEMA,
        "document_count": document_count,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "sampling": "uniform_documents_with_replacement",
        "draws_per_sample": document_count,
        "index_encoding": "big_endian_uint32_sample_major",
    }


def _bootstrap_lcbs(
    reference: Sequence[dict], populations: Sequence[Sequence[dict]],
) -> tuple[list[float], dict]:
    """Score every population row with one shared deterministic resample plan."""
    document_count = len(reference)
    plan = _bootstrap_plan_identity(document_count)
    plan_digest = hashlib.sha256()
    pack_indices = struct.Struct(f">{document_count}I").pack
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap_deltas = [[] for _ in populations]
    reference_sums = [row["sum_token_logprob"] for row in reference]
    token_counts = [row["scored_token_count"] for row in reference]
    population_sums = [
        [row["sum_token_logprob"] for row in population]
        for population in populations
    ]

    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(document_count) for _ in range(document_count)]
        plan_digest.update(pack_indices(*indices))
        sample_tokens = sum(token_counts[index] for index in indices)
        sample_reference = math.fsum(
            reference_sums[index] for index in indices
        ) / sample_tokens
        for row_index, sums in enumerate(population_sums):
            sample_population = math.fsum(
                sums[index] for index in indices
            ) / sample_tokens
            bootstrap_deltas[row_index].append(
                sample_population - sample_reference
            )

    lower_bounds = [
        linear_percentile(values, LOWER_PERCENTILE)
        for values in bootstrap_deltas
    ]
    plan["indices_sha256"] = plan_digest.hexdigest()
    plan["plan_sha256"] = canonical_sha256(plan)
    return lower_bounds, plan


def standardize_robust_scores(
    scores: Sequence[float], *, epsilon: float = STANDARDIZATION_EPSILON,
) -> dict:
    """Standardize population fitness and fail closed on zero spread."""
    scores = [float(score) for score in scores]
    if len(scores) < 2:
        raise ValueError("robust scores require at least two values")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("robust scores must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("standardization epsilon must be finite and positive")
    mean = math.fsum(scores) / len(scores)
    variance = math.fsum((score - mean) ** 2 for score in scores) / len(scores)
    standard_deviation = math.sqrt(variance)
    zero_spread = standard_deviation <= epsilon
    standardized = (
        [0.0] * len(scores)
        if zero_spread
        else [
            (score - mean) / (standard_deviation + epsilon)
            for score in scores
        ]
    )
    return {
        "schema": "eggroll-es-robust-score-standardization-v1",
        "count": len(scores),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "epsilon": epsilon,
        "zero_spread": zero_spread,
        "standardized_scores": standardized,
    }


def _build_result(reference: list[dict], population: list[dict]) -> dict:
    """Build complete provenance from validated, canonically ordered rows."""
    for row in population:
        _require_alignment(reference, row["documents"])

    lower_bounds, bootstrap_plan = _bootstrap_lcbs(
        reference, [row["documents"] for row in population],
    )
    total_tokens = sum(row["scored_token_count"] for row in reference)
    reference_mean = math.fsum(
        row["sum_token_logprob"] for row in reference
    ) / total_tokens
    scored_population = []
    for row, lower_bound in zip(population, lower_bounds):
        population_mean = math.fsum(
            document["sum_token_logprob"] for document in row["documents"]
        ) / total_tokens
        numeric_summary = {
            "schema": DOCUMENT_SUMMARY_SCHEMA,
            "documents": row["documents"],
        }
        scored_population.append({
            "seed": row["seed"],
            "documents": row["documents"],
            "numeric_summary_sha256": canonical_sha256(numeric_summary),
            "mean_token_logprob": population_mean,
            "mean_delta": population_mean - reference_mean,
            "bootstrap_lower_confidence_bound": lower_bound,
        })

    robust_scores = [
        {
            "seed": row["seed"],
            "score": row["bootstrap_lower_confidence_bound"],
        }
        for row in scored_population
    ]
    standardization = standardize_robust_scores(
        [row["score"] for row in robust_scores],
    )
    standardized_scores = [
        {"seed": row["seed"], "score": score}
        for row, score in zip(
            robust_scores, standardization["standardized_scores"],
        )
    ]
    reference_summary = {
        "schema": DOCUMENT_SUMMARY_SCHEMA,
        "documents": reference,
    }
    document_manifest = [
        {
            "document_id_sha256": row["document_id_sha256"],
            "scored_token_count": row["scored_token_count"],
        }
        for row in reference
    ]
    result = {
        "schema": DOCUMENT_LCB_RESULT_SCHEMA,
        "config": document_lcb_config(),
        "config_sha256": DOCUMENT_LCB_CONFIG_SHA256,
        "document_manifest": document_manifest,
        "document_manifest_sha256": canonical_sha256(document_manifest),
        "reference": {
            "documents": reference,
            "document_count": len(reference),
            "scored_token_count": total_tokens,
            "mean_token_logprob": reference_mean,
            "numeric_summary_sha256": canonical_sha256(reference_summary),
        },
        "bootstrap_plan": bootstrap_plan,
        "population": scored_population,
        "population_numeric_summary_sha256": canonical_sha256([
            {
                "seed": row["seed"],
                "documents": row["documents"],
            }
            for row in scored_population
        ]),
        "robust_scores": robust_scores,
        "robust_scores_sha256": canonical_sha256(robust_scores),
        "standardization": standardization,
        "standardized_scores": standardized_scores,
        "standardized_scores_sha256": canonical_sha256(standardized_scores),
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def score_population_document_lcbs(
    reference_documents: Iterable[Mapping],
    population_rows: Iterable[Mapping],
) -> dict:
    """Return auditable document-LCB fitness for one complete population.

    Inputs contain only ``document_id``, ``scored_token_count``, and
    ``sum_token_logprob``.  The result replaces document IDs with hashes and
    retains the numeric summaries needed for independent recomputation.
    """
    reference = _validated_documents(
        reference_documents, label="reference",
    )
    population = _validated_population_rows(population_rows)
    return _build_result(reference, population)


def recompute_document_lcb_result(result: Mapping) -> dict:
    """Regenerate every numeric statistic and hash from persisted summaries."""
    if not isinstance(result, Mapping):
        raise ValueError("document-LCB result must be an object")
    reference_block = result.get("reference")
    if not isinstance(reference_block, Mapping):
        raise ValueError("document-LCB result has no reference summary")
    reference = _validated_persisted_documents(
        reference_block.get("documents"), label="persisted reference",
    )
    population = _validated_persisted_population(result.get("population"))
    return _build_result(reference, population)


def validate_document_lcb_result(result: Mapping) -> dict:
    """Fail closed unless persisted provenance exactly recomputes."""
    expected = recompute_document_lcb_result(result)
    if result != expected:
        raise ValueError("document-LCB result does not exactly recompute")
    return expected
