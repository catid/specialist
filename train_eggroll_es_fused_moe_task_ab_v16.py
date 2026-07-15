#!/usr/bin/env python3
"""Timed, no-update V13 diagnostic adapter for the V16 fused-MoE A/B."""

from __future__ import annotations

import copy
import math
import os
import statistics
import sys
import time
from pathlib import Path

import eggroll_es_fused_moe_task_ab_preregistration_v16 as prereg_v16
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
canonical_sha256 = anchor_v13.canonical_sha256
_DEFAULT_LAYER_PLAN_BUNDLE = None


def set_default_layer_plan_bundle_v16(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    anchor_v13.validate_frozen_layer_plan_bundle_v13(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def _metric_summary_v16(values, count):
    values = [float(value) for value in values]
    if (
        len(values) != count
        or not all(
            math.isfinite(value) and -1.0 - 1e-12 <= value <= 1.0 + 1e-12
            for value in values
        )
    ):
        raise RuntimeError("v16 compact estimator endpoint changed")
    values = [max(-1.0, min(1.0, value)) for value in values]
    return {
        "count": count,
        "median": float(statistics.median(values)),
        "worst": min(values),
    }


def _validate_generation_timing_v16(timing):
    waves = timing.get("wave_seconds") if isinstance(timing, dict) else None
    if (
        set(timing or {}) != {
            "clock", "boundary", "warmup_generation_calls_per_engine",
            "wave_seconds", "total_seconds",
        }
        or timing.get("clock") != "time.perf_counter_ns"
        or timing.get("boundary") != "blocking_four_engine_generation_resolve_only"
        or timing.get("warmup_generation_calls_per_engine") != 1
        or not isinstance(waves, list)
        or len(waves) != prereg_v16.TIMED_SIGNED_WAVE_COUNT_V16
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in waves
        )
        or not math.isclose(
            float(timing.get("total_seconds", -1.0)),
            math.fsum(float(value) for value in waves),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("v16 generation timing contract changed")
    return timing


class TimedTaskABMixinV16(anchor_v13.TrainPanelDiagnosticMixinV13):
    configure_task_ab_v16 = (
        anchor_v13.TrainPanelDiagnosticMixinV13.configure_train_panels_v13
    )

    def estimate_task_ab_v16(self, seeds):
        seeds = [int(seed) for seed in seeds]
        if seeds != anchor_v13.PERTURBATION_SEEDS_V13:
            raise RuntimeError("v16 fixed V13 perturbation basis changed")
        prepared, combined_prompts = self._prepared_panels_v13()

        # Exactly one reference-weight generation call per engine warms the
        # task shapes. It is intentionally outside the measured boundary.
        warmup = self._resolve([
            engine.generate.remote(
                list(combined_prompts), self._dense_sampling_params_v4(0),
                use_tqdm=False,
            )
            for engine in self.engines
        ])
        if (
            not isinstance(warmup, list)
            or len(warmup) != anchor_v13.REQUIRED_ENGINE_COUNT
            or any(len(batch) != len(combined_prompts) for batch in warmup)
        ):
            raise RuntimeError("v16 task warmup generation incomplete")
        warmup = None

        pre_probe = self._base_probe_v13(prepared, combined_prompts)
        captures = {
            name: {
                "weighted_sign_scores": {
                    sign: [] for sign in anchor_v13.SIGNS_V13
                },
                "unweighted_sign_scores": {
                    sign: [] for sign in anchor_v13.SIGNS_V13
                },
                "stratum_sign_scores": {
                    stratum: {
                        sign: [] for sign in anchor_v13.SIGNS_V13
                    }
                    for stratum in anchor_v13.panel_sampler.STRATA
                },
                "weighted_stratum_contributions": {
                    stratum: {
                        sign: [] for sign in anchor_v13.SIGNS_V13
                    }
                    for stratum in anchor_v13.panel_sampler.STRATA
                },
                "dense_result_sha256": {
                    sign: [] for sign in anchor_v13.SIGNS_V13
                },
            }
            for name in anchor_v13.PANEL_NAMES_V13
        }
        wave_seconds = []
        for start in range(
            0, len(seeds), anchor_v13.REQUIRED_ENGINE_COUNT
        ):
            wave = anchor_v13.anchor_v10.validate_full_engine_wave_v10(
                seeds[start:start + anchor_v13.REQUIRED_ENGINE_COUNT],
                anchor_v13.REQUIRED_ENGINE_COUNT,
            )
            for sign, negate in (("plus", False), ("minus", True)):
                batches = None
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights",
                            args=(int(seed), self.sigma, negate),
                        )
                        for index, seed in enumerate(wave)
                    ])
                    started_ns = time.perf_counter_ns()
                    batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(combined_prompts),
                            self._dense_sampling_params_v4(0),
                            use_tqdm=False,
                        )
                        for index, _seed in enumerate(wave)
                    ])
                    completed_ns = time.perf_counter_ns()
                    wave_seconds.append((completed_ns - started_ns) / 1e9)
                finally:
                    self._restore_all_engines_exact()
                if (
                    not isinstance(batches, list)
                    or len(batches) != anchor_v13.REQUIRED_ENGINE_COUNT
                    or any(
                        len(batch) != len(combined_prompts) for batch in batches
                    )
                ):
                    raise RuntimeError("v16 resident generation wave incomplete")
                for engine_index, _seed in enumerate(wave):
                    for name in anchor_v13.PANEL_NAMES_V13:
                        panel = prepared[name]
                        slice_start, slice_end = panel["slice"]
                        score = anchor_v13._score_panel_outputs_v13(
                            panel["panel"], panel["dense_items"],
                            batches[engine_index][slice_start:slice_end],
                        )
                        capture = captures[name]
                        capture["weighted_sign_scores"][sign].append(
                            score["weighted_mean"]
                        )
                        capture["unweighted_sign_scores"][sign].append(
                            score["unweighted_mean"]
                        )
                        capture["dense_result_sha256"][sign].append(
                            score["dense_result_sha256"]
                        )
                        for stratum in anchor_v13.panel_sampler.STRATA:
                            capture["stratum_sign_scores"][stratum][sign].append(
                                score["stratum_unweighted_means"][stratum]
                            )
                            capture["weighted_stratum_contributions"][stratum][
                                sign
                            ].append(
                                score["weighted_stratum_contributions"][stratum]
                            )

        checks = self._resolve([
            engine.collective_rpc.remote(
                "verify_self_exact_reference", args=()
            )
            for engine in self.engines
        ])
        if not anchor_v13.anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks,
            lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v16 post-population exact reference check failed")
        boundary = self._population_boundary_audit_v4(0)
        post_probe = self._base_probe_v13(prepared, combined_prompts)
        if pre_probe != post_probe:
            raise RuntimeError("v16 alpha-zero five-panel base probe drifted")
        analysis = anchor_v13.analyze_panel_responses_v13(captures)
        panel_contract = {
            name: {
                "role": prepared[name]["panel"]["role"],
                "rows": anchor_v13.panel_sampler.PANEL_SIZE,
                "ordered_row_identity_sha256": prepared[name]["panel"][
                    "ordered_row_identity_sha256"
                ],
                "templated_prompt_answer_sha256": prepared[name][
                    "templated_prompt_answer_sha256"
                ],
            }
            for name in anchor_v13.PANEL_NAMES_V13
        }
        artifact = {
            "schema": "eggroll-es-five-panel-resident-sign-diagnostic-v13",
            "iteration": 0,
            "alpha": 0.0,
            "model_update_applied": False,
            "applications": [],
            "perturbation_basis": {
                "basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
                "basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
                "seeds": seeds,
                "seed_sha256": canonical_sha256(seeds),
                "sign_order": list(anchor_v13.SIGNS_V13),
            },
            "panel_bundle_content_sha256": self._v13_panel_bundle[
                "content_sha256_before_self_field"
            ],
            "panel_contract": panel_contract,
            "common_random_numbers": {
                "generation_seed": 43,
                "temperature": 0.0,
                "same_order_every_direction_and_sign": True,
                "combined_panel_order": list(anchor_v13.PANEL_NAMES_V13),
            },
            "responses": captures,
            "analysis": analysis,
            "identity_audit": {
                "pre_probe": pre_probe,
                "post_probe": post_probe,
                "exact_reference_checks": checks,
                "passed": True,
            },
            "population_boundary_audit_v4": boundary,
            "hardware_coverage": {
                "engine_count": 4,
                "tp_per_engine": 1,
                "gpu_ids": [0, 1, 2, 3],
                "population_waves": 8,
                "signed_waves": 16,
                "partial_waves": 0,
                "all_engines_generate_every_signed_wave": True,
            },
            "interpretation": "diagnostic_only_no_promotion_decision",
        }
        artifact["content_sha256_before_self_field"] = canonical_sha256(artifact)
        anchor_v13.validate_diagnostic_v13(artifact)
        timing = {
            "clock": "time.perf_counter_ns",
            "boundary": "blocking_four_engine_generation_resolve_only",
            "warmup_generation_calls_per_engine": 1,
            "wave_seconds": wave_seconds,
            "total_seconds": math.fsum(wave_seconds),
        }
        _validate_generation_timing_v16(timing)
        return artifact, timing

    def configure_train_panels_v13(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v16 requires the task A/B entrypoint")

    def estimate_train_panels_v13(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v16 requires timed task A/B generation")


def compact_arm_v16(diagnostic, timing):
    anchor_v13.validate_diagnostic_v13(diagnostic)
    _validate_generation_timing_v16(timing)
    analysis = diagnostic["analysis"]
    pairwise = list(analysis["optimization_pairwise"].values())
    aggregate = analysis["robust_optimization_aggregate"]
    aggregate_values = aggregate["coefficients"]
    optimization_values = [
        analysis["panel_analysis"][name]["coefficients"]
        for name in anchor_v13.OPTIMIZATION_PANELS_V13
    ]
    screens = list(analysis["train_screen_transfer"].values())
    stability = {
        "optimization_pairwise_cosine": _metric_summary_v16(
            [item["cosine"] for item in pairwise], 3,
        ),
        "optimization_pairwise_sign_agreement": _metric_summary_v16(
            [
                item["sign_agreement"]["all_coordinate_fraction"]
                for item in pairwise
            ],
            3,
        ),
        "aggregate_to_optimization_cosine": _metric_summary_v16(
            [
                anchor_v13._cosine_v13(aggregate_values, values)
                for values in optimization_values
            ],
            3,
        ),
        "aggregate_to_optimization_sign_agreement": _metric_summary_v16(
            [
                anchor_v13._sign_agreement_v13(
                    aggregate_values, values,
                )["all_coordinate_fraction"]
                for values in optimization_values
            ],
            3,
        ),
        "train_screen_cosine": _metric_summary_v16(
            [
                item["cosine_to_frozen_optimization_aggregate"]
                for item in screens
            ],
            2,
        ),
        "train_screen_sign_agreement": _metric_summary_v16(
            [
                item["sign_agreement_to_frozen_optimization_aggregate"]
                ["all_coordinate_fraction"]
                for item in screens
            ],
            2,
        ),
    }
    dense_manifest = {
        name: {
            sign: diagnostic["responses"][name]["dense_result_sha256"][sign]
            for sign in anchor_v13.SIGNS_V13
        }
        for name in anchor_v13.PANEL_NAMES_V13
    }
    dense_hash = canonical_sha256(dense_manifest)
    compact_estimator = {
        "stability": stability,
        "robust_aggregate": {
            "coefficient_sha256": aggregate["coefficient_sha256"],
            "l2_norm": aggregate["l2_norm"],
            "nonzero_coordinate_count": aggregate["nonzero_coordinate_count"],
        },
    }
    task_output_hash = canonical_sha256({
        "dense_result_manifest_sha256": dense_hash,
        "pre_post_base_probe": diagnostic["identity_audit"]["pre_probe"],
        "panel_contract": diagnostic["panel_contract"],
    })
    integrity = {
        "alpha_zero_no_applications": (
            diagnostic["alpha"] == 0.0 and diagnostic["applications"] == []
        ),
        "model_update_applied_false": (
            diagnostic["model_update_applied"] is False
        ),
        "pre_post_base_probe_equal": (
            diagnostic["identity_audit"]["pre_probe"]
            == diagnostic["identity_audit"]["post_probe"]
        ),
        "population_boundary_passed": diagnostic[
            "population_boundary_audit_v4"
        ]["passed"] is True,
        "hardware_contract_passed": diagnostic["hardware_coverage"] == {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "population_waves": 8, "signed_waves": 16, "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        },
        "dense_hash_coverage_passed": all(
            len(dense_manifest[name][sign]) == 32
            for name in anchor_v13.PANEL_NAMES_V13
            for sign in anchor_v13.SIGNS_V13
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError("v16 task arm integrity failed")
    return {
        "diagnostic_content_sha256": diagnostic[
            "content_sha256_before_self_field"
        ],
        "dense_result_manifest_sha256": dense_hash,
        "task_output_sha256": task_output_hash,
        "compact_estimator": compact_estimator,
        "generation_timing": {
            "wave_seconds": copy.deepcopy(timing["wave_seconds"]),
            "total_seconds": timing["total_seconds"],
        },
        "all_integrity_audits_passed": True,
        "persisted_raw_content": False,
    }


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    anchor_v13.validate_frozen_layer_plan_bundle_v13(captured)
    parent = anchor_v13.load_trainer(captured)

    class TimedTaskABTrainerV16(TimedTaskABMixinV16, parent):
        pass

    return TimedTaskABTrainerV16
