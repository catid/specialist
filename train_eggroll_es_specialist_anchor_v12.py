#!/usr/bin/env python3
"""Consensus-direction and train-only candidate-screen contracts for V12."""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import numpy as np

import eggroll_es_robust_anchor as robust_anchor
import eggroll_es_worker_v12 as worker_v12
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v11c as anchor_v11c


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v12.ConsensusCandidateWorkerExtensionV12"
)
REQUIRED_ENGINE_COUNT = 4
SCREEN_ROWS_V12 = 64
SCREEN_LABELS_V12 = ("C45", "C46")
ALPHA_GRID_V12 = [0.0, 7.8125e-7, 1.5625e-6]
EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12 = (
    "1a85502564020048f634b0a8ced3952343f135f67ffd1faad1aaa697aebea8a8"
)
EXPECTED_CELL_COSINE_V12 = 0.938139575931748
EXPECTED_ANCHOR_COSINE_V12 = 0.8527515739920155
FROZEN_TRAIN_ARROW_SHA256_V12 = (
    "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"
)
FROZEN_TRAIN_ROWS_V12 = 794
SCREEN_MANIFESTS_V12 = {
    "C45": {
        "seed": 45,
        "rows": 64,
        "sha256": (
            "45944527c98eaf7446a89f85672f83ec6c42d28047288cddbc66f1d547b09490"
        ),
        "row_identity_sha256": (
            "baa447245cdc53f168a676dc9c8d1ab57bd1c4f38ee8cac427875e7e8fd03b3c"
        ),
    },
    "C46": {
        "seed": 46,
        "rows": 64,
        "sha256": (
            "be80a4973772d0457d769fcb12462192bc1153f7eec9e301f961825c67e95e74"
        ),
        "row_identity_sha256": (
            "b3c1b7e84835b80d296254c9dc90c484b3b15526f4be7c0f306cf1ea4b566354"
        ),
    },
}
EXPECTED_CONSENSUS_COEFFICIENTS_V12 = [
    -0.3351909573474715, 0.549533653720475, 1.1106392502352487,
    1.4656329819941813, 0.0198887494129567, -0.9286947426544143,
    0.9416469827512188, -0.24189446719627244, -0.6297520877582911,
    0.1380089616802588, -0.453336373330877, -1.5110211348031342,
    -1.8551404740981603, 1.3819983529597273, 1.1314454215148866,
    -1.7708723737215681, 1.214093829379483, -0.08369237064291975,
    0.46928587272594874, 0.04681362949617495, -0.44980926309339553,
    0.6672629588224755, -0.6223861805769366, 2.0346783512283335,
    0.6217549643732979, -1.0339652477269574, -0.9932247685543888,
    -0.019375048636848843, -0.3893596216489531,
    -0.26449771913412007, -1.5473661990064362, 1.3368950696364776,
]

canonical_sha256 = anchor_v11.canonical_sha256
coefficient_sha256 = anchor_v11.coefficient_sha256
file_sha256 = anchor_v11.file_sha256
load_anchor_prose = anchor_v11.load_anchor_prose
anchor_v4 = anchor_v11.anchor_v4
anchor_v5 = anchor_v11.anchor_v5
anchor_v10 = anchor_v11.anchor_v10
_DEFAULT_LAYER_PLAN_BUNDLE = None


def _unit(values, label):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (32,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"v12 {label} vector is invalid")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"v12 {label} vector has zero or invalid norm")
    return vector / norm


def _cosine(left, right):
    return float(np.dot(_unit(left, "left"), _unit(right, "right")))


def _centered_cosine(left, right):
    vector = np.asarray(right, dtype=np.float64)
    return _cosine(left, vector - float(np.mean(vector)))


def consensus_from_resident_cross_v12(cross):
    """Collapse duplicate anchor cells and build sqrt(32)-norm consensus."""
    if not isinstance(cross, dict):
        raise RuntimeError("v12 resident-sign cross artifact is missing")
    cells = cross.get("cells")
    if not isinstance(cells, dict) or set(cells) != {
        "D43xA43", "D43xA44", "D44xA43", "D44xA44",
    }:
        raise RuntimeError("v12 crossed-cell coverage changed")
    c43_a43 = cells["D43xA43"].get("coefficients")
    c43_a44 = cells["D43xA44"].get("coefficients")
    c44_a43 = cells["D44xA43"].get("coefficients")
    c44_a44 = cells["D44xA44"].get("coefficients")
    if c43_a43 != c43_a44 or c44_a43 != c44_a44:
        raise RuntimeError("v12 anchor-cell duplicates no longer collapse exactly")
    c43 = _unit(c43_a43, "D43")
    c44 = _unit(c44_a43, "D44")
    consensus = math.sqrt(32.0) * _unit(c43 + c44, "unit-sum")
    coefficients = [float(value) for value in consensus]
    seeds = cross.get("base_perturbation_seeds")
    if seeds != anchor_v11.PERTURBATION_SEEDS_V11:
        raise RuntimeError("v12 perturbation seed order changed")
    identity = coefficient_sha256(seeds, coefficients)
    central_anchor = cross.get("central_anchor_scores", {})
    if (
        not isinstance(central_anchor, dict)
        or central_anchor.get("A43") != central_anchor.get("A44")
    ):
        raise RuntimeError("v12 anchor central vectors no longer collapse exactly")
    cell_cosines = {
        "D43": _cosine(coefficients, c43_a43),
        "D44": _cosine(coefficients, c44_a43),
    }
    anchor_cosine = _centered_cosine(coefficients, central_anchor["A43"])
    if (
        identity != EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
        or coefficients != EXPECTED_CONSENSUS_COEFFICIENTS_V12
        or not all(
            math.isclose(value, EXPECTED_CELL_COSINE_V12, rel_tol=0.0, abs_tol=3e-16)
            for value in cell_cosines.values()
        )
        or not math.isclose(
            anchor_cosine, EXPECTED_ANCHOR_COSINE_V12,
            rel_tol=0.0, abs_tol=3e-16,
        )
    ):
        raise RuntimeError("v12 frozen consensus numeric identity changed")
    result = {
        "schema": "eggroll-es-two-domain-unit-consensus-v12",
        "construction": "sqrt(32)*unit(unit(D43)+unit(D44))",
        "duplicate_anchor_cells_collapsed": {
            "D43": ["D43xA43", "D43xA44"],
            "D44": ["D44xA43", "D44xA44"],
        },
        "seeds": list(seeds),
        "coefficients": coefficients,
        "coefficient_sha256": identity,
        "l2_norm": float(np.linalg.norm(consensus)),
        "cell_cosines": cell_cosines,
        "anchor_cosine": anchor_cosine,
        "resident_cross_content_sha256": cross.get(
            "content_sha256_before_self_field"
        ),
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_consensus_v12(consensus):
    if (
        not isinstance(consensus, dict)
        or consensus.get("schema") != "eggroll-es-two-domain-unit-consensus-v12"
        or consensus.get("construction")
        != "sqrt(32)*unit(unit(D43)+unit(D44))"
        or consensus.get("seeds") != anchor_v11.PERTURBATION_SEEDS_V11
        or consensus.get("coefficients")
        != EXPECTED_CONSENSUS_COEFFICIENTS_V12
        or consensus.get("coefficient_sha256")
        != EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
        or consensus.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in consensus.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v12 consensus artifact identity changed")
    if (
        set(consensus.get("cell_cosines", {})) != {"D43", "D44"}
        or not all(
            math.isclose(value, EXPECTED_CELL_COSINE_V12,
                         rel_tol=0.0, abs_tol=3e-16)
            for value in consensus["cell_cosines"].values()
        )
        or not math.isclose(
            consensus.get("anchor_cosine", float("nan")),
            EXPECTED_ANCHOR_COSINE_V12, rel_tol=0.0, abs_tol=3e-16,
        )
    ):
        raise RuntimeError("v12 consensus cosine identity changed")
    if coefficient_sha256(
        consensus["seeds"], consensus["coefficients"],
    ) != EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12:
        raise RuntimeError("v12 consensus coefficient hash does not recompute")
    return consensus


def _row_identity_v12(row):
    return canonical_sha256({
        "question": row["question"],
        "answer": row["answer"],
        "fact_id": row["fact_id"],
    })


def build_disjoint_screens_v12(dataset, loader_builder):
    """Create C45/C46 while excluding D43/D44 and each prior screen."""
    if len(dataset) != FROZEN_TRAIN_ROWS_V12:
        raise RuntimeError("v12 frozen training row count changed")

    def ordered_rows(seed):
        rows = []
        loader = loader_builder(dataset, 64, seed)
        # Reconstruct row identities by mapping the unique Q/A pairs back to
        # the frozen dataset; fact IDs are used only for disjointness hashing.
        by_pair = {}
        for row in dataset:
            key = (row["question"], row["answer"])
            if key in by_pair:
                raise RuntimeError("v12 frozen train contains duplicate Q/A pairs")
            by_pair[key] = row
        for questions, answers in loader:
            for question, answer in zip(questions, answers):
                rows.append(by_pair[(question, answer)])
        if len(rows) != len(dataset):
            raise RuntimeError("v12 train permutation changed length")
        return rows

    blocked = {
        _row_identity_v12(row)
        for seed in (43, 44) for row in ordered_rows(seed)[:64]
    }
    screens = {}
    for label in SCREEN_LABELS_V12:
        spec = SCREEN_MANIFESTS_V12[label]
        selected = [
            row for row in ordered_rows(spec["seed"])
            if _row_identity_v12(row) not in blocked
        ][:SCREEN_ROWS_V12]
        identities = [_row_identity_v12(row) for row in selected]
        questions = [row["question"] for row in selected]
        answers = [row["answer"] for row in selected]
        if (
            len(selected) != SCREEN_ROWS_V12
            or len(set(identities)) != SCREEN_ROWS_V12
            or canonical_sha256(identities) != spec["row_identity_sha256"]
            or canonical_sha256({
                "questions": questions, "answers": answers,
            }) != spec["sha256"]
        ):
            raise RuntimeError(f"v12 frozen {label} screen changed")
        blocked.update(identities)
        screens[label] = (questions, answers)
    return screens


def _documents_from_dense_v12(result):
    if result.get("schema") != "eggroll-es-dense-gold-reward-result-v4":
        raise RuntimeError("v12 dense screen result schema changed")
    documents = []
    for row in result.get("examples", []):
        documents.append({
            "document_id": canonical_sha256({
                "prompt_sha256": row["prompt_sha256"],
                "answer_sha256": row["answer_sha256"],
                "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            }),
            "scored_token_count": row["answer_token_count"],
            "sum_token_logprob": row["sum_answer_token_logprob"],
        })
    if len(documents) != SCREEN_ROWS_V12:
        raise RuntimeError("v12 direct screen did not score exactly 64 rows")
    return documents


def _paired_lcb_v12(reference, candidate):
    result = robust_anchor.score_population_document_lcbs(
        reference,
        [
            {"seed": 0, "documents": candidate},
            {"seed": 1, "documents": candidate},
        ],
    )
    robust_anchor.validate_document_lcb_result(result)
    scores = [row["score"] for row in result["robust_scores"]]
    if len(scores) != 2 or scores[0] != scores[1]:
        raise RuntimeError("v12 duplicate paired-LCB rows diverged")
    return {"lower_confidence_bound": scores[0], "result": result}


def paired_state_comparison_v12(reference, candidate):
    if set(reference.get("screens", {})) != set(SCREEN_LABELS_V12):
        raise RuntimeError("v12 reference screen coverage changed")
    if set(candidate.get("screens", {})) != set(SCREEN_LABELS_V12):
        raise RuntimeError("v12 candidate screen coverage changed")
    if set(reference.get("anchors", {})) != {"A43", "A44"}:
        raise RuntimeError("v12 reference anchor coverage changed")
    if set(candidate.get("anchors", {})) != {"A43", "A44"}:
        raise RuntimeError("v12 candidate anchor coverage changed")
    screens = {
        label: _paired_lcb_v12(
            _documents_from_dense_v12(reference["screens"][label]),
            _documents_from_dense_v12(candidate["screens"][label]),
        )
        for label in SCREEN_LABELS_V12
    }
    anchors = {
        label: _paired_lcb_v12(
            reference["anchors"][label], candidate["anchors"][label],
        )
        for label in ("A43", "A44")
    }
    eligible = (
        all(row["lower_confidence_bound"] > 0.0 for row in screens.values())
        and all(row["lower_confidence_bound"] >= 0.0 for row in anchors.values())
    )
    result = {
        "schema": "eggroll-es-paired-train-anchor-gate-v12",
        "screen_rule": "both_independent_95pct_paired_LCB_strictly_positive",
        "anchor_rule": "both_generation_seed_95pct_paired_LCB_nonnegative",
        "screens": screens,
        "anchors": anchors,
        "eligible": eligible,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def select_smallest_eligible_v12(states):
    if [row.get("alpha") for row in states] != ALPHA_GRID_V12:
        raise RuntimeError("v12 alpha state order changed")
    eligible = [
        row for row in states if row["alpha"] > 0.0
        and row.get("gate", {}).get("eligible") is True
    ]
    return eligible[0] if eligible else None


def validate_frozen_layer_plan_bundle_v12(bundle):
    return anchor_v11c.validate_frozen_layer_plan_bundle_v11c(bundle)


def load_frozen_layer_plan_v12(*args, **kwargs):
    bundle = anchor_v11c.load_frozen_layer_plan_v11c(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v12(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v12(argv):
    bundle, remaining = anchor_v11c.parse_frozen_layer_plan_cli_v11c(argv)
    validate_frozen_layer_plan_bundle_v12(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v12(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v12(bundle)
    anchor_v11c.set_default_layer_plan_bundle_v11c(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


class ConsensusCandidateContractMixinV12:
    def set_consensus_evidence_v12(self, evidence):
        consensus = validate_consensus_v12(copy.deepcopy(evidence["consensus"]))
        self._v12_consensus_evidence = copy.deepcopy(evidence)
        self._v12_expected_consensus = consensus

    def estimate_step_coefficients(self, iteration, seeds, input_text, target_text):
        if not hasattr(self, "_v12_expected_consensus"):
            raise RuntimeError("v12 consensus evidence was not configured")
        plan = super().estimate_step_coefficients(
            iteration, seeds, input_text, target_text,
        )
        local = consensus_from_resident_cross_v12(
            plan["resident_sign_cross_v11"]
        )
        expected = self._v12_expected_consensus
        if local != expected:
            raise RuntimeError("v12 live V11 responses differ from bound consensus")
        plan["v11_source_coefficient_sha256_v12"] = plan["coefficient_sha256"]
        plan["coefficients"] = list(local["coefficients"])
        plan["coefficient_sha256"] = local["coefficient_sha256"]
        plan["consensus_candidate_v12"] = copy.deepcopy(local)
        boundary = plan["population_boundary_audit_v4"]
        metadata = plan["distributed_update_v4"]
        plan_id = canonical_sha256({
            "schema": "eggroll-es-distributed-plan-id-v4",
            "iteration": int(iteration),
            "coefficient_sha256": plan["coefficient_sha256"],
            "reference_generation": self._v3_reference_generation,
            "reference_sha256": self._v3_reference_identity["sha256"],
            "layer_plan_sha256": self._v4_layer_plan["plan_sha256"],
            "reward_config_sha256": self._v4_reward_config_sha256,
            "runtime_mapping_sha256": canonical_sha256(
                self._v4_layer_plan_install
            ),
            "population_boundary_audit_sha256": boundary["audit_sha256"],
        })
        metadata["plan_id"] = plan_id
        self._v3_active_plan_id = plan_id
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        plan["applied_alpha"] = 0.0
        plan["applications"] = []
        self._latest_anchor_plan = plan
        self._persist_anchor_plan(plan)
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        validate_consensus_v12(plan.get("consensus_candidate_v12"))
        if plan.get("coefficient_sha256") != (
            EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
        ):
            raise RuntimeError("v12 refuses a non-consensus update")
        return anchor_v4.FrozenLayerDenseRewardMixinV4.apply_seed_coefficients(
            self, plan, target_alpha,
        )

    def capture_train_only_state_v12(self, screens, generation_seed):
        del generation_seed
        if set(screens) != set(SCREEN_LABELS_V12):
            raise RuntimeError("v12 direct-screen coverage changed")
        dense_results = {}
        for offset, label in enumerate(SCREEN_LABELS_V12):
            questions, answers = screens[label]
            prompts = [
                anchor_v4.anchor_v3.anchor_v2.anchor_v1.base.specialist_template(q)
                for q in questions
            ]
            items = anchor_v4.prepare_gold_answer_items_v4(
                self.tokenizer, prompts, answers,
            )
            outputs = anchor_v11.anchor_v1.dispatch_eval_batch(
                self.engines,
                [{"prompt_token_ids": row["prompt_token_ids"]} for row in items],
                self._dense_sampling_params_v4(offset),
                self._resolve,
            )
            dense_results[label] = anchor_v4.score_gold_answer_outputs_v4(
                items, outputs,
            )
        anchors = {}
        prompts = [
            {"prompt_token_ids": row["prompt_token_ids"]}
            for row in self.anchor_items
        ]
        for label, seed in (("A43", 43), ("A44", 44)):
            outputs = anchor_v11.anchor_v1.dispatch_eval_batch(
                self.engines, prompts,
                self._anchor_sampling_seed_v10(0, seed),
                self._resolve,
            )
            anchors[label] = anchor_v5.summarize_anchor_documents_v5(
                self.anchor_items, outputs,
            )
        return {"screens": dense_results, "anchors": anchors}


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v12(captured)
    # V11b fixed the raw-question versus templated-prompt manifest split and
    # V11c exported the complete substituted-anchor runtime API.  Returning to
    # the original V11 parent here would deterministically fail before the
    # population estimate, so V12 must extend the successful V11c facade.
    parent = anchor_v11c.load_trainer(captured)
    parent.launch_engines = anchor_v10._clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v12",
    )

    class ConsensusCandidateTrainerV12(ConsensusCandidateContractMixinV12, parent):
        pass

    return ConsensusCandidateTrainerV12
