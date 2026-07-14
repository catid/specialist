#!/usr/bin/env python3
"""Fail-closed V12 consensus screen, confirmation, and release stages."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from torch.utils.data import DataLoader

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_stability_v8 as driver_v8
import run_eggroll_es_anchor_variance_v10 as driver_v10
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v12 as anchor_v12


ROOT = Path(__file__).resolve().parent
FROZEN_TRAIN_DATASET_V12 = driver_v10.FROZEN_TRAIN_DATASET_V10
FROZEN_EVAL_DATASET_V12 = driver_v10.FROZEN_EVAL_DATASET_V10
FROZEN_OUTPUT_DIRECTORY_V12 = driver_v10.FROZEN_OUTPUT_DIRECTORY_V10
MIDDLE_LATE_PLAN_SHA256_V12 = driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
PERTURBATION_SEEDS_V12 = list(driver_v10.PERTURBATION_SEEDS_V10)
V11_EVIDENCE_PATH_V12 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11_EVIDENCE_V12.json"
).resolve()
# Minted once from the canonical completed V11g attempt and journal. A real
# stage fails before engine launch if either immutable identity changes.
EXPECTED_V11_EVIDENCE_FILE_SHA256_V12 = (
    "d68dafd50e229bd444b5ff0a666aabb508d3d021d44cd2060d7aace391fc6745"
)
EXPECTED_V11_EVIDENCE_CONTENT_SHA256_V12 = (
    "b6212a4bdafaf234f8445b11c18ef96e15526d450f89d366542c63fae2d15e8f"
)
EXPERIMENT_NAMES_V12 = {
    "preseal": "snapshot794_layer_v12_middle_late_consensus_preseal_c45c46",
    "confirm": "snapshot794_layer_v12_middle_late_consensus_fresh_confirm",
    "release": "snapshot794_layer_v12_middle_late_consensus_release_eval",
}
V12_IMPLEMENTATION_PATHS = {
    "worker_v12": ROOT / "eggroll_es_worker_v12.py",
    "trainer_v12": ROOT / "train_eggroll_es_specialist_anchor_v12.py",
    "driver_v12": Path(__file__).resolve(),
    "evidence_builder_v12": ROOT / "build_eggroll_es_v11_evidence_v12.py",
    "contract_tests_v12": ROOT / "test_eggroll_es_consensus_candidate_v12.py",
    "protocol_v12": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_CONSENSUS_CANDIDATE_V12_PROTOCOL.md"
    ),
}
V11G_EVIDENCE_KEYS_V12 = {
    "schema", "passed", "validation", "downstream_validation",
    "v11g_attempt", "v11g_journal", "consensus", "builder",
    "contains_validation_ood_or_heldout_content", "selection_surface",
    "content_sha256_before_self_field",
}
CANDIDATE_SEAL_KEYS_V12 = {
    "schema", "alpha", "coefficient_sha256", "selection",
    "preseal_report", "v11_evidence_content_sha256",
    "implementation_bundle_sha256", "benchmark_content_opened_before_seal",
    "fresh_direct_confirmation_required", "release_fallback_allowed",
    "content_sha256_before_self_field",
}
CONFIRMATION_KEYS_V12 = {
    "schema", "stage", "experiment_name", "model", "train_dataset",
    "train_arrow_sha256", "screen_manifests", "screen_disjointness",
    "alpha_grid", "consensus", "v11_evidence_content_sha256",
    "implementation", "hardware_contract", "runtime_integrity",
    "benchmark_content_opened_before_candidate_seal", "heldout_opened",
    "passed", "alpha", "coefficient_sha256", "gate", "application",
    "candidate_seal", "benchmark_content_opened", "fallback_attempted",
    "content_sha256_before_self_field",
}


def _file_sha256(path):
    return driver_v1.file_sha256(path)


def implementation_identity_v12():
    files = {
        key: {"path": str(path.resolve()), "file_sha256": _file_sha256(path)}
        for key, path in V12_IMPLEMENTATION_PATHS.items()
    }
    return {
        "files": files,
        "bundle_sha256": driver_v1.canonical_sha256(files),
    }


def _fixture_evidence_v12():
    """Dry-run-only evidence; no real stage may use this path."""
    journal = json.loads((
        ROOT / "experiments/eggroll_es_hpo/runs/"
        "snapshot794_layer_v10_middle_late_antithetic_cross_"
        "d43d44_a43a44_basis20260714/alpha_line_search.json"
    ).read_text())
    cross = anchor_v11._resident_artifact_v11(
        journal["coefficient_plan"]["antithetic_cross_v10"]
    )
    consensus = anchor_v12.consensus_from_resident_cross_v12(cross)
    fixture = {
        "schema": "eggroll-es-v11g-compact-equivalence-evidence-v12",
        "passed": True,
        "validation": "fixture_dry_run_only",
        "downstream_validation": "fixture_dry_run_only",
        "v11g_attempt": {"fixture": True},
        "v11g_journal": {"fixture": True},
        "consensus": consensus,
        "contains_validation_ood_or_heldout_content": False,
        "selection_surface": "V11g_train_and_anchor_responses_only",
    }
    fixture["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        fixture
    )
    return fixture


def load_v11_evidence_v12(path, *, dry_run=False, fixture=False):
    if fixture:
        if not dry_run:
            raise RuntimeError("v12 fixture evidence is forbidden for real stages")
        return _fixture_evidence_v12()
    if (
        EXPECTED_V11_EVIDENCE_FILE_SHA256_V12 is None
        or EXPECTED_V11_EVIDENCE_CONTENT_SHA256_V12 is None
    ):
        raise RuntimeError("v12 real V11g evidence binding is pending")
    path = Path(path).resolve()
    driver_v8.offline_audit._assert_no_heldout(str(path), "v12 V11 evidence")
    if (
        path != V11_EVIDENCE_PATH_V12
        or _file_sha256(path) != EXPECTED_V11_EVIDENCE_FILE_SHA256_V12
    ):
        raise RuntimeError("v12 compact V11g evidence file identity changed")
    evidence = json.loads(path.read_text())
    if (
        set(evidence) != V11G_EVIDENCE_KEYS_V12
        or evidence.get("schema")
        != "eggroll-es-v11g-compact-equivalence-evidence-v12"
        or evidence.get("passed") is not True
        or evidence.get(
            "contains_validation_ood_or_heldout_content"
        ) is not False
        or evidence.get("selection_surface")
        != "V11g_train_and_anchor_responses_only"
        or evidence.get("content_sha256_before_self_field")
        != EXPECTED_V11_EVIDENCE_CONTENT_SHA256_V12
        or evidence.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v12 compact V11g evidence content changed")
    anchor_v12.validate_consensus_v12(evidence.get("consensus"))
    return evidence


def _assert_preseal_surface_v12(argv, stage):
    forbidden = ("heldout",)
    if stage in ("preseal", "confirm"):
        forbidden += ("validation", "ood", "--eval-dataset", "--eval-splits")
    for token in argv:
        lowered = str(token).lower()
        if any(piece in lowered for piece in forbidden):
            raise ValueError(
                f"v12 {stage} stage rejects benchmark/OOD surface: {token}"
            )


def _stage_parser_v12(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-stage", choices=(stage,), required=True)
    parser.add_argument("--v11-evidence", default=str(V11_EVIDENCE_PATH_V12))
    parser.add_argument("--v12-fixture-evidence", action="store_true")
    parser.add_argument("--v12-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--model-name", default=str(ROOT / "models/Qwen3.6-35B-A3B"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V12))
    parser.add_argument("--sigma", type=float, default=0.0003)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--target-alphas", default="0,0.00000078125,0.0000015625")
    parser.add_argument(
        "--anchor-prose-jsonl",
        default=str(ROOT / "data/general_prose_anchor_v1.jsonl"),
    )
    parser.add_argument(
        "--anchor-prose-report",
        default=str(ROOT / "data/general_prose_anchor_v1.report.json"),
    )
    parser.add_argument("--anchor-items-per-step", type=int, default=128)
    parser.add_argument("--anchor-max-input-tokens", type=int, default=512)
    parser.add_argument("--min-anchor-cosine", type=float, default=0.8)
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V12))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAMES_V12[stage])
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    if stage in ("confirm", "release"):
        parser.add_argument("--candidate-seal", required=True)
        parser.add_argument("--expected-candidate-seal-file-sha256", required=True)
        parser.add_argument("--expected-candidate-seal-content-sha256", required=True)
    if stage == "release":
        parser.add_argument("--confirmation", required=True)
        parser.add_argument("--expected-confirmation-file-sha256", required=True)
        parser.add_argument("--expected-confirmation-content-sha256", required=True)
        parser.add_argument("--eval-dataset", default=str(FROZEN_EVAL_DATASET_V12))
        parser.add_argument("--eval-splits", default="validation,ood_qa")
        parser.add_argument(
            "--ood-prose-jsonl", default=str(ROOT / "data/ood_prose_v3.jsonl")
        )
        parser.add_argument("--ood-prose-max-input-tokens", type=int, default=1024)
    return parser


def validate_runtime_v12(args, bundle, implementation):
    anchor_v12.validate_frozen_layer_plan_bundle_v12(bundle)
    targets = driver_v1.parse_target_alphas(args.target_alphas)
    expected_bundle = args.expected_implementation_bundle_sha256
    if (
        bundle["plan_sha256"] != MIDDLE_LATE_PLAN_SHA256_V12
        or Path(args.model_name).resolve()
        != (ROOT / "models/Qwen3.6-35B-A3B").resolve()
        or args.checkpoint is not None
        or Path(args.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V12
        or args.sigma != 0.0003
        or args.population_size != 32
        or args.batch_size != 128
        or args.mini_batch_size != 64
        or args.max_tokens != 32
        or args.seed != 43
        or args.n_vllm_engines != 4
        or args.n_gpu_per_vllm_engine != 1
        or args.use_gpus != "0,1,2,3"
        or targets != anchor_v12.ALPHA_GRID_V12
        or args.anchor_items_per_step != 128
        or args.anchor_max_input_tokens != 512
        or args.min_anchor_cosine != 0.8
        or args.reward_function_timeout != 10
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V12
        or args.experiment_name != EXPERIMENT_NAMES_V12[args.v12_stage]
        or args.logging != "none"
        or args.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v12 frozen four-GPU runtime recipe changed")
    if not args.v12_dry_run and expected_bundle is None:
        raise ValueError("v12 real stage requires a frozen implementation bundle hash")
    if expected_bundle is not None and expected_bundle != implementation["bundle_sha256"]:
        raise ValueError("v12 implementation bundle hash changed")
    return targets


def _load_binding_v12(path, expected_file, expected_content, schema, label):
    path = Path(path).resolve()
    driver_v8.offline_audit._assert_no_heldout(str(path), f"v12 {label}")
    if _file_sha256(path) != expected_file:
        raise RuntimeError(f"v12 {label} file identity changed")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != schema
        or value.get("content_sha256_before_self_field") != expected_content
        or value.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError(f"v12 {label} content identity changed")
    return value


def load_candidate_seal_v12(args):
    seal = _load_binding_v12(
        args.candidate_seal,
        args.expected_candidate_seal_file_sha256,
        args.expected_candidate_seal_content_sha256,
        "eggroll-es-immutable-candidate-seal-v12", "candidate seal",
    )
    if (
        set(seal) != CANDIDATE_SEAL_KEYS_V12
        or set(seal.get("selection", {}))
        != {"eligible", "gate_content_sha256", "policy"}
        or set(seal.get("preseal_report", {}))
        != {"path", "file_sha256", "content_sha256"}
        or seal.get("alpha") not in anchor_v12.ALPHA_GRID_V12[1:]
        or seal.get("coefficient_sha256")
        != anchor_v12.EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
        or seal.get("selection", {}).get("eligible") is not True
        or seal.get("selection", {}).get("policy")
        != "smallest_positive_eligible_alpha"
        or not all(
            isinstance(seal["selection"].get(key), str)
            and len(seal["selection"][key]) == 64
            for key in ("gate_content_sha256",)
        )
        or not all(
            isinstance(seal["preseal_report"].get(key), str)
            and len(seal["preseal_report"][key]) == 64
            for key in ("file_sha256", "content_sha256")
        )
        or not isinstance(seal.get("v11_evidence_content_sha256"), str)
        or len(seal["v11_evidence_content_sha256"]) != 64
        or not isinstance(seal.get("implementation_bundle_sha256"), str)
        or len(seal["implementation_bundle_sha256"]) != 64
        or seal.get("benchmark_content_opened_before_seal") is not False
        or seal.get("fresh_direct_confirmation_required") is not True
        or seal.get("release_fallback_allowed") is not False
    ):
        raise RuntimeError("v12 sealed candidate is not release-eligible")
    return seal


def load_confirmation_v12(args, seal):
    confirmation = _load_binding_v12(
        args.confirmation,
        args.expected_confirmation_file_sha256,
        args.expected_confirmation_content_sha256,
        "eggroll-es-fresh-direct-confirmation-v12", "confirmation",
    )
    gate = confirmation.get("gate", {})
    application = confirmation.get("application", {})
    seal_binding = confirmation.get("candidate_seal", {})
    if (
        set(confirmation) != CONFIRMATION_KEYS_V12
        or confirmation.get("stage") != "confirm"
        or confirmation.get("passed") is not True
        or confirmation.get("alpha") != seal["alpha"]
        or confirmation.get("coefficient_sha256") != seal["coefficient_sha256"]
        or set(gate) != {
            "schema", "screen_rule", "anchor_rule", "screens", "anchors",
            "eligible", "content_sha256_before_self_field",
        }
        or gate.get("eligible") is not True
        or gate.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in gate.items()
            if key != "content_sha256_before_self_field"
        })
        or set(application) != {
            "target_alpha", "alpha_increment", "update_sequence",
            "manifest_sha256", "coefficient_sha256", "final_identity",
            "post_commit_states",
        }
        or application.get("target_alpha") != seal["alpha"]
        or application.get("coefficient_sha256") != seal["coefficient_sha256"]
        or set(seal_binding) != {"path", "file_sha256", "content_sha256"}
        or seal_binding.get("file_sha256")
        != args.expected_candidate_seal_file_sha256
        or seal_binding.get("content_sha256")
        != args.expected_candidate_seal_content_sha256
        or confirmation.get("benchmark_content_opened_before_candidate_seal")
        is not False
        or confirmation.get("benchmark_content_opened") is not False
        or confirmation.get("heldout_opened") is not False
        or confirmation.get("fallback_attempted") is not False
    ):
        raise RuntimeError("v12 fresh direct confirmation is not a pass")
    return confirmation


def _load_train_v12(args):
    train_dict = load_from_disk(args.train_dataset)
    if list(train_dict) != ["train"]:
        raise RuntimeError("v12 training artifact must contain exactly train")
    dataset = train_dict["train"]
    if len(dataset.cache_files) != 1 or _file_sha256(
        dataset.cache_files[0]["filename"]
    ) != anchor_v12.FROZEN_TRAIN_ARROW_SHA256_V12:
        raise RuntimeError("v12 frozen training Arrow identity changed")
    screens = anchor_v12.build_disjoint_screens_v12(
        dataset, base.build_train_loader,
    )
    questions, answers = next(iter(driver_v10._crossed_train_loader_v10(
        dataset, 128, 43,
    )))
    prompts = [base.specialist_template(question) for question in questions]
    return dataset, screens, questions, answers, prompts


def _make_trainer_v12(args, bundle, dataset, eval_loaders):
    train_loader = base.build_train_loader(dataset, 128, 43)
    trainer_class = anchor_v12.load_trainer(bundle)
    trainer = trainer_class(
        model_name=args.model_name, checkpoint=args.checkpoint,
        sigma=args.sigma, alpha=anchor_v12.ALPHA_GRID_V12[-1],
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=32, batch_size=128, mini_batch_size=64,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=train_loader, eval_dataloader_dict=eval_loaders,
        eval_freq=1, n_vllm_engines=4, n_gpu_per_vllm_engine=1,
        logging="none", global_seed=43, use_gpus="0,1,2,3",
        experiment_name=args.experiment_name,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=args.output_directory,
    )
    anchor_dataset = anchor_v12.load_anchor_prose(
        args.anchor_prose_jsonl, args.anchor_prose_report,
    )
    trainer.configure_anchor(
        anchor_dataset, items_per_step=128, max_input_tokens=512,
        min_anchor_cosine=0.8,
    )
    return trainer


def _plan_v12(trainer, evidence, prompts, answers):
    trainer.set_consensus_evidence_v12(evidence)
    plan = trainer.estimate_step_coefficients(
        0, list(PERTURBATION_SEEDS_V12), prompts, answers,
    )
    anchor_v12.validate_consensus_v12(plan["consensus_candidate_v12"])
    return plan


def _application_summary_v12(plan):
    if not plan.get("applications"):
        return None
    application = copy.deepcopy(plan["applications"][-1])
    return {
        "target_alpha": application["target_alpha"],
        "alpha_increment": application["alpha_increment"],
        "update_sequence": application["update_sequence"],
        "manifest_sha256": application["manifest_sha256"],
        "coefficient_sha256": application["coefficient_sha256"],
        "final_identity": application["final_identity"],
        "post_commit_states": application["post_commit_states"],
    }


def _base_report_v12(args, evidence, implementation, plan):
    boundary = plan["population_boundary_audit_v4"]
    return {
        "stage": args.v12_stage,
        "experiment_name": args.experiment_name,
        "model": str(Path(args.model_name).resolve()),
        "train_dataset": str(Path(args.train_dataset).resolve()),
        "train_arrow_sha256": anchor_v12.FROZEN_TRAIN_ARROW_SHA256_V12,
        "screen_manifests": copy.deepcopy(anchor_v12.SCREEN_MANIFESTS_V12),
        "screen_disjointness": "C45,C46 exclude D43,D44 and each other",
        "alpha_grid": list(anchor_v12.ALPHA_GRID_V12),
        "consensus": copy.deepcopy(plan["consensus_candidate_v12"]),
        "v11_evidence_content_sha256": evidence[
            "content_sha256_before_self_field"
        ],
        "implementation": implementation,
        "hardware_contract": {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "partial_waves_forbidden": True,
        },
        "runtime_integrity": {
            "identity_audit_passed": plan.get("identity_audit", {}).get(
                "passed"
            ) is True,
            "population_exact_restore_passed": boundary.get("passed") is True,
            "population_boundary_audit_sha256": boundary["audit_sha256"],
            "restored_reference_identity": boundary["current_identity"],
            "resident_cross_content_sha256": plan[
                "resident_sign_cross_v11"
            ]["content_sha256_before_self_field"],
            "distributed_plan_id": plan["distributed_update_v4"]["plan_id"],
            "anchor_plan": {
                "path": plan["journal_path"],
                "file_sha256": _file_sha256(plan["journal_path"]),
            },
        },
        "benchmark_content_opened_before_candidate_seal": False,
        "heldout_opened": False,
    }


def execute_preseal_v12(args, bundle, evidence, implementation):
    dataset, screens, _questions, answers, prompts = _load_train_v12(args)
    trainer = None
    try:
        trainer = _make_trainer_v12(args, bundle, dataset, {})
        plan = _plan_v12(trainer, evidence, prompts, answers)
        reference = trainer.capture_train_only_state_v12(screens, 0)
        states = [{
            "alpha": 0.0,
            "gate": anchor_v12.paired_state_comparison_v12(reference, reference),
            "application": None,
        }]
        for index, alpha in enumerate(anchor_v12.ALPHA_GRID_V12[1:], 1):
            trainer.apply_seed_coefficients(plan, alpha)
            candidate = trainer.capture_train_only_state_v12(screens, index)
            states.append({
                "alpha": alpha,
                "gate": anchor_v12.paired_state_comparison_v12(
                    reference, candidate,
                ),
                "application": _application_summary_v12(plan),
            })
        selected = anchor_v12.select_smallest_eligible_v12(states)
        report = {
            "schema": "eggroll-es-consensus-preseal-screen-v12",
            **_base_report_v12(args, evidence, implementation, plan),
            "status": "candidate_selected" if selected else "no_eligible_candidate",
            "states": states,
            "selection_policy": "smallest_positive_eligible_alpha",
            "selected_alpha": selected["alpha"] if selected else None,
            "fallback_selected": False,
            "anchor_plan": {
                "path": plan["journal_path"],
                "file_sha256": _file_sha256(plan["journal_path"]),
            },
        }
        report["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
            report
        )
        run_dir = Path(args.output_directory) / args.experiment_name
        report_path = run_dir / "preseal_screen.json"
        if report_path.exists() or report_path.with_name(report_path.name + ".tmp").exists():
            raise ValueError("v12 preseal report already exists")
        driver_v1.atomic_write_json(report_path, report)
        seal = None
        if selected:
            seal = {
                "schema": "eggroll-es-immutable-candidate-seal-v12",
                "alpha": selected["alpha"],
                "coefficient_sha256": (
                    anchor_v12.EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
                ),
                "selection": {
                    "eligible": True,
                    "gate_content_sha256": selected["gate"][
                        "content_sha256_before_self_field"
                    ],
                    "policy": "smallest_positive_eligible_alpha",
                },
                "preseal_report": {
                    "path": str(report_path.resolve()),
                    "file_sha256": _file_sha256(report_path),
                    "content_sha256": report[
                        "content_sha256_before_self_field"
                    ],
                },
                "v11_evidence_content_sha256": evidence[
                    "content_sha256_before_self_field"
                ],
                "implementation_bundle_sha256": implementation["bundle_sha256"],
                "benchmark_content_opened_before_seal": False,
                "fresh_direct_confirmation_required": True,
                "release_fallback_allowed": False,
            }
            seal["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
                seal
            )
            driver_v1.atomic_write_json(run_dir / "candidate_seal.json", seal)
        return report, seal
    finally:
        if trainer is not None:
            base.close_trainer(trainer)


def execute_confirm_v12(args, bundle, evidence, implementation, seal):
    dataset, screens, _questions, answers, prompts = _load_train_v12(args)
    trainer = None
    try:
        trainer = _make_trainer_v12(args, bundle, dataset, {})
        plan = _plan_v12(trainer, evidence, prompts, answers)
        reference = trainer.capture_train_only_state_v12(screens, 100)
        trainer.apply_seed_coefficients(plan, seal["alpha"])
        candidate = trainer.capture_train_only_state_v12(screens, 101)
        gate = anchor_v12.paired_state_comparison_v12(reference, candidate)
        report = {
            "schema": "eggroll-es-fresh-direct-confirmation-v12",
            **_base_report_v12(args, evidence, implementation, plan),
            "passed": gate["eligible"] is True,
            "alpha": seal["alpha"],
            "coefficient_sha256": seal["coefficient_sha256"],
            "gate": gate,
            "application": _application_summary_v12(plan),
            "candidate_seal": {
                "path": str(Path(args.candidate_seal).resolve()),
                "file_sha256": args.expected_candidate_seal_file_sha256,
                "content_sha256": args.expected_candidate_seal_content_sha256,
            },
            "benchmark_content_opened": False,
            "fallback_attempted": False,
        }
        report["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
            report
        )
        output = (
            Path(args.output_directory) / args.experiment_name
            / "fresh_direct_confirmation.json"
        )
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise ValueError("v12 confirmation output already exists")
        driver_v1.atomic_write_json(output, report)
        return report
    finally:
        if trainer is not None:
            base.close_trainer(trainer)


def _release_eval_loaders_v12(args):
    if Path(args.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V12:
        raise ValueError("v12 release evaluation root changed")
    splits = driver_v1.validate_eval_splits(args.eval_splits)
    datasets = driver_v1.load_allowlisted_eval_datasets(args.eval_dataset, splits)
    loaders = {
        split: DataLoader(
            datasets[split], batch_size=64,
            collate_fn=base.specialist_collate, shuffle=False,
        )
        for split in splits
    }
    return datasets, loaders


def _qa_state_v12(trainer, iteration):
    trainer.eval_step(iteration=iteration)
    return {
        split: driver_v1.summarize_eval_file(path)
        for split, path in driver_v1._eval_paths(
            trainer, iteration, driver_v1.ALLOWED_EVAL_SPLITS,
        ).items()
    }


def execute_release_v12(
    args, bundle, evidence, implementation, seal, confirmation,
):
    del confirmation
    dataset, _screens, _questions, answers, prompts = _load_train_v12(args)
    _eval_datasets, eval_loaders = _release_eval_loaders_v12(args)
    ood_prose = base.load_ood_prose(args.ood_prose_jsonl)
    trainer = None
    try:
        trainer = _make_trainer_v12(args, bundle, dataset, eval_loaders)
        plan = _plan_v12(trainer, evidence, prompts, answers)
        baseline_qa = _qa_state_v12(trainer, 0)
        baseline_prose = base.score_ood_prose(
            trainer, ood_prose, "v12_release_baseline",
            args.ood_prose_max_input_tokens,
        )
        trainer.apply_seed_coefficients(plan, seal["alpha"])
        candidate_qa = _qa_state_v12(trainer, 1)
        candidate_prose = base.score_ood_prose(
            trainer, ood_prose, "v12_release_candidate",
            args.ood_prose_max_input_tokens,
        )
        validation_gate = driver_v1.strict_qa_gate(
            baseline_qa["validation"], candidate_qa["validation"],
        )
        validation_gate["requires_strict_mean_improvement"] = True
        validation_gate["passed"] = (
            validation_gate["passed"]
            and validation_gate["deltas"]["mean_reward"] > 0.0
        )
        ood_qa_gate = driver_v1.strict_qa_gate(
            baseline_qa["ood_qa"], candidate_qa["ood_qa"],
        )
        prose_gate = base.compare_ood_prose(
            baseline_prose, candidate_prose, max_degradation=0.0,
        )
        passed = all((
            validation_gate["passed"], ood_qa_gate["passed"],
            prose_gate["passed"],
        ))
        report = {
            "schema": "eggroll-es-fixed-candidate-release-eval-v12",
            **_base_report_v12(args, evidence, implementation, plan),
            "passed": passed,
            "decision": "release" if passed else "reject_without_fallback",
            "alpha": seal["alpha"],
            "coefficient_sha256": seal["coefficient_sha256"],
            "baseline_qa": baseline_qa,
            "candidate_qa": candidate_qa,
            "validation_gate": validation_gate,
            "ood_qa_gate": ood_qa_gate,
            "baseline_prose": driver_v1.summarize_prose_evaluation(
                baseline_prose
            ),
            "candidate_prose": driver_v1.summarize_prose_evaluation(
                candidate_prose
            ),
            "ood_prose_gate": prose_gate,
            "application": _application_summary_v12(plan),
            "fixed_candidate_only": True,
            "fallback_attempted": False,
            "heldout_opened": False,
        }
        report["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
            report
        )
        output = (
            Path(args.output_directory) / args.experiment_name
            / "fixed_candidate_release_eval.json"
        )
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise ValueError("v12 release output already exists")
        driver_v1.atomic_write_json(output, report)
        return report
    finally:
        if trainer is not None:
            base.close_trainer(trainer)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument(
        "--v12-stage", choices=("preseal", "confirm", "release"), required=True,
    )
    stage_args, _ = stage_probe.parse_known_args(argv)
    _assert_preseal_surface_v12(argv, stage_args.v12_stage)
    bundle, remaining = anchor_v12.parse_frozen_layer_plan_cli_v12(argv)
    args = _stage_parser_v12(stage_args.v12_stage).parse_args(remaining)
    implementation = implementation_identity_v12()
    validate_runtime_v12(args, bundle, implementation)
    evidence = load_v11_evidence_v12(
        args.v11_evidence, dry_run=args.v12_dry_run,
        fixture=args.v12_fixture_evidence,
    )
    seal = None
    confirmation = None
    if args.v12_stage in ("confirm", "release"):
        seal = load_candidate_seal_v12(args)
        if (
            seal.get("v11_evidence_content_sha256")
            != evidence["content_sha256_before_self_field"]
            or seal.get("implementation_bundle_sha256")
            != implementation["bundle_sha256"]
        ):
            raise RuntimeError(
                "v12 candidate seal differs from current evidence or implementation"
            )
    if args.v12_stage == "release":
        confirmation = load_confirmation_v12(args, seal)
        if (
            confirmation.get("v11_evidence_content_sha256")
            != evidence["content_sha256_before_self_field"]
            or confirmation.get("implementation", {}).get("bundle_sha256")
            != implementation["bundle_sha256"]
        ):
            raise RuntimeError(
                "v12 confirmation differs from current evidence or implementation"
            )
    if args.v12_dry_run:
        payload = {
            "schema": "eggroll-es-consensus-candidate-dry-run-v12",
            "stage": args.v12_stage,
            "alpha_grid": list(anchor_v12.ALPHA_GRID_V12),
            "consensus_coefficient_sha256": evidence["consensus"][
                "coefficient_sha256"
            ],
            "screen_manifests": copy.deepcopy(anchor_v12.SCREEN_MANIFESTS_V12),
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "four_gpu_contract": {"engines": 4, "tp": 1, "gpu_ids": [0, 1, 2, 3]},
            "benchmark_surface_available": args.v12_stage == "release",
        }
        print(json.dumps(payload, sort_keys=True))
        return payload
    run_dir = Path(args.output_directory) / args.experiment_name
    if run_dir.exists():
        raise ValueError("v12 output exists; resume and overwrite are forbidden")
    if args.v12_stage == "preseal":
        return execute_preseal_v12(args, bundle, evidence, implementation)
    if args.v12_stage == "confirm":
        return execute_confirm_v12(
            args, bundle, evidence, implementation, seal,
        )
    return execute_release_v12(
        args, bundle, evidence, implementation, seal, confirmation,
    )


if __name__ == "__main__":
    main()
