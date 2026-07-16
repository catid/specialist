#!/usr/bin/env python3
"""Pure contracts for the nested P8-vs-P16 LoRA-ES V52 experiment.

This module is deliberately free of torch, Ray, model, dataset, and GPU imports.
It defines the one-variable scientific comparison, train-only projection and
reliability math, post-population gates, and the sealed compute estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import eggroll_es_multi_anchor_v43h as multi_anchor


ROOT = Path(__file__).resolve().parent
REQUIRED_PYTHON_V52 = (
    ROOT / "es-at-scale/.venv/bin/python"
).absolute()
RETRY_REVISION_V52 = (
    "retry4_calibration_bounds_schema_path_repair"
)
RETRY1_GPU_LOG_V52 = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v52_matched_lora_es_nested_p8_vs_p16_retry1/gpu_activity_v52.jsonl"
).resolve()
RETRY1_GPU_LOG_SHA256_V52 = (
    "bb1e9b4cb88273998346966755564fb547f34215381dfadfdba67f500466357e"
)
RETRY2_GPU_LOG_V52 = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v52_matched_lora_es_nested_p8_vs_p16_retry2/gpu_activity_v52.jsonl"
).resolve()
RETRY2_GPU_LOG_SHA256_V52 = (
    "2d2df6c965d25edbdb35b6e44a6b892db42eef10dd336cd0c821c5bc34d54a75"
)
RETRY3_GPU_LOG_V52 = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v52_matched_lora_es_nested_p8_vs_p16_retry3/gpu_activity_v52.jsonl"
).resolve()
RETRY3_GPU_LOG_SHA256_V52 = (
    "ad407725a4f1ad566da64cc187a1ac4b1677573d93e3ff81633b3e8f8ab5ce51"
)
POPULATION_SIZES_V52 = (8, 16)
P8_SEEDS_V52 = (
    140002291, 1028842752, 480373990, 1037026679,
    759861149, 227761095, 428721957, 150663570,
)
P16_SEEDS_V52 = P8_SEEDS_V52 + (
    863156398, 658045682, 947615772, 615729462,
    958574585, 1048698881, 573870406, 938961107,
)
SIGMA_V52 = 0.0006
ALPHA_V52 = 0.00015
SCALE_ORDER_V52 = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
ACTORS_V52 = 4
PHASES_V52 = ("materialize", "generate", "score", "restore", "drain")
MASTER_SHA256_V52 = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)
MASTER_RUNTIME_SHA256_V52 = (
    "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
)
DATASET_SHA256_V52 = (
    "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a"
)
MEMBERSHIP_SHA256_V52 = (
    "e9b073369966e21912a0bda86da501ab0975646df2a7d80bf5675c3dfec8c121"
)
MEMBERSHIP_CONTENT_SHA256_V52 = (
    "a8870fdce8fbf631b3d3472fd03690f6987590ee6e8758dc8fdcb4556dcc9096"
)
TRAIN_BUNDLE_CONTENT_SHA256_V52 = (
    "c94a7e8e8c30dbc2586351c4ef0ff13e3e9cc8551a21b9e055d34fbfa94bf44a"
)
SUBSET_FILE_SHA256_V52 = (
    "dd2c857b75617351d64cfce29f5a8e5d79ce9da212e4db50d22f2de3795c70a1"
)
SUBSET_CONTENT_SHA256_V52 = (
    "cdfa9d10669171d5d814b55df1f674a89dfa557c5376b45c8d0073e5d1acaec7"
)
REQUEST_ORDER_SHA256_V52 = (
    "0a08356fafc4086a48dc75f5fd8e4875505b25af1196de4db509b1c64a4e97ca"
)
SPLIT_MANIFEST_FILE_SHA256_V52 = (
    "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d"
)
SPLIT_MANIFEST_CONTENT_SHA256_V52 = (
    "3fcc2820e8dffe6a21198d0520365aace049735ac84bda179ea44bc8ad0881eb"
)
SOURCE_WEIGHTS_SHA256_V52 = (
    "0d6efd4d5be626f41cdd711843f799b9b3c09e9ecf6a7a8e6e9aeeff09f6dc5b"
)
SOURCE_CONFIG_SHA256_V52 = (
    "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
)
SOURCE_V52 = (
    ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5/"
    "v434_equal_r32_seed17_init20260715041/final"
).resolve()
SOURCE_WEIGHTS_V52 = (SOURCE_V52 / "adapter_model.safetensors").resolve()
SOURCE_CONFIG_V52 = (SOURCE_V52 / "adapter_config.json").resolve()
STAGED_V52 = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d"
).resolve()
STAGED_WEIGHTS_V52 = (STAGED_V52 / "adapter_model.safetensors").resolve()
STAGED_CONFIG_V52 = (STAGED_V52 / "adapter_config.json").resolve()
STAGED_MANIFEST_V52 = (STAGED_V52 / "stage_manifest_v44a.json").resolve()
STAGED_WEIGHTS_SHA256_V52 = (
    "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
)
STAGED_CONFIG_SHA256_V52 = SOURCE_CONFIG_SHA256_V52
STAGED_MANIFEST_FILE_SHA256_V52 = (
    "e30ba44563b5db56f4a487b26f4e2310fd3755b15f8db69d9400facd8baa3813"
)
STAGED_MANIFEST_CONTENT_SHA256_V52 = (
    "ea328ada018e1c0d182d329d2a9cb81f8f0375aef93738f3b9c0a00f63c82da3"
)
STAGED_TRANSFORMED_IDENTITY_SHA256_V52 = (
    "f210bf05e7fe38481d0a7d9c641a7f902e575521b50e98bdc021bf11b49cb1c8"
)
STAGED_ORDERED_VALUES_SHA256_V52 = (
    "26daf52fac11a584891f745e9682c4409ff4aee3119814f0a083a91a192bdf45"
)
MASTER_ORDERED_KEY_SHA256_V52 = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)
RUNTIME_ASSIGNMENT_SHA256_V52 = (
    "bac008805d7fc7c6279c47255d8d1563b0be978cb21109e8c013114f143e09df"
)
TRAIN_DATASET_V52 = (
    ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5/"
    "train_v434_fold3_v49d.jsonl"
).resolve()
TRAIN_MEMBERSHIP_V52 = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v52_v434_train_row_conflict_unit_membership.json"
).resolve()
TRAIN_GENERATION_PANEL_V52 = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v52_v434_train_generation_panel.json"
).resolve()
RUNTIME_V52 = {
    "tuned_folder": (
        "/home/catid/specialist/experiments/vllm_moe_tuning/"
        "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c"
    ),
    "tuned_table_content_sha256": (
        "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ),
}

OBJECTIVE_PATHS_V52 = {
    "domain": ("domain", "aggregate", "equal_unit_mean"),
    "fragile_generation_f1": (
        "fragile_generation", "equal_conflict_unit_mean_f1",
    ),
    "prose_lm": ("prose_lm", "mean_token_logprob"),
    "qa_answer_logprob": ("qa_answer_logprob", "mean_example_logprob"),
    "qa_generation_f1": ("qa_generation", "mean_f1"),
}
TRAIN_GATE_NAMES_V52 = (
    "domain_point_improvement",
    "prose_lm_noninferiority",
    "qa_logprob_noninferiority",
    "qa_generation_f1_noninferiority",
    "qa_generation_exact_noninferiority",
    "qa_generation_nonzero_noninferiority",
    "fragile_generation_f1_noninferiority",
    "fragile_generation_exact_noninferiority",
    "fragile_generation_nonzero_noninferiority",
)
EDGE_IDENTITY_KEYS_V52 = (
    "document_sha256", "normalized_url", "raw_lineage", "semantic_cluster",
)

SEALED_NUMERIC_PARENTS_V52 = {
    "v52_retry3_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_nested_p8_vs_p16_v52_retry3.json",
        "file_sha256": "f98bc057b9082d0a819d17972c3de643f4919aec418324a5957362892657cd63",
        "content_sha256": "a1e04573d1191c84a3d6119331ae539bc77af38c7312f4fd214cbda9604cd96e",
    },
    "v52_retry3_attempt": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/.v52_matched_lora_es_nested_p8_vs_p16_retry3.attempt.json",
        "file_sha256": "718d6f4319a6aef72f28bcddb1aedf3d64346944430b6c2d4aeaab9a123149a2",
        "content_sha256": "e3e0f9bb53394875ebd9d389b6694b708704e1561e17466ab8722cc8d588d700",
    },
    "v52_retry3_preinstall_baseline": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry3/preinstall_actor_baseline_v52.json",
        "file_sha256": "1440aa58371b799ff6a406af5239bbe158c9920e938e18c886a923e19c7223a4",
        "content_sha256": "05b85bebc5011eb75a71e744d3fa4dcf08064cf9f50069c92a280dc436290f71",
    },
    "v52_retry3_master_identity_audit": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry3/master_identity_audit_v52.json",
        "file_sha256": "d5b2c14e60ac34901adb89f89255a4f4f01b03e15ca69d7fe1fb609a38ca966b",
        "content_sha256": "65dd53e3d0f5ef7452d5db08459468db7b0c27835d29fb37ed4caa4dd21b4d30",
    },
    "v52_retry3_numeric_calibration": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry3/numeric_calibration_v52.json",
        "file_sha256": "b5ba4758ed101c4cea300cd64ff867afb8bcbc6d4144818d30538376ecc0b433",
        "content_sha256": "b35ffe5a352d6404853ea6831f73763c8d3b67c0bbfcc7a87e44441097eff7eb",
    },
    "v52_retry3_failure": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry3/failure_v52.json",
        "file_sha256": "e6eb487495d63fa0e198ace775119397be85c27c4d003be474b94eb9115eae97",
        "content_sha256": "961b4c3e424ac3d37b39d650e4704de7823967da6ac6128db65675a8a28c8f71",
    },
    "v52_retry2_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_nested_p8_vs_p16_v52_retry2.json",
        "file_sha256": "a9bcae8c3d178163d585456006d6f2e94e198179b824a96bb2ce95b771bf3035",
        "content_sha256": "d754114e3d4f8ab8776016dbf787d952cf2c6a4365d7c83f4d143d0318050cf5",
    },
    "v52_retry2_attempt": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/.v52_matched_lora_es_nested_p8_vs_p16_retry2.attempt.json",
        "file_sha256": "f4166abe06260db6a3ec3f4690905eb17d42015684a2f0d3ed3eec50d8134025",
        "content_sha256": "30e83962ce0c179b7bcd642f40eb862eee553aaa0fd231ca95a11019076aa0cd",
    },
    "v52_retry2_preinstall_baseline": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry2/preinstall_actor_baseline_v52.json",
        "file_sha256": "4193d63f09b0a9adbcd417ac58480ddf3236bccee29cb5d091fb4cfe17fc9653",
        "content_sha256": "276f2fe9d00c84b220e66f18a2546e78c16024ffad56191e5394d896c2dd8cc7",
    },
    "v52_retry2_failure": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry2/failure_v52.json",
        "file_sha256": "e87f99cddde971a57c849073a7a7a89044bc8946f069e6dc4307376ce81a92de",
        "content_sha256": "2e627c460f3e5f104beaf5c50b375e03bca4850e4ca4c78f1a65122b3f78ffc3",
    },
    "v52_retry1_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_nested_p8_vs_p16_v52_retry1.json",
        "file_sha256": "c051ed9a595735f18cc721e6fbcc09a73ed3cc197c66375f3168323a2c306f94",
        "content_sha256": "feaead0c4ef1b9aabb46adcd8b7c0923794e2f29478cbdb2193522d44769e6eb",
    },
    "v52_retry1_attempt": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/.v52_matched_lora_es_nested_p8_vs_p16_retry1.attempt.json",
        "file_sha256": "fcba8593976019f086df95b104745cbc341ee738153be0f119b27b2509c0cda4",
        "content_sha256": "651fc5056314633af4a8694eedaa2b7c35a26a9a11b36e920a182bd18f964a86",
    },
    "v52_retry1_failure": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry1/failure_v52.json",
        "file_sha256": "0f5bdb6b5d42621f1aa25d627ee11706c0203b2372cfec27a5dc126ff0399ac4",
        "content_sha256": "fc6804d0861b1e1f1e9b71ecd222792e7f8937ce8338161aad461d84dcd19ac3",
    },
    "v52_original_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_nested_p8_vs_p16_v52.json",
        "file_sha256": "b8ea48b11a9ea91ba3ece09bc854d74a7a17bb28e6f9496e4f186e0c574eb15d",
        "content_sha256": "007f89c39593219040ca10783d7b4ae7bd3e7c5163a383ad90a1594302304f5f",
    },
    "v52_pre_model_attempt": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/.v52_matched_lora_es_nested_p8_vs_p16.attempt.json",
        "file_sha256": "39584acfac8051010973a26b6660d94fcd7d77d5e5a40795e6bd9cdd155434bd",
        "content_sha256": "958d966296edad689c0d9f3b18bfdd8b6c085c9b2de99b3b803c6f02f59a22ec",
    },
    "v52_pre_model_failure": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16/failure_v52.json",
        "file_sha256": "0b5ef9458665174757bdb57b44df9ceb98c6f7f1cc36230890cfded226604727",
        "content_sha256": "d1f4e6e9e8489dffb2cff2457e48f00691ae618de467a1245091a01121422e1a",
    },
    "v49d_equal_train_report": {
        "path": ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5/runtime_report_v434_equal.json",
        "file_sha256": "0f669d188046849ccb6a3013938f9979214711f5e83c0d9e54290c9c20c850d8",
        "content_sha256": "a8de8805238335ba27db11359fc9f75ad65431d64b25f6a6c10eb5e0f3bdba0e",
    },
    "v49d_stage_manifest": {
        "path": STAGED_MANIFEST_V52,
        "file_sha256": STAGED_MANIFEST_FILE_SHA256_V52,
        "content_sha256": STAGED_MANIFEST_CONTENT_SHA256_V52,
    },
    "v40c_topology_report": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v40c_v37_lora_topology_probe_tuned_projection_retry/lora_topology_report_v40c.json",
        "file_sha256": "7672f835239b91e66a03a512ad9fbe3cbbaff31783a7b1a26bdd136d98b55050",
        "content_sha256": "9394ed06a80fffb1c4cc1532ac59741ab4c4a1c5a481136f29415b463eaf747d",
    },
    "v49e_replicated_shadow_report": {
        "path": ROOT / "experiments/eval_reports/sft_v434_equal_replicated_shadow_only_v49e.json",
        "file_sha256": "9bb57c48d77c45ff91b8296a7f7fe65c67e91a26473c286c84dc074107c6a6eb",
        "content_sha256": "fed27cb9202ba7f7f0e7cc3b882b62c50c4748ea101a366c077a75679bb1aa3c",
    },
    "v48b_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_generation_boundary_pop8_v48b.json",
        "file_sha256": "34e19fe84ff061b98a8627f07daab59f5cbb8c718668fc479454114fef67c3d0",
        "content_sha256": "4d5e17a07551377f0ef39c3dfa306fda68b9b669eeafd3c78ebbfff894072d1e",
    },
    "v48e_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_generation_boundary_reprojection_backtracking_v48e.json",
        "file_sha256": "bd3b4ebb0c2d125c809fc2a4f80d816be3a59b235706cbde1ab9800b30bc3b91",
        "content_sha256": "48f694dc9bd8a4466f413315b8a9a507df9f8a2f3111e54f43f8a752ee5fac7f",
    },
    "v48e_report": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v48e_generation_boundary_reprojection_backtracking/generation_boundary_reprojection_backtracking_report_v48e.json",
        "file_sha256": "8d78b6588ce05df17130ca7f3e0fb6c0827cf34e55a57620292f94aa917b7e64",
        "content_sha256": "0ff2f5effe7cb4318f9c3f9c25353312c382db73916891e1cdb1fed16a91aa28",
    },
    "v51_acceptance_audit": {
        "path": ROOT / "experiments/eggroll_es_hpo/audits/v51_direct_master_transition_vs_v50.json",
        "file_sha256": "ceb5ada0f4b6a3a994260079f62a614be1bf91eb6d954a3e9ffd5c53e5a147c2",
        "content_sha256": "60e55b62bfec09be64acd5c612a394dc54a3cf9596bdf2ef949c6ef93f5bd25d",
    },
    "v51_report": {
        "path": ROOT / "experiments/eggroll_es_hpo/runs/v51_direct_pinned_master_transition_microbenchmark_retry1/transition_microbenchmark_report_v51.json",
        "file_sha256": "a727990745f6d41e0eb36104a19ce1d92c1645c2bf01bfb7ce3b02e4bbe29b1f",
        "content_sha256": "42b31602793aa727dcfc5622fce4c0e1c1a094e2b8234f3f6f402f52db68ba37",
    },
    "v46f_ood_first_preregistration": {
        "path": ROOT / "experiments/eggroll_es_hpo/preregistrations/base_vs_lora_es_v43l_replicated_ood_first_eval_v46f.json",
        "file_sha256": "8e9533dc64435a3c0c18c51b5fe100c7cead77555f4a6d63d2923bf30a3cd1c8",
        "content_sha256": "db5a8d1489b4a92b135ca0f977c7c4335f42d6200e1bb7fe473f35da725c542e",
    },
}


def canonical_sha256_v52(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v52(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_json_v52(label: str) -> dict:
    """Load only a content-addressed numeric/protocol parent artifact."""
    sealed = SEALED_NUMERIC_PARENTS_V52[label]
    path = sealed["path"]
    if file_sha256_v52(path) != sealed["file_sha256"]:
        raise RuntimeError(f"v52 sealed parent file changed: {label}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != sealed["content_sha256"]
        or canonical_sha256_v52(compact) != sealed["content_sha256"]
    ):
        raise RuntimeError(f"v52 sealed parent content changed: {label}")
    return value


def state_derivations_v52() -> list[dict]:
    """Freeze state derivations without fabricating CPU/GPU-equal hashes."""
    result = []
    for direction, seed in enumerate(P16_SEEDS_V52):
        for label, sign in (("plus", 1), ("minus", -1)):
            result.append({
                "state_index": len(result),
                "direction": direction,
                "label": label,
                "sign": sign,
                "seed": seed,
                "sigma": SIGMA_V52,
                "master_sha256": MASTER_SHA256_V52,
                "derivation": "V41A antithetic candidate from pinned FP32 master",
                "candidate_identity_policy": (
                    "four-actor GPU-runtime consensus required before scoring"
                ),
                "runtime_identity_policy": (
                    "four-actor BF16-runtime consensus required before scoring"
                ),
            })
    return result


def scientific_arms_v52() -> dict:
    shared = {
        "initial_master_sha256": MASTER_SHA256_V52,
        "initial_master_runtime_sha256": MASTER_RUNTIME_SHA256_V52,
        "train_dataset_sha256": DATASET_SHA256_V52,
        "train_bundle_content_sha256": TRAIN_BUNDLE_CONTENT_SHA256_V52,
        "generation_panel_request_order_sha256": REQUEST_ORDER_SHA256_V52,
        "shared_fresh_prepopulation_calibration": True,
        "sigma": SIGMA_V52,
        "alpha": ALPHA_V52,
        "primary": (
            "equal average of unit-norm domain and fragile generated-F1 "
            "centered-rank coefficient vectors"
        ),
        "projection_halfspaces": list(OBJECTIVE_PATHS_V52),
        "scale_order": list(SCALE_ORDER_V52),
        "train_gate_names": list(TRAIN_GATE_NAMES_V52),
        "signed_replicates_per_direction": ACTORS_V52,
    }
    return {
        "p8": {**shared, "population_size": 8, "seeds": list(P8_SEEDS_V52)},
        "p16": {**shared, "population_size": 16, "seeds": list(P16_SEEDS_V52)},
    }


def assert_one_scientific_variable_v52(arms: dict) -> dict:
    left, right = dict(arms["p8"]), dict(arms["p16"])
    left_size = left.pop("population_size")
    right_size = right.pop("population_size")
    left_seeds = left.pop("seeds")
    right_seeds = right.pop("seeds")
    if (
        left != right or (left_size, right_size) != POPULATION_SIZES_V52
        or tuple(left_seeds) != P8_SEEDS_V52
        or tuple(right_seeds) != P16_SEEDS_V52
        or tuple(right_seeds[:left_size]) != tuple(left_seeds)
    ):
        raise RuntimeError("v52 arms differ by more than nested population size")
    return {
        "sole_scientific_variable": "antithetic_direction_population_size",
        "control": 8,
        "treatment": 16,
        "p8_is_exact_nested_prefix_of_p16": True,
        "all_other_scientific_fields_equal": True,
    }


def _average_tie_ranks_v52(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = 0.5 * (cursor + stop - 1)
        for index in order[cursor:stop]:
            result[index] = rank
        cursor = stop
    return result


def _correlation_v52(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(dtype=np.float64)
    right = right - right.mean(dtype=np.float64)
    denominator = math.sqrt(float(np.mean(left * left) * np.mean(right * right)))
    return float(np.mean(left * right) / denominator) if denominator else 0.0


def reliability_gate_v52(
    central_replicates: list[list[float]],
    fresh_calibration_observed_maximum: float,
) -> dict:
    values = np.asarray(central_replicates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] not in POPULATION_SIZES_V52 or values.shape[1] != 4:
        raise ValueError("v52 reliability requires an 8x4 or 16x4 matrix")
    if not np.isfinite(values).all():
        raise ValueError("v52 reliability received non-finite values")
    means = values.mean(axis=1, dtype=np.float64)
    observed = float(np.mean((means - means.mean()) ** 2))
    single_noise = float(np.var(values, axis=1, ddof=1, dtype=np.float64).mean())
    mean_noise = single_noise / 4.0
    signal = max(0.0, observed - mean_noise)
    signal_standard_deviation = math.sqrt(signal)
    reliability = signal / observed if observed > 0.0 else 0.0
    left = values[:, :2].mean(axis=1, dtype=np.float64)
    right = values[:, 2:].mean(axis=1, dtype=np.float64)
    spearman = _correlation_v52(
        _average_tie_ranks_v52(left.tolist()),
        _average_tie_ranks_v52(right.tolist()),
    )
    calibration_safe = bool(
        math.isfinite(fresh_calibration_observed_maximum)
        and 0.0 <= fresh_calibration_observed_maximum <= 0.001802103667415178
    )
    signal_clears_fresh_calibration = bool(
        calibration_safe
        and signal_standard_deviation > fresh_calibration_observed_maximum
    )
    result = {
        "schema": "nested-population-reliability-v52",
        "population_size": int(values.shape[0]),
        "central_replicates": values.tolist(),
        "reliability": reliability,
        "minimum_reliability": 0.8,
        "estimated_signal_standard_deviation": signal_standard_deviation,
        "split_half_spearman": spearman,
        "minimum_split_half_spearman": 0.7,
        "fresh_calibration_observed_maximum_actor_spread": float(
            fresh_calibration_observed_maximum
        ),
        "historical_calibration_ceiling": 0.001802103667415178,
        "fresh_calibration_inside_historical_ceiling": calibration_safe,
        "estimated_signal_standard_deviation_clears_fresh_calibration_maximum": (
            signal_clears_fresh_calibration
        ),
        "passed": bool(
            reliability >= 0.8
            and spearman >= 0.7
            and calibration_safe
            and signal_clears_fresh_calibration
        ),
    }
    result["content_sha256"] = canonical_sha256_v52(result)
    return result


def _nested_v52(value: dict, path: tuple[str, ...]) -> float:
    current = value
    for key in path:
        current = current[key]
    result = float(current)
    if not math.isfinite(result):
        raise ValueError("v52 objective contains a non-finite score")
    return result


def extract_arm_sign_scores_v52(signed_scores: dict, population_size: int) -> dict:
    if population_size not in POPULATION_SIZES_V52:
        raise ValueError("v52 population arm must be P8 or P16")
    if set(signed_scores) != {"plus", "minus"}:
        raise ValueError("v52 signed population labels changed")
    for sign in ("plus", "minus"):
        if len(signed_scores[sign]) != 16:
            raise ValueError("v52 requires one complete P16 population")
        if any(len(row) != ACTORS_V52 for row in signed_scores[sign]):
            raise ValueError("v52 signed population actor coverage changed")
    return {
        objective: {
            sign: [[
                _nested_v52(signed_scores[sign][direction][actor], path)
                for actor in range(ACTORS_V52)
            ] for direction in range(population_size)]
            for sign in ("plus", "minus")
        }
        for objective, path in OBJECTIVE_PATHS_V52.items()
    }


def objective_coefficients_v52(sign_scores: dict) -> dict:
    if set(sign_scores) != {"plus", "minus"}:
        raise ValueError("v52 objective sign labels changed")
    arrays = {
        sign: np.asarray(sign_scores[sign], dtype=np.float64)
        for sign in ("plus", "minus")
    }
    shapes = {item.shape for item in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("v52 objective sign shapes differ")
    population_size, actors = next(iter(shapes))
    if population_size not in POPULATION_SIZES_V52 or actors != ACTORS_V52:
        raise ValueError("v52 objective requires P8/P16 by four actors")
    if not all(np.isfinite(item).all() for item in arrays.values()):
        raise ValueError("v52 objective contains non-finite signed scores")
    fixed_four_actor_mean_signed = {
        sign: [math.fsum(row.tolist()) / ACTORS_V52 for row in item]
        for sign, item in arrays.items()
    }
    signed = (
        fixed_four_actor_mean_signed["plus"]
        + fixed_four_actor_mean_signed["minus"]
    )
    ranks = _average_tie_ranks_v52(signed)
    utilities = (ranks / (len(signed) - 1) - 0.5).tolist()
    plus = utilities[:population_size]
    minus = utilities[population_size:]
    coefficients = [
        left - right for left, right in zip(plus, minus, strict=True)
    ]
    return {
        "schema": "nested-population-centered-ranks-v52",
        "actor_reducer": "fixed mean in ascending actor-rank order",
        "fixed_four_actor_mean_signed_scores": fixed_four_actor_mean_signed,
        "signed_centered_rank_utilities": {"plus": plus, "minus": minus},
        "coefficients": coefficients,
        "zero_spread": all(value == 0.0 for value in coefficients),
    }


def project_arm_v52(signed_scores: dict, population_size: int) -> dict:
    sign_scores = extract_arm_sign_scores_v52(signed_scores, population_size)
    objectives = {
        name: objective_coefficients_v52(scores)
        for name, scores in sign_scores.items()
    }
    if any(item.get("zero_spread") is True for item in objectives.values()):
        raise RuntimeError("v52 objective has zero signed-population spread")
    domain = np.asarray(objectives["domain"]["coefficients"], dtype=np.float64)
    fragile = np.asarray(
        objectives["fragile_generation_f1"]["coefficients"], dtype=np.float64,
    )
    primary = 0.5 * (
        domain / np.linalg.norm(domain) + fragile / np.linalg.norm(fragile)
    )
    anchors = {name: item["coefficients"] for name, item in objectives.items()}
    projection = multi_anchor.project_multi_anchor_trust_region_v43h(
        primary.tolist(), anchors, max_norm_ratio=0.5,
    )
    diagnostics = projection["diagnostics"]
    if (
        diagnostics.get("decision") != "project_and_trust_region"
        or diagnostics.get("anchor_order") != list(OBJECTIVE_PATHS_V52)
        or diagnostics.get("all_anchor_halfspaces_satisfied") is not True
    ):
        raise RuntimeError("v52 five-halfspace projection failed closed")
    result = {
        "schema": "nested-five-halfspace-projection-v52",
        "population_size": population_size,
        "seeds": list(P16_SEEDS_V52[:population_size]),
        "sign_scores_sha256": canonical_sha256_v52(sign_scores),
        "objective_fitness": objectives,
        "primary_coefficients": primary.tolist(),
        "projection": projection,
        "coefficients": projection["coefficients"],
    }
    result["content_sha256"] = canonical_sha256_v52(result)
    return result


def scale_plans_v52(projected: dict) -> list[dict]:
    coefficients = np.asarray(projected["coefficients"], dtype=np.float64)
    unconstrained = float(
        projected["projection"]["diagnostics"]["unconstrained_domain_norm"]
    )
    source_ratio = float(np.linalg.norm(coefficients) / unconstrained)
    if not math.isclose(source_ratio, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("v52 source projection did not reach the trust cap")
    result = []
    for ratio in SCALE_ORDER_V52:
        values = (coefficients * (ratio / source_ratio)).tolist()
        actual = float(np.linalg.norm(values) / unconstrained)
        if not math.isclose(actual, ratio, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v52 scale plan norm changed")
        result.append({
            "target_norm_ratio": ratio,
            "coefficients": values,
            "coefficient_sha256": canonical_sha256_v52({
                "seeds": projected["seeds"], "coefficients": values,
            }),
            "actual_norm_ratio": actual,
            "five_halfspaces_preserved_by_positive_scaling": True,
        })
    return result


def selected_train_ratio_v52(results: list[dict]) -> float | None:
    if len(results) > len(SCALE_ORDER_V52):
        raise ValueError("v52 evaluated too many scales")
    for index, result in enumerate(results):
        if result.get("target_norm_ratio") != SCALE_ORDER_V52[index]:
            raise ValueError("v52 scale evaluation order changed")
        checks = result.get("checks", {})
        if set(checks) != set(TRAIN_GATE_NAMES_V52):
            raise ValueError("v52 nine-check train gate inventory changed")
        passed = all(checks.values()) and result.get("candidate_consensus_passed") is True
        if passed:
            if index != len(results) - 1:
                raise ValueError("v52 continued after its largest passing scale")
            return SCALE_ORDER_V52[index]
        if result.get("exact_abort_readback_passed") is not True:
            raise ValueError("v52 rejected candidate was not exactly aborted")
    return None


def ood_eligible_v52(gate: dict) -> bool:
    qa = gate.get("ood_qa", {})
    prose = gate.get("ood_prose", {})
    protocol = gate.get("protocol", {})
    return bool(
        qa.get("exact_count_delta", -1) >= 0
        and qa.get("mean_reward_delta", -1.0) >= 0.0
        and prose.get("point_delta", -1.0) >= 0.0
        and prose.get("paired_document_bootstrap_95_ci", [-1.0])[0] >= 0.0
        and protocol.get("counter_increase", 1) == 0
        and gate.get("raw_questions_answers_or_generations_persisted") is False
    )


def document_disjoint_shadow_eligible_v52(shadow: dict) -> bool:
    intersections = shadow.get("edge_identity_intersections", {})
    return bool(
        shadow.get("rows") == 83
        and shadow.get("conflict_units") == 51
        and shadow.get("split_manifest_file_sha256")
        == SPLIT_MANIFEST_FILE_SHA256_V52
        and set(intersections) == set(EDGE_IDENTITY_KEYS_V52)
        and all(intersections[key] == 0 for key in EDGE_IDENTITY_KEYS_V52)
        and shadow.get("document_disjoint_from_fold3_train") is True
    )


def treatment_success_v52(result: dict) -> bool:
    p8, p16 = result.get("p8", {}), result.get("p16", {})
    if not p16.get("ood_eligible") or not p16.get("shadow_better_than_master"):
        return False
    if p8.get("ood_eligible"):
        return bool(p16.get("shadow_better_than_p8"))
    return True


def compute_plan_v52() -> dict:
    v51_population_seconds = 208.916206
    v51_total_seconds = 292.9445971029927
    startup_cleanup_seconds = v51_total_seconds - v51_population_seconds
    p16_population_seconds = 2.0 * v51_population_seconds
    v48e_six_scale_total_seconds = 273.64169292800943
    per_arm_incremental = v48e_six_scale_total_seconds - startup_cleanup_seconds
    train_before_fresh_calibration_seconds = (
        startup_cleanup_seconds + p16_population_seconds
        + 2.0 * per_arm_incremental
    )
    # V48B attempt receipt -> sealed anchor calibration artifact.  Adding the
    # whole interval is conservative because it includes part of startup that
    # is already represented above.
    fresh_shared_calibration_seconds = 232.43436460199998
    worst_train_seconds = (
        train_before_fresh_calibration_seconds
        + fresh_shared_calibration_seconds
    )
    total_low = worst_train_seconds + 70.0
    total_high = worst_train_seconds + 240.0
    result = {
        "schema": "sealed-v51-linear-compute-plan-v52",
        "states": 32,
        "directions": 16,
        "actors_per_state": 4,
        "state_actor_receipts": 32 * 4,
        "phase_actor_receipts": 32 * 4 * len(PHASES_V52),
        "v51_observed_16_state_population_seconds": v51_population_seconds,
        "linear_32_state_population_seconds": p16_population_seconds,
        "observed_startup_cleanup_seconds": startup_cleanup_seconds,
        "train_before_fresh_calibration_seconds": (
            train_before_fresh_calibration_seconds
        ),
        "conservative_shared_fresh_calibration_seconds": (
            fresh_shared_calibration_seconds
        ),
        "worst_case_two_arm_train_only_seconds": worst_train_seconds,
        "estimated_end_to_end_wall_seconds": [total_low, total_high],
        "estimated_four_gpu_hours": [
            4.0 * total_low / 3600.0,
            4.0 * total_high / 3600.0,
        ],
        "interpretation": (
            "linear estimate from accepted V51 population timing plus two "
            "worst-case six-scale V48E backtracking arms, the full observed "
            "V48B attempt-to-anchor interval for one shared fresh calibration, "
            "and aggregate-only OOD/shadow evaluation"
        ),
    }
    result["content_sha256"] = canonical_sha256_v52(result)
    return result
