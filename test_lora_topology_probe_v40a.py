import json

from safetensors import safe_open

import build_lora_topology_probe_preregistration_v40a as build
import eggroll_es_worker_lora_topology_v40a as worker
import run_lora_topology_probe_v40a as runtime


def test_exact_v37_peft_surface_maps_to_expected_runtime_targets():
    elements = 0
    keys = []
    with safe_open(runtime.ADAPTER_FILE, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            logical, side = worker._source_parts(key)
            target, slices = worker._runtime_target(logical)
            assert side in {"A", "B"}
            assert target.startswith("model.layers.")
            assert slices and min(slices) >= 0
            keys.append(key)
            elements += handle.get_tensor(key).numel()
    assert len(keys) == 70
    assert elements == 4_528_128
    assert len(set(keys)) == len(keys)


def test_preregistration_is_train_only_and_self_hashed():
    value = build.build()
    assert value["schema"] == "lora-topology-preregistration-v40a"
    assert value["dataset_or_evaluation_access_authorized"] is False
    assert value["synthetic_prompt_only"] is True
    assert value["runtime"]["physical_gpu_ids"] == [0, 1, 2, 3]
    assert value["runtime"]["engine_count"] == 4
    assert value["probe"]["peft_elements_expected"] == 4_528_128
    assert runtime.file_sha256(runtime.ADAPTER / "adapter_config.json") == runtime.file_sha256(
        runtime.STAGED_ADAPTER / "adapter_config.json"
    )
    content = value.pop("content_sha256_before_self_field")
    assert content == runtime.canonical_sha256(value)
    forbidden = ("heldout", "holdout", "shadow", "ood")
    paths = json.dumps(value["implementation_bindings"]).lower()
    assert not any(item in paths for item in forbidden)


def test_output_record_is_content_only():
    class Logprob:
        def __init__(self, value): self.logprob = value

    class Sample:
        token_ids = [7]
        cumulative_logprob = -0.25
        logprobs = [{7: Logprob(-0.25)}]

    class Output:
        prompt_token_ids = [1, 2]
        prompt_logprobs = [None, {2: Logprob(-0.5)}]
        outputs = [Sample()]

    assert runtime.output_record(Output()) == {
        "prompt_token_ids": [1, 2],
        "prompt_logprobs": [None, [(2, -0.5)]],
        "outputs": [{
            "token_ids": [7], "cumulative_logprob": -0.25,
            "logprobs": [[(7, -0.25)]],
        }],
    }
