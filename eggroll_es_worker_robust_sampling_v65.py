#!/usr/bin/env python3
"""GPU-free numeric scorer for the V65 robust-sampling population."""

from __future__ import annotations

import hashlib
import math
import os
import time


def score_generation_batch_v65(
    panel_items: list[dict], answers: list[str], batch: list,
) -> list[dict]:
    """Reduce transient generations to hashes and F1/exact/nonzero only."""
    import run_lora_es_generation_boundary_v48b as v48b

    if (
        len(panel_items) != 64 or len(answers) != 64 or len(batch) != 64
        or any(item.get("request_index") != index
               for index, item in enumerate(panel_items))
        or any(not isinstance(answer, str) or not answer.strip()
               for answer in answers)
        or any(
            not isinstance(item.get(key), str)
            or len(item[key]) != 64
            for item in panel_items
            for key in ("row_sha256", "unit_identity_sha256")
        )
    ):
        raise RuntimeError("v65 scoring batch coverage changed")
    rows = []
    for item, answer, output in zip(panel_items, answers, batch, strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise RuntimeError("v65 completion multiplicity changed")
        prediction = v48b.v43i.fused._extract_answer(str(generated[0].text))
        f1 = float(v48b.v43i.fused._f1(prediction, answer))
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise RuntimeError("v65 generated-F1 reducer changed")
        answer_tokens = v48b.v43i.fused._tokens(answer)
        prediction_tokens = v48b.v43i.fused._tokens(prediction)
        exact = int(bool(answer_tokens) and prediction_tokens == answer_tokens)
        rows.append({
            "request_index": item["request_index"],
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": item["unit_identity_sha256"],
            "prediction_sha256": hashlib.sha256(
                prediction.encode("utf-8")
            ).hexdigest(),
            "f1": f1,
            "exact": exact,
            "nonzero": int(f1 > 0.0),
        })
    return rows


class RobustGenerationScoringActorV65:
    """A Ray CPU actor that never returns question, answer, or generation text."""

    def __init__(
        self, actor_rank: int, panel_items: list[dict], answers: list[str],
    ) -> None:
        import ray

        gpu_ids = list(ray.get_gpu_ids())
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.actor_rank = int(actor_rank)
        if (
            self.actor_rank not in range(4) or gpu_ids or visible != ""
            or len(panel_items) != 64 or len(answers) != 64
            or any(not isinstance(answer, str) or not answer.strip()
                   for answer in answers)
        ):
            raise RuntimeError("v65 CPU scoring actor isolation changed")
        self.panel_items = [dict(item) for item in panel_items]
        self.answers = list(answers)
        self.gpu_ids = gpu_ids
        self.visible = visible

    def runtime_identity_v65(self) -> dict:
        return {
            "schema": "v65-robust-generation-cpu-scorer",
            "actor_rank": self.actor_rank,
            "gpu_ids": self.gpu_ids,
            "cuda_visible_devices": self.visible,
        }

    def score_v65(self, state: dict, batch: list) -> dict:
        expected = dict(state)
        expected["actor_rank"] = self.actor_rank
        if state != expected:
            raise RuntimeError("v65 scoring actor/state assignment changed")
        started_ns = time.monotonic_ns()
        process_started_ns = time.process_time_ns()
        metrics = score_generation_batch_v65(
            self.panel_items, self.answers, batch,
        )
        process_ended_ns = time.process_time_ns()
        ended_ns = time.monotonic_ns()
        return {
            "schema": "v65-robust-generation-actor-score",
            "state": expected,
            "gpu_ids": self.gpu_ids,
            "unit_metrics": metrics,
            "question_answer_or_generation_text_persisted": False,
            "timing": {
                "actor_rank": self.actor_rank,
                "state_index": state["state_index"],
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
                "process_elapsed_ns": process_ended_ns - process_started_ns,
            },
        }
