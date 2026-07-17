#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${ROOT}/es-at-scale/.venv/bin/python}"
RUN="${RUN:-${ROOT}/experiments/eggroll_es_hpo/runs/v80_bf16_kv_mamba_capacity_0479_r1}"

if [[ -e "${RUN}" ]]; then
    echo "V80 requires a fresh run directory: ${RUN}" >&2
    exit 2
fi
mkdir -p "${RUN}"

pids=()
monitor_pid=""

stop_children() {
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
    fi
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
handle_signal() {
    stop_children
    trap - HUP INT TERM
    exit 130
}
trap handle_signal HUP INT TERM

for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
        "${ROOT}/probe_vllm_bf16_kv_mamba_capacity_v80.py" \
        --precision-arm fp8_serialized \
        --actor-label "gpu-${gpu}" \
        --output "${RUN}/gpu_${gpu}.json" \
        >"${RUN}/gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
    printf '%s,%s\n' "${gpu}" "${pids[-1]}" \
        >>"${RUN}/actor_pids.csv"
done

"${PYTHON}" "${ROOT}/monitor_qwen36_fp8_kv_capacity_v79.py" \
    --actor-pids "${RUN}/actor_pids.csv" \
    --output "${RUN}/gpu_telemetry_v80.jsonl" \
    --sample-interval-seconds 0.5 \
    --cleanup-batches 3 \
    --max-cleanup-wait-seconds 60 \
    --require-pcie \
    >"${RUN}/gpu_telemetry_monitor_v80.log" 2>&1 &
monitor_pid="$!"

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
wait "${monitor_pid}"

if [[ "${status}" -ne 0 ]]; then
    echo "At least one V80 actor failed; evidence is retained at ${RUN}" >&2
    exit "${status}"
fi

echo "V80 data-free live artifacts written to ${RUN}"
