# Qwen3.6 V73C exact-phase profiler: CPU-only sealing evidence

Status: prospectively sealed; no V73C model, Ray, CUDA, GPU, dataset, semantic evaluation, HPO, checkpoint, or promotion run was performed while producing this evidence.

## Immutable identities

- Preregistration file SHA-256: `b134a263b1548905d8f9c2373d9f8a8dd79ec2ba74808a5ca23c84f31dc71983`
- Preregistration content SHA-256: `6ea79bd1e948f728af0f93087d0e28ad576cef62bfc04be79180f30a0a93785a`
- External systems-only closure content SHA-256: `23325b31209df8f3e67b44152226119c714000872e1b926003c3a7c4f5137d67`
- Runtime closure module count: 101
- Opaque historical reference bindings: 3 (nonzero; does not authorize resolution or access)

## Measurement contract

- Exact NVTX phase count: 22 in domain `eggroll_es_v73c_phase`.
- Timeline arm: unprivileged CUDA/NVTX/legacy-NCCL/kernel/memcpy/allocation trace, launch-authorized only after fresh-path, exact toolchain, four-GPU identity, idle, and no-compute-process gates.
- Timeline command-template SHA-256: `f8528e034a0e1c0830e5608bd7e46e67915f2cfc84010c4a9cc1602f570cbaa0`
- HBM/DRAM arm: blocked before directory creation or subprocess; `RmProfilingAdminOnly=1` and `ERR_NVGPUCTRPERM`.
- No bandwidth bytes are inferred from utilization percentages; no NCCL transport/path is inferred from topology capability.

## Exact LoRA geometry and host topology

- FP32 tensors/elements: 70 / 4,528,128.
- Logical collective payload per rank: 18,112,512 B; nominal ring schedule 27,168,768 B/rank (not measured physical-link bytes).
- Four physical GPUs: all cross-pairs report NODE, peer read/write capability OK, NVLink capability NS, and NUMA affinity 0.
- Topology matrix SHA-256: `57127e112509330246aafbc4c2be974cfd9643248f88878fb356dbf1f0f0992a`.

## Exact future timeline launch (not executed)

```bash
/home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_qwen36_v73c_exact_phase_profiler.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/qwen36_v73c_exact_phase_profiler.json --preregistration-sha256 b134a263b1548905d8f9c2373d9f8a8dd79ec2ba74808a5ca23c84f31dc71983 --preregistration-content-sha256 6ea79bd1e948f728af0f93087d0e28ad576cef62bfc04be79180f30a0a93785a --arm timeline --execute
```

The launcher re-hashes the sealed `.bashrc` and 11 other timeline runtime inputs before any output directory or profiler subprocess. Its exact inner Nsight command is:

```bash
/usr/bin/bash --noprofile --norc -i -c 'source "$HOME/.bashrc" >/dev/null 2>&1 && exec "$@"' v73c-nsys-launch /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,nccl --nccl-trace=none --cuda-trace-scope=process-tree --cuda-memory-usage=true --cuda-event-trace=false --cuda-graph-trace=graph --sample=none --cpuctxsw=none --backtrace=none --python-sampling=false --python-backtrace=none --pytorch=none --discard-environment=true --force-overwrite=false --export=sqlite --stats=false --show-output=false --wait=all --output=/home/catid/specialist/experiments/eggroll_es_hpo/profiles/v73c_timeline_lora_es_same_live_qwen36_exact_phase/v73c_timeline_lora_es_same_live_qwen36_exact_phase /usr/bin/env PYTHONPATH=/home/catid/specialist/v73c_sitecustomize:/home/catid/specialist SPECIALIST_V73C_SYSTEMS_ONLY_GUARD=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL NCCL_DEBUG_FILE=/home/catid/specialist/experiments/eggroll_es_hpo/profiles/v73c_timeline_lora_es_same_live_qwen36_exact_phase/nccl_debug.%h.%p.log /home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_lora_es_v71_v72_profile_calibration_v73c.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/qwen36_v73c_exact_phase_profiler.json --preregistration-sha256 b134a263b1548905d8f9c2373d9f8a8dd79ec2ba74808a5ca23c84f31dc71983 --preregistration-content-sha256 6ea79bd1e948f728af0f93087d0e28ad576cef62bfc04be79180f30a0a93785a --arm timeline --execute
```

The HBM arm remains unauthorized. An administrator permission change would require a new additive preregistration; it does not silently enable this contract.
