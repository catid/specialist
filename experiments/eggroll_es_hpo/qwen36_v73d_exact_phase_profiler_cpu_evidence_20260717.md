# Qwen3.6 V73D exact-phase profiler: CPU-only sealing evidence

Status: prospectively sealed; no V73D model, Ray, CUDA, GPU, dataset, semantic evaluation, HPO, checkpoint, or promotion run was performed while producing this evidence.

## Immutable identities

- Preregistration file SHA-256: `d1810d51ecc49615d4067c4b8b151fa9154cb708a6658c02a497210238768c0a`
- Preregistration content SHA-256: `512287c5a438cd1a22b099f61fe55e777de86a71b52555b49a47959425deb740`
- External systems-only closure content SHA-256: `2c6f08b578ca7dfc0ff1aaa6fdcca3acc7618ceaded90d6eda3c975fc633aab2`
- Runtime closure module count: 101
- Opaque historical reference bindings: 9 (nonzero; does not authorize resolution or access)

## Additive repair and actor bootstrap

- V73C attempt 1 remains immutable and failed closed because the controller sitecustomize guard did not run in the prestarted Ray actor before its V72 import.
- Immutable V73C run/profile failure file SHA-256: `5708f456e7736944b7304c5a17486b82c15c5a451a51679734ff00e86cdefef7` / `ae76cdcf38a71ac7e27035a38d48867a91ed68248c91ef4535ce25b0d79a0c17`.
- V73D controller mechanism: `controller_sitecustomize`; actor mechanism: `ray_actor_worker_extension_pre_parent_import`. They are separately attributed.
- The Ray job runtime environment injects the exact guard hash before `ray.init`, while actor-specific runtime environments remain merged.
- The actor extension refuses V72/V71/V41A or the three historical reference modules if any was imported before its guard install.
- Boundary registry content SHA-256: `5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7`; required successful protected open/resolve/metadata/enumeration operations: 0.

## GPU-time accounting

- Reserved wall GPU-seconds are reported independently from directly measured model-resident and useful GPU-seconds.
- The legacy allocation/residency estimate is diagnostic and is not accepted as either direct residency or useful compute evidence.
- Profiled event time is not relabeled as unprofiled useful time; promotion-charged GPU-seconds remain 0.

## Measurement contract

- Exact NVTX phase count: 22 in domain `eggroll_es_v73d_phase`.
- Timeline arm: unprivileged CUDA/NVTX/legacy-NCCL/kernel/memcpy/allocation trace, launch-authorized only after fresh-path, exact toolchain, four-GPU identity, idle, and no-compute-process gates.
- Timeline command-template SHA-256: `1ca7f2bf9c5d187ba1b42c90d8d43e1ebb310d9ee21cf7e6dba18c5378d18bb7`
- HBM/DRAM arm: blocked before directory creation or subprocess; `RmProfilingAdminOnly=1` and `ERR_NVGPUCTRPERM`.
- No bandwidth bytes are inferred from utilization percentages; no NCCL transport/path is inferred from topology capability.

## Exact LoRA geometry and host topology

- FP32 tensors/elements: 70 / 4,528,128.
- Logical collective payload per rank: 18,112,512 B; nominal ring schedule 27,168,768 B/rank (not measured physical-link bytes).
- Four physical GPUs: all cross-pairs report NODE, peer read/write capability OK, NVLink capability NS, and NUMA affinity 0.
- Topology matrix SHA-256: `57127e112509330246aafbc4c2be974cfd9643248f88878fb356dbf1f0f0992a`.

## Exact future timeline launch (not executed)

```bash
/home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_qwen36_v73d_exact_phase_profiler.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/qwen36_v73d_exact_phase_profiler.json --preregistration-sha256 d1810d51ecc49615d4067c4b8b151fa9154cb708a6658c02a497210238768c0a --preregistration-content-sha256 512287c5a438cd1a22b099f61fe55e777de86a71b52555b49a47959425deb740 --arm timeline --execute
```

The launcher re-hashes the sealed `.bashrc` and 11 other timeline runtime inputs before any output directory or profiler subprocess. Its exact inner Nsight command is:

```bash
/usr/bin/bash --noprofile --norc -i -c 'source "$HOME/.bashrc" >/dev/null 2>&1 && exec "$@"' v73d-nsys-launch /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,nccl --nccl-trace=none --cuda-trace-scope=process-tree --cuda-memory-usage=true --cuda-event-trace=false --cuda-graph-trace=graph --sample=none --cpuctxsw=none --backtrace=none --python-sampling=false --python-backtrace=none --pytorch=none --discard-environment=true --force-overwrite=false --export=sqlite --stats=false --show-output=false --wait=all --output=/home/catid/specialist/experiments/eggroll_es_hpo/profiles/v73d_timeline_lora_es_same_live_qwen36_exact_phase/v73d_timeline_lora_es_same_live_qwen36_exact_phase /usr/bin/env PYTHONPATH=/home/catid/specialist/v73d_sitecustomize:/home/catid/specialist SPECIALIST_V73D_SYSTEMS_ONLY_GUARD=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL NCCL_DEBUG_FILE=/home/catid/specialist/experiments/eggroll_es_hpo/profiles/v73d_timeline_lora_es_same_live_qwen36_exact_phase/nccl_debug.%h.%p.log /home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_lora_es_v71_v72_profile_calibration_v73d.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/qwen36_v73d_exact_phase_profiler.json --preregistration-sha256 d1810d51ecc49615d4067c4b8b151fa9154cb708a6658c02a497210238768c0a --preregistration-content-sha256 512287c5a438cd1a22b099f61fe55e777de86a71b52555b49a47959425deb740 --arm timeline --execute
```

The HBM arm remains unauthorized. An administrator permission change would require a new additive preregistration; it does not silently enable this contract.
