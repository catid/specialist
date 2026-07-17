# Qwen3.6 V73E exact-phase profiler: CPU-only sealing evidence

Status: prospectively sealed; no V73E model, Ray, CUDA, GPU, dataset, semantic evaluation, HPO, checkpoint, or promotion run was performed while producing this evidence.

## Immutable identities

- Preregistration file SHA-256: `cc31616b59c53da930e6d672f0f6913332b3db5c395e2568317b51366bc269c2`
- Preregistration content SHA-256: `e898c1984cb87b3ed83425b851cf972d3969f6d070df551fd1a0cd97b6ab94d2`
- External systems-only closure content SHA-256: `5a658da53bac17e8020b26ab9dd9dd596d816bf6f9e898f55b637399f437c537`
- Runtime closure module count: 101
- Opaque historical reference bindings: 9 (nonzero; does not authorize resolution or access)

## Additive repair and actor bootstrap

- V73D attempt 1 remains immutable and failed closed during setup because its inherited adapter verifier tried to open a quarantined legacy adapter before Ray, model loading, or GPU work.
- Immutable V73D run/profile failure file SHA-256: `08c0a3e2ca3832486e6b94bfd4ad0529c8420d3537574c7f2dda6ad4a8ed3886` / `025a38adce724d931a644dd090f6eb5b347f0d1ebd7bcc500a149af67f22bd50`.
- V73E controller mechanism: `controller_sitecustomize`; actor mechanism: `ray_actor_worker_extension_pre_parent_import`. They are separately attributed.
- The Ray job runtime environment injects the exact guard hash before `ray.init`, while actor-specific runtime environments remain merged.
- The actor extension refuses V72/V71/V41A or the three historical reference modules if any was imported before its guard install.
- Boundary registry content SHA-256: `5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7`; required successful protected open/resolve/metadata/enumeration operations: 0.

## GPU-time accounting

- Reserved wall GPU-seconds are reported independently from directly measured model-resident and useful GPU-seconds.
- The legacy allocation/residency estimate is diagnostic and is not accepted as either direct residency or useful compute evidence.
- Profiled event time is not relabeled as unprofiled useful time; promotion-charged GPU-seconds remain 0.

## Measurement contract

- Exact NVTX phase count: 22 in domain `eggroll_es_v73e_phase`.
- Timeline arm: unprivileged CUDA/NVTX/legacy-NCCL/kernel/memcpy/allocation trace, launch-authorized only after fresh-path, exact toolchain, four-GPU identity, idle, and no-compute-process gates.
- Timeline command-template SHA-256: `7bbaf5635a799b1fca4a9c238fb4e0cce07b5e084fc0a616a85c53ae0f87af0d`
- HBM/DRAM arm: blocked before directory creation or subprocess; `RmProfilingAdminOnly=1` and `ERR_NVGPUCTRPERM`.
- No bandwidth bytes are inferred from utilization percentages; no NCCL transport/path is inferred from topology capability.

## Exact LoRA geometry and host topology

- FP32 tensors/elements: 70 / 4,528,128.
- Logical collective payload per rank: 18,112,512 B; nominal ring schedule 27,168,768 B/rank (not measured physical-link bytes).
- Four physical GPUs: all cross-pairs report NODE, peer read/write capability OK, NVLink capability NS, and NUMA affinity 0.
- Topology matrix SHA-256: `57127e112509330246aafbc4c2be974cfd9643248f88878fb356dbf1f0f0992a`.

## Exact future timeline launch (not executed)

```bash
/home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_qwen36_v73e_exact_phase_profiler.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/qwen36_v73e_exact_phase_profiler.json --preregistration-sha256 cc31616b59c53da930e6d672f0f6913332b3db5c395e2568317b51366bc269c2 --preregistration-content-sha256 e898c1984cb87b3ed83425b851cf972d3969f6d070df551fd1a0cd97b6ab94d2 --arm timeline --execute
```

The launcher re-hashes the sealed `.bashrc` and 11 other timeline runtime inputs before any output directory or profiler subprocess. Its exact inner Nsight command is:

```bash
/usr/bin/bash --noprofile --norc -i -c 'source "$HOME/.bashrc" >/dev/null 2>&1 && exec "$@"' v73e-nsys-launch /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,nccl --nccl-trace=none --cuda-trace-scope=process-tree --cuda-memory-usage=true --cuda-event-trace=false --cuda-graph-trace=graph --sample=none --cpuctxsw=none --backtrace=none --python-sampling=false --python-backtrace=none --pytorch=none --discard-environment=true --force-overwrite=false --export=sqlite --stats=false --show-output=false --wait=all --output=/home/catid/specialist/experiments/eggroll_es_hpo/profiles/v73e_timeline_lora_es_content_free_qwen36_exact_phase/v73e_timeline_lora_es_content_free_qwen36_exact_phase /usr/bin/env PYTHONPATH=/home/catid/specialist/v73e_sitecustomize:/home/catid/specialist SPECIALIST_V73E_SYSTEMS_ONLY_GUARD=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL NCCL_DEBUG_FILE=/home/catid/specialist/experiments/eggroll_es_hpo/profiles/v73e_timeline_lora_es_content_free_qwen36_exact_phase/nccl_debug.%h.%p.log /home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_lora_es_v71_v72_profile_calibration_v73e.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/qwen36_v73e_exact_phase_profiler.json --preregistration-sha256 cc31616b59c53da930e6d672f0f6913332b3db5c395e2568317b51366bc269c2 --preregistration-content-sha256 e898c1984cb87b3ed83425b851cf972d3969f6d070df551fd1a0cd97b6ab94d2 --arm timeline --execute
```

The HBM arm remains unauthorized. An administrator permission change would require a new additive preregistration; it does not silently enable this contract.
