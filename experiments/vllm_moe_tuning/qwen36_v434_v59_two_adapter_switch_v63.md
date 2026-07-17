# Qwen3.6 V434/V59 two-adapter switching probe (V63)

## Scope and safety

This is a data-free runtime feasibility probe. It generates 68 synthetic
prompts and persists only token counts and SHA-256 receipts. It opens no
training, validation, OOD, shadow, or terminal-holdout rows; persists no
prompt, generated text, or token IDs; and performs no adapter update or HPO.
These results establish runtime feasibility and noise behavior only. They are
not evidence that either adapter is better on the specialist task.

The common runtime is Qwen3.6-35B-A3B, vLLM 0.25.0, BF16, TP1,
synchronous FCFS scheduling, `VLLM_BATCH_INVARIANT=0`, 68 concurrent requests,
greedy temperature-zero decoding, and 64 generated tokens. One GPU-resident
LoRA slot and two CPU-resident LoRA slots are used. Each actor discards one
full-batch warmup per adapter, then executes
`R,C,C,R,R,C,C,R`, giving four recorded calls per adapter and balanced call
positions.

The bound adapters are:

- V434 weights: `7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a`
- V59 weights: `c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8`
- Shared adapter-config identity: `b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5`

Probe implementation SHA-256:
`115774a63f54480fa4796f24f5b47a82fda1c2a761db4cf3ae0b6b83e85165d6`.
Test implementation SHA-256:
`b5d044f86e5de9ef8f0a071ef190c1e11e495c2797c01e54e323c1e63b3e1f1c`.
The probe and its base receipt helpers pass 12 focused tests.

## Four-actor results

“Changed rows” counts synthetic rows whose token receipt differed across the
four calls for the same adapter. “Between-state rows” compares the first
recorded receipt from each adapter. The swapped controls exchange the adapter
paths while leaving labels and call positions unchanged.

| Runtime and path order | V434 changed rows by actor | V59 changed rows by actor | Between-state differing rows by actor | Mean post-load seconds |
|---|---:|---:|---:|---:|
| Eager, V434 then V59 | 1, 1, 2, 1 | 0, 0, 0, 0 | 6, 6, 5, 6 | 45.1571 |
| Eager, paths swapped | 0, 1, 1, 1 | 0, 0, 0, 0 | 6, 5, 5, 5 | 45.1893 |
| CUDA graphs, V434 then V59 | 1, 1, 1, 1 | 0, 0, 0, 0 | 4, 3, 4, 4 | 22.7320 |
| CUDA graphs, paths swapped | 1, 1, 1, 0 | 0, 0, 0, 0 | 4, 3, 4, 4 | 22.7091 |

V59 was internally repeatable for every actor in every runtime and path-order
control. V434 showed small receipt drift in 15 of the 16 actor/control cells.
The swapped controls therefore localize the drift to the V434 adapter state,
not to the `reference` label, call position, or which adapter was loaded first.
Both adapters remained observably distinct after repeated switching, so there
is no evidence of stale-adapter reuse or state aliasing.

Graph mode reduced mean post-load time by about 49.7%, but it produced a
different numerical regime and only 3--4 separating synthetic rows rather than
the eager regime's 5--6. V63 therefore retains eager execution: V62B calibrated
that exact runtime, and graph mode would require a separately sealed null
calibration before it could support model-selection claims.

Every final actor reported successful vLLM engine shutdown. Engine shutdown
had already removed the torch process group, so no residual explicit process
group destruction was needed. A post-run `nvidia-smi` check showed all four
GPUs back at 4 MiB and no compute processes.

## Receipt file identities

Eager V434-then-V59, actors 0 through 3:

- `49974739372f3d1d579959b7244749ee486ac661b13cef73168c313d9143d049`
- `cf7d833908ef0e5b201db0ad1693071e5519f3fa2ecc96457d8240e7bd39053f`
- `4d37e54c253ba2d4b38aa7334168e0ff1418de32e4a1db648bcbff9903643bb9`
- `2485c706d03d461d6ca9db85bebbe0649a7259ecf64c0f5fb8a928a9d1c5ef48`

Eager swapped paths, actors 0 through 3:

- `4ae6c3561d9e8eec55f7777f128640348d0d6a95077fdfe60c84905fb5d63924`
- `5a86a1f597df028d046ac80146add32c797ac315c040e7ff444f8f10821e7d0a`
- `edfbab1cce46ce2681bfaedd7e92a344552cc253028405a076a7845b2a34f79e`
- `19315f32a66ee137eb4208a3735442b0cf7a6d0f605bc3bad0c4599f4c419757`

Graph V434-then-V59, actors 0 through 3:

- `e7f19dd07a68e76c0aae16bbe018cbb81ffcfe0c584e0623071598954fe3d449`
- `9b545faabfc93c463465ec13bf335867883e807bcdbb24fa55cffd3b9ba89126`
- `db7be72e25f91cc80c1c9670fe91be9c213824556c5cd6108bd8082c4bf0fc5f`
- `7ec3b49bee15c911dcffd94880909c9060e6e93258d4cc7a8247481692020da6`

Graph swapped paths, actors 0 through 3:

- `67056df05e234b18070996b5d5427768f0a1d8f77c1e5d0b17781b270d5aa047`
- `2d0d3d4e5d92552924277c170a3096f9470d17fa40f60dfb90e6f4864cc1a94d`
- `660be06483f918c8652dc1fff635d8d6660b7a247963ef7bea7c54e8d2e707f6`
- `20742bc9ed0040d22b00b7f8496775bae4f92cb663c0f3833eb45ef3290076ae`

## Decision

Standard vLLM LoRA requests can switch V434 and V59 safely inside one TP1
engine with `max_loras=1` and `max_cpu_loras=2`. The sealed V63 confirmation
can therefore compare the two adapters in each actor without duplicate model
loads. It must retain the V62B eager identity, fixed counterbalancing, all
preregistered replicas, conflict-unit bootstrap, actor leave-one-out penalty,
and exact adapter-hash verification. This probe does not authorize HPO,
adapter updates, protected-data access, or any threshold change.
