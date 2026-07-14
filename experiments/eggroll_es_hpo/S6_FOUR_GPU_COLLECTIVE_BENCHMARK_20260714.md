# Four-GPU EGGROLL-ES collective benchmark, 2026-07-14

This isolated benchmark tested whether NCCL configuration or bucketing was a
material speed opportunity for the Qwen3.6-35B-A3B middle-late EGGROLL-ES
recipe. It did not read or modify training, evaluation, OOD, or heldout data.

The host has four RTX PRO 6000 Blackwell Max-Q GPUs on `NODE` PCIe topology,
with peer reads and writes enabled and no NVLink. All measured runs used one
rank per GPU, FP32 accumulation, exact rank parity, and no co-tenant.

## Representative update

The runtime update is 23 separate parameter collectives, not one flat tensor.
The benchmark reconstructs the packed runtime sizes from the 35 frozen source
weights: 23 tensors, 142,999,552 total elements, with size-list SHA-256
`7871fe2d020537cebb72053b8c3866b7b1d4296fee284833ac0b18c8c6a5c240`.
Every successful run produced the expected checksum `235520.0` on every rank.

Three 20-second default runs measured `10.5781`, `8.1990`, and `10.8347`
algorithm GiB/s. One forced-Ring run measured `10.8721` GiB/s. The best default
run is within `0.35%` of forced Ring, while between-run default variance is much
larger, so this evidence rejects an NCCL algorithm override for the training
recipe. At the slower default rate, the 572 MB update payload is about 65 ms;
it is not the dominant runtime cost.

Long-duration repetitions strengthen that conclusion. Four 300-second default
runs measured `7.0289`, `8.0087`, `7.6790`, and `8.4345` algorithm GiB/s (mean
`7.7878`). Forced Ring measured `7.5630` over 240 seconds and `7.8773` and
`7.9403` over 300 seconds (mean `7.7935`). The long-run means differ by less
than `0.1%`, much less than run-to-run variance. The final default run completed
4,750 synchronized iterations on every rank with checksum `235520.0`; all four
GPUs were sampled at 100% utilization throughout.

## Flat and bucket controls

- A synchronized flat 142,999,552-element default run measured `5.7163`
  algorithm GiB/s; a same-size 35-bucket run measured `5.7651` GiB/s. The
  roughly `0.85%` difference does not justify bucketing complexity.
- Controlled 20-second flat runs measured `6.5521` GiB/s for default,
  `6.4739` for Tree, and `7.0943` for Ring. These flat-buffer results do not
  represent the actual 23-collective execution and are retained only as
  controls.
- A 16 MiB single-message run measured `6.4775` GiB/s. That rate did not carry
  over when the full payload was split into 35 messages.

An early duration-based version queued asynchronous NCCL work before its final
CUDA synchronization. Its output is excluded from the comparisons above. The
checked-in benchmark synchronizes every iteration and uses a separate Gloo
group for JSON reporting, so forced NCCL algorithms cannot invalidate the
report gather.

## Hardware health controls

Before the collective study, all four GPUs sustained 90 seconds of concurrent
8192-square BF16 matrix multiplication at `260.1`--`271.7` TFLOP/s per GPU and
60 seconds of concurrent tensor-add traffic at `1401.5`--`1405.2` effective
GiB/s per GPU. No device error, nonfinite result, or rank mismatch occurred.

The actionable speed priority remains resident perturbation reuse (V11), which
halves inference-bearing perturb/restore cycles. Collective tuning should be
revisited only with controlled clocks and the exact 23-tensor pattern, as its
own preregistered A/B.
