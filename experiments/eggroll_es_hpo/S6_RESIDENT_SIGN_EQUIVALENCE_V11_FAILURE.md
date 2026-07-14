# S6 resident-sign exact-equivalence v11 failure

V11 failed closed before estimating a coefficient plan or applying any update.
The four Qwen3.6-35B-A3B TP=1 actors loaded successfully, reproduced the
alpha-zero benchmark, and then raised:

`RuntimeError: v11 captured D43 manifest changed`

The main process exited `1`; cleanup removed all four actors and CUDA contexts.
The failure journal has file SHA-256
`2c9680fc55b6f90ef306ecf7f7493c7f448f48c36790af9992ffe4236ed19924`,
status `failed`, `coefficient_plan_estimated: false`, and target alpha exactly
zero. The frozen train Arrow identity remained
`6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6`.
No heldout data was opened.

## Root cause

The V10 loader correctly binds raw question/answer manifests before prompt
formatting. V11 then captured the trainer inputs after chat-template formatting
but incorrectly compared those prompts with the raw-question hashes. The two
surfaces are deterministic and intentionally different:

- Raw D43: `b864cfcc4ebcd987d8091f1067f631366c128d63d09fb7160a09561d10063a0f`.
- Templated D43: `54f53464e479fa9dd0c80263f0e424a3d225681c1d8f15554b171f6d5b40c637`.
- Raw D44: `3574ff126f727a262957f34ab83fbefce6754ae9e4be790f810f42656e692bc2`.
- Templated D44: `44cc0ba38c7b2c685a2c44699be9f6dd6313c1391765e13c046812f06e280c23`.

The V11 guard therefore worked as designed, but guarded the wrong identity at
that boundary. A fresh V11b retry must retain the raw loader binding, add the
templated trainer binding, use a new run directory, and include a regression
test proving the two identity domains cannot be confused. The failed directory
must not be resumed or overwritten.

## Frozen failed implementation

- Worker: `1bda6397fd5f1478d47de929babe509e6a3475c882ffa642d5252a3f283cd8d5`.
- Trainer: `c3c4cde40408eeb91dd51cd48c17294cdc4389ee51359a0c560fa1ff94f4fec9`.
- Driver: `78c25003485045e42f7dc9481be6d45d710c05a4a97ccfbab58c38150bde28f0`.
- Reporter: `b1b3a50cd6383d44b29ec9d2dcf34ab9857ad7f2d73ef5feb9f2a908b8fbc1f8`.
- Focused tests: `1dc150b33c81c691be14d1117ff75c2e04c931ceb340c7a37a7a3521dad4f047`.
- Protocol: `9d58455bd3321fa73becf6ce819b7182138a823fb7399de5d82b6ae3cd02764e`.
