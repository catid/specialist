# Agent Instructions

## Web searches

- Use the Exa MCP for web searches.

## Protected evaluation data

- Ordinary unit, regression, and compatibility tests must use synthetic
  fixtures. Never run a broad test selection that can resolve a real protected
  holdout or terminal source path.
- Real terminal evaluation is allowed only through its explicit one-use,
  content-addressed claim runner after the exact command and output boundary
  are sealed. Do not print or persist protected text, URLs, answers, generated
  completions, or per-item metrics.
- Treat every unexpected protected-source read as irreversible. Record and
  quarantine the touched source; never reset an access counter or relabel the
  source as untouched.
- Before running evaluation-related tests, inspect their fixture and loader
  paths. If a real source can be reached, stop and replace that path with a
  synthetic fixture before continuing.
- Apply protected, holdout, OOD, terminal, incident, and manual-review path
  exclusions in the file-selection command itself. A downstream `rg` filter
  only hides output; it does not prevent the upstream process from reading
  excluded files.
- Prefer explicitly named safe manifests over repository-wide content searches
  for dataset inventories. Do not use broad `rg` globs across the repository
  when the same facts are available from a registry or build report.

## EGGROLL-ES program status

- The user halted the EGGROLL-ES HPO and training program on 2026-07-17 after
  reviewing its runtime and quality results.
- Do not launch or resume EGGROLL-ES, V73 profiling, or related GPU backfill
  benchmarks unless the user explicitly reopens this direction. Earlier
  instructions to resume work or keep GPUs busy do not override this halt.
- Preserve existing experiment artifacts, unfinished V73I files, and every
  protected-data quarantine. Do not delete them or present unfinished work as
  completed.
- The replacement training direction is the supervised low-regression protocol
  in `plan.md`. EGGROLL-ES remains abandoned even while that new work proceeds.

## Active low-regression training policy

- Treat `plan.md` as the canonical training protocol for Qwen3.6-35B-A3B.
- Use BF16, MLP-only expert-aware LoRA, frozen routing and attention, supervised
  knowledge transfer, general-behavior replay, and verifier-filtered refinement.
- Do not use RL to inject the domain corpus. GRPO is optional and gated on an
  objective verifier after SFT and offline refinement have passed regression
  gates.
- Complete the data inventory, source-disjoint split, replay construction,
  baseline evaluation, exact adapter-scope assertions, and memory smoke test
  before launching the four-GPU pilot sweep.
- A request to plan, audit, or create Beads tasks does not by itself authorize a
  live training run. Keep GPUs idle until an execution task is explicitly
  started.

## Website Markdown training data

- Treat `data/site_corpora/registry/site_corpus_registry_v1.json` as the
  complete inventory for raw-Markdown training. A corpus is not incorporated
  merely because its Markdown is committed or because Q&A was derived from it.
- After any eligible corpus or registry change, rebuild
  `data/site_corpora/training/site_markdown_cpt_v1` with
  `build_site_markdown_training_dataset_v1.py`, then run the builder with
  `--check`. Do not declare a dataset refresh complete while an eligible
  registry artifact lacks a training chunk.
- Train Markdown only through the raw causal-LM/CPT path. Never disguise a
  Markdown document as an assistant answer or feed it to the QA-only
  EGGROLL-ES reward collator.
- Preserve complete document content, source-document split groups,
  attribution, rights status, safety-transfer flags, and chunk lineage. Never
  pack two source documents into one example.
- Keep rights-review and policy-blocked corpora in the snapshot's explicit
  exclusion ledger. Do not silently omit them or silently override their gate.
- When the user explicitly authorizes named rights-review corpora for training,
  preserve the original license/rights metadata and add a separate
  content-addressed authorization override recording the scope, rationale,
  date, and attribution requirements. Never rewrite an unverified license as
  verified, and never let an override for named corpora implicitly authorize
  other blocked material.
- A materialized snapshot is not evaluation-disjointness authority. Require a
  fresh opaque source-disjoint contract extension before launching adaptation
  on newly added corpus bytes.

## Claude CLI

- Claude CLI requests can take a long time. Inspect the session more carefully
  than relying on a short timeout such as 300 seconds.
- Retry a Claude CLI request if a recent attempt failed because it timed out.
- Before a review, source `~/.bashrc` and verify `claude auth status`. When OAuth
  billing is requested, unset API-key, auth-token, base-URL, and third-party
  provider overrides for the Claude process while retaining the stored
  first-party `claude.ai` OAuth session.
- Pass long review prompts through stdin. Do not put a positional prompt after
  `--allowed-tools`: that option is variadic and can consume the prompt as
  another tool name, causing a local parse failure before any model request.
- Specify the requested model and effort explicitly (for example,
  `--model fable --effort max`) and never silently substitute another model.
- Record the Claude session ID, keep polling the same live session through long
  quiet periods, and inspect Claude's edits plus focused tests before committing.

## Forwarded SSH agent

- Treat the forwarded SSH agent socket as ephemeral. The client SSH connection
  cycles frequently, so a previously exported `SSH_AUTH_SOCK` is usually stale.
- Before any authenticated Git operation, source `~/.bashrc` again, confirm
  that `SSH_AUTH_SOCK` names an existing socket, and verify the currently
  forwarded identity. Never assume a cached socket path still works.
- If the re-sourced path is stale, locate the newest live user-owned
  `/tmp/ssh-*/agent.*` socket, verify its identity with `ssh-add -l`, and scope
  that socket to the authenticated command. Do not keep retrying the stale path.
