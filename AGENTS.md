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
