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
