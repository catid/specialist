# Direct-training Markdown registry

`site_corpus_registry_v1.json` is the deterministic inventory for the
first-class `canonical_markdown_source_corpus` dataset layer. It inventories
every Git-tracked corpus manifest that declares a direct-training Markdown
artifact. It does not merge, chunk, sample, or assign any document to a split.

Each record binds:

- the Markdown path, byte length, SHA-256, Unicode word count, whitespace word
  count, and exact Qwen3.6-35B-A3B token count;
- its manifest path, manifest SHA-256, and manifest schema/version;
- a stable source-document identity rather than a filename-only identity;
- the source's rights basis, limitations, and promotion gate;
- explicit safety and cross-domain transfer flags; and
- one required source-document split group inherited by the complete Markdown,
  every later chunk, and every derived QA or other descendant.

Omission is a hard error. `build_registry_v1.py --check` discovers all tracked
corpus manifests and fails if a direct-training manifest or non-auxiliary
Markdown artifact is absent from the reviewed config. It also fails on any
artifact, manifest, tokenizer, count, or registry drift.

## Files

- `source_registry_config_v1.json` is the manually reviewed normalization
  layer for source identity, rights, and safety/transfer semantics that legacy
  manifests do not express uniformly.
- `build_registry_v1.py` deterministically discovers and hashes tracked
  artifacts, counts words and tokens, validates completeness, and renders the
  registry to standard output. It has no merge or snapshot-writing path.
- `site_corpus_registry_v1.json` is the sealed machine-readable registry.
- `test_site_corpus_registry_v1.py` enforces reproducibility, complete
  inventory, rights metadata, tokenizer identity, hash/count integrity,
  exclusion of policy notices, and source-document split inheritance.

Run the sealed check from the repository root:

```bash
.venv/bin/python data/site_corpora/registry/build_registry_v1.py --check
.venv/bin/python -m unittest \
  data.site_corpora.registry.test_site_corpus_registry_v1
```

To review a proposed regeneration without changing the registry:

```bash
.venv/bin/python data/site_corpora/registry/build_registry_v1.py
```

## Rights boundary

The registry reports the rights basis; it does not invent one. Explicit CC BY
articles and eligible federal text carry their recorded terms and limitations.
Legacy public-web corpora whose manifests lack an explicit reuse basis remain
visible but carry `rights_review_required_before_new_snapshot`. Public access,
robots permission, an archive capture, paraphrase, or an earlier
`direct_training_ready` label is not itself a copyright license or legal
opinion.

## Split and sampling boundary

Snapshot builders must assign the registry's `group_id` before any Markdown
chunking or QA derivation and must keep all descendants in that one split.
They must reject cross-split reuse and reject any snapshot that silently omits
a registered eligible artifact. Token volume alone cannot set sampling weight:
source/category caps must be preregistered so a very large site digest cannot
overwhelm smaller clinical, technical, or practitioner sources.
