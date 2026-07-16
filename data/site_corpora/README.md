# Canonical rope source corpora

This directory contains the non-Q&A layer of the specialist dataset. Each
accepted site or media resource gets a dedicated directory whose canonical
Markdown is suitable for direct language-model training. The Markdown is a
first-class dataset artifact, not temporary input to a question generator.

The source contract and approved knowledge taxonomy are defined in
[`sources/site_corpus_queue_v1.json`](../../sources/site_corpus_queue_v1.json).
Source discovery is governed separately by
[`sources/site_discovery_queue_v1.json`](../../sources/site_discovery_queue_v1.json).

## Dataset layers

1. **Canonical Markdown source corpus.** Clean, dense, page-attributed prose
   preserves substantive facts, steps, cautions, alternatives, and limitations
   while removing navigation, SEO, cookie/footer, commerce, and repeated UI.
2. **Derived manual Q&A.** Natural learner questions are written only after the
   source corpus is sealed. Every answer must be checked against and linked to
   its source record. URL, title, sitemap, archive, and navigation recall is not
   a valid learning objective.

Both layers are retained for training. Q&A does not replace the Markdown, and
the Markdown is not converted wholesale into automatically generated Q&A.

## Completed corpora

- [`rope_topia/rope_topia.md`](rope_topia/rope_topia.md)
- [`shibari_atlas/shibari_atlas.md`](shibari_atlas/shibari_atlas.md)
- [`rope365/rope365.md`](rope365/rope365.md)
- [`crash_restraint/crash_restraint.md`](crash_restraint/crash_restraint.md)

Each directory also contains a manifest, report, provenance or source snapshot,
and deterministic coverage tests. Corpus completeness means complete coverage
of substantive accessible text, not a verbatim website mirror and not an
attempt to reconstruct visual-only or gated instructions.

## Completed policy exclusions

Some requested sites can be inventoried but cannot supply direct-training
text under their current access terms. These artifacts retain the policy
snapshot, sitemap coverage, and an explicit zero-content disposition so a
later worker cannot silently treat public readability as training permission:

- [`theduchy/CORPUS.md`](theduchy/CORPUS.md) — `direct_training_ready=false`;
  the live robots content signal disallows AI training.
- [`shibari_study/CORPUS.md`](shibari_study/CORPUS.md) —
  `direct_training_ready=false`; the live terms prohibit the requested
  collection, copying, storage, and database compilation.

Neither exclusion notice, its URL inventory, nor URL/title metadata is a
substitute training corpus. If a site's policy later changes, it requires a
fresh dated audit and a new corpus build rather than reinterpreting these
snapshots.

## Training-boundary requirements

Before a corpus is promoted into a model-training mixture:

- assign train/validation membership at the source-document level, using the
  canonical page URL and document hash before deriving Q&A;
- keep every Markdown section and every Q&A row derived from the same page in
  the same split;
- exclude all documents committed to validation, OOD, shadow, or sealed
  holdout membership from every training layer;
- chunk long Markdown only at heading/paragraph boundaries and never join text
  from different source pages into one example;
- retain source, canonical URL, document hash, corpus hash, taxonomy categories,
  and derivation lineage in each training record;
- preregister Markdown-versus-Q&A sampling weights and source/category caps so
  a large site cannot overwhelm smaller high-value sources;
- require unchanged document-disjoint QA and prose OOD gates before promotion;
  the terminal holdout remains unopened until a candidate is fixed by those
  prior gates.

Canonical Markdown may be committed while an experiment is running, but it is
not silently inserted into that experiment's sealed dataset.
