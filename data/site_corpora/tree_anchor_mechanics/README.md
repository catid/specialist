# TreeConsult tree-anchor mechanics audit

This directory is a deterministic **zero-body rights-deferred artifact**, not
an instructional corpus. TreeConsult's live legal notice requires written
permission for reuse of site text and excerpts. No research PDF body was
requested, stored, copied, summarized, or paraphrased.

The metadata audit inventories the nineteen resources displayed in the site's
“Climbing and rigging” downloads category. It retains the public robots file,
hash-only evidence for the inspected legal and downloads pages, displayed
document provenance, and `HEAD` access results. The ordinary build is entirely
offline:

```sh
python3 data/site_corpora/tree_anchor_mechanics/build.py
python3 -m unittest discover -s data/site_corpora/tree_anchor_mechanics/tests -v
```

`capture_metadata.py` is deliberately separate from the build. Its network
scope is restricted to `GET` requests for robots/legal/download-index metadata
and `HEAD` requests for the nineteen listed resources. It refuses research
document `GET` routes.

`CORPUS.md` is only an access and provenance notice. The exact inventory is
compliance metadata, not a title/URL training layer. A future corpus requires
documented written permission, a fresh capture, body-level technical review,
and document-disjoint split assignment before Markdown chunking or QA creation.
