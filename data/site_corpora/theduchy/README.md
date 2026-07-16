# The Duchy policy-exclusion artifact

This directory is intentionally **not** an instructional corpus. The live robots response captured on 2026-07-16 declares `ai-train=no` and disallows GPTBot across the site. Accordingly, canonical pages and assets were not retrieved, and `content_records.jsonl` is empty.

`source_snapshot/` contains only the exact robots response, sitemap XML, and capture provenance. `inventory.jsonl` assigns every canonical URL discovered through those sitemaps the disposition `robots_and_content_signal_ai_training_disallowed`. `CORPUS.md` is a short access/provenance notice; it contains no substantive site knowledge.

The ordinary build is offline and deterministic:

```sh
python3 data/site_corpora/theduchy/build.py
python3 -m unittest discover -s data/site_corpora/theduchy/tests -v
```

`capture_metadata.py` is not part of the ordinary build. It is a deliberately narrow acquisition utility that accepts only `robots.txt`, `sitemap_index.xml`, and same-domain `*-sitemap.xml` metadata paths. It rejects canonical content routes.
