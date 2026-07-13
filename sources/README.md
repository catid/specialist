# Curated rope-resource sources

`rope_resources_v1.json` is the owner-curated directory and collection policy
for the 23 supplied resources. It preserves all 23 supplied URLs verbatim as
`supplied_url`, stores a normalized `canonical_url`, and defines a bounded,
relevance-first collection plan. General commerce sites are never mirrored;
only named product/specification pages are in scope.

`hip_harness_playlist_v1.json` is a discovery inventory, not endorsed training
content. It records the playlist's 105 titled items and four hidden/unavailable
IDs. Each video still needs its own provenance, transcript, factual review, and
safety review before it can supply instructional QA.

`manual_facts/*.jsonl` contains small, independently assigned, manually read
fact packets. Each line has a self-contained question, evidence-backed answer,
short official-page evidence, evidence URL, reviewer, date, resource ID, and
claim type. Vendor/manufacturer claims stay attributed; prices, stock,
schedules, ratings, unsupported load/safety claims, and medical marketing are
excluded.
The final Group A, B, and C packets contain 8, 18, and 12 facts respectively,
for 38 facts total.

The answer policy is exact and evidence-backed: an answer is normally a
contiguous source phrase, allowing only case, whitespace, punctuation, and
quotation normalization. Three reviewed synthesis exceptions combine facts
that appear explicitly in one cited evidence excerpt: joining Rope365's three
building-block bullets into one list, expanding Amazon's compound “strap and
ring cutter” wording into the two named cutters, and consolidating X-POLE's two
height values with the millimetre, inch, and foot units printed in its manual.
No exception may introduce a fact or value absent from its cited evidence.

## Collection boundaries

- The Duchy is recommendation-only because its robots policy declares
  `ai-train=no` and `use=reference`.
- Crash Restraint is recommendation-only because its terms prohibit automated
  and manual scraping/copying.
- ROPECRAFT remained recommendation-only because live DNS/robots verification
  failed and its visible event information was stale during review.
- My Nawashi remained recommendation-only after Etsy returned access
  challenges/rate limits.
- Paid/member lessons and purchased videos are not copied. Public storefront or
  catalog descriptions may be retained as resource metadata.
- YouTube collection is anonymous and never supplies cookies or credentials.
  Age gates, unavailable captions, and client-attestation failures are recorded
  as blockers, not bypassed.
- Event dates, meeting schedules, prices, stock, shipping, and availability are
  volatile metadata, not timeless training answers.

The live policy-respecting crawl fetched 165 relevant public documents,
including 136 English Rope 365 curriculum pages. Raw pages and per-page
coverage remain under ignored `data/raw/`; the compact tracked report
`data/rope_resources_coverage_v1.json` records all 23 outcomes and reason
counts. There were no network/extraction failures: nine resources produced
documents, eleven were policy/access blocked, and three rendered too little
static text for safe ingestion.

## Rebuild

```bash
.venv/bin/python collect_resource_corpus.py --disable-youtube
.venv/bin/python build_resource_coverage_report.py
.venv/bin/python build_resource_qa.py
.venv/bin/python build_resource_facts.py
.venv/bin/python build_curated_qa.py
```

The collector obeys robots.txt, `Content-Signal`, crawl-delay/request-rate,
same-site redirects, explicit page caps, access challenges, and manifest URL
filters. It performs no HTML link traversal. The builders fail closed on stale
manifest digests, incomplete URL coverage, malformed QA, protocol tokens,
duplicates, weak evidence support, time-sensitive questions, and evaluation
leakage.
