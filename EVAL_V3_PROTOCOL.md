# S5 document-disjoint evaluation protocol

[`build_eval_v3.py`](build_eval_v3.py) builds the S5 evaluation gate entirely
from frozen local evidence. It never regenerates questions and never changes
the historical `eval_qa_v2.jsonl`, `heldout_docs.jsonl`, `ood_qa.jsonl`, or
`ood_prose.jsonl` artifacts.

The builder extracts every URL-valued provenance field from a candidate
training JSONL and normalizes schemes, hosts, default ports, paths, tracking
parameters, semantic query parameters, fragments, and common YouTube aliases.
Only legacy eval-v2 `heldout` questions whose frozen source document has no
normalized identity in the candidate training set are eligible.

URL isolation is followed by a complete manual review pinned in
[`eval_qa_v3.review.json`](data/eval_qa_v3.review.json). The review's cohort
hash covers every candidate item ID, normalized URL, question, and answer; a
changed candidate cohort therefore cannot silently inherit old decisions.
The current pass reviewed 167 rows from 68 documents, dropped 108, and kept
59. It rejects 29 contextless/trivia items, 25 volatile promotions or resource
claims, 16 non-domain facts, 14 event/recency facts, nine source misreads, nine
unsupported/high-risk safety items, five mixed-language targets, and one
semantic duplicate. Only after this filter are retained documents
hash-partitioned within each source into 41 `validation` rows (22 documents)
and 18 sealed `heldout` rows (11 documents). Questions from one URL can never
cross the boundary, and every source cohort is represented on both sides.

Run a deterministic rebuild, or verify the tracked artifacts byte for byte:

```bash
.venv/bin/python build_eval_v3.py
.venv/bin/python build_eval_v3.py --check
```

The tracked [`eval-v3 report`](data/eval_v3.report.json) pins every input,
review, and output hash; records legacy overlap and manual outcome counts;
proves zero source-identity collisions; and gives source and quality-bucket
counts. The companion artifacts are:

- [`domain QA`](data/eval_qa_v3.jsonl), used for HPO validation and a
  document-separated sealed result;
- [`general-knowledge OOD QA`](data/ood_qa_v3.jsonl), retaining the fixed
  repository-authored questions; and
- [`general-prose OOD`](data/ood_prose_v3.jsonl), retaining the fixed Wikipedia
  texts with deterministic source URLs.

Checkpoint selection may use only the `validation` rows. The `heldout` rows
are opened once after all hyperparameters, seeds, and stopping decisions are
fixed. A candidate also has to pass both OOD non-inferiority checks in the
report: the paired 95% confidence-interval lower bound may not be below
`-0.02` for general-knowledge mean reward or general-prose mean token
log-probability, and OOD exact answers may fall by at most one item. Metrics
are also reported by source and by the keyword-routed
`safety_relevant_grounded` versus
`standard_grounded`, so an aggregate gain cannot hide concentrated domain
damage.

URL disjointness is necessary but cannot detect copied or paraphrased content
served from a different URL. Eval-v3 also inherits eval-v2's automated
question generation and its two verifier gates; it is a cleaner development
and cycle-final benchmark, not a claim of a new human-authored gold standard.
Its small manually retained cohorts require paired confidence intervals and
multi-seed confirmation; point estimates alone are not a sufficient result.
