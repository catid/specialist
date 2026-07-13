# Manual QA curation

The large generated QA corpus is a candidate pool, not training data.  Every
source document admitted to the manual dataset must be read by a reviewer in a
small, source-grouped packet.  Reviewers make explicit decisions; no model or
script writes the questions or answers.

## Review rules

For each packet, read the complete source before judging its candidate Q&A.

- Keep only facts that the source states directly.
- Make every question self-contained.  Avoid phrases such as "the article",
  "the context", or "described as" when the subject is not named.
- Ask one unambiguous question about one useful fact.  Do not retain multiple
  paraphrases of the same fact.
- Use a short extractive answer whose wording occurs in the quoted evidence.
- Copy the smallest source passage that proves the answer into `evidence`.
- Search both evaluation JSONL files for the answer, spelling variants, aliases,
  and the underlying fact before keeping or adding it.  The lexical leakage
  gate is useful but does not replace this manual semantic check.
- Drop promotions, prices, schedules, stale availability claims, trivia with
  no durable value, and questions whose premise overstates the source.
- Treat safety and medical claims conservatively.  Do not turn a personal
  account or commercial blog into general medical advice.
- Add important facts missed by the generator when they meet the same rules.

Each line in a review file is one JSON decision:

```json
{"action":"edit","candidate_fact_ids":["fact-a","fact-b"],"question":"Why may antibacterial wipes fail to reach all parts of jute rope?","answer":"because of gaps between strands and yarns","evidence":"UV and antibac wipes almost certainly won’t reach every spot of your ropes, because of the gaps between strands (and even the gaps between yarns within the strands).","reason":"merged duplicate paraphrases and named the subject","source":"anatomiestudio","url":"https://example.test/source","reviewer":"reviewer-name","batch":"batch-001"}
```

Actions have these meanings:

- `keep`: retain one candidate unchanged; include its one fact ID plus evidence.
- `edit`: replace or merge one or more candidates; include every consumed ID.
- `drop`: reject one or more candidates; include their IDs and a specific reason.
- `add`: add a missed source fact; `candidate_fact_ids` must be empty.

Every candidate in every reviewed source document must be consumed exactly
once.  `build_manual_qa.py` checks coverage, source evidence, extraction,
leakage, duplication, serialization, and provenance before emitting training
records.

## Workflow

1. Rebuild the candidate pool with `build_leakfree_qa.py`.
2. Use `prepare_manual_qa_review.py` with a short URL list.  Its packets contain
   one complete source document apiece.
3. Give different packets to different reviewers and write JSONL decisions.
4. Run `build_manual_qa.py` over all completed review files.
5. Train only on the validated `qa_manual` output, never directly on packets or
   the generated candidate pool.
