# USDA trail-rigging corpus report

## Outcome

This package adds one manually curated, non-Q&A training document from the 2024 USDA Forest Service manual *Rigging for Trail Work: Principles, Techniques, and Lessons from the Backcountry*. The corpus concentrates on rope construction and selection, ratings, inspection, slings, blocks, connectors, vector geometry, mechanical advantage, planning, communication, failure zones, and emergency readiness.

The training document contains 6,082 whitespace-delimited words in 350 lines. It is a synthesis, not raw PDF extraction. Every substantive section cites printed manual pages. The official PDF is referenced by its canonical GovInfo location, 22,614,422-byte length, and SHA-256 digest; the large PDF is not committed.

## Why this source was selected

The manual fills genuine gaps in load-path reasoning:

- force has direction as well as magnitude;
- line angles can dominate tension;
- redirects impose a vector resultant on their supports;
- attachment height changes bending moment;
- nominal diameter does not establish rope strength;
- WLL, MBS, condition, and configuration are different facts;
- several components do not necessarily share load equally;
- a safe plan must include inspection, braking, communication, exclusion zones, and failure consequences.

These lessons complement bondage-specific material, but the source's domain is trail material rigging. The corpus therefore preserves mechanics without presenting Forest Service field methods as human-suspension instructions.

## Manual cleanup performed

Automated PDF text extraction was used only for navigation and verification. The final Markdown was written and checked by hand. Cleanup included:

- replacing fragmented extraction with coherent original prose;
- using printed manual pages rather than PDF viewer offsets;
- separating source facts from application-specific cautions;
- attaching assumptions to every retained equation and numeric example;
- rebuilding one small line-angle table from reproducible trigonometry;
- removing vendor names, contact details, product promotion, and URL trivia;
- excluding all photographs, illustrations, figure artwork, and expressive table layouts;
- excluding detailed construction recipes for soil anchors, deadmen, rock anchors, tree or stump anchors, spars, A-frames, tripods, and trail cable systems;
- excluding wire-rope cutting, seizing, clipping, and torque procedures;
- excluding source-specific retirement heuristics that could be mistaken for universal criteria;
- excluding heat-finishing instructions and device-specific operating procedures.

The source has several editorial inconsistencies. The corpus explicitly records that:

- page 128's reciprocal of sine should be cosecant, not cosine;
- page 133's “30 percent” should be an angle in degrees;
- page 149's repeated sine expression conflicts with the shown geometry, which requires sine for the lateral component and cosine for the inline component under that page's convention;
- ambiguous appendix captions on pages 152 and 160 were not copied.

## Rights treatment

The publication is an official USDA Forest Service work distributed through GovInfo. Eligible federal text is treated as presumptively public domain under 17 U.S.C. 105. That basis is intentionally not extended to every component in the PDF.

The acknowledgments name an illustrator and visual or editorial contributors, and the publication contains third-party references, a decorative quotation, vendor descriptions, images, and figures. None of those expressive components is retained. The corpus uses original synthesis plus independently typeset equations and derived values. This is a conservative corpus policy, not a blanket legal opinion.

The retained MODS XML and robots text are compact provenance snapshots. Both official remote text objects lacked a terminal newline when fetched; the committed text snapshots add one terminal LF, so provenance records both the remote raw digest and the normalized local digest. Robots access was checked separately and is not treated as a copyright license.

## Domain-transfer controls

The first section of the training document says directly that this source does not certify:

- bondage rope or a human-suspension system;
- a tree, limb, stump, or rock;
- a ceiling, joist, beam, slab, fastener, or indoor hardpoint;
- a frame, tripod, anchor, connector, or other hardware;
- any numerical design value for supporting a person.

Source-specific values such as a preferred 10:1 trail-rigging factor, approximate sheave ratios, per-sheave friction estimates, generic shackle side-load examples, or dynamic-load heuristics remain labeled as contextual examples. Manufacturer instructions, applicable standards, and qualified structural or rigging evaluation control.

The corpus offers no procedure for constructing an improvised human-support point. Tree and indoor-hardpoint paragraphs teach uncertainty and professional evaluation rather than selection recipes.

## Split and anti-trivia controls

The artifact is direct-training Markdown rather than generated Q&A. It contains no chat-control tokens, answer wrappers, canonical-URL questions, title-recall prompts, or protected examples. The canonical document is the unit of split assignment: place all chunks and any later derived Q&A from this source in one split.

No validation, OOD, shadow, sealed-holdout, or protected-QA files were inspected or used. Tests enforce the non-Q&A surface, source-level split rule, warnings, arithmetic, disposition coverage, snapshot hashes, and absence of embedded images.

## Package contents

- `CORPUS.md`: 6,082-word manually synthesized training document.
- `dispositions.jsonl`: 25 explicit include, narrow-include, and exclude decisions.
- `manifest.json`: training role, source identity, rights, scope, hashes, and split rules.
- `source_snapshot/provenance.json`: retrieval, review, rights, safety, and remote-object record.
- `source_snapshot/mods.xml`: official GovInfo MODS snapshot, terminal-LF normalized.
- `source_snapshot/robots.txt`: GovInfo robots snapshot, terminal-LF normalized.
- `tests/test_corpus.py`: deterministic integrity, quality, safety, and math checks.

## Remaining limitations

This package does not independently validate the Forest Service manual's empirical values. It does not replace current manufacturer documentation or standards. It does not contain source figures, so any lesson that depended on an ambiguous diagram was excluded or reduced to a reproducible equation. It is one source document and should not dominate sampling merely because it is long.

Later QA, if any, should ask about mechanics, assumptions, inspection, and limits—not the publication title, authors, page numbers, identifiers, or URLs. Any QA derived from this source must inherit the same source-document split assignment and domain-transfer warnings.
