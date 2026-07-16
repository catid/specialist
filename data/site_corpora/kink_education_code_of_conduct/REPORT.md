# Kink Education Code of Conduct v. 1 corpus report

## Outcome

This package adds one 4,129-whitespace-word, non-Q&A Markdown training document derived manually from the Kink Education Code of Conduct, version 1. The direct document preserves the v. 1 KECC Collective's educator/producer/discussion structure and its substantive guidance on consent modeling, student demonstrations, competence disclosure, risk autonomy, inclusion, accountability, boundaries, vetting, incident disclosure, and privacy.

The PDF is visibly dated 25 April 2019 and calls version 1 incomplete and subject to evolution. The package therefore labels the material as a historical versioned ethics framework. It does not describe the code as current consensus, law, professional certification, central enforcement, or proof that any person, venue, class, demonstration, relationship, or activity is safe.

The package contains 50 manual disposition records: 38 included or narrowly included decisions and 12 exclusions. `CORPUS.md` contains 58 claim-level source citations.

## Source ledger and integrity

Five allowlisted records were retrieved successfully on 16 July 2026. The three KECC records were used for substantive review; the two Creative Commons records were used only for licence context.

| Source ID | Role | Bytes | SHA-256 | Last-Modified |
|---|---|---:|---|---|
| `KECC-FULL` | Primary full-version HTML | 38,172 | `903d1c07dfada79ceee507ad13cbd5d41aa7c8c722228b038122221be93ca2cc` | 21 March 2025 18:08:47 GMT |
| `KECC-PDF` | Version/date/FAQ/adoption source and full-text crosscheck | 174,888 | `2429e5806ecf2427828676b9aaa8e95c5ac247a8406c6d3b4dd6e1d51b973657` | 25 April 2019 22:14:50 GMT |
| `KECC-HOME` | Collective authorship and first-version context | 3,945 | `9e6e9b1f281b7042ce5a8b3e41f398ff82d4d03be319f8f3d10ef14ebb4287a8` | 21 March 2025 18:08:47 GMT |
| `CC-DEED` | Licence metadata only | 35,744 | `17de6b7071e8f4816b103fb45aff75f71c94821f303534fc3e21cb68bd0f7148` | 9 June 2026 14:26:11 GMT |
| `CC-LEGAL` | Licence legal metadata only | 51,859 | `a7dbad04e9a44a69a06d2ea5f20cceccb163091550591ed41ac610f112789246` | 9 June 2026 14:26:11 GMT |

The Creative Commons URL linked by the KECC pages redirected from HTTP to HTTPS. The resolved deed directly linked the English legal-code route. No other Creative Commons page was accessed.

The total retrieval was 304,608 bytes: 217,005 KECC bytes and 87,603 licence-metadata bytes. Exact requested and resolved URLs, access timestamps, response metadata, hashes, deterministic normalization metrics, roles, and treatments are recorded in `sources.jsonl`.

Raw HTML, the raw PDF, and audit extractions are not retained. The sources are referenced by identity and integrity metadata only.

## Complete review method

The full-version HTML was reviewed from introduction through all nine substantive topics. An audit-only standard-library parser enumerated headings, paragraphs, and lists and produced 4,978 normalized words. The homepage produced 278 normalized words. These metrics supported completeness and navigation; the normalized text was not used as the training artifact.

All 24 PDF pages were extracted for audit with pypdf 6.14.2 and manually reviewed. The 6,197-word extraction exposed the visible 2019 date, homepage, embedded FAQ, adoption explanation, and full code. It also contained browser-generated headers, footers, file paths, and repeated full-version wording; none was copied as training prose.

The final Markdown was composed by hand. Accepted ideas were selected, paraphrased, reorganized, integrated with their rationale, and annotated with source-section citations. No bulk HTML-to-Markdown or PDF-to-Markdown conversion was accepted as final prose.

## Rights and ShareAlike metadata

The KECC pages label the work Creative Commons Attribution-ShareAlike 4.0 International. The adapted Markdown does all of the following:

- credits the creator as the v. 1 KECC Collective;
- names the source work and version;
- links the full HTML and PDF source records;
- links CC BY-SA 4.0;
- describes the manual selection, paraphrase, condensation, reorganization, and omissions;
- states that no endorsement is implied;
- licenses the adapted Markdown under CC BY-SA 4.0;
- adds no claim of legal advice or extra technical restriction.

The deed and legal code are used only to validate rights metadata. Their legal prose is not direct training content. This package does not offer a legal opinion.

Individual Collective member names were omitted from training prose because the collective signature supplies creator attribution and member-name recall is not a useful learning objective. The source ledger attributes the work to the Collective without converting personal identities or affiliations into trivia prompts.

## Component and access audit

The full HTML contains one shared-footer Creative Commons badge image; the homepage contains the same badge. Neither image was fetched or reused. The site stylesheet, JavaScript, externally hosted font, navigation, contact links, and shared chrome were excluded. The PDF contained no raster XObject occurrence in its 24 pages.

The full page linked shorter educator and producer versions, FAQ, adoption, contact, and a named third-party producer network. Those routes were not accessed. The allowlisted PDF itself contained its FAQ and adoption pages, so they were reviewed within the PDF without requesting separate HTML pages. No third-party kink page was fetched.

## Version and authority cleanup

Three source facts control interpretation:

1. The web title identifies the material as version 1.
2. The PDF carries a visible 25 April 2019 date.
3. The PDF FAQ calls the version imperfect and incomplete, anticipates changes in practice and terminology, and says the Collective expected further revision.

The source also acknowledges that its core group did not represent the kink community's diversity as well as desired. These limitations are retained near the start of the training document and again in the derivation rules.

The PDF says the Collective did not monitor adopters or centrally enforce the code. The training document uses that only as an authority boundary. It does not teach sanctions, investigations, or legal enforcement, and it does not imply that declaring adoption proves compliance.

## Educator, producer, and discussion structure

The source's three-part organization is preserved because it assigns different controls to different roles:

- educators control their teaching, touch, negotiation, demonstrations, disclosures, classroom boundaries, and personal conduct;
- producers control hiring, vetting, policies, staffing, reporting routes, attendee support, and organizational communication;
- discussion sections explain power, modeling, privacy, inclusion, status, and other reasons behind the conduct proposals.

The final Markdown does not flatten institutional controls into an educator's personal responsibility or treat an educator's declaration as a substitute for producer vetting.

## Consent and demonstrations

The corpus retains the version-1 formulation of consent as informed, capacity-aware, explicit or enthusiastic, continuous, and free of coercion or manipulation. It preserves the classroom rule that touch is not implied, specific verbal agreement precedes touch, new negotiations should not be conducted before the audience, and students need enough context to recognize that a demonstration was negotiated.

Student demonstrations retain the complete power analysis: educator status, classroom social pressure, difficulty withdrawing after a demo begins, entertainment pressure, and alienation through apparent attractiveness-based selection. The preference for existing partners or educators, advance written negotiation, and in-class students only as a last resort remains explicit.

Anti-pressure safeguards include advance notice, explaining the activity and risk before asking, not selecting an individual, asking without repeated room pressure, skipping the demonstration when no one volunteers, blocking heckling or coercion, requiring unambiguous enthusiasm, preserving withdrawal, and selecting primarily for safety, agreement, and relevant ability. The producer's standards and duty to support a student after a bad experience are retained.

The source's exact advance-time threshold is excluded as a portable sufficiency rule. Advance selection and written negotiation are retained as the durable safeguard. Aftercare remains a planning and availability duty, not an operational protocol.

## Competence and risk autonomy

The corpus preserves topic-specific competence, disclosure when departing from commonly accepted practice, and a three-way distinction among relevant professional qualifications, professional review, and personal opinion. It also preserves the source's recognition that much kink education is supplied by expert amateurs because adjacent professions do not cover every kink subject.

That context is used to block invented certification. No credential, professional review, or personal experience is described as approval of an entire class or technique.

Risk-profile respect includes no pressure to perform, the option to audit, transparent risk communication before hands-on participation, and respect for venue policy. Named medical and kink activities in the source are excluded because they illustrate disclosure categories; they are not converted into operational instruction.

## Inclusion and accessibility

Educator-side material includes nondiscrimination, no retaliation, pronoun respect, gender-neutral teaching when appropriate, support across body types and physical abilities, different learning needs, and classroom response to microaggressions. The source's microaggression categories are summarized conceptually, while culture-specific examples are omitted.

Producer-side material includes diverse staff and educators, varied perspectives in decisions, inclusive design and promotion, openness to feedback, a published conduct policy, reporting mechanisms, trained response staff, and support planning around triggering or trauma. The training document explicitly says that policies and roles do not themselves prove accessibility, impartial handling, medical competence, or safety.

## Accountability and boundaries

Accountability is presented as feedback, acknowledgment, repair, learning, and changed behavior rather than perfection. The provisional contact and circle definitions are retained along with their conflict-of-interest design: independent contacts, more than one route, diverse and approachable participants, community trust, and a designated information repository to help reveal patterns.

The source itself says these terms were inconsistent and evolving. The corpus therefore does not claim that a contact or circle establishes investigator competence, due process, confidentiality, or enforcement authority.

Educator-student boundaries retain professional conduct in classes and private lessons, disclosure of existing relationships where relevant, a formal relationship policy developed with outside input, explicit discussion of power, and negotiated mentorship boundaries. The source's preference for broad prohibitions in many circumstances is retained without turning it into a universal legal rule. Its exact waiting-period suggestion is excluded as a safe-harbor threshold.

## Vetting, incident disclosure, and privacy

Producer vetting covers competence versus confidence, personal and professional misconduct relevant to the role, appropriate scope for new educators, community reports, and transparent information for attendee choice. Intake topics include significant injuries, accusations of consent violation, accountability histories, exclusions, changes intended to prevent recurrence, consent processes, experience, and qualifications.

These topics are not a clearance checklist. Self-report, references, credentials, or a completed form do not prove safety. The source's named external network and people are omitted. The durable lesson is that references supplied by an educator are insufficient on their own and that vetting should use appropriately independent information.

Bidirectional vetting is retained: booking can communicate venue endorsement of an educator, while teaching can communicate educator endorsement of a venue.

Disclosure retains both caused and alleged consent incidents and injuries, accountability processes, and exclusions. Non-retaliation, no pressure for silence, respect for private discussion, and careful access controls are preserved. The source's unresolved privacy tension remains explicit: stigma and intimate information create real privacy needs, while secrecy can deny others information needed for informed choice. Victim privacy, community safety, proportionality, other affected people, and fairness must be considered together.

The corpus does not answer evidentiary, mandatory-reporting, data-protection, defamation, medical-confidentiality, or due-process questions. Those exclusions prevent ethical discussion from being misrepresented as legal advice.

## Claim-level citations

Every substantive source-derived paragraph or compact list in `CORPUS.md` carries a citation in the form `[SOURCE-ID: source section]`. `sources.jsonl` maps each source ID to its exact URL, retrieval time, content type, size, digest, and treatment. Licence citations are explicitly metadata-only.

Uncited prose is limited to adaptation notices, conservative safety limitations added during editing, and dataset derivation rules. Those additions are clearly distinguishable from claims attributed to the KECC.

## Data hygiene

`CORPUS.md` is prose, not chat or Q&A. It contains no contributor-name drill, contact-address drill, source-order exercise, file-hash recall, or URL recall prompt. URLs appear only where necessary for source and licence attribution.

The KECC homepage, full-version HTML, PDF, this Markdown, and every future derivative form one source-family unit. They must be assigned to one split before chunking or question generation. QA, trainer, training-projection, validation, evaluation, holdout, OOD, shadow, benchmark, probe, and dataset artifacts were not accessed during construction.

## Package inventory

- `CORPUS.md`: one directly trainable CC BY-SA 4.0 adapted Markdown document.
- `sources.jsonl`: five-record source ledger with access metadata and hashes.
- `dispositions.jsonl`: 50 manual inclusion, narrowing, and exclusion decisions.
- `source_snapshot/provenance.json`: retrieval, review, rights, component, scope, and split audit.
- `manifest.json`: deterministic package metadata and output hashes.
- `tests/test_corpus.py`: deterministic integrity and semantic-boundary checks.
