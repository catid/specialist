# Manual acquisition and corpus audit report

## Outcome

This package is accepted as a direct-training-ready historical ropework vocabulary corpus with a strict non-operational boundary. Its sole content evidence is the static Unicode transcription of Project Gutenberg ebook 77729. The acquired body was manually read from first line through last line; only source-supported, exact-new vocabulary and teaching decomposition survived. The source body, images, automated item summary, and complete procedures are not redistributed.

## Pre-acquisition access audit

The access and rights audit was completed before the ebook body was requested on 2026-07-16.

The current official `MIRRORS.ALL` page listed `https://gutenberg.pglaf.org/` as Project Gutenberg’s high-speed United States mirror and stated that it includes generated files. The main terms-of-use and robot-access pages said that the main website is intended for human users and directed automated or multi-book access to the registered mirrors and documented harvesting mechanisms. The main ebook body was therefore not acquired from `gutenberg.org`.

The main robots file was reviewed manually and disallowed the ebook-search route for the general user agent. That narrow robots rule was not treated as permission to ignore the stronger human-only terms. The selected mirror had no published robots file at `/robots.txt`: the request returned HTTP 404. The exact 162-byte 404 response is checksummed in the provenance record. A missing robots file was not interpreted as a grant of rights; use of this host rested on its listing in the official mirror registry and was limited to one known static item.

Two known candidate artifacts were checked by headers before acquisition. The generated cache text was available but had a later regeneration timestamp. The static Unicode release artifact at `/7/7/7/2/77729/77729-0.txt` matched the ebook identifier and release-era timestamp, so it was selected. Exactly one body GET was made for that artifact.

## Exact item and artifact identity

- Ebook: Project Gutenberg 77729
- Author: William N. Brady (official record displays death year 1887)
- Title: *The kedge-anchor: or, Young sailors’ assistant*
- Edition: sixth
- Print imprint in text: New York, published by the author, 1852
- Copyright-entry statement in text: 1847
- Official ebook release: 2026-01-18
- Official record rights label: Public domain in the USA
- Static Unicode mirror URL: `https://gutenberg.pglaf.org/7/7/7/2/77729/77729-0.txt`
- Retrieval: `2026-07-16T16:21:10Z`
- HTTP status: 200
- Content type: text/plain, UTF-8
- Last-Modified: `Sun, 18 Jan 2026 17:22:06 GMT`
- ETag: `"696d16be-1691ab"`
- Bytes: 1,479,083
- Lines: 26,071
- Whitespace-delimited words: 239,369
- SHA-256: `7a1c173c1f3c73201ab21f741130094a57765a923ff26d503c4c0f4c93a58cd2`
- SHA-512: `bd81079fc2faaf79c319df659f97ab4fe52598c5cbd18c01b1491d572d7f1dc96aa61e920d6dffafd9d1689f01b57633e1a6ad0e97fe51e1f87acdd4934f0c1a`

The title page in the exact text independently confirms the title, author, sixth edition, 1852 imprint, and 1847 entry. The official item record additionally identifies the production credits and source-image lineage. The automatically generated summary on that record was deliberately excluded.

## Complete manual reading

The text was divided once to keep reviewer context bounded. Lines 1–13,000 and 13,001–26,071 were read sequentially in manageable, ordered chunks. The two ranges are contiguous, non-overlapping, and exhaust all 26,071 lines. The second reviewer made no edits. Neither reviewer inspected any existing `CORPUS.md` body; novelty checks were restricted to existing package manifests, reports, and readmes.

The review reconciled front matter and contents, Parts I through XI, every text illustration marker, the complete glossary, all numerical tables, later recipes and reviews, the transcriber notes, and the end marker. There are 101 `[Illustration` markers in the transcription; every one was excluded. Part XI was checked through its final tables rather than skipped as a homogeneous appendix.

The second-half audit found no eligible exact-new content after line 13,000. Lines 13,001–17,396 are the tail of Table 499 and Tables 500–521; no Table 522 heading appears in the body. Lines 17,397–21,217 are Table 523, lines 21,218–24,962 are Table 524, lines 24,963–25,391 are Tables 525–526, lines 25,392–25,553 are recipes, lines 25,554–25,914 are press reviews, and lines 25,915–26,071 are transcriber notes and the end marker. Block names occurring in equipment inventories were not mistaken for definitions.

## Novelty check

Before retention, candidate terms were searched only across existing `manifest.json`, `REPORT.md`, and `README.md` files. No existing corpus body was opened. The exact phrases `Spanish fox`, `knittle`, `sennit`, `grommet`, `Flemish eye`, `artificial eye`, `worming`, `parceling`, `snaking`, `fiddle block`, `shoe block`, `sister block`, `snatch block`, `throat seizing`, `quarter seizing`, `wrought mat`, and `sword mat` each returned zero metadata/report/readme matches at audit time.

Common names already represented elsewhere—such as ordinary bowline and figure-eight families—were not expanded into redundant directions. They appear only where necessary to describe Brady’s sequence or historical grouping. The retained value is therefore the missing historical component vocabulary, contrasts among families, and pedagogical decomposition.

## Retained evidence

The corpus preserves the source’s progression from rope-yarn through fox, Spanish fox, and knittle; its movement from elementary named forms to splices and eyes; its separation of worming, parceling, serving, and seizing; its strand-end and flatwork families; and its transition into block anatomy.

It records conceptual contrasts rather than executable recipes: short versus long strand-path strategies; eye splice versus cut-splice collar; yarn-level Flemish eye versus strand-path artificial eye; ordinary cross-turn seizing versus snaking; sea versus harbor gasket plaiting; wall-and-crown composition; shell, sheave, pin, score, and groove; and the structural differences among shoulder, fiddle, shoe, sister, tail, snatch, dead-eye, bull’s-eye, and heart forms.

Historical spellings and inconsistencies are preserved as linked search clues, not normalized into false modern distinctions.

## Controlled exclusions

- all 101 illustration markers, plates, engravings, captions as visual evidence, and figure-dependent steps;
- every complete tying, splicing, eye-making, seizing, serving, mat-making, gasket-making, or block-strapping procedure;
- tools, material specifications, dimensions, turn and tuck counts, equipment construction, chemical treatments, and maintenance recipes;
- purchase calculations, mechanical-advantage instruction, cable and rope strength values, load rules, weights, dimensions, tables, and inventories;
- ship launching, rigging, anchoring, sailing, naval combat, emergency, salvage, heaving-down, and management operations;
- asserted strength, reliability, preference, or safety claims that have not been validated for modern use;
- obsolete advice, including tar, blacking, and other treatment recipes;
- body lowering, restraint, man-rope use, human support, and human suspension;
- advertisements, endorsements, press reviews, promotional claims, the item page’s automatic summary, and Project Gutenberg or transcriber boilerplate except for provenance and fidelity warnings; and
- every inference about present bondage, shibari, rescue, arborist, marine, industrial, or life-support suitability.

## Rights, trademark, jurisdiction, and fidelity

The official item record marks ebook 77729 public domain in the United States. The 1852 text and the official author death-year display support that United States determination, but this package does not declare worldwide public-domain status. Users outside the United States must evaluate local law.

Project Gutenberg’s permission and license pages distinguish public-domain source text from the Project Gutenberg registered trademark and associated distribution conditions. This independent derivative uses the name only for attribution and source identification, does not imply endorsement, and does not redistribute the source text or imagery.

Project Gutenberg cautions that an ebook may not exactly reproduce its print source because of spelling, hyphenation, or formatting changes. The exact transcription additionally says that it preserves many archaic and inconsistent forms while documenting moved illustrations, recombined tables, expanded ditto marks, silent punctuation and typo corrections, and specific editorial changes. The corpus therefore attributes its lexical evidence to this exact transcription and does not claim diplomatic print fidelity.

## Absolute application boundary

This is a historical vocabulary and pedagogy corpus. It cannot be used to choose, teach, validate, rate, load, install, certify, or approve a knot, hitch, bend, splice, eye, seizing, serving, mat, gasket, block, rope, anchor, hardpoint, upline, restraint, body-lowering system, or human-suspension system.

No source body, source image, or automated summary is redistributed.
