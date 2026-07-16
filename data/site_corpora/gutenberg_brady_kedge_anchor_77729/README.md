# Project Gutenberg 77729: bounded historical ropework corpus

This package is a manually written, information-dense extraction of exact-new historical rope vocabulary and teaching decomposition from William N. Brady’s *The kedge-anchor: or, Young sailors’ assistant*, sixth edition (New York, 1852). It is intentionally not a copy of the ebook and not a tying or rigging manual.

The sole content-evidence artifact was the static Unicode text retrieved once from the Project Gutenberg high-speed mirror at `https://gutenberg.pglaf.org/7/7/7/2/77729/77729-0.txt`. That mirror was selected only after the current official mirror list identified `https://gutenberg.pglaf.org/` as a registered Project Gutenberg mirror. The main `gutenberg.org` ebook body was not downloaded or automated because Project Gutenberg’s current terms describe the main website as intended for human users and direct automated users to mirrors.

The source body is not stored in this package. Its exact checksums, response metadata, line coverage, mirror audit, rights review, and manual dispositions are recorded in `sources.jsonl`, `surfaces.jsonl`, `dispositions.jsonl`, and `source_snapshot/provenance.json`.

## What is retained

`CORPUS.md` retains only manually paraphrased, source-supported material about:

- the source’s component-to-family teaching sequence;
- fox, Spanish fox, knittle, and sennit;
- historical knot, hitch, bend, and bight word use;
- conceptual distinctions among short and long splices, splice eyes, Flemish and artificial eyes, and grommets;
- worming, parceling, serving, seizing, and snaking as distinct operations;
- wall-and-crown composition, sea and harbor gasket structure, wrought and sword mats;
- shell, sheave, pin, score, groove, and contrasting block forms; and
- transcription and spelling caveats.

Every evidence paragraph ends in a source locator. The corpus is direct-training-ready only for this bounded historical vocabulary and teaching architecture.

## What is excluded

The package excludes the raw text, all illustrations and plates, image-dependent steps, complete procedures, tools and material recipes, chemical treatments, numerical and load rules, strength tables, equipment inventories, naval operations, promotions, automated summary text, body lowering or restraint, human support, and human suspension. It does not establish current terminology, suitability, strength, safety, working load, or correctness for bondage, shibari, rescue, arborist, marine, industrial, or life-support use.

## Rights and attribution

The [official ebook record](https://www.gutenberg.org/ebooks/77729) identifies ebook 77729 as public domain in the United States. The record identifies William N. Brady as the author, the sixth edition, original publication information, release date January 18, 2026, and transcription credits to Chris Curnow, Harry Lamé, and the Online Distributed Proofreading Team, using source images made available by the Internet Archive.

Project Gutenberg’s [permission page](https://www.gutenberg.org/policy/permission), [license page](https://www.gutenberg.org/policy/license), and [terms](https://www.gutenberg.org/policy/terms_of_use.html) distinguish public-domain text from the Project Gutenberg trademark and associated distribution conditions. “Project Gutenberg” is a registered trademark; no endorsement is implied. Public-domain status is jurisdiction-specific, so users outside the United States must make their own determination.

This package is a quotation-light, manually rewritten and narrowed derivative created on 2026-07-16. Project Gutenberg also cautions that its electronic texts may contain editorial changes such as modernization, dehyphenation, or formatting changes and are not guaranteed to be exact print facsimiles. The source transcription itself documents additional changes and unresolved inconsistencies.

## Verification

Run:

```bash
python3 -m unittest discover -s data/site_corpora/gutenberg_brady_kedge_anchor_77729/tests -v
```
