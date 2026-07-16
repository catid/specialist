# NIST Manila-rope fiber color and serviceability boundary (1933)

This package is a concise, manually rewritten historical evidence-and-limitations digest of Genevieve Becker and William D. Appel’s *Evaluation of Manila-Rope Fiber for Color*. It is designed to prevent a common but unsupported inference: the paper did not test rope serviceability, and its prepared abaca-fiber measurement was not the same thing as finished-rope surface appearance.

The sole content-evidence body is the official NIST PDF at `https://nvlpubs.nist.gov/nistpubs/jres/11/jresv11n6p811_a2b.pdf`. Official NIST volume, journal, and technical-series rights pages provide identity and rights provenance. No third-party paper, specification, standard, report, product page, or cited source body was acquired or used as evidence.

## Retained scope

`CORPUS.md` retains only attributed historical evidence that:

- the authors explicitly excluded a fiber-color/serviceability relationship from their study;
- constituent fiber color and finished-rope surface appearance are different targets;
- construction, yarn and strand structure, twist, lubricant, dust, exposure, and bleaching confound visible appearance;
- the source’s dark-rope discussion is incomplete and cannot support either a discard rule or a safety claim; and
- the samples came from a commercial manufacturer context with rope furnished by cordage manufacturers and the Boston Navy Yard.

It also preserves the authors, paper and manuscript dates, DOI, the Cordage Institute research-associate relationship, and the Bureau of Standards context.

## Excluded scope

The package excludes all four figures, all five tables, every experimental value, color scale, visual rank, instrument, product, solvent, specimen-preparation or sampling direction, procurement detail, specification, third-party or reference body, and modern recommendation. It contains no color acceptance, discard, retirement, cleaning, hygiene, strength, working-load, or remaining-life rule.

This source concerns abaca-based Manila rope. It must not be transferred to jute, true hemp, body contact, uplines, anchors, hardpoints, bondage, restraint, body lowering, or human suspension.

## Source identity

- Becker G, Appel WD (1933), *Evaluation of Manila-Rope Fiber for Color*.
- Bureau of Standards Research Paper RP627.
- *Bureau of Standards Journal of Research* 11:811–822.
- DOI: `10.6028/jres.011.057`.
- Publication issue: December 1933; article dateline: Washington, September 16, 1933.
- Official volume record: `https://www.nist.gov/nist-research-library/journal-research-volume-11`.

The official PDF is checksum-bound in `sources.jsonl` and `source_snapshot/provenance.json`; it is not redistributed.

## Rights and change notice

NIST’s journal information page states that papers in the *Journal of Research of NIST* are not subject to copyright in the United States and requests attribution. NIST’s technical-series rights page notes possible foreign rights, grants broad reuse rights to the extent NIST may assert them, and cautions that third-party works may differ. This package makes no worldwide public-domain claim.

This is a manually rewritten, quotation-light, narrowed derivative created on 2026-07-16. Every scan image, figure, table, reference body, product name, and potentially separable third-party component is excluded. No endorsement by NIST, the Cordage Institute, any manufacturer, the Navy Department, or the Boston Navy Yard is implied.

## Verification

Run:

```bash
python3 -m unittest discover -s data/site_corpora/nist_manila_rope_color_serviceability_1933/tests -v
```
