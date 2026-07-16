# Manual audit report

## Outcome

This package is accepted as a bounded text-only direct-training source with `direct_training_ready: true`. It uses only the two explicitly allowed public Bureau of Reclamation bodies: the official project page and the public two-page research bulletin. Both exact bodies were normally accessible and checksumable.

The protected final report identified as Document 1018 is a hard exclusion. It was not requested or accessed. No corpus claim infers, reconstructs, paraphrases, or quotes it.

## Verified allowed bodies

### Project page

- Title: *Testing/Verification of Rope Access Anchors*
- Project ID: 6390
- Corporate author: Bureau of Reclamation
- URL: `https://usbr.gov/research/projects/detail.cfm?id=6390`
- Retrieved: `2026-07-16T14:37:52Z`
- HTTP status: 200
- Bytes: 23040
- SHA-256: `451093515a67b4043988cc9122f3f567e7df17eb9fdcffd26f096f2f8f8bab0f`

### Public bulletin

- Title: *Testing and Verifying Rope Access Anchors*
- Series: *The Knowledge Stream Research Update*, Bulletin 2014-09
- Publisher: Bureau of Reclamation
- URL: `https://usbr.gov/research/projects/download_product.cfm?id=2259`
- Retrieved: `2026-07-16T14:37:53Z`
- HTTP status: 200
- Physical pages: 2
- Bytes: 1258428
- SHA-256: `95758fafca146ded6a672a9660f46e233d497cae3012a641dcc3ef17f13ff4e5`

Neither source body is redistributed.

## Manual review method

The project-page DOM and visible text were manually divided into ten material surfaces. Every surface was reviewed and assigned a disposition in `surfaces.jsonl`. The two bulletin pages were extracted, read manually, rendered, and visually inspected together in one ordered contact sheet. `dispositions.jsonl` records one decision for each PDF page.

The component audit contains sixteen records. It excludes three photographs and their captions; agency branding; contact blocks; attributed quotation; incorporated legal and standards material; capacities and numerical results; brands and products; configurations; test and installation procedures; detailed failure mechanics; future operational tests; equipment; linked operational material; and all protected-report content.

## Retained qualitative evidence

- Simulated installation defects were included as a research variable.
- Comparatively weak and stronger concrete were included as a test variable.
- Aging-concrete condition and installation quality were framed as sources of uncertainty.
- Different system families were expected to have different installation-error sensitivity.
- Failure modes differed qualitatively, including progressive versus abrupt outcomes.
- Multi-anchor load sharing was angle-sensitive and could not be represented by simply adding individual results.
- The project was not intended to provide one fixed standard for every anchoring situation.

Every retained paragraph is labeled `Non-bondage rope-access evidence` or `Non-bondage rope-access evidence boundary` and carries a source-surface or physical-page citation.

## Exclusions and safety boundary

- Every capacity, threshold, numerical result, unit, configuration, reduction factor, or sizing inference.
- Every proof-load, installation, testing, inspection, acceptance, or field-verification procedure.
- Every product, brand, commercial identity, item of equipment, and future operational test.
- Every photograph, caption, agency mark, person-attributed quotation, and image-derived claim.
- All incorporated standards, legal text, organization requirements, and external practice manuals.
- Project history, contacts, endorsements, injury-history claim, and named-person details not needed for evidence provenance.
- Protected final report Document 1018 in its entirety, including any request, access, inference, reconstruction, paraphrase, or quotation.
- Any DIY proof testing or certification of a ceiling, hardpoint, anchor, bondage, body-support, or human-suspension application.

The retained observations are federal occupational rope-access research evidence. They do not provide a design, capacity, procedure, or approval for any real installation.

## Rights and transparency

The project page is Bureau of Reclamation corporate-authored material. The official Bureau bulletin is credited on the project page to a Reclamation researcher and was prepared and published as agency work. Eligible federal prose is treated as public domain in the United States under 17 U.S.C. 105. All third-party or uncertain components are excluded.

The public bodies are summaries rather than a complete experimental record. The package preserves that limitation and does not fill omitted detail from protected or linked sources.
