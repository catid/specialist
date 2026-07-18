# Additional Rope-Knowledge Resource Discovery v2

Status: **candidate metadata only; not training authority**

Search provider: **Exa MCP exclusively**

Created: 2026-07-18

## Outcome

This second, gap-targeted pass records 35 additional-resource decisions:

- 10 `collect` candidates, meaning only that a source is worth a later rights, component, safety, and provenance workflow.
- 18 `manual-review` candidates.
- 7 explicit `reject` decisions.

Every entry remains `candidate_only_not_authorized_not_training_eligible` or an explicit rejected equivalent. A `collect` decision does **not** place bytes in the corpus, authorize reuse, establish source-disjointness, or make a source eligible for training.

The pass filled the requested gaps without recursively crawling a site, copying source content into a dataset, changing the authoritative registry, rebuilding a training snapshot, promoting any source, or extending a source-disjoint contract.

## Immutable baselines

The discovery pass was deduplicated against these exact inputs:

| Baseline | Count | SHA-256 |
|---|---:|---|
| `data/site_corpora/registry/site_corpus_registry_v1.json` | 33 resources | `633b191178ee378dcd058b56d2a58b7a76d6e05daad154a4eb93c619fc8ee06f` |
| `data/site_corpora/registry/additional_resource_discovery_v1/candidates.json` | 32 candidates | `92ae9fac5498e35ee520978786a2f0adc1c86c1d53f548bb61dd6e308a0df1c5` |
| `data/site_corpora/registry/additional_resource_discovery_v1/report.md` | — | `7998068a7cf1e09bed54c76c1fdbffdee4cdad446b40076c9e9374da57a8a1a7` |
| `data/site_corpora/registry/additional_resource_discovery_v1/manifest.json` | — | `a4be0913b7980c909b166ba122c43a1604ae532de622b61e66e22bba7c64a38a` |

The v2 inventory uses document and bibliographic identity, not URL equality alone. That caught a second National Diet Library PID for the same 1948 *Taihojutsu Instruction Manual*: PID 1711696 and v1 PID 1078906 share title, author, publisher, year, call number, and NDL bibliographic ID, so the second scan is rejected as a duplicate.

## Method

The pass used narrow Exa searches and targeted Exa fetches to establish:

- exact canonical URL and domain;
- publisher, authorship, edition, DOI/PID, and authority evidence;
- scope and the specific knowledge gap addressed;
- exact, domain-level, semantic, and bibliographic duplication against both baselines;
- visible license, copyright, access, paywall, and repository signals;
- robots/TOS/AI-reuse questions that remain unresolved;
- safety-transfer risks and a bounded future collection depth;
- a priority and one of `collect`, `manual-review`, or `reject`.

Targeted page reads were used only for metadata verification. No retrieved prose, images, diagrams, tables, or source files were materialized under `data/site_corpora`, converted to Markdown, or added to any training artifact.

## Findings by gap

### Indoor hardpoints and structural evaluation

The most important new pair is the official OSHA fall-arrest appendix and the Wood I-Joist Manufacturers Association guide. Together they make the load path explicit: substrate, member, blocking, fasteners, attachment hardware, and force direction all matter. The WIJMA guide is especially valuable because it explains why sheathing-only attachment, flange drilling, fastener withdrawal, head pull-through, cross-grain tension, and poorly planned proof loading are failure paths for engineered joists.

The current DoD UFGS section adds post-installed concrete and masonry anchor qualification, installation, and QA. The Sport-Thieme guide adds a useful comparative ceiling taxonomy and warnings about rotation, simple hooks, suspended ceilings, and beam systems, but it is proprietary, product-oriented, and cites standards that need a current-edition audit.

The X-POLE manual is retained only as an explicit negative boundary: a floor-to-ceiling pressure-mounted dance pole is not evidence of a human-suspension hardpoint. The uncredentialed YVEX article is rejected for lack of engineering provenance.

None of the occupational or building-anchor numbers can be relabeled as bondage working-load limits. A local qualified structural engineer, exact substrate/member identification, complete load path, load direction, installation quality, inspection, and rescue plan remain controlling.

### Upline and mechanical advantage

The Massachusetts Firefighting Academy operational guide and Connecticut instructor packet provide the clearest focused treatment of simple, compound, and complex mechanical advantage; change-of-direction pulleys; Z-rigs; piggyback systems; progress capture; smooth hauling; lockoff; and resets.

The key transferable lesson is conceptual rather than numeric: theoretical mechanical advantage is reduced by friction, and a redirect changes the loads seen by anchors. These rescue materials assume trained teams, rated hardware, redundancy, communication, and system checks. They are not drop-in shibari upline recipes.

The Texas skills chapter is rejected as checklist-heavy and rights-sensitive. The older Massachusetts technician guide is rejected as redundant and likely tied to superseded standards.

### Rope construction, splicing, inspection, and material failure

The 2024 Cordage Institute CI-2001 guideline is the strongest current inspection-and-retirement authority found, but its catalog explicitly prohibits reproduction without written permission and sells the document. It therefore remains metadata/citation only. The institute's splicing hub is valuable as an authority map and emphasizes construction-specific and manufacturer-specific instructions, but it is not a reusable splice corpus.

The available NAVSEA Chapter 613 Rev 3 mirror is broad and dense, covering laid, plaited, braided, and double-braid ropes, splicing, inspection, and replacement. It is nevertheless a superseded 1999 mirror; an official current Rev 5 or later copy must be located before any collection decision.

Three experiments add a useful evidence ladder:

- a 2024 study tests actual finished jute, nettle, coir, sisal, and cotton ropes, but its exact reusable license is unresolved and its concrete-reinforcement treatments do not transfer to body-contact rope;
- a 2015 experiment measures finished hemp and sisal ropes at different temperatures, but is small and not a wet/abrasion/knot/cyclic validation;
- a CC BY 4.0 MDPI study measures uncertainty and failure behavior in jute yarn, which is useful for teaching natural-material variability but is not finished-rope evidence.

A marine-aged jute/polyester laminate paper is explicitly rejected: a resin composite is not rope, despite sharing the word “jute.”

### Position-related nerve and circulation risk

The CC BY 4.0 postoperative-ulnar-neuropathy systematic review adds current anatomy, compression/ischemia mechanisms, positioning, padding, and monitoring evidence. Its prevention evidence is explicitly low quality, and perioperative unconscious-patient conditions differ from consensual rope.

The hand/wrist restraint systematic review is unusually direct evidence of radial, median, and ulnar injuries associated with handcuffs and zip ties. It is valuable but remains manual-review because the exact reuse license is not visible and the studies largely comprise injured-case reports and series. Their proportions are not population incidence and cannot define safe tightness or duration.

The BMC tourniquet review supplies mechanism-level material on applied pressure, edge gradients, cuff width, mechanical compression, and ischemia. Pneumatic cuffs are not rope, so the source can explain mechanisms but cannot yield a bondage pressure/time rule.

### Accessibility and adaptation

Three practical guides add complementary coverage:

- Northcott's 2025 *Love Rights* resource addresses complex communication, Easy Read presentation, supported decision-making, boundaries, privacy, and support people.
- Enfold's disability sexuality handbook supplies a broad rights-based curriculum but is CC BY-NC-ND 4.0; the NoDerivatives term blocks automatic transformation and requires legal review or permission.
- Holland Bloorview's clinical guide offers concrete patterns for alternate and eye-gaze communication, fatigue and weakness, energy conservation, caregiver privacy, supportive positioning, and adaptive tools.

Each has jurisdiction, age, rights, and clinical-scope limits. None validates a rope tie or suspension for a disabled person. Adaptation must remain person-specific and must not equate disability with incapacity.

The open 2025 chronic-pain/BDSM study adds directly relevant participant evidence. It is exploratory, cross-sectional, convenience-sampled, and based on self-reported chronic pain. Perceived benefits are not treatment evidence, are not universal, and must not be used to encourage higher-intensity or edge play.

### Consent and privacy-preserving incident learning

The CC BY 4.0 sexual-consent-norms study adds measured differences among BDSM, other sexual-minority, and majority samples. It is suitable for misconception correction only if the model is taught that group norms never establish an individual's consent and that explicit agreement does not erase power or coercion.

Three federal sources form an incident-learning governance stack:

1. AHRQ CANDOR provides system-focused, prevention-oriented event investigation and analysis.
2. HHS OCR explains Expert Determination, Safe Harbor, actual knowledge, free text, and re-identification risk.
3. NIOSH FACE provides a concise model for fact gathering, qualified investigation, anonymous public reports, no blame determination, and prevention recommendations.

These are methods, not rope facts. HIPAA does not govern every community report, healthcare confidentiality/privilege does not automatically transfer, and rare kink incidents can remain identifiable after obvious names are removed.

### Primary Japanese people, lineage, and history

Three first-person or eyewitness interviews materially strengthen the lineage evidence base:

- Naka Akira discusses Nureki Chimuo, Minomura Kou, Kinbiken, his own entry into rope, and responsibility for danger.
- Akechi Denki discusses his own practice history, the term *nawashi*, Kitan Club, Tsujimura Takashi, Ito Seiu, hojojutsu research, and stage development.
- Saikatsu, Akechi's long-term assistant, supplies a detailed oral chronology of postwar media, performance, Akechi, and Bakuyukai.

All three remain manual-review because copyright, translation, speaker/interviewer rights, and recollection-versus-corroboration must be resolved. First-person testimony is authoritative about what a speaker recalls or reports, not independent proof that every historical claim is true.

Two NDL items add primary historical artifacts:

- *Tokugawa Bakufu Keiji Zufu* (1893) is visible under a Japanese Copyright Act Article 67 adjudication, not an open license. It is metadata/citation only absent permission.
- Ito Seiu's *Bijin Ranbu* (1932) is marked protection-period-expired, but NDL reproduction/IIIF policy, explicit-content handling, relevance, translation, and historian review still gate any use. Bulk image ingestion is excluded by default.

The CiNii globalization conference abstract is rejected because it is too thin and its record explicitly says abstract reuse is disallowed.

Historical penal restraint, erotic artwork, and hojojutsu must not be presented as modern consensual technique, proof of safety, or a simple continuous kinbaku lineage.

## Decision ledger

| Candidate | Decision | Priority | Domain | Primary gap |
|---|---|---|---|---|
| OSHA Subpart M Appendix C | collect | high | `www.osha.gov` | steel/masonry/wood anchorage and tension forces |
| WIJMA engineered-joist anchorage | manual-review | high | `i-joist.org` | I-joists, SCL, blocking, fastener failure |
| UFGS 05 05 20 | collect | high | `www.wbdg.org` | post-installed concrete and masonry anchors |
| Sport-Thieme ceiling suspensions | manual-review | medium | `pimage.sport-thieme.de` | ceiling types, beam systems, rotation, redundancy |
| X-POLE XPERT manual | reject | none | `xpole.com` | unsafe pressure-mount transfer boundary |
| YVEX room anchor article | reject | none | `in.yvex.de` | uncredentialed hardpoint advice |
| Massachusetts operational rescue guide | manual-review | high | `www.mass.gov` | practical mechanical advantage and system checks |
| Connecticut mechanical-advantage packet | manual-review | high | `portal.ct.gov` | Z-rig, progress capture, actual vs theoretical MA |
| Texas rescue skills chapter | reject | none | `www.tcfp.texas.gov` | thin/redundant checklist |
| Massachusetts 2015 technician guide | reject | none | `www.mass.gov` | older/redundant rescue material |
| NAVSEA NSTM Chapter 613 Rev 3 | manual-review | high | `maritime.org` | rope construction, splicing, inspection |
| Cordage Institute CI-2001 | manual-review | high | `ropecord.com` | current inspection and retirement authority |
| Cordage Institute splicing hub | manual-review | medium | `ropecord.com` | construction-specific splice authority map |
| Plant-based natural-fiber ropes study | manual-review | high | `doi.org` | direct finished-rope comparison |
| Hemp/sisal temperature study | manual-review | medium | `doi.org` | finished natural-rope temperature response |
| Jute-yarn uncertainty study | collect | medium | `doi.org` | natural-material variability and failure uncertainty |
| Marine-aged jute composite laminate | reject | none | `doi.org` | nontransferable composite keyword match |
| Postoperative ulnar neuropathy review | collect | high | `pmc.ncbi.nlm.nih.gov` | position-related ulnar risk and monitoring |
| Hand/wrist restraint injury review | manual-review | high | `pmc.ncbi.nlm.nih.gov` | direct wrist restraint injuries |
| Tourniquet compression review | collect | high | `doi.org` | pressure gradients and nerve-injury mechanisms |
| Northcott *Love Rights* | manual-review | high | `northcottwebsiteprod.blob.core.windows.net` | complex communication and supported decisions |
| Enfold disability sexuality handbook | manual-review | high | `enfoldindia.org` | accessible rights-based curriculum |
| Holland Bloorview neuromuscular guide | manual-review | high | `hollandbloorview.ca` | communication, fatigue, positioning, privacy |
| Chronic-pain/BDSM exploratory study | collect | high | `pmc.ncbi.nlm.nih.gov` | direct adaptation and chronic-pain evidence |
| Sexual consent norms study | collect | high | `pmc.ncbi.nlm.nih.gov` | empirical consent norms and misconceptions |
| AHRQ CANDOR event investigation | collect | medium | `www.ahrq.gov` | systems-focused incident learning |
| HHS de-identification guidance | collect | medium | `www.hhs.gov` | privacy and re-identification risk |
| NIOSH FACE program model | collect | medium | `www.cdc.gov` | anonymous prevention-oriented reports |
| Naka Akira interview | manual-review | high | `www.kinbakutoday.com` | direct Nureki/Naka/Kinbiken testimony |
| Akechi Denki interview | manual-review | high | `www.osada-ryu.com` | direct postwar practice and influence testimony |
| Saikatsu interview | manual-review | high | `www.kinbakutoday.com` | eyewitness chronology of Akechi and Bakuyukai |
| NDL *Tokugawa Bakufu Keiji Zufu* | manual-review | high | `dl.ndl.go.jp` | historical restraint primary artifact |
| NDL Ito Seiu *Bijin Ranbu* | manual-review | high | `dl.ndl.go.jp` | primary artist work and iconography |
| NDL duplicate Taihojutsu PID | reject | none | `dl.ndl.go.jp` | exact bibliographic duplicate of v1 |
| CiNii kinbaku/shibari abstract | reject | none | `cir.nii.ac.jp` | thin and reuse-disallowed abstract |

## Promotion gates for a future task

This report does not authorize the following work. A separate task would need to:

1. Resolve exact source/version and successor status.
2. Resolve license, copyright, repository policy, robots, TOS, and AI/training reuse.
3. Exclude or separately clear figures, standards, photos, cases, translations, and other components.
4. Perform medical, structural, accessibility, Japanese-language/history, privacy, and explicit-content review as applicable.
5. Define bounded collection depth and preserve attribution, version, and claim-level transfer limits.
6. Deduplicate by document/bibliographic identity across all existing and proposed sources.
7. Obtain a fresh opaque source-disjoint contract extension for any newly authorized bytes.
8. Only then consider registry promotion, corpus materialization, Markdown conversion, or training-snapshot rebuild.

No item in v2 bypasses those gates.
