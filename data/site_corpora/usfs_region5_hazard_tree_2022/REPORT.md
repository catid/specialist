# USDA Region 5 hazard-tree corpus report

## Outcome

This package adds a 4,226-whitespace-word, non-Q&A Markdown digest of *Hazard Tree Identification and Mitigation*, USDA Forest Service Pacific Southwest Region, FHP Report RO-22-01, revised March 2022. The direct-training artifact preserves the report's prediction limits, reduce-not-eliminate-risk framing, terminology, inspection context, target exposure, defect interactions, site change, monitoring, and specialist-referral principles.

The source concerns Region 5 hazard-tree management around people and built infrastructure. It does not study a tree or branch as a load-bearing anchor. The corpus explicitly states that tree health is not anchor capacity and provides no tree, branch, sling, anchor, hardpoint, load, bondage, or human-suspension certification.

The exact official PDF was reviewed across all 40 digital pages but is referenced rather than retained. Only the manually written `CORPUS.md` is direct-training ready.

## Source and integrity

- Title: *Hazard Tree Identification and Mitigation*
- Issuer: USDA Forest Service, Pacific Southwest Region, Forest Health Protection
- Report: FHP Report RO-22-01
- Revision: March 2022
- Official PDF: `https://www.fs.usda.gov/sites/nfs/files/legacy-media/r05/Hazard%20Tree%20Identification%20and%20Mitigation%20-%202022.pdf`
- HTTP status: 200
- Content type: `application/pdf`
- Content length: 1,597,581 bytes
- SHA-256: `a743018dd717a8c101c2a714dd6d37ed06bb981795a816d8cc43939adaa28612`
- Last modified: `Tue, 18 Feb 2025 11:52:44 GMT`
- PDF structure: version 1.7, 40 digital pages, unencrypted

The visible cover names Peter A. Angwin, Daniel R. Cluck, James Rosen, William C. Woodruff, Ashley E. Hawkins, Charles W. Barnes, Philip G. Cannon, and Sherry Hazelhurst. The generic embedded PDF author value `fsdefaultUser` is recorded in provenance but is not used as authorship metadata.

No publisher HTML page, alternate PDF, figure route, external reference, or other body route was accessed. PDF extraction supported navigation, page verification, caption review, and structural audit only. Automated extraction was not emitted as training prose.

## Rights and component review

The report is issued as an official USDA Forest Service technical report. Eligible federal text is treated as presumptively public domain under 17 U.S.C. 105. That is not treated as a blanket rights conclusion for every embedded or derived component, and the package is not a legal opinion.

The extracted report contains no explicit rights statement or component-credit statement. Absence of a credit is not proof that a figure, form, screenshot, or cited element is federally authored. The corpus therefore uses only manually paraphrased explanatory text and excludes all source visuals, layouts, forms, and external material.

The component audit found:

- six numbered figures;
- one numbered table rendered as a raster image;
- a cover graphic;
- appendix form, map, and quick-list graphics;
- a Survey123 application screenshot;
- 18 embedded raster-object occurrences across nine digital pages.

Figure 1 is a two-panel pre/post risk diagram. Figures 2 through 5 depict numerical potential-failure zones. Figure 6 illustrates included-bark formation. Table 1 combines a score, category, and suggested action. None was retained, traced, transcribed as a layout, or fetched separately.

The raster count is a structural count, not a claim that every object is a distinct expressive work; some are labels or graphic fragments. Vector and text-based layouts were reviewed with their parent pages and excluded as well. No photograph caption was found and no photograph was reused.

## Manual content audit

All 40 digital pages were reviewed. The review covered the cover and contents, introduction, definitions, prioritization, inspection frequency, target assessment, every defect section, rating system, mitigation section, monitoring, records, specialist assistance, references, and all five appendices.

The direct-training digest has 43 printed-page citations. Its 30 disposition records contain 21 included or narrowed decisions and nine exclusions.

The following source material was retained with context:

- Region 5, California, National Forest System, and agency-program scope;
- inability to predict every tree failure;
- risk reduction rather than risk elimination;
- defect, failure, target, loss, hazard, and rating-tool distinctions;
- inspection timing as a documented local decision;
- post-storm, post-fire, reopening, construction, and material-change context;
- target occupancy, duration, frequency, reach, and consequence;
- cracks, unions, architecture, decay, cankers, dead parts, roots, lean, and interacting defects;
- compaction, erosion, fire, flooding, saturation, excavation, construction, equipment, vehicle and foot traffic, soil movement, and root lifting;
- monitoring, longitudinal records, and learning from failures;
- Forest Health Protection, certified-arborist, and professional-arborist referral.

## Prediction and terminology gates

The report says that even with high care, all tree failures cannot be predicted. A professional program can significantly reduce but cannot eliminate injury or property risk. Later sections acknowledge that failures occur regardless of inspection intensity and that post-catastrophe forecasting can be both overinclusive and underinclusive.

The digest keeps that uncertainty beside every inspection concept. A visual inspection is a snapshot, hidden defects can exist, and absence of a visible defect is not proof that no relevant defect exists.

It also prevents five concepts from collapsing:

- a defect is a strength-reducing flaw;
- failure is mechanical breakage of a tree or part;
- a target is a person or built asset that could be struck;
- loss is injury or property damage;
- hazard combines tree-side failure concern with target-side exposure and consequence.

Region 5's hazard rating is represented only as an administrative prioritization device. Its arithmetic, score bands, suggested actions, and raster table are excluded because a score is not a physical breaking-strength measurement.

## Inspection frequency and monitoring

The source discusses inspection frequency in relation to public use, past failure, local insects and disease, fire, weather, site condition, facility type, resources, and program priorities. It gives agency examples ranging from more frequent developed-site review to longer roadway cycles and supplemental checks after major storms or fires.

The corpus does not turn those examples into a universal calendar. It retains the rule that the interval and any change to it need a documented local rationale. A date cannot override material change in weather, soil, roots, occupancy, construction, nearby failures, or target exposure.

Monitoring is retained as an observation trail, not reassurance. A useful record identifies the tree and target context, observed condition, date, environmental conditions, positive and negative findings, professional interpretation, and trigger for reassessment. It cannot promise that failure will wait for the next visit.

The agency evaluation form, mapping codes, mobile application, scoring sheet, and suggested-action workflow are excluded.

## Defect and site-change cleanup

The source contains extensive species, disease, size, percentage, shell-thickness, circumference, distance, and point rules. They are highly specific to the Region 5 program and sometimes contain explicit exceptions. None is reproduced as a portable threshold.

The digest instead retains relations that remain useful when scoped:

- cracks have multiple origins and become more concerning when through a member, paired around decay, at a leaning base, or connected to another defect;
- branch-union shape is not enough by itself, and old top loss, included bark, cracks, cankers, decay, and replacement growth need joint interpretation;
- crown vigor does not reliably reveal internal decay;
- decay significance depends on extent, location, remaining sound tissue, cavities, orientation, and interacting defects;
- cankers may conceal decay and can interact with cracks, lean, fire injury, and other damage;
- timing of dead-tree or dead-part failure is nearly impossible to predict;
- root problems are often hidden, and a healthy-looking crown does not exclude root disease;
- recent or changing lean is different evidence from long-compensated lean, but compensating growth does not certify stability;
- connected defects can matter more than a raw defect count.

Site history remains attached. Compaction, erosion, fire, flooding, prolonged saturation, excavation, construction, heavy equipment, vehicle traffic, and concentrated foot traffic can alter roots or soil support. Soil cracks, mounding, root lifting, partial windthrow, and an increasing lean are change indicators requiring professional interpretation.

## Exclusions

The corpus excludes:

- invasive or specialized testing instructions;
- tree-height multipliers and all numerical failure-zone distances;
- species, host, disease, fungus, insect, size, percentage, circumference, decay-shell, and rating thresholds;
- the seven-point rating system and suggested-action table;
- felling, removal, topping, pruning, cabling, bracing, support-pole, treatment, and closure instructions;
- catastrophic-treatment areas, mortality probabilities, marking, contractor, and cutting workflows;
- administrative forms, mapping codes, quick lists, and Survey123 instructions;
- references, cited standards, other-region comparisons, and external prose;
- figures, cover art, table layout, forms, graphics, maps, and screenshots.

These exclusions remove both rights uncertainty and unsafe operational transfer.

## Anchor and human-suspension boundary

Hazard-tree assessment asks whether a tree or part might fail and affect an exposed target. Anchor assessment asks whether a specific load path, under defined forces and dynamics, is adequate. The source addresses only the first question.

The digest prohibits deriving:

- tree, branch, sling, anchor, hardpoint, or load certification;
- working-load limits or safety factors from a hazard score;
- sling placement, wrap, knot, direction, or component choice;
- an anchor, backup, frame, or suspension design;
- a safe site from a Region 5 failure-zone distance;
- capacity from healthy foliage, corrected lean, or lack of visible defects;
- approval of bondage or human suspension.

Any contemplated tree-supported load remains subject to two separate professional domains:

1. qualified arboricultural assessment of the living tree, root environment, defects, disease, site change, and foreseeable tree-failure modes;
2. qualified rigging or structural assessment of the actual load path, directions, magnitudes, dynamics, components, attachment effects, redundancy, and consequence.

Neither domain substitutes for the other. This corpus provides no load-bearing approval.

## Split and derivation controls

FHP Report RO-22-01 is one canonical source-document unit. Assign it to one split before chunking or deriving QA and keep every descendant in that split. No QA artifact, training projection, evaluation set, validation set, holdout, OOD set, shadow set, benchmark, probe, or dataset was read during the build.

Later QA may test uncertainty, terminology, defect interaction, inspection context, target exposure, site change, monitoring, and professional boundaries. It must not ask for title, author, report number, URL, page, species, disease, score, distance, mitigation, cutting, anchor, load, or embedded-component recall.

## Package

- `CORPUS.md`: 4,201 Unicode-regex words and 4,226 whitespace-delimited words of manually written direct-training Markdown.
- `dispositions.jsonl`: 30 manual inclusion, narrowing, rights, component-audit, and exclusion decisions.
- `manifest.json`: source identity, rights, role, hashes, scope, safety gates, and split rules.
- `source_snapshot/provenance.json`: exact remote integrity, PDF structure, rights review, component audit, evidence audit, and training boundary.
- `tests/test_corpus.py`: offline identity, integrity, anti-universalization, no-mitigation, no-anchor-certification, visual-exclusion, and split-hygiene tests.

The package is deliberately much smaller than the report. It keeps the information that improves hazard reasoning while excluding material that could turn a regional inspection document into a universal tree rating, cutting manual, or human-suspension certification.
