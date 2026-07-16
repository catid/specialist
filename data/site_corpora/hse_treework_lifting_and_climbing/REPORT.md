# HSE treework lifting and climbing corpus report

## Outcome

This package adds a 3,605-whitespace-word, non-Q&A Markdown synthesis of current Health and Safety Executive guidance on arboricultural lifting equipment, aerial tree work, tree climbing, work at height, and lifting-equipment examination. It preserves UK occupational scope and focuses on system boundaries, planning, supervision, competence, anchor and line roles, ground communication, inspection layers, stop-work judgment, and rescue readiness.

The corpus does not certify a tree, branch, tree union, anchor, hardpoint, rope, sling, backup, bondage setup, or human suspension. It expressly rejects presenting HSE occupational treework procedure as a bondage or recreational-suspension standard.

Eight HTML pages were reviewed. Six contributed bounded substantive concepts, the LOLER hub was retained only as routing provenance, and the HSE copyright page was used only for rights review. Raw HTML and normalized extraction are not direct-training artifacts and are not retained.

## Page-level source integrity

All pages returned HTTP 200 as `text/html` without redirection on 16 July 2026.

| Page | Raw bytes | Raw SHA-256 | Last-Modified | Training treatment |
|---|---:|---|---|---|
| Lifting equipment in arboriculture | 46,310 | `0cf03b91b2d1aff23902a3bb4879c0ffc5c4023c76d36fb39760748d821fb0f8` | 24 June 2026 19:00:57 GMT | Manually transformed |
| Aerial work: Overview | 39,764 | `1130f6592362623506ccb03e5713ff1697509f7ac429820607ca8f1b4354a06c` | 2 July 2026 13:24:28 GMT | Manually transformed |
| Tree climbing | 41,485 | `5d090257d25ad947c242ce2033a1efa20f8d43613c125fa36f1f53800cb8cfd1` | 24 June 2026 19:00:58 GMT | Manually transformed |
| LOLER hub | 34,572 | `b0291c079851a9c0509db36afcde82e0639d6570972c067b9d6c74d56662e633` | 24 June 2026 19:01:53 GMT | Routing provenance only |
| Thorough examinations and inspections | 47,486 | `ac6b66cdc1f6ee505703f9fff7a2b081254c94977c7cab8f972c1af46eb9382e` | 24 June 2026 19:01:57 GMT | Manually transformed |
| Working at height: Overview | 40,112 | `1bd19f428053cac56338d1c89e0035d9af7558ec4cc2f654c8eca6ff889e4543` | 24 June 2026 19:00:59 GMT | Manually transformed |
| Inspecting fall arrest equipment made from webbing or rope | 36,865 | `2074fd759c0fc828883ffdcc3e54ceac17fb6030ab8573b359f85848979dd5b2` | 24 June 2026 18:56:56 GMT | One landing-page principle retained |
| Copyright | 39,349 | `8cdc470fc361541b29a3fac77c27a93f80eda59cd480eed45bc975d76b522849` | 24 June 2026 18:43:43 GMT | Rights provenance only |

The total raw retrieval was 325,943 bytes. A deterministic audit-only text normalization produced 28,162 bytes and 4,454 words across the eight main-content containers. Per-page normalized hashes and counts are recorded in provenance. That extraction was used to navigate and verify claims; none was copied into `CORPUS.md`.

Exact URLs, retrieval timestamps, raw and normalized digests, and page treatments are in `source_snapshot/provenance.json`.

## Rights

The HSE copyright page says website information is Crown-owned and Crown-copyright-protected unless otherwise indicated and allows reuse of Crown material under the Open Government Licence v3.0. It provides the preferred acknowledgement used in the corpus: “Contains public sector information published by the Health and Safety Executive and licensed under the Open Government Licence.”

The same page warns that some images, illustrations, and multimedia may not be Crown-owned and says the HSE logo requires prior written permission. The corpus therefore applies OGL only to eligible substantive Crown text. This is not a blanket conclusion about linked or embedded components and is not a legal opinion.

Excluded rights components include:

- HSE logo and agency marks;
- header, footer, navigation, breadcrumbs, search, and site chrome;
- shared inline back-to-top SVG icons;
- all images, illustrations, video, and multimedia;
- linked PDFs and archived guidance;
- external industry code, technical guide, learning products, and qualification providers;
- legislation, standards, and third-party pages;
- copyright contact details and print-product boilerplate.

No linked component was fetched or retained.

## Media and link audit

The eight substantive main-content regions contained no content `img`, `video`, or `iframe` tag. Seven pages contained the shared back-to-top SVG inside the broad main container; it is site chrome and is excluded. The HSE logo sits in the site header outside substantive content and is also excluded.

The tree-climbing page links to one external training video, an industry code reader, association learning, HSE research PDFs, a chainsaw publication, and legislation. The aerial-work page links external qualification bodies and archived material. The examination page links legislation, a standard provider, and HSE publications. The INDG367 landing page links its PDF. None was accessed.

This audit matters because a link on an HSE page does not place the linked work under the source page's OGL treatment or make its operational instructions part of the accepted corpus.

## Manual content decisions

The 33 disposition records contain 20 included or narrowed decisions and 13 exclusions. The source pages were reviewed section by section, and linked content was reviewed at access and disposition level.

The corpus retains:

- planned, appropriately supervised lifting work;
- risk assessment, hierarchy, competent people, suitable equipment, inspection, maintenance, emergency planning, and rescue;
- HSE's statement that a load includes a person;
- supporting, fixing, and anchoring attachments inside the lifting-system boundary;
- tree anchor points as lifting-equipment components in the page's arboriculture framework;
- professional attention to species, age, condition, disease, and foreseeable loading;
- two load-bearing anchors where possible, distinct working and safety lines, separately anchored line roles, and supplementary-anchor swing context;
- common-mode reasoning added as a safety gate against assuming independence from quantity;
- ground-team planning, continuous communication, line housekeeping, exclusion control, observation, workload, and stop-work role;
- daily competent checks, written weekly high-wear rope inspection records, interim inspection, maintenance, thorough examination, and traceable records;
- defect communication, withdrawal, stop-and-reassess, and rescue readiness.

It excludes operational climbing, anchor construction, line connection, device use, equipment marking, exact ratings, exact legal intervals, regulatory forms, enforcement procedure, brand or device endorsement, and rescue technique.

## System boundary

The arboriculture page defines lifting equipment to include equipment that lifts or lowers a load and attachments that anchor, fix, or support it. It also says that the load includes a person and that tree anchor points used for rope access, positioning, or rigging form part of the lifting equipment.

The corpus turns this into a bounded system lesson:

- a person cannot be omitted from the load case;
- an anchor cannot be treated as scenery outside the system;
- material-handling suitability does not automatically mean person-support suitability;
- a component marking does not rate an unmarked tree or the complete system.

No safe working load, rating, marking rule, or safety factor is reproduced. HSE's classification does not provide a field calculation for branch capacity.

## Anchor and line-role cleanup

The tree-climbing page discusses two load-bearing anchors where possible, work-positioning backup, and separately anchored working and safety lines in rope-access work. The arboriculture and climbing pages discuss supplementary anchors in relation to work position and pendulum swing.

The corpus preserves functions but blocks three common overclaims:

1. Two branches are not necessarily independent if they share a weak union, stem, root plate, connector, or attachment.
2. A supplementary anchor that changes geometry is not automatically a full-strength backup.
3. A formal occupational exception to a second line is not a recreational single-line instruction.

The direct-training document contains no anchor-selection method, climbing route, line connection, ascent, descent, changeover, connector, knot, harness, fall-arrest, or single-line operating procedure.

## Planning, ground roles, and reassessment

HSE places planning, supervision, competence, communication, and rescue around the equipment. The aerial-work page makes the ground team an active participant: ground workers plan with the climber, understand the task, maintain communication, watch and anticipate, manage line condition and position, maintain exclusion controls, and share workload.

Line housekeeping retains the HSE concerns about knots, kinks, tangles, loose wood, machinery, vehicles, equipment, obstructions, and the public. The source warning not to wrap a working rope around the body for extra grip is preserved without teaching a hauling or friction method.

The core changing-condition rule is also preserved: continue assessing the operation, modify the plan and risk assessment when necessary, and stop and reassess when unsure. The digest adds examples of what can invalidate assumptions—anchor or line-route change, visible tree change, line damage, communication loss, worker or rescuer unavailability, public entry, weather, access, or equipment quarantine—without inventing a go/no-go score.

## Inspection layers and records

The arboriculture page calls for competent daily inspection of the main climbing rope and associated equipment, trained daily pre-use checks, and a written weekly inspection record for high-wear ropes. The broader examination page distinguishes ordinary checks and inspections from a systematic, detailed thorough examination by a competent person.

The digest keeps four layers distinct:

- pre-use check;
- interim inspection;
- maintenance;
- thorough examination with a written report.

It retains competence as practical and theoretical knowledge plus experience sufficient to detect weakness and judge its importance. It also retains independence and impartiality: a routine maintainer should not simply validate their own maintenance work.

Exact statutory examination intervals, deadlines, test methods, report-field counts, enforcement reporting, and record-retention periods are excluded. They are time-sensitive legal details outside the corpus's systems purpose.

The HSE page's dangerous-defect control is retained in general form: communicate the defect and withdraw or quarantine the equipment until the risk is addressed. Records link the item, examiner, date, scope, finding, decision, restriction, follow-up, and next review; they do not add physical strength.

## Rescue boundary

The HSE aerial-work page describes at least two people during aerial work and an available, competent, equipped ground rescuer. Other source pages require a reliable means of rescue, trained tree-rescue personnel at the work site, task-specific rescue training, and emergency planning.

The corpus preserves readiness before ascent: staffing, competence, equipment, communication, access, and foreseeable casualty position need consideration before the emergency. It contains no steps for reaching, transferring, lowering, disconnecting, treating, or handing over a casualty.

Tree rescue is an occupational discipline. Its staffing and procedure cannot be presented as a sufficient bondage emergency plan.

## Cross-domain gate

The corpus prohibits:

- certification of a tree, branch, union, anchor, hardpoint, rope, sling, backup, bondage setup, or human suspension;
- capacity inference from species, age, appearance, health, condition, or absence of obvious disease;
- assuming two anchors or two lines are sufficient without capacity and common-mode assessment;
- assuming a supplementary point is independent or full strength;
- copying treework climbing, rescue, or equipment configuration into recreational suspension;
- treating HSE treework as a bondage standard;
- using the digest as legal advice beyond the current UK page scope.

Domain-specific competent assessment, governing standards, manufacturer information, and the actual load path continue to control. The corpus supplies systems reasoning, not load-bearing approval.

## Split and derivation controls

All eight pages form one HSE source family for split purposes. Assign them and every derivative to one split before chunking or QA generation. No QA artifact, training projection, evaluation set, validation set, holdout, OOD set, shadow set, benchmark, probe, or dataset was read during the build.

Later QA may test system boundaries, roles, common-mode reasoning, inspection layers, communication, reassessment, and rescue readiness. It must not ask for URL, title, retrieval date, regulation number, equipment rating, statutory interval, device, brand, knot, technique, legal citation, or source order.

## Package

- `CORPUS.md`: 3,592 Unicode-regex words and 3,605 whitespace-delimited words of manually written direct-training Markdown with 28 HSE page-label citations.
- `dispositions.jsonl`: 33 manual include, narrow, rights, media, linked-content, and exclusion decisions.
- `manifest.json`: source family, page integrity, rights, role, hashes, scope, cross-domain gates, and split rules.
- `source_snapshot/provenance.json`: page-level retrieval and normalization digests, network boundary, OGL review, component and link audit, evidence audit, and training boundary.
- `tests/test_corpus.py`: offline page-integrity, OGL, media-exclusion, system-boundary, inspection-layer, ground-role, rescue, no-device, no-legal-overreach, cross-domain, and split-hygiene tests.

The result is information-dense but deliberately non-operational. It teaches how HSE places people, anchors, lines, competence, communication, inspection, change, and rescue inside one occupational system while refusing to certify any tree-supported or recreational suspension setup.
