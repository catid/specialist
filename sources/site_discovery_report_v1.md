# Rope source discovery report v1

Initial review at: 2026-07-16T06:04:06Z
Initial batch: `discovery_batch_001`
Latest review at: 2026-07-16T09:17:20Z
Latest batch: `discovery_batch_006`

## Outcome

This first bounded batch manually reviewed 20 candidate websites or official site-hosted resources by opening their actual public pages. After the evidence-quality correction described in batch 005, its decisions are 6 `accept_high_priority`, 9 `accept_targeted_scope`, 2 `defer`, and 3 `reject`. The 15 accepted sources are added to the canonical corpus queue as separate pending extraction jobs. Discovery did not copy page bodies into training data, enter a course, sign in, start a trial, defeat a challenge, or infer instructions from images or video.

All 20 canonical candidate URLs were compared with all 25 entries that were already in `site_corpus_queue_v1.json`; none duplicated an existing queue URL or resource ID. Rope365 was separately sealed as complete with its accepted commit and artifacts. The expanded full-site Crash Restraint scope remains unchanged.

Scores use the approved formula exactly:

`2 × information density + 2 × technical specificity + provenance quality + durability + 2 × novel coverage + safety context − 2 × duplication penalty − 2 × commerce-noise penalty`

The machine-readable ledger contains every required field, all component scores, representative page notes, access status, supported taxonomy categories, decision reasons, and the complete proposed crawl scope.

## Accepted high-priority sources

| Source | Score | Novel coverage | Representative public pages reviewed | Why it was accepted | Corpus boundary |
|---|---:|---:|---|---|---|
| [Kinbaku Today](https://www.kinbakutoday.com/) | 38 | 5 | [source-critical history essay](https://www.kinbakutoday.com/kinbaku-history-memory-and-origin-stories/), journal index | Deep translated, interviewed, archival, and source-critical history of Japanese kinbaku, its people, publications, terminology, and disputed origin narratives. | History, translations, interviews, substantive articles, and text-bearing how-to only; exclude announcements, galleries, comments, account/cart, and media-only detail. |
| [Nawapedia](https://nawapedia.net/index.php?title=Main_Page) | 33 | 5 | [Chimuo Nureki](https://nawapedia.net/index.php?title=Chimuo_Nureki), [Akira Naka](https://nawapedia.net/index.php?title=Akira_Naka) | Structured named-person, chronology, alias, publication, and lineage coverage with print and first-party references. | Substantive article namespace only; exclude edit/history/talk, category shells, venue/event directories, and media-only pages. |
| [Kinbaku Books](https://kokoro-kinbaku.com/) | 34 | 5 | [translation index](https://kokoro-kinbaku.com/translations/), [Urado Hiroshi guide context](https://kokoro-kinbaku.com/2014/09/17/urado-hiroshi-sm-play-you-can-play-sm-1972/) | Rare commissioned English translations and bibliographic research on historical Japanese books, magazines, interviews, performers, and postwar SM culture. | Historical essays, research, and public translations; deduplicate the old WordPress domain and exclude cover-only listings, comments, tags, and untranslated image scans. |
| [La quarta corda / Nawame](https://www.laquartacorda.it/en/) | 37 | 4 | [safety guide](https://www.laquartacorda.it/en/safety-guidelines-for-bondage/), [suspension-line syllabus](https://www.laquartacorda.it/en/all-about-suspensions-in-bondage-and-shibari/), [webbook structure](https://www.laquartacorda.it/en/how-to-use-this-webbook/) | Broad progressive webbook spanning safety, consent, anatomy, terminology, technique, bottoming, suspension lines, hardpoints, and pulley systems. | English educational corpus plus unique untranslated Italian facts; deduplicate translations and never reconstruct video-only steps. |
| [Self Suspension](https://www.selfsuspend.com/) | 41 | 5 | [safety](https://www.selfsuspend.com/safety), [self-suspension uplines](https://www.selfsuspend.com/uplines-for-selfsuspension), [equipment](https://www.selfsuspend.com/supplies/) | The clearest discovered public source for self-suspension risk, critical-line continuity, spotters, jams, rated uplines, hardware, movement, and emergency planning. | Public safety, preparation, equipment, upline, body-mechanics, emergency, and technique pages; exclude galleries, directories, comments, and visual-only instructions. |
| [Helsinki Shibari](https://www.helsinkishibari.com/) | 33 | 4 | [bottom-perspective risks](https://www.helsinkishibari.com/articles/title), [nerve safety](https://www.helsinkishibari.com/articles/nerve-safety-for-rope-bondage), [social safety practice](https://www.helsinkishibari.com/safety-practices) | Bottom-centered fall, nerve, fainting, neck, equipment, and suspension risk guidance paired with operational consent, reporting, confidentiality, accountability, and sober-space policies. | Durable educational and policy pages in English plus unique Finnish facts; exclude calendar, current staff, contacts, and repeated event material. |

## Accepted with targeted scope

| Source | Score | Novel coverage | Representative public pages reviewed | Why it was accepted | Corpus boundary |
|---|---:|---:|---|---|---|
| [The Rope Bottom Guide](https://theropebottomguide.com/) | 40 | 5 | [official version 4.2 PDF](https://theropebottomguide.com/downloads/rope_bottom_guide.pdf) | Dense authored bottom-centered coverage of partner evaluation, responsibility, communication, body knowledge, placement, temperature, risk, and recovery. | Official guide and version metadata only; preserve cautions and structure, without reproducing or inferring photography. |
| [Anatomie Studio London](https://www.anatomiestudio.com/) | 28 | 4 | [education blog](https://www.anatomiestudio.com/blog), [code of conduct](https://www.anatomiestudio.com/code-of-conduct), [accessibility](https://www.anatomiestudio.com/accessibility) | Unusually strong public bottoming, disability, accessibility, pedagogy, consent, performance, and rope-space governance material. | Education/community articles, code, accessibility, nerve safety, and durable history only; exclude tickets, schedules, store, testimonials, galleries, and venue news. |
| [Twisted Windows Bondage Safety](https://www.twistedwindows.com/bondagesafety) | 29 | 4 | [six nerve-injury factors](https://www.twistedwindows.com/bondagesafety/six-contributing-factors), [health considerations](https://www.twistedwindows.com/bondagesafety/health-considerations-for-bondage), [nerve first aid](https://www.twistedwindows.com/bondagesafety/first-aid-for-nerve-damage) | A second body audit retained useful practitioner operational guidance but found mixed citation quality and unsupported health, timing, treatment, and prognosis claims. | Quotation-light attributed digest with a claim-to-citation matrix; quarantine two-finger/capillary-refill heuristics, exact timings, harness-hang claims, drugs/supplements, ice/heat, prognosis, exercise, and uncorroborated medical advice. |
| [The Twisted Monk](https://www.twistedmonk.com/) | 26 | 4 | [rope care](https://www.twistedmonk.com/pages/ropecare), [rated synthetic uplines](https://www.twistedmonk.com/blogs/news/the-joy-of-posh), [partner green flags](https://www.twistedmonk.com/blogs/news/green-lights) | A small valuable core covers fiber-specific maintenance, critical versus secondary lines, rated uplines, dynamic load, redundancy, partner evaluation, and inclusion. | Strict article allowlist; exclude the Shopify catalog, prices, stock, reviews, customer stories, marketing, and volatile commerce. Attribute vendor and guest claims. |
| [RopeLab Online](https://www.ropelab.com.au/) | 40 | 5 | [two-point anchor loads](https://www.ropelab.com.au/two-point-anchor-loads/), [pulley analysis](https://www.ropelab.com.au/pulley-system-analysis/), [mechanical-advantage quiz](https://www.ropelab.com.au/ropelab-quiz-1-mechanical-advantage-2/) | Professional rigging research fills major mechanics gaps: vectors, anchor loads, redirects, friction, pulley efficiency, mechanical advantage, abrasion, testing, and rescue systems. | Public/free human-suspension-relevant mechanics only. Never access members-only content or convert industrial/rescue values into bondage guarantees. |
| [Discover Kinbaku](https://discoverkinbaku.com/en/) | 27 | 4 | [kinbaku and lineage](https://discoverkinbaku.com/en/what-is-kinbaku/), [declared influences](https://discoverkinbaku.com/en/about_us/) | First-party Nureki–Naka–Sugiura lineage perspective, semenawa philosophy, and uncommon somatic resources for bottoms. | Substantive terminology, declared lineage, philosophy, Elements, and somatics only; exclude bookings, galleries, promotion, thin class listings, and untranscribed embeds. |
| [Frozen Meursault](https://www.frozenmeursault.com/) | 36 | 5 | [reference hub](https://www.frozenmeursault.com/references/), [rope anatomy](https://www.frozenmeursault.com/anatomyforrope/), [restraint-removal tools](https://www.frozenmeursault.com/wp-content/uploads/2015/03/Restraint-Removal-Tools.pdf) | A paramedic educator's practical nerve-assessment, anatomy, restraint-removal, and emergency material adds strong professional provenance. | Exact rope anatomy, nerve, removal/cutting, and emergency allowlist; exclude breath/choke, waterboarding, blood, rape play, impact, and other unrelated high-risk topics. |
| [MI Ropes](https://www.miropes.com/) | 26 | 3 | [official 2023 rope-safety handout](https://static1.squarespace.com/static/5fbab2c13b0a9d7ed1c1a22e/t/65348be5aea53543b18b8ff8/1697942504872/Rope%2BSafety%2BMIRopes%2B-%2B2023.pdf) | A compact current community safety cross-check with explicit box-tie, communication, and nerve/circulation cautions. | This one official rope-safety PDF only; exclude unrelated handouts, event information, and its recommendation list as training trivia. |
| [Shibari Studio Berlin Journal](https://www.shibari-studio.com/) | 23 | 3 | [nerve and circulation safety](https://www.shibari-studio.com/shibari-journal/shibari-safety-nerve-risk), journal index | Selected long journal posts give clear risk multipliers, warning signs, function checks, and response protocols, though provenance is lighter and overlap is high. | Manually allowlisted complete safety, consent, bottoming, pedagogy, or performance articles only; exclude services, store, bookings, galleries, portfolios, testimonials, and city SEO. |

## Deferred and rejected candidates

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [Shibari Academy](https://www.shibariacademy.com/) | Defer | 11 | Public prose is extensive but heavily duplicates accepted sources, funnels into paid courses/storefronts, and includes medical and fiber claims requiring specialist adjudication. No enrollment, sign-in, or trial is permitted. |
| [Shibari Safety](https://shibarisafety.com/) | Defer | 20 | Promising structure, but named expertise and visible citations are weak; the site disclaims professional qualification, and a key page returned a challenge rather than readable content. The challenge was not bypassed. |
| [Shibari.ph nerve card](https://www.shibari.ph/wp-content/uploads/2018/07/PDC-nervedamage-EN.pdf) | Reject | 17 | Useful warning signs are mixed with medication, supplement, cooling, and rubbing recommendations that conflict with more cautious cited sources; most unique value is visual. |
| [KinbakuPedia](https://kinbakupedia.com/history/) | Reject | 0 | The reviewed history page is extremely thin, unsourced, grammatically incomplete, and overly certain about a contested hojōjutsu origin narrative. |
| [ShibariNews](https://shibarinews.com/) | Reject | 4 | Anonymous generic articles, high duplication, course marketing, and unsupported massage/anti-inflammatory advice make it inferior to accepted first-party and cited specialist sources. |

## Taxonomy gains and cautions

The historical cluster—Kinbaku Today, Nawapedia, and Kinbaku Books—materially expands `lineage_history_people` and `terminology_cultural_context`, including named relationships, aliases, magazines, publishers, translations, and evidence-quality distinctions. These sources disagree in framing and should remain source-attributed rather than being flattened into one universal origin story.

The body and safety cluster—Twisted Windows, Self Suspension, The Rope Bottom Guide, Helsinki Shibari, Anatomie, Frozen Meursault, and MI Ropes—adds bottom agency, body variability, nerve mechanisms, health negotiation, emergency planning, spotter competence, community accountability, and accessibility. Medical advice must stay attributed and undergo later manual safety review; discovery acceptance does not certify a claim.

The mechanics cluster—Self Suspension, RopeLab, Twisted Monk, and La quarta corda—adds critical-line continuity, rated versus unrated rope, redirects, multi-point anchor forces, friction, mechanical advantage, pulleys, dynamic loading, and redundancy. RopeLab is deliberately cross-domain: its professional rigging physics can explain mechanics, but its numbers and equipment contexts cannot be silently converted into bondage-specific guarantees.

No accepted source removes the need for qualified, in-person instruction or structural evaluation. Discovery did not establish that a particular ceiling, beam, tree, frame, connector, or rope system is safe for human suspension.

## Access and clean-room statement

Filesystem reads for this task were confined to the governing source queues/documentation/test and the new discovery outputs. No existing QA, training projection, manual review, evaluation, heldout, holdout, OOD, shadow, benchmark, or probe artifact was read. Public web review opened candidate-owned pages and official asset URLs, followed openly linked source pages, and respected access gates. No page body was written to a training artifact during discovery.

## Discovery batch 002 outcome

`discovery_batch_002` manually reviewed 17 additional, nonduplicate public sources by opening their actual pages or official documents. It produced 5 `accept_high_priority`, 7 `accept_targeted_scope`, 3 `defer`, and 2 `reject` decisions. The 12 accepted candidates are separate pending jobs in the canonical corpus queue; the queue grows from 40 to 52 resources. Scores use the same formula documented above.

The review emphasized structural hardpoints, tree anchors, quantitative load paths, redirects, dynamic loading, friction, mechanical advantage, lowering/rescue, text-supported knots and lockoffs, Japanese practitioner testimony, qualitative bottoming evidence, and disability adaptation. Cross-domain sources were accepted only for narrowly transferable facts with explicit context. No industrial, arborist, circus, climbing, rescue, manufacturer, or government source was treated as bondage certification.

### Accepted high-priority sources

| Source | Score | Distinct evidence reviewed | Why it was accepted | Suggested extraction worker |
|---|---:|---|---|---|
| [TreeConsult rigging research](https://www.tree-consult.org/downloads.htm) | 43 | Author-hosted research, HSE-funded report, measured-load presentation | Primary evidence for anchor, redirect, friction, dynamic-event, stem-response, and safety-margin mechanics. | `tree_anchor_mechanics_capture` |
| [Petzl Professional Technical Tips](https://www.petzl.com/INT/en/Professional/) | 41 | Manufacturer instructions and setup-specific test results | Rescue/lowering configurations, imperfect sharing, redirect forces, slack, friction, and explicit competence limits. | `rescue_loadpath_capture` |
| [U.S. Bureau of Reclamation Rope Access Guidelines](https://www.usbr.gov/rope/docs/rope_guidelines.pdf) | 40 | Public federal manual | Anchors, deviations, knots, lockoffs, mechanical advantage, friction, lowering/raising, inspection, rescue, and training in one auditable source. | `rescue_loadpath_capture` |
| [DMM Technical Knowledge](https://us.dmmwales.com/blogs/knowledge) | 39 | Authored rigging derivation and manufacturer test table | Worked pulley ratios, practical friction, anchor reaction, sling material, slack loading, and configuration-dependent failure. | `rescue_loadpath_capture` |
| [Arboricultural Association guides](https://www.trees.org.uk/Trees.org.uk/media/Trees-org.uk/Documents/Tech%20Guides/Safety-Guide-3-Form.pdf) | 37 | Two directly public official safety checklists | Tree condition, anchor selection/testing, system compatibility, foreseeable or peak loading, slack/fall control, inspection, and rescue. | `tree_anchor_mechanics_capture` |

### Accepted with targeted scope

| Source | Score | Corpus contribution and boundary | Suggested extraction worker |
|---|---:|---|---|
| [Hilti anchor technical guides](https://www.hilti.com/content/hilti/W1/US/en/business/business/engineering/product-technical-guides.html) | 36 | Mechanical/chemical anchors, concrete/masonry/steel, edge/spacing, embedment, cleaning/curing, corrosion, and failure variables only; exclude catalog and capacity transfer. | `indoor_anchor_engineering_capture` |
| [Animated Knots](https://www.animatedknots.com/) | 34 | Text-supported rescue/climbing friction hitches, releasable lockoffs, tensionless anchors, and figure-eight family; never copy or reconstruct animations. | `knot_lockoff_capture` |
| [Equity Fit to Fly](https://www.equity.org.uk/advice-and-support/health-and-safety/fit-to-fly-a-performers-checklist) | 32 | Performer due diligence for venue evidence, competent installation, whole load path, dynamic load, automation, and rescue; retain UK context. | `indoor_anchor_engineering_capture` |
| [The Senses of Shibari dissertation](https://www.esinem.com/wp-content/uploads/2012/06/SensesShibari3.pdf) | 32 | Transformed, citation-preserving digest of participant ethnography, interviews, body communication, negotiation, materiality, and performance; no chapter republication. | `lineage_lived_experience_capture` |
| [FEDEC Safety and Rigging](https://www.fedec.eu/en/article/205-safety-and-rigging-intents-project) | 31 | Durable circus risk-management, competence, independent-system, inspection, and rescue principles; quarantine dated legal/product claims. | `rescue_loadpath_capture` |
| [Sabukaru practitioner interviews](https://sabukaru.online/articles/bound-to-be-beautiful) | 30 | Concise, quotation-light digest of direct testimony from Tenma Haru and Hina, Kinoko Hajime, and Go Arisue; exclude weak external history/health framing. | `lineage_lived_experience_capture` |
| [Giddy disabled-bondage interviews](https://getmegiddy.com/disability-and-sex/week-3) | 25 | Attributed disabled/chronic-pain experience, body-first positioning, access needs, communication, nonverbal stop signals, pacing, and adaptation; exclude medical and therapeutic generalization. | `accessibility_bottoming_capture` |

Suggested worker names describe independent extraction lanes, not permission to merge sources. Each queued resource still needs its own manifest, citation trail, exclusions, version/access notes, and later manual safety review.

### Deferred and rejected candidates

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [Introduction to Rigging: Aerialist Essentials](https://www.researchgate.net/publication/384293183_Introduction_to_Rigging_Aerialist_Essentials) | `defer` | 41 | Excellent wood/concrete/steel, beam/joist/truss, frame, hardware, and rigging-math coverage, but the author-uploaded book explicitly says all rights reserved and forbids reproduction without written permission. Rights clearance or a documented transformed-use decision is required. |
| [Circus Rescues](https://www.researchgate.net/publication/385710267_Circus_Rescues) | `defer` | 37 | Strong rescue-preplanning and lowering material, but no permissive license is displayed and related books carry explicit reuse restrictions. |
| [Flying Squirrel freestanding rig](https://www.flyingsquirrelconsortium.com/freestandingrig.html) | `defer` | 14 | Both the official page and manual failed direct TLS/server review. Search snippets were not used, and no access bypass was attempted. |
| [Vvolfy aerial-rig manual](https://vvolfy.com/aerial-rig-setup-manual/) | `reject` | 17 | Robots signal `ai-train=no`; independent engineering is absent, and the page uses improvised vehicle/jumping tests and hazardous assurances as support. |
| [Performance of the Real Kinbaku](https://performancereal.pubpub.org/pub/9cqnzv7e/release/2) | `reject` | 15 | Robots signal `ai-train=no`; weak historical, physiological, and therapeutic sourcing plus duplication make the remaining yield unsuitable. |

### Gap changes and remaining work

Tree/outdoor anchor coverage improves substantially through public professional-association checklists and measured arboriculture research. Two indexed Association consultation-draft URLs returned 404 in the final audit and were excluded rather than used as current evidence. Indoor coverage gains mechanical and chemical fastener engineering in concrete, masonry, and steel plus a venue due-diligence checklist. A high-density source covering wood joists, trusses, and varied indoor point construction was found but deliberately deferred on rights grounds, so those topics—and tension/compression ceiling systems in particular—remain discovery priorities.

Petzl, DMM, USBR, FEDEC, and Animated Knots add complementary layers: measured or derived loads, device/configuration-specific behavior, system planning, releasable lowering, rescue governance, and text-supported knot/lockoff construction. Values must remain tied to each source's geometry, equipment, material, test, date, and professional domain.

The cultural and lived-experience additions are evidence-type bounded. Sabukaru is direct journalism, the dissertation is undergraduate participant ethnography, and Giddy is named lifestyle journalism with educator interviews. None should be flattened into universal history, physiology, or practice rules.

### Batch 002 access and clean-room statement

Actual public pages, PDFs, access failures, robots directives, and visible reuse notices were reviewed. Public readability was not treated as automatic permission: two high-yield manuals were deferred for rights clearance, explicit `ai-train=no` signals were honored, and an inaccessible manual was not inferred from search snippets. Accepted copyrighted journalism is limited to concise, attributed factual digests; visual assets and extended quotations are excluded.

Repository inspection remained limited to the governing source-discovery artifacts and contract test. No QA, training projection, manual-review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, or unrelated corpus artifact was read or modified. Discovery wrote no source page body into a training artifact.

## Discovery batch 003 outcome

`discovery_batch_003` manually reviewed 16 additional, nonduplicate sources and the endpoint-specific access and reuse terms needed for each. It produced 4 `accept_high_priority`, 6 `accept_targeted_scope`, 4 `defer`, and 2 `reject` decisions. The ten accepted candidates are separate pending extraction jobs; all rights-limited sources use narrow, attributed scopes and all cross-domain material retains its original operating context.

The most important access correction is that Europe PMC and US PubMed Central are not interchangeable capture endpoints. The two accepted clinical sources have licensed JATS available from the robots-allowed EMBL-EBI Europe PMC API. The Cureus publisher copy of the neuropathy paper prohibits AI-training collection, US PMC disallows crawling, and the Springer copy of the suspension review blocks named AI agents. Those restricted copies are excluded even though the same works can lawfully and technically be acquired from the licensed Europe PMC route.

### Accepted sources

| Source | Decision | Score | Distinct value and boundary |
|---|---|---:|---|
| [ICAR suspension syndrome review](https://europepmc.org/article/MED/38071341) | `accept_high_priority` | 43 | CC BY 4.0 review-level evidence, graded recommendations, and myth correction via EMBL-EBI JATS only; harness/rescue evidence is not silently generalized to rope bondage. JATS binds PMCID PMC10710713, DOI 10.1186/s13049-023-01164-z, and PMID 38071341. |
| [Rope neuropathy study](https://europepmc.org/article/MED/37384078) | `accept_high_priority` | 42 | CC BY 3.0 directly relevant survey evidence via EMBL-EBI JATS only; self-report associations retain denominators, uncertainty, and study limits. JATS binds PMCID PMC10294117, DOI 10.7759/cureus.39588, and PMID 37384078. |
| [Ontario live-performance safety guidelines](https://www.ontario.ca/document/safety-guidelines-live-performance-industry) | `accept_targeted_scope` | 41 | Official performer-flying, rigging, anchors, temporary/outdoor structure, competent-role, inspection, and rescue facts; transformed and jurisdiction-attributed under Crown terms. |
| [NCSF Consent Counts](https://ncsfreedom.org/consent-counts/) | `accept_high_priority` | 41 | Consent policy, explicit prior permission, negotiation, red flags, incident reporting, response teams, confidentiality, trauma response, and survey findings; item-level licenses must be retained. |
| [USFS Rigging for Trail Work](https://www.govinfo.gov/app/details/GOVPUB-A13-PURL-gpo235248) | `accept_high_priority` | 41 | Durable federal manual on knots, bend radius, slings, anchors, friction, blocks, mechanical advantage, inspection, and planning; credited third-party material must be audited and forestry ratings must not transfer to human suspension. |
| [Actsafe performer-flying and rigging bulletins](https://actsafe.ca/resources/performer-flying-and-aerial-stunts/) | `accept_targeted_scope` | 38 | Qualified roles, passive secondary systems, live loading, abrasion, inspection, rehearsal, communication, and rescue; concise attributed facts only. |
| [ICAR rope-connection recommendation](https://www.alpine-rescue.org/articles/10--rope-connections-for-kernmantle-rope-extension-tercom-recommendation-nr-4) | `accept_targeted_scope` | 35 | Revision-controlled context for joining kernmantle rescue ropes; noncommercial editorial digest only and no transfer to natural-fiber rope or human suspension. |
| [Exploring Bondage-Related Injury Risks](https://academic.oup.com/jsm/article/22/Supplement_1/qdaf068.070/8119573) | `accept_targeted_scope` | 32 | Recent two-survey conference abstract; bibliographic metadata and minimal factual digest only, explicitly preliminary and noncausal. |
| [Disability and Bondage](https://enhancetheuk.org/disability-and-bondage/) | `accept_targeted_scope` | 30 | Occupational-therapy prompts for positioning, support, pressure, stability, sensation, grip, pacing, and planning; excludes the unsafe solo-pulley scenario and individualized advice is not universalized. |
| [Healing Experiences in Japanese Rope Bondage Practice](https://www.journal.aleftrust.org/index.php/cstp/article/view/46) | `accept_targeted_scope` | 28 | CC BY 4.0 qualitative methodology and attributed participant experience; no therapeutic-efficacy claim is permitted. |

### Deferred and rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [ITRA rope-rescue documents](https://www.technicalrescue.org/documents/) | `defer` | 41 | A high-value 2024 syllabus was identified, but direct TLS validation fails and the terms do not authorize corpus-scale reuse; require valid transport and written permission. |
| [ANZCOR harness-suspension first aid](https://www.anzcor.org/home/first-aid/guideline-9-1-5-first-aid-management-of-harness-suspension-trauma) | `defer` | 37 | Strong clinical guidance, but its copyright page explicitly bars incorporation into another work without written permission; licensed ICAR evidence covers its central myth correction. |
| [Carleton rope-bondage thesis](https://carleton.scholaris.ca/items/51e42f12-3cf6-400a-b101-5db266aaa109) | `defer` | 34 | Institutional rights allow research/teaching and link sharing but bar adaptation/derivatives and commercial use; explicit permission is needed. |
| [Kanna/Kagura bakushi biographies](https://kanna-kagura.blogspot.com/2020/04/biographies-of-kinbakushi-of-japan.html) | `defer` | 27 | Rare translated lineage testimony, but rights are layered across the interview, translation, annotations, magazine, and later book. |
| [Durham comparative rope thesis](https://etheses.durham.ac.uk/id/eprint/15763/) | `reject` | 32 | Repository robots explicitly block named OpenAI/AI-training agents; no body was opened and no bypass was attempted. |
| [Cleveland Clinic suspension syndrome](https://my.clevelandclinic.org/health/diseases/suspension-syndrome) | `reject` | 29 | Reuse terms prohibit storing, repackaging, or editing, and a stronger CC BY review already covers the topic. |

### Endpoint and evidence controls

| Work or host | Permitted capture route | Explicitly excluded route or use |
|---|---|---|
| Rope neuropathy paper, PMCID PMC10294117 | `www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML`, CC BY 3.0 | Cureus publisher AI-training collection; US PMC crawling |
| ICAR suspension review, PMCID PMC10710713 | `www.ebi.ac.uk/europepmc/webservices/rest/PMC10710713/fullTextXML`, CC BY 4.0 | Springer and US PMC crawler-restricted copies |
| Ontario | Transformed, attributed facts with jurisdiction/version | Wholesale Crown-text republication or universal regulatory claims |
| NCSF | Per-item license handling; CC BY-SA items under their terms | Assuming the whole mixed-license hub has one license |
| USFS/GovInfo | Federal text plus official MODS metadata | Unreviewed third-party credited diagrams, tables, excerpts, or standards |
| ICAR TERCOM | Narrow noncommercial editorial factual digest | Expressive/commercial reuse without permission; domain transfer |

### Batch 003 access and clean-room statement

Every decision distinguishes discovery from extraction. Public availability was not treated as permission, and alternate copies were not used to evade endpoint-specific robots or terms. No challenge, paywall, authentication, invalid TLS path, or crawler restriction was bypassed. Discovery used only governing source-discovery files and public web evidence; no existing QA, training projection, manual review, evaluation, heldout, holdout, OOD, shadow, benchmark, or probe artifact was read. No source body was written into a training artifact.

## Discovery batch 004 outcome

`discovery_batch_004` manually reviewed 12 additional, nonduplicate public
sources, including the actual body or official metadata, current access policy,
and visible reuse terms. It produced 1 `accept_high_priority`, 5
`accept_targeted_scope`, 5 `defer`, and 1 `reject` decisions. The six accepted
candidates expand the canonical corpus queue from 62 to 68 resources.

This batch concentrates on first-party Japanese practitioner testimony and
lineage, named rope-school progressions, knot/friction and upline curriculum
maps, and rope construction/care. Discovery acceptance remains only permission
to perform a bounded extraction review; it does not certify medical advice,
historical claims, tying instructions, hardware, hardpoints, or suspension
systems.

### Accepted sources

| Source | Decision | Score | Distinct value and extraction boundary |
|---|---|---:|---|
| [Heartland Kinbaku public guides](https://www.heartlandkinbaku.com/resources) | `accept_high_priority` | 36 | Explicitly shareable-with-attribution beginner, bottom, and physical-preparation guides plus a 20-class progression; every medical, anatomy, exercise, timing, and load claim must be independently adjudicated or quarantined. |
| [Osada-ryu primary writings](https://www.osada-ryu.com/?page_id=184) | `accept_targeted_scope` | 34 | Quotation-light, fully attributed digest of first-person Akechi, Nureki, Yukimura, and Osada-ryu testimony; speaker, interviewer, date, framing, recollection, and later synthesis remain distinct. |
| [Devil Mask Studio curriculum](https://devilmaskstud.io/preflight/) | `accept_targeted_scope` | 30 | Detailed prerequisite map for body-adapted harnesses, uplines, lockoffs, force vectors, ground tests, go/no-go reasoning, bamboo, hashira, inversions, and transitions; descriptions are not reconstructed tutorials. |
| [Rope Study progression](https://ropestudy.com/) | `accept_targeted_scope` | 29 | Current 101–401 competency progression and partnered-practice pedagogy; capture only normally browsable pages during the incomplete May 2026 rebuild, without bypassing ModSecurity. |
| [Hajime Kinoko official profile](https://shibari.jp/en/profile/index.html) | `accept_targeted_scope` | 24 | Narrow first-party digest of declared teachers, influences, artistic statement, Ichinawa-kai, and durable milestones; lineage is labeled self-described and promotion is excluded. |
| [Go Arisue official profile](https://www.arisuego.com/home/%E6%9C%89%E6%9C%AB%E5%89%9B%E3%81%AE%E4%B8%96%E7%95%8C/) | `accept_targeted_scope` | 19 | Stable Japanese first-party biography and role chronology from a provisional 2026 relaunch; image, event, store, and forthcoming paid content is excluded. |

### Deferred and rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [USDA National Tree Climbing Guide](https://research.fs.usda.gov/treesearch/49660) | `defer` | 39 | The official 88-page manual is highly relevant, but its current host publishes a universal robots prohibition and the former official path is 404; no mirror or workaround was used. |
| [Ritsumeikan Nureki biography/interview study](https://ritsumei.repo.nii.ac.jp/records/2003261) | `defer` | 37 | Exceptional source-critical scholarship, but the repository displays no reuse license and university policy requires prior permission for duplication, modification, or distribution. |
| [Early *Kitan Club* bibliography](https://www.jstage.jst.go.jp/article/antitled/4/0/4_273/_article/-char/ja) | `defer` | 33 | Public metadata is retained, but ordinary copyright applies and J-STAGE robots disallow the `*_pdf` capture route; body use requires both a permitted endpoint and permission. |
| [RebornRopes technical guides](https://www.rebornropes.com/pages/shibari-how-to-jute-rope-care) | `defer` | 25 | Broad materials and friction coverage is mixed with disputed knot equivalences, unsafe single-column framing, unsupported respiratory claims, uncited load claims, and heavy commerce; specialist adjudication is required first. |
| [Full Circle beginner handout](https://www.fullcirclekink.com/pdfs/Bondage%20for%20Beginners%20handout.pdf) | `defer` | 21 | Search indexing suggests a formerly CC BY-SA handout, but the official asset is 404 and the source body and license cannot now be revalidated; no cache, archive, or third-party copy was used. |
| [German Wikibooks Shibari manual](https://commons.wikimedia.org/wiki/File:Shibari.pdf) | `reject` | 21 | Despite CC BY-SA 3.0, the incomplete legacy manual contains hazardous or unsupported rope processing, hardware, safety-factor, neck-adjacent pattern, gender, and origin guidance. |

### Batch 004 evidence, rights, and quality controls

The first-party lineage cluster intentionally preserves claim type. Hajime
Kinoko and Go Arisue profiles are self-descriptions, while the Osada-ryu pages
contain layered interviews, interviewer narration, repost history, recollection,
and school framing. These sources may correct thin secondary summaries, but
they do not independently prove every date, relationship, or contested origin.

Heartland, Devil Mask Studio, and Rope Study provide curriculum architecture:
prerequisites, sequencing, competency language, partnered practice, harness
adaptation, uplines, load reasoning, and decisions to stop. Their maps can guide
later educational organization without inventing technique from images or
marketing descriptions. Heartland's explicit sharing statement supports
capture of its named public documents, but practitioner claims still require
claim-level evidence review.

Permissive licensing alone was not treated as a quality signal. The German
Wikibooks manual was rejected after body review, and RebornRopes was deferred
despite its accessible technical library. Conversely, strong relevance did not
override access or rights: USDA robots, J-STAGE's PDF restriction, Ritsumeikan's
reuse policy, and Full Circle's missing official asset all block corpus capture.

### Batch 004 access and clean-room statement

Repository inspection remained limited to the governing source-discovery
ledger, queue, report, and contract test. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, or source
corpus artifact was read. Public-web review respected current robots, ordinary
browsing boundaries, authentication, licenses, and rightsholder endpoints. No
challenge, cache, archive, third-party mirror, or access workaround was used,
and no reviewed page body was written into a training artifact.

## Discovery batch 005 outcome

`discovery_batch_005` manually reviewed 12 additional source candidates by
opening the actual public body, official metadata, or live failure endpoint and
checking current access and reuse boundaries. It produced 2
`accept_high_priority`, 4 `accept_targeted_scope`, 4 `defer`, and 2 `reject`
decisions. The six accepted candidates expand the canonical corpus queue from
68 to 74 resources.

This batch prioritizes primary lineage testimony, source-critical history,
licensed forensic evidence, structural-wood engineering, progressive rope
pedagogy, and narrowly recoverable rope-maker observations. It also rejects a
dead URL index and unsafe community class notes rather than treating page titles
or public availability as training value.

### Accepted sources

| Source | Decision | Score | Distinct value and extraction boundary |
|---|---|---:|---|
| [BDSM fatality literature review](https://europepmc.org/article/MED/34383118) | `accept_high_priority` | 42 | CC BY 4.0 forensic review binding PMCID PMC8813685, DOI 10.1007/s00414-021-02674-0, and PMID 34383118. Preserve the 17-case publication-derived denominator, selection bias, absence of an incidence denominator, category distinctions, and questionable inferences; omit sensational or instructional case reconstruction. |
| [USDA Wood Handbook](https://www.govinfo.gov/app/details/GOVPUB-A13-PURL-gpo158673) | `accept_high_priority` | 40 | Official 2021 engineering reference for wood variability, grade, moisture, duration, connectors, engineered wood, deterioration, and structural assumptions. Capture only relevant federal text after third-party-rights audit; it does not certify a ceiling, beam, joist, tree, connector, hardpoint, bondage system, or human-suspension use. |
| [Shibaru history series](https://shibaru.life/category/history/) | `accept_targeted_scope` | 33 | Four signed Traditional-Chinese practitioner essays connect hojojutsu, Kabuki, ukiyo-e, ero-guro, and the postwar *Kitan Club* network. Produce a quotation-light source-labeled synthesis, not a full translation; exclude images and historical instruction and quarantine unsupported bruising, slippage, and safety claims. |
| [Yukimura-ryu official archive](https://yukimura-ryu.com/) | `accept_targeted_scope` | 32 | Rare Yukimura, family, and student testimony plus named school progression. Preserve first-person/recollection/interviewer claim types and the unresolved 2006-versus-2007 formal-student date instead of flattening the conflict. |
| [Willcat / Tension curriculum](https://willcatropes.art/) | `accept_targeted_scope` | 26 | Concrete prerequisites, assessment, two-sided fundamentals, integration labs, and readiness progression before uplines or load. Store only an attributed pedagogy map; exclude unsupported history, anatomy, body-reading, absolute safety, construction, and promotion. |
| [ESINEM rope-maker articles](https://www.esinem.com/category/articles/hardware/) | `accept_targeted_scope` | 21 | A small unique core of firsthand observations on lay, strand imbalance, high-stranding, tangling, and handling. Cross-check cordage terminology and exclude product claims, load/size recommendations, anatomy, gote safety, assumed analogies, tutorials, and sales. This is distinct from the already queued ESINEM-hosted dissertation. |

### Deferred and rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [Rope Bondage and Affective Embodiments](https://revistas.udistrital.edu.co/index.php/CORPO/article/view/14228) | `defer` | 32 | Valuable comparative fieldwork and cultural analysis, but CC BY-NC-ND 4.0 does not license the intended derivative training Markdown; retain DOI/metadata while seeking author/editor permission. |
| [Akechi Kanna official Note essay](https://note.com/nawashi_kanna/n/nbce4efeb3aad) | `defer` | 25 | Useful first-person lineage and practice perspective, but Note's terms retain creator copyright and restrict reproduction, publication, modification, and adaptation without consent. |
| [Tanaka globalization abstract](https://cir.nii.ac.jp/crid/1390282680688408192) | `defer` | 22 | CiNii metadata is durable, but `Abstract License Flag: Disallowed` and J-STAGE's `*_pdf` robots rule leave no authorized body-capture route. |
| [Rope Bondage the Smart Way](https://www.ropeconnections.com/wp-content/uploads/2018/05/Rope-Bondage-the-Smart-Way-Pete-Riggs.pdf) | `defer` | 20 | The 345-page author-hosted manual expressly prohibits reproduction and retrieval-system storage without permission, and its self-qualified, dated body needs a complete safety and duplication audit. |
| [RopeBite Pittsburgh](https://ropebitepgh.com/) | `reject` | 5 | Heavy duplication and video dependence are compounded by ibuprofen/B12 advice and instructions to boil, gas-flame singe, and oven-bake treated rope and paper towels. |
| [Remedial Ropes WordPress archive](https://remedialropes.wordpress.com/) | `reject` | 2 | The landing page is under construction and all six promised article bodies return 404. URL titles, feeds, snippets, archives, and mirrors are not corpus knowledge; the successor Twisted Windows source is already separately reviewed. |

### Corrections and strengthened evidence gates

The Europe PMC identity audit now checks every queued MED route against the
authorized EMBL-EBI JATS rather than trusting an index number. The three bound
records are:

| Work | Required identity binding | Authorized full-text route |
|---|---|---|
| Rope-bondage neuropathy survey | PMCID PMC10294117; DOI 10.7759/cureus.39588; PMID 37384078 | `www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML` |
| ICAR suspension-syndrome review | PMCID PMC10710713; DOI 10.1186/s13049-023-01164-z; PMID 38071341 | `www.ebi.ac.uk/europepmc/webservices/rest/PMC10710713/fullTextXML` |
| BDSM fatality review | PMCID PMC8813685; DOI 10.1007/s00414-021-02674-0; PMID 34383118 | `www.ebi.ac.uk/europepmc/webservices/rest/PMC8813685/fullTextXML` |

The prior ICAR queue PMID `38081341` was corrected to `38071341` everywhere,
and regression coverage rejects the stale identifier. The earlier rope
neuropathy identity remains corrected to PMID `37384078`, not `37324199`.

Twisted Windows was re-audited after the dead Remedial Ropes predecessor was
found. Its score is reduced from 40 to 29 and its decision changes from
`accept_high_priority` to `accept_targeted_scope`. A capture worker must build a
claim-to-citation matrix and quarantine unsupported exact onset/recovery times,
the two-finger rule, capillary-refill heuristics, harness-hang timing, drugs or
supplements, ice/heat treatment, exercise, prognosis, and other medical advice.
This retains useful community operational knowledge without laundering
practitioner synthesis into clinical fact.

### Batch 005 access and clean-room statement

Repository inspection remained restricted to the governing discovery contract,
candidate ledger, corpus queue, report, and discovery tests. No QA, training
projection, manual review, evaluation, heldout, holdout, OOD, shadow,
benchmark, probe, or existing source-corpus artifact was read. Public-web review
used normal public pages and official endpoints, respected robots and
rightsholder terms, and did not bypass a login, purchase, challenge, missing
body, license, or platform restriction. No reviewed page body was copied into a
training artifact during discovery.

## Discovery batch 006 outcome

`discovery_batch_006` manually reviewed 11 additional candidates by opening
the actual public body where access and rights allowed, or only official
metadata and governing terms where they did not. It produced 0
`accept_high_priority`, 3 `accept_targeted_scope`, 5 `defer`, and 3 `reject`
decisions. The three accepted candidates expand the canonical corpus queue from
74 to 77 resources.

This batch adds two complementary curriculum architectures and an official
museum chronology. It also records unusually valuable but unusable leads rather
than treating visibility as permission: House Cordee asks readers not to copy
its articles; the ESTA standards require a restrictive EULA; Edinburgh blocks
this agent class; and the Colombian ergonomics thesis is NoDerivatives.

### Accepted sources

| Source | Decision | Score | Distinct value and extraction boundary |
|---|---|---:|---|
| [Edo-Tokyo Museum Seiu Ito record](https://www.edo-tokyo-museum.or.jp/en/s-exhibition/ito-seiu/) | `accept_targeted_scope` | 30 | Official chronology and bibliography for Seiu Ito: born in Asakusa in 1882, theater/sign-painting and newspaper work, historical-custom research, seme-e from his mid-thirties, and death in 1961 at age 78. Create only an original factual digest; copy no prose or images, and do not label him the “father” of modern kinbaku or infer a direct technique lineage. |
| [Temple New York core curriculum](https://www.templenewyork.org/core-rope-curriculum) | `accept_targeted_scope` | 26 | Concrete Level 0–4 architecture from community care and consent through knots/frictions, body bamboo, harness reasoning, partials, uplines, static/dynamic suspension, and collaborative labs. Capture the progression, not implied instructions or certification. |
| [VoxBody core curriculum](https://www.voxbody.com/core-curriculum) | `accept_targeted_scope` | 24 | A deliberately slow, roughly 14-month sequence in which both partners progress from floorwork through Fundamentals, Progressions, Ascent, and Onward. Preserve inclusive/non-ryu pedagogy while excluding schedules, promotion, unsupported readiness heuristics, and any certification claim. |

### Deferred sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [ESTA E1.43 and E1.56 standards](https://tsp.esta.org/tsp/documents/published_docs.php) | `defer` | 43 | The official index identifies the strongest current performer-flying and facility-support-point standards found, but the EULA grants single-computer, non-transferable use and prohibits merging, adaptation, translation, modification, network use, or transfer. Neither standard was downloaded or accepted; seek a separate written training license. |
| [Binding Practice](https://digicoll.lib.berkeley.edu/record/265621) | `defer` | 35 | The 77-page architecture thesis has rare spatial, hardpoint, material, friction, sanitation, thermal, and pedagogy ideas, but Copyright © 2022 and speculative engineering require author permission plus qualified structural review. |
| [House Cordee bottoming advice](https://www.housecordee.com/bottoming-advice) | `defer` | 32 | Nineteen bottom-centered articles cover risk profiling, communication, negotiation, injury planning, documentation, emotion, drop, and performance, but each explicitly says not to copy it. Permission and clarification of the Peter Martin versus Mya/Fox attribution are required. |
| [Edinburgh holding-of-rope thesis](https://era.ed.ac.uk/items/713fab98-99f8-4b08-953b-a4742d4cad29) | `defer` | 32 | Public metadata identifies a 2022 DClinPsychol autoethnography, but repository robots globally disallow GPTBot and ChatGPT-User and ordinary copyright applies. The body was not accessed; a single practitioner narrative cannot establish therapeutic efficacy. |
| [UAN shibari ergonomics thesis](https://redcol.minciencias.gov.co/Record/UAntonioN2_b4e4a4e542aef2d96adb592e04f7d0ff) | `defer` | 29 | National metadata marks the 2021 Spanish undergraduate thesis CC BY-NC-ND 4.0. Retain metadata only while seeking author/university permission; NoDerivatives does not authorize translation or transformed Markdown, and technical claims need review. |

### Rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [Repeated compression radial neuropathies abstract](https://www.sciencedirect.com/science/article/pii/S1388245719306935) | `reject` | 20 | This ordinary-copyright single-case conference abstract overgeneralizes prevalence and is superseded by the already queued CC BY case article. Retain its DOI and unresolved 95.3-versus-77.3-percent conduction-block discrepancy only. |
| [LB Shibari Dojo class resources](https://lbshibaridojo.com/class-resources/) | `reject` | 16 | Anonymous worksheets mix useful reflection prompts with neck-rope and breath-control-with-rope interests, pre-scene stretching, and timed endurance/bodyweight “readiness” scores. A partner-sharing note on one file is not a corpus license. |
| [Nuts and Bolts Rope Type Report](https://www.alltiedupsandiego.com/wp-content/uploads/2020/10/Nuts-and-Bolts-Rope-Type-Report-v2-oct-22-2020a.pdf) | `reject` | 2 | Manual review showed a commercial personality-quiz result and course lead magnet, not technical facts about rope type, fibers, knots, friction, rigging, or safety. Its search title and URL are not knowledge. |

### Batch 006 rights, evidence, and safety controls

Public visibility was again separated from reuse permission. Discovery did not
accept the ESTA EULA, download the standards, access Edinburgh's thesis body,
transform the NoDerivatives UAN thesis, copy House Cordee, or infer useful
content from a search snippet. The Berkeley thesis remains a permission lead,
not an engineering source for certifying a human-suspension point.

The two accepted studio pages are pedagogical maps only. Class names and
prerequisites can help organize knowledge, but they are not construction steps,
clinical evidence, or proof of learner competence. The museum page supplies
institutional chronology, not tying authority or proof of every later lineage
claim. Rejected material cannot become QA, URL-trivia, safety advice, or a
backdoor source through a cache or mirror.

### Batch 006 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, and discovery tests. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, or source
corpus artifact was read. Public-web review used ordinary public routes,
respected current robots and rightsholder terms, and did not bypass a login,
purchase, challenge, EULA, missing body, license, or crawler restriction. No
reviewed page body was copied into a training artifact during discovery.
