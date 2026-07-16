# Rope source discovery report v1

Initial review at: 2026-07-16T06:04:06Z
Initial batch: `discovery_batch_001`
Latest review at: 2026-07-16T08:02:21Z
Latest batch: `discovery_batch_003`

## Outcome

This first bounded batch manually reviewed 20 candidate websites or official site-hosted resources by opening their actual public pages. Its decisions were 7 `accept_high_priority`, 8 `accept_targeted_scope`, 2 `defer`, and 3 `reject`. The 15 accepted sources are added to the canonical corpus queue as separate pending extraction jobs. Discovery did not copy page bodies into training data, enter a course, sign in, start a trial, defeat a challenge, or infer instructions from images or video.

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
| [Twisted Windows Bondage Safety](https://www.twistedwindows.com/bondagesafety) | 40 | 5 | [six nerve-injury factors](https://www.twistedwindows.com/bondagesafety/six-contributing-factors), [health considerations](https://www.twistedwindows.com/bondagesafety/health-considerations-for-bondage), [nerve first aid](https://www.twistedwindows.com/bondagesafety/first-aid-for-nerve-damage) | Practitioner-authored safety material with medical citations, explicit uncertainty, adaptations, emergency reasoning, and both top and bottom perspectives. | Complete safety hub plus directly relevant rope/hardpoint pages; exclude events, performances, sales/logistics, unrelated kink, and migrated Remedial Ropes duplicates. |
| [Self Suspension](https://www.selfsuspend.com/) | 41 | 5 | [safety](https://www.selfsuspend.com/safety), [self-suspension uplines](https://www.selfsuspend.com/uplines-for-selfsuspension), [equipment](https://www.selfsuspend.com/supplies/) | The clearest discovered public source for self-suspension risk, critical-line continuity, spotters, jams, rated uplines, hardware, movement, and emergency planning. | Public safety, preparation, equipment, upline, body-mechanics, emergency, and technique pages; exclude galleries, directories, comments, and visual-only instructions. |
| [Helsinki Shibari](https://www.helsinkishibari.com/) | 33 | 4 | [bottom-perspective risks](https://www.helsinkishibari.com/articles/title), [nerve safety](https://www.helsinkishibari.com/articles/nerve-safety-for-rope-bondage), [social safety practice](https://www.helsinkishibari.com/safety-practices) | Bottom-centered fall, nerve, fainting, neck, equipment, and suspension risk guidance paired with operational consent, reporting, confidentiality, accountability, and sober-space policies. | Durable educational and policy pages in English plus unique Finnish facts; exclude calendar, current staff, contacts, and repeated event material. |

## Accepted with targeted scope

| Source | Score | Novel coverage | Representative public pages reviewed | Why it was accepted | Corpus boundary |
|---|---:|---:|---|---|---|
| [The Rope Bottom Guide](https://theropebottomguide.com/) | 40 | 5 | [official version 4.2 PDF](https://theropebottomguide.com/downloads/rope_bottom_guide.pdf) | Dense authored bottom-centered coverage of partner evaluation, responsibility, communication, body knowledge, placement, temperature, risk, and recovery. | Official guide and version metadata only; preserve cautions and structure, without reproducing or inferring photography. |
| [Anatomie Studio London](https://www.anatomiestudio.com/) | 28 | 4 | [education blog](https://www.anatomiestudio.com/blog), [code of conduct](https://www.anatomiestudio.com/code-of-conduct), [accessibility](https://www.anatomiestudio.com/accessibility) | Unusually strong public bottoming, disability, accessibility, pedagogy, consent, performance, and rope-space governance material. | Education/community articles, code, accessibility, nerve safety, and durable history only; exclude tickets, schedules, store, testimonials, galleries, and venue news. |
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
| [ICAR suspension syndrome review](https://europepmc.org/article/MED/38081341) | `accept_high_priority` | 43 | CC BY 4.0 review-level evidence, graded recommendations, and myth correction via EMBL-EBI JATS only; harness/rescue evidence is not silently generalized to rope bondage. |
| [Rope neuropathy study](https://europepmc.org/article/MED/37324199) | `accept_high_priority` | 42 | CC BY 3.0 directly relevant survey evidence via EMBL-EBI JATS only; self-report associations retain denominators, uncertainty, and study limits. |
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
