# Rope source discovery report v1

Initial review at: 2026-07-16T06:04:06Z
Initial batch: `discovery_batch_001`
Latest review at: 2026-07-16T19:03:38Z
Latest batch: `discovery_batch_028`

## Outcome

This first bounded batch manually reviewed 20 candidate websites or official site-hosted resources by opening their actual public pages. After the evidence-quality correction described in batch 005 and the three complete practitioner-site re-audits recorded in batch 014, its decisions are 6 `accept_high_priority`, 9 `accept_targeted_scope`, 0 `defer`, and 5 `reject`. The 15 accepted sources are added to the canonical corpus queue as separate pending extraction jobs. Discovery did not copy page bodies into training data, enter a course, sign in, start a trial, defeat a challenge, or infer instructions from images or video.

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
| [Shibari Academy](https://www.shibariacademy.com/) | Reject | 11 | The completed public-site re-audit found restrictive no-reproduction terms plus delayed-care, instant-nerve-injury, continue-after-moving-a-knot, categorical rope, and affiliate-shop claims. Rights and evidence quality each independently fail. |
| [Shibari Safety](https://shibarisafety.com/) | Reject | 20 | The complete eight-route re-audit found anonymous, explicitly unqualified authorship, weak circular references, unsupported medical and structural numbers, and boiling or oil recipes. Crawl permission supplies no reuse license. |
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

`discovery_batch_003` manually reviewed 16 additional, nonduplicate sources and the endpoint-specific access and reuse terms needed for each. After the full-body safety correction recorded below, it contains 4 `accept_high_priority`, 5 `accept_targeted_scope`, 4 `defer`, and 3 `reject` decisions. The nine accepted candidates are separate pending extraction jobs; all rights-limited sources use narrow, attributed scopes and all cross-domain material retains its original operating context.

The complete 11-page body of the previously queued healing phenomenology paper
was re-audited after discovery. Its ten regular practitioners came from practice
groups, acquaintances and some friends, all were Caucasian, and the interview
guide expressly prompted felt healing effects. Although the article acknowledges
that medical therapeutic properties are unproven and the sample is not
generalizable, it repeatedly promotes healing or therapy and positively frames a
participant temporarily losing consciousness during suspension as a
transformative near-death experience. That combination of positive selection,
leading prompts, missing adverse perspectives, therapeutic overclaim and medical-
emergency normalization outweighs its CC BY license. It is now rejected and
removed from the extraction queue.

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

### Deferred and rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [ITRA rope-rescue documents](https://www.technicalrescue.org/documents/) | `defer` | 41 | A high-value 2024 syllabus was identified, but direct TLS validation fails and the terms do not authorize corpus-scale reuse; require valid transport and written permission. |
| [ANZCOR harness-suspension first aid](https://www.anzcor.org/home/first-aid/guideline-9-1-5-first-aid-management-of-harness-suspension-trauma) | `defer` | 37 | Strong clinical guidance, but its copyright page explicitly bars incorporation into another work without written permission; licensed ICAR evidence covers its central myth correction. |
| [Carleton rope-bondage thesis](https://carleton.scholaris.ca/items/51e42f12-3cf6-400a-b101-5db266aaa109) | `defer` | 34 | Institutional rights allow research/teaching and link sharing but bar adaptation/derivatives and commercial use; explicit permission is needed. |
| [Kanna/Kagura bakushi biographies](https://kanna-kagura.blogspot.com/2020/04/biographies-of-kinbakushi-of-japan.html) | `defer` | 27 | Rare translated lineage testimony, but rights are layered across the interview, translation, annotations, magazine, and later book. |
| [Durham comparative rope thesis](https://etheses.durham.ac.uk/id/eprint/15763/) | `reject` | 32 | Repository robots explicitly block named OpenAI/AI-training agents; no body was opened and no bypass was attempted. |
| [Cleveland Clinic suspension syndrome](https://my.clevelandclinic.org/health/diseases/suspension-syndrome) | `reject` | 29 | Reuse terms prohibit storing, repackaging, or editing, and a stronger CC BY review already covers the topic. |
| [Healing Experiences in Japanese Rope Bondage Practice](https://www.journal.aleftrust.org/index.php/cstp/article/view/46) | `reject` | 15 | The licensed body has severe positive-selection and prompting bias, overstates therapeutic implications, and positively narrates suspension-related loss of consciousness. Metadata and rejection rationale only; no participant narrative or healing claim enters training. |

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
and visible reuse terms. After the official-access correction recorded in
batch 008, it contains 2 `accept_high_priority`, 5 `accept_targeted_scope`, 4
`defer`, and 1 `reject` decisions. The seven accepted candidates expand the
canonical corpus queue from 61 to 68 resources.

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
| [USDA National Tree Climbing Guide](https://www.govinfo.gov/app/details/GOVPUB-A13-PURL-gpo215987) | `accept_high_priority` | 39 | An official GovInfo package now supplies the 2015 federal guide through a durable permitted route. Capture only bounded planning, role, inspection/retirement, knot-function, line-management, communication, and rescue-readiness concepts; exclude operational climbing, ratings, body-weight anchor tests, credited material, and all transfer into bondage or human suspension. |
| [Osada-ryu primary writings](https://www.osada-ryu.com/?page_id=184) | `accept_targeted_scope` | 34 | Quotation-light, fully attributed digest of first-person Akechi, Nureki, Yukimura, and Osada-ryu testimony; speaker, interviewer, date, framing, recollection, and later synthesis remain distinct. |
| [Devil Mask Studio curriculum](https://devilmaskstud.io/preflight/) | `accept_targeted_scope` | 30 | Detailed prerequisite map for body-adapted harnesses, uplines, lockoffs, force vectors, ground tests, go/no-go reasoning, bamboo, hashira, inversions, and transitions; descriptions are not reconstructed tutorials. |
| [Rope Study progression](https://ropestudy.com/) | `accept_targeted_scope` | 29 | Current 101–401 competency progression and partnered-practice pedagogy; capture only normally browsable pages during the incomplete May 2026 rebuild, without bypassing ModSecurity. |
| [Hajime Kinoko official profile](https://shibari.jp/en/profile/index.html) | `accept_targeted_scope` | 24 | Narrow first-party digest of declared teachers, influences, artistic statement, Ichinawa-kai, and durable milestones; lineage is labeled self-described and promotion is excluded. |
| [Go Arisue official profile](https://www.arisuego.com/home/%E6%9C%89%E6%9C%AB%E5%89%9B%E3%81%AE%E4%B8%96%E7%95%8C/) | `accept_targeted_scope` | 19 | Stable Japanese first-party biography and role chronology from a provisional 2026 relaunch; image, event, store, and forthcoming paid content is excluded. |

### Deferred and rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
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
despite its accessible technical library. Strong relevance did not override
access or rights: J-STAGE's PDF restriction, Ritsumeikan's reuse policy, and
Full Circle's missing official asset all block corpus capture. The USDA guide's
original Research Station route remains crawler-restricted, but batch 008 found
the same federal edition in an official, durable GovInfo package; this is a new
authorized route, not a mirror or workaround.

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
## Discovery batch 014 outcome

discovery_batch_014 reviewed 11 additional, nonduplicate candidates against
their actual first-party record or body, article-level license, methods,
conflicts, access path, redundancy, and domain-transfer risk. It produced 1
accept_high_priority, 3 accept_targeted_scope, 6 defer, and 1 reject decision.
The four accepted candidates expand the canonical corpus queue from 109 to 113
resources.

This batch adds two complementary evidence lanes. Two rope-engineering sources
cover three-strand contact force and condition-monitoring methods while keeping
yarn, rope construction and material scales separate. Two
consent studies add perceived community norms and within-person variability
without turning group identity, arousal, body language or one survey into proof
of an individual's consent.

The manual practitioner-site screen also closed three old uncertainties.
Shibari Safety and Shibari Academy move from defer to reject; ShibariNews
remains rejected with a stronger sitemap, terms and provenance audit. Deadheavy's
resource page is newly rejected as a body but its outbound destinations remain
independent discovery leads. Large page counts and public robots access were not
mistaken for authority, reuse permission or information density.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Three-strand aramid contact-force experiment](https://research.utwente.nl/en/publications/experimental-analysis-of-contact-forces-between-strands-in-three-/) | accept_high_priority | 39 | CC BY 4.0, 17-page primary evidence using pressure film in exact Twaron three-strand ropes. Preserve construction hierarchy, calibration limits, qualitative twist and contact distinctions, model limits, request-only data, and the NWO or Teijin funding and declared financial-support conflict. Exclude values, fitted corrections, lifetime or rating claims, supplier endorsement and all natural-fiber or human-use transfer. |
| [Synthetic-rope condition-monitoring review](https://papers.phmsociety.org/index.php/ijphm/article/view/2619) | accept_targeted_scope | 30 | CC BY 3.0 US historical review of rope hierarchy, interacting damage mechanisms, embedded versus external sensors and continuous versus discrete monitoring. Exclude every permission-reproduced figure, patent, vendor claim, standard, discard number and remaining-life rule. |
| [Sexual Consent Norms in a Sexually Diverse Sample](https://europepmc.org/article/MED/38017253) | accept_targeted_scope | 34 | CC BY 4.0 EMBL-EBI JATS contributes a 388-person online sample with 116 BDSM participants. Preserve overlapping identities, perceived descriptive versus injunctive norms, small or null findings, indirect measurement and sensitivity discrepancies; never infer that BDSM identity proves consent. |
| [Within-person variability of sexual consent](https://eprints.gla.ac.uk/238155/) | accept_targeted_scope | 30 | CC BY 4.0, 28-day experience-sampling evidence from 113 adults and 1,189 analytic partnered events distinguishes internal feelings from external communication and shows event-level variability. It is not a BDSM or rope study and does not validate body-language inference, a check-in schedule or a rope policy. |

### Deferred sources

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Yarn-on-yarn abrasion mechanisms](https://journals.sagepub.com/doi/10.1177/15589250241228263) | 35 | The publisher-rendered body identifies CC BY 4.0 and a relevant controlled yarn-abrasion study, but direct retrieval and an ordinary unmodified browser both receive a Cloudflare 403 challenge. The first-party PDF could not be retrieved or checksummed, so no page audit, mirror, or snippet reconstruction substitutes for legitimate body access. |
| [Hemp fiber-to-rope hierarchy experiment](https://www.sciencedirect.com/science/article/pii/S1359835X24002008) | 40 | The publisher metadata and CC BY 4.0 license are clear, and the 120-fiber, 55-yarn, 33-strand and 64-rope hierarchy is unusually valuable. The official PDF returns HTTP 403 and the API returns only unauthorized minimized metadata. No mirror, snippet reconstruction or access bypass substitutes for a page-level body audit. |
| [The ancient art of laying rope](https://arxiv.org/abs/1004.0814) | 34 | Helical maximum-rotation and zero-twist ideas are relevant, but the arXiv copy carries only a nonexclusive distribution license and the identified publisher copy restricts redistribution and commercial use. Permission is required before transformation. |
| [Self-locking and stability of the bowline knot](https://www.sciencedirect.com/science/article/pii/S2352431625001257) | 40 | The seven-page EPFL final PDF is CC BY-NC 4.0 and tests six quasi-static composite elastomeric rods, not textile rope. NonCommercial compatibility is unresolved; equations, thresholds, tying diagrams, secure claims and rescue-harness language remain quarantined. |
| [Autistic adults' BDSM and kink experiences](https://durham-repository.worktribe.com/output/1177508/comforting-reassuring-andhot-a-qualitative-exploration-of-engaging-in-bondage-discipline-domination-submission-sadism-and-sadomasochism-and-kink-from-the-perspective-of-autistic-adults) | 38 | Rare six-person interpretative evidence may help accessibility coverage, but the accepted manuscript is noncommercial-only and direct review now meets a Cloudflare challenge. No participant narrative or identity generalization enters without permission and legitimate access. |
| [Plant-based natural-fiber rope treatments](https://www.tandfonline.com/doi/abs/10.1080/15440478.2024.2397703) | 37 | CC BY 4.0 does not cure domain contamination. Alkali and boiling-water treatments for concrete reinforcement could be mislearned as bondage-rope cleaning or performance advice; independent textile and safety adjudication is required before body use. |

### Rejected source and prior-site corrections

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [Deadheavy resources](https://deadheavy.com/resources/) | reject | 2 | Roughly 21 outbound links, book names, quotations and recommendations make a discovery directory, not a factual corpus. Copying it would create URL trivia, and its useful code of conduct names upstream adaptations with unresolved rights. |
| [Shibari Safety](https://shibarisafety.com/) | prior defer to reject | 20 | Eight public routes were fully screened. Anonymous and expressly unqualified authorship, weak references, unsupported breathing or anatomy typologies, numeric suspension guidance, treatment recipes and absent reuse rights bar inclusion. |
| [Shibari Academy](https://www.shibariacademy.com/) | prior defer to reject | 11 | A 143-URL commercial surface, restrictive terms and public delayed-care, nerve, knot-continuation, rope-selection and cleaning claims independently fail the rights and safety gates. |
| [ShibariNews](https://shibarinews.com/) | reject retained | 4 | The roughly 1,094-post surface lacks named editorial or specialist provenance, repeats reductive history and unsupported body advice, funnels to Shibari Academy, and permits only personal noncommercial extracts. |

### Batch 014 rights, evidence, and safety controls

Article-level licenses were followed rather than inferred from an open-access
badge. CC BY sources advance only with attribution, change indication and
component audit. CC BY-NC, noncommercial repository terms and arXiv distribution
permission remain metadata-only. A clear CC BY license does not authorize
reconstructing an inaccessible first-party body from snippets or mirrors.

Rope scale remains explicit. Synthetic yarn is not finished rope; an aramid
three-strand specimen does not represent natural-fiber rope; a composite
elastomeric rod is not textile cordage; an offshore monitoring review does not
supply visual retirement rules for bondage rope. Static and quasi-static
evidence does not establish cyclic, dynamic, security, capsizing, body-contact,
upline, anchor or human-suspension behavior.

Consent evidence remains population and measurement bound. Perceived group
norms are not observed individual behavior, overlapping identity groups are not
independent populations, internal willingness cannot be read from arousal or
body language, and event-level variability does not prescribe a universal
script. The studies support careful reassessment and uncertainty, not identity
stereotypes or a guarantee of ethical conduct.

### Batch 014 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, tests, and first-class registry work for an independently
completed corpus. No QA, training projection, manual-review, evaluation,
heldout, holdout, OOD, shadow, benchmark, probe, trainer or unrelated corpus
artifact was read. Public review used ordinary first-party, official repository,
publisher, DOI, licensed API and license routes. No login, paywall, challenge,
Cloudflare gate, unauthorized API, cache, archive, snippet reconstruction or
third-party mirror was bypassed, and discovery wrote no reviewed source body
into a training artifact.

## Discovery batch 015 outcome

`discovery_batch_015` reviewed 14 exact-new candidates against the existing
ledger and corpus queue. It produced 1 `accept_high_priority`, 8
`accept_targeted_scope`, 5 `defer`, and 0 `reject` decisions. The nine accepted
sources expand the canonical Markdown queue from 113 to 122 resources.

The batch emphasizes facts and transferable evaluation habits rather than URL
recall. Two Crown investigation reports add unusually strong evidence about
hidden rope construction defects, inspection blind spots, component mismatch,
quality control, and inappropriate test-method transfer. Federal textile and
maritime sources add bounded fiber-aging and rope-construction context. Three
rights-clear studies add accessibility, agency, negotiation, and uncertainty
while preserving their non-rope populations, methods, jurisdictions, and
small-sample limits. One high-risk strangulation study advances only an
extremely narrow epistemic-safety lesson and none of its participant beliefs.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [Zarga HMPE mooring-line investigation](https://www.gov.uk/maib-reports/failure-of-mooring-line-on-board-lng-carrier-zarga-with-1-person-injured) | 39 | OGL v3 Crown-authored failure analysis of a jacket hiding the load-bearing core, axial-compression or cyclic contact damage, line-and-fitting incompatibility, incomplete supplier/builder information, visual-inspection limits, and invalid test-method transfer. Exclude identities, third-party annexes and images, standards, vendors, every number or rating, snapback procedures, and any human-suspension equivalence. |
| [Defective throw-bag rescue-line investigation](https://www.gov.uk/maib-reports/failure-of-a-throw-bag-rescue-line-during-a-boat-capsize-drill) | 41 | OGL v3 text supports a narrow case study of hidden thermally fused segment joints, why intact-rope tests do not validate joints, and the limits of visual or random quality control. Exclude brands, ratios, loads, standards, images, manufacturing recipes, rescue procedures, and body-rope transfer. |
| [Consent and abuse reflections among Swedish young adults with intellectual disabilities](https://europepmc.org/article/PMC/12397527) | 39 | CC BY 4.0 EMBL-EBI JATS supports agency, contextual ambiguity, clarification, accessible communication and education, with the 22-interview, gender-imbalanced, verbally communicating Swedish sample inseparable from every finding. Exclude participant quotations, abuse narratives, legal-capacity advice, deficit framing, and BDSM inference. |
| [Disability sexual rights as access, choice, and pleasure](https://link.springer.com/article/10.1007/s11195-024-09874-7) | 35 | CC BY 4.0 theoretical framework adds positive access needs, heterogeneity and agency to accessibility and scene-design coverage. Preserve its theoretical, non-rope and jurisdiction-specific limits; exclude legal advice, universalization, entitlement to another person, and third-party components. |
| [NPS curatorial care of textile objects, Appendix K](https://www.nps.gov/subjects/museums/upload/MHI_AppK_TextilesObjects.pdf) | 36 | Eligible federal text supplies a tightly bounded taxonomy of plant, animal and synthetic fibers, construction, processing damage, molecular degradation, dimensional stress, light, heat, moisture, pollution, pests and abrasion. Label it museum-textile evidence; exclude figures, credited matter, environmental numbers, chemical, cleaning, treatment, storage and emergency recipes, and rope-strength or retirement inference. |
| [BSEE auxiliary-line abrasion alert](https://www.bsee.gov/safety-alerts/bsee-safety-alert-454-auxiliary-line-abrading-causes-rope-guard-failure) | 37 | Eligible federal text supports only the industrial contact-path to chronic abrasion to component damage to dropped-part or line-jump failure chain, plus inspection of adjacent components and line-of-fire reasoning. Exclude wire-rope transfer, photos, vendors, standards, heights, machinery procedure and replacement-design prescriptions. |
| [Three trans men's kink-negotiation case studies](https://www.mdpi.com/1660-4601/19/18/11382) | 32 | CC BY 4.0 evidence from three Toronto participants may preserve reported deliberate negotiation, explicit consent, sobriety, affirming partners and community support as perceptions, not proof of causality or safety. Exclude quotations, stories, HIV advice, unnecessary demographics and claims that kink frameworks are validated risk controls. |
| [Sexual strangulation safety perceptions](https://link.springer.com/article/10.1007/s10508-025-03097-3) | 30 | CC BY 4.0 material is limited to the proposition that consent, trust, benign intent and participant belief do not establish physical safety, attached to its selected responses, risk-prompt priming, single final open question and unreported coding-reliability limitation. Exclude every quotation, prevalence or injury statistic, anatomy, pressure, location, duration, onset, medical, first-aid, legal or technique claim and every unsafe participant belief. |
| [NPS maritime rope-construction history](https://www.nps.gov/articles/000/charlestown-navy-yard-ropewalk.htm) | 29 | Eligible federal prose supports yarn-to-strand-to-rope hierarchy, spinning, forming, laying and counter-twist construction, historical hemp ropewalk production, and clearly labeled maritime line, cable and hawser vocabulary. Exclude third-party images or excerpts, production trivia and numbers, modern safety claims, and tying or rigging recipes. |

### Deferred sources

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [BSEE TAP-591 polyester subrope inspection study](https://www.bsee.gov/research-record/tap-591-evaluate-accuracy-polyester-subrope-damage-detection-performed-rovs) | 38 | The accessible 117-page contractor report could add inspection uncertainty, FMEA and measurement limits, but federal commissioning and review do not place Stress Engineering Services or TTI text in the public domain. No reuse license was found; permission and a component audit are required. |
| [Shibari Lounge risk profile](https://www.shibarilounge.co.uk/journal/shibari-risk-profile) | 14 | A CC BY footer conflicts with June 2025 terms granting limited noncommercial, no-modification use. Useful accessibility and policy prose remains unavailable for transformation pending written clarification; unsupported medical, breath-play, hardware, hardpoint, bamboo, hygiene and product content stays excluded regardless. |
| [FEMA structural-collapse search-technician qualification](https://rtlt.preptoolkit.fema.gov/Public/Position/View/8-509-1164) | 31 | A federal competency framework could support demonstrated skills, team context, exercises, currency and recertification, but the RTLT robots policy disallows all automated access. Retain metadata only until an allowed static route or permission exists; never import named standards, occupational credentials or rescue technique. |
| [USFS-hosted Jute and Kenaf chapter](https://research.fs.usda.gov/treesearch/30670) | 38 | The CRC/Taylor & Francis chapter reserves reproduction and electronic-use rights and disclaims claims over original U.S. Government works, leaving the joint-author and component boundary unresolved; USFS robots also disallows access. Publisher permission is required before any fiber-science transformation. |
| [USCG cutter seamanship line handling](https://media.defense.gov/2020/Nov/09/2002532100/-1/-1/0/CIM_3120_9.PDF) | 27 | Federal authorship is promising, but the official media host returns HTTP 403 and the material substantially duplicates the accepted USFS rigging manual. Seek an allowed route before considering only team communication and system-level reassessment; exclude operations, commands, numbers, ratings and human-support transfer. |

### Batch 015 rights, evidence, and safety controls

Public access, public funding and an agency landing page were never treated as
reuse permission. Crown text advances under OGL v3 only after excluding
credited third-party matter. Eligible federal prose advances under the
public-domain presumption only with component audit. CC BY articles preserve
attribution, changes, methods and limitations. Conflicting site terms,
contractor authorship, robots disallows and HTTP 403 routes remain hard gates.

Industrial HMPE mooring rope, wire rope, rescue line and museum textiles are
not bondage rope. Their accepted records preserve only source-supported
failure-analysis or material-taxonomy lessons, never ratings, recipes,
retirement rules, hardpoint approval or human-suspension validation. The
strangulation source is quarantined from all technique and medical content.
Sensitive consent and disability studies exclude participant narratives and
retain sample selection, communication, jurisdictional and causal limits.

### Batch 015 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests
plus ordinary first-party public sources and their explicit licenses or terms.
No QA, trainer, projection, evaluation, heldout, holdout, OOD, shadow,
benchmark, probe, unrelated corpus body or unrelated repository artifact was
read. No login, purchase, robots rule, challenge, HTTP 403, archive, mirror,
cache or snippet reconstruction was bypassed, and no source body entered a
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

## Discovery batch 007 outcome

`discovery_batch_007` manually reviewed 11 additional candidates against their
actual public body, authoritative metadata, current robots policy, and governing
reuse terms. It produced 2 `accept_high_priority`, 2
`accept_targeted_scope`, 4 `defer`, and 3 `reject` decisions. The four accepted
candidates expand the canonical corpus queue from 77 to 81 resources.

This batch fills three especially useful gaps: a current clinical review of
entrapment-neuropathy mechanisms and diagnostic limits, a current federal
hazard-tree guide that explicitly preserves prediction uncertainty, and exact
National Diet Library authority identities for recurring Japanese people and a
publication title. A short NPS curriculum is accepted only as an assessment
map. It is not treated as rope instruction or human-suspension certification.

### Accepted sources

| Source | Decision | Score | Distinct value and extraction boundary |
|---|---|---:|---|
| [USDA Region 5 Hazard Tree Identification and Mitigation](https://www.fs.usda.gov/sites/nfs/files/legacy-media/r05/Hazard%20Tree%20Identification%20and%20Mitigation%20-%202022.pdf) | `accept_high_priority` | 41 | Current official framework for failure, defects, targets, inspection context, roots, leans, decay, site disturbance, multiple defects, weather, and monitoring. Preserve its statement that even excellent care cannot predict all failures or eliminate risk; capture general screening concepts only, audit credited components, and never certify a tree, branch, anchor, load, or human-suspension setup. |
| [Entrapment neuropathies contemporary review](https://europepmc.org/article/MED/32766466) | `accept_high_priority` | 41 | CC BY 4.0 clinical review, bound to PMCID PMC7382548, PMID 32766466, and DOI 10.1097/PR9.0000000000000829 through the authorized EMBL-EBI JATS route. Preserve human-versus-preclinical evidence, heterogeneous mechanisms and presentation, assessment limits, and uncertainty; derive no safe rope pressure, placement, timing, prognosis, or treatment. |
| [Web NDL Authorities rope-history identities](https://id.ndl.go.jp/auth/ndla/) | `accept_targeted_scope` | 22 | Rights-clear structured identity fields for Seiu Ito (00023065), Chimuo Nureki (00731050), and the *Kitan Club* uniform title (034485143). Capture exact fields and dates with NDL attribution, not biography, lineage, teaching relationships, or publication roles. |
| [NPS New River Gorge climbing-guide curriculum](https://www.nps.gov/neri/planyourvisit/climbing-guide-certification-guidelines.htm) | `accept_targeted_scope` | 23 | Compact professional competency map covering named knot families, backup instruction, anchor assessment, redundancy and load distribution, belaying, rappelling, assistance, pulley raising, inspection, environmental hazards, and measured proficiency. It remains specific to an old park permit-equivalency context and supplies no construction steps or certification transfer. |

### Deferred sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [Army TC 21-24 Tower and Helicopter Rappelling](https://rdl.train.army.mil/catalog-ws/view/100.ATSC/718741EE-0DC1-4458-86AA-43F7DEA0F451-1740684333183/TC21_24.pdf) | `defer` | 38 | Official indexing identifies the current 25 February 2025 public-release edition and useful roles, checks, communication, inspection, and emergency workflow, but the RDL host did not resolve. No mirror was used; retry the official route and never directly transfer military rappelling into bondage. |
| [Live Performance Australia HG06 Performer Hazards](https://liveperformance.com.au/wp-content/uploads/2019/03/HG06-Performers-Sept-2024.pdf) | `defer` | 35 | The September 2024 guide has strong lifecycle risk-assessment, engineered-point, cross-check, rescue-plan, stop, rehearsal, and touring practices. LPA terms reserve rights and limit reuse to personal/internal informational use, so derivative corpus and training permission must be obtained first. |
| [Ritsumeikan Murakami/*Kitan Club* paper](https://ritsumei.repo.nii.ac.jp/records/2005134) | `defer` | 32 | Unique source-critical postwar intellectual history, but the file row displays no license and university policy requires permission for reproduction, modification, or distribution. Retain DOI/metadata only and do not enter robots-disallowed API, OAI, export, or data routes. |
| [Shibari kinbaku: el bondage entra en escena](https://sedici.unlp.edu.ar/handle/10915/174364?show=full) | `defer` | 23 | Peer-reviewed Spanish-language analysis of two La Plata performance/teaching experiences adds a valuable local perspective, but CC BY-NC-SA is not assumed compatible with the intended use. Retain metadata and seek commercial/derivative permission; do not promote its broad historical framing into technical or lineage authority. |

### Rejected sources

| Source | Decision | Score | Reason |
|---|---|---:|---|
| [OpenStax Anatomy and Physiology](https://openstax.org/books/anatomy-and-physiology/pages/13-4-the-peripheral-nervous-system) | `reject` | 34 | Book pages explicitly prohibit LLM or generative-AI training without permission, and robots disallow GPTBot from `/books/`. This use restriction overrides the otherwise attractive CC license for this corpus; no body, mirror, export, or older copy may be used. |
| [Army TM 3-34.86 Rigging Techniques, Procedures, and Applications](https://rdl.train.army.mil/catalog-ws/view/100.ATSC/1676C9DD-7ADF-4E41-9130-6BCEE7BBC5A9-1343151324082/tm3_34x86.pdf) | `reject` | 23 | Its own front matter says the 2012 conversion is identical to 1995 content, integrates no later doctrine, and may cite obsolete sources. It is also currently unreachable, heavily duplicates the accepted 2024 USFS rigging corpus, and includes hazardous improvised structures. |
| [RopeWiki Riggings](https://ropewiki.com/Category%3ARiggings) | `reject` | 20 | Anonymous CC BY-NC-SA canyoneering pages cover FiddleStick/toggle, retrievable, releasable, ghosting, block, and other high-consequence systems. Stronger accepted sources exist, and direct transfer into bondage, performer flying, hardpoints, human suspension, or rescue would be unsafe. |

### Batch 007 rights, evidence, and safety controls

Permissive presentation did not override explicit use limits. OpenStax's AI ban
caused rejection despite its Creative Commons textbook license. LPA,
Ritsumeikan, and SEDICI remain citation-and-permission leads, not body-text
sources. The two Army manuals were not fetched from mirrors when the official
host failed: the current circular is deferred for a clean official retry, while
the explicitly unchanged 1995 rigging manual is rejected as obsolete and
duplicative.

The accepted safety sources preserve domain boundaries. The clinical review is
not rope-specific and cannot supply a safe pressure or time threshold. The
tree report evaluates potential failure and targets, not load-rated anchors;
tree health is not hardpoint capacity. The NPS page and NDL records provide a
curriculum map and authority fields respectively, not technique, biography,
lineage proof, certification, or QA trivia.

### Batch 007 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, and discovery tests. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, or source
corpus artifact was read. Public-web review used ordinary public pages and the
authorized EMBL-EBI API, respected current robots, crawl delays, licenses, and
rightsholder terms, and did not bypass a login, purchase, challenge, missing
body, license, crawler restriction, or unreachable host. No reviewed page body
was copied into a training artifact during discovery.

## Discovery batch 008 outcome

`discovery_batch_008` manually reviewed 12 new, nonduplicate source candidates
against their substantive pages, primary metadata, current reuse terms, and
training suitability. It produced 1 `accept_high_priority`, 3
`accept_targeted_scope`, 7 `defer`, and 1 `reject` decisions. The four new
accepted candidates expand the corrected canonical corpus queue from 81 to 85
resources.

This batch makes reuse status explicit rather than equating public readability
with permission. Eligible HSE Crown text is OGL v3.0, two Project Gutenberg
works are public domain in the USA, and unmarked NIST federal information is
reusable. Strong CDC, SAAIL, Cordage Institute, Samson, practitioner, clinical,
and local-history sources remain reference-only where policy, copyright, or
license terms conflict with transformed training use.

Batch 008 also found the exact 2015 USDA National Tree Climbing Guide in an
official GovInfo package. Candidate
`usfs_national_tree_climbing_guide` is therefore promoted from `defer` to
`accept_high_priority` in its original batch-004 record. The Research Station
route remains disallowed; GovInfo is an independent official federal endpoint,
not a mirror or access workaround. This correction adds one queue entry and
shifts the historical queue boundaries in batches 004–007 without changing
their newly reviewed candidate counts.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [HSE treework lifting and climbing guidance](https://www.hse.gov.uk/treework/safety-topics/arboriculture.htm) | `accept_high_priority` | 33 | Current OGL-licensed Crown text on planning, competence, inspection records, separately anchored lines, load and anchor-system framing, ground-team communication, reassessment, rope management, and rescue readiness. Preserve UK occupational treework context; exclude images, linked third-party material, operational climbing, ratings, device endorsement, and any certification or direct transfer to bondage. |
| [The Use of Ropes and Tackle](https://www.gutenberg.org/ebooks/56585) | `accept_targeted_scope` | 33 | Public-domain historical terminology, topology, classification, idealized tackle geometry, and handling categories. Exclude every historical strength/load/efficiency table, wire-rope or industrial procedure, figure-derived instruction, body restraint, and human-suspension claim; cross-check mechanics against current sources. |
| [Knots, Splices and Rope Work](https://www.gutenberg.org/ebooks/13510) | `accept_targeted_scope` | 26 | Public-domain historical vocabulary for rope construction, whipping, seizing, bends, hitches, splices, lashings, crowns, and decorative families. Exclude nooses, body restraint, values, repair-strength or absolute-security claims, fire/chemical processing, and current-safety presentation. |
| [NBS Report 8308: Heating Characteristics of Cordage Fibers](https://www.nist.gov/publications/heating-characteristics-cordage-fibers) | `accept_targeted_scope` | 35 | Primary federal measurements and modeling for three abaca and one Haitian sisal sample in bulk-bale/storage contexts. Preserve method, assumptions, sample identity, and historical limits; do not generalize to jute or finished rope or derive boiling, singeing, treatment, storage, fire, strength, or care advice. |

### Deferred sources

| Source | Score | Why it is reference-only |
|---|---:|---|
| [Cordage Institute publications](https://ropecord.com/publications-catalog/) | 34 | The strongest current standards lead found, including 2024–2025 editions, but the publisher expressly prohibits reproduction without prior written permission and many documents are paid. Retain identifiers, titles, dates, scope, and a permission list only. |
| [Samson technical documents](https://www.samsonrope.com/resources/technical-documents) | 34 | Excellent manufacturer evidence on inspection, retirement, abrasion, compression, cuts, glazing, chemicals, friction, twist, measurement, and creep, but manuals are all-rights-reserved and product/application specific. Retain metadata and seek training permission. |
| [Clinical Guidelines for Working with Clients Involved in Kink](https://doi.org/10.1080/0092623X.2023.2232801) | 34 | Strong peer-reviewed clinical and anti-stigma framing, PMID 37439228, under CC BY-NC 4.0. NonCommercial compatibility with intended training use is not assumed; retain citation and permission/use review only. |
| [SAAIL autism, kink, and BDSM toolkit](https://www.autlives.com/kink-and-bdsm) | 31 | Valuable six-participant, co-produced work on direct communication, sensory needs, overwhelm, pacing, privacy, and event access, but © 2023–2026 with no open reuse license. Preserve study metadata and seek permission. |
| [NIOSH suspension-scaffold failure Alert](https://www.cdc.gov/niosh/docs/92-108/) | 30 | Five incident reports and six deaths support inspection and hazard concepts, but CDC's no-substantive-changes reuse guidance conflicts with the required original paraphrase and its 1992 regulatory context is obsolete. Defer pending policy clarification. |
| [WykD practitioner archive](https://wykd.com/) | 19 | First-person pedagogy, declared influences, injury responsibility, and terminology are mixed with heavy promotion and overlap. All rights are reserved; retain metadata and seek permission for a strict article allowlist. |
| [Ropewalk / Story of Rope](https://www.storyofrope.org/) | 17 | Useful Hooven & Allison, Xenia, fiber-transition, film, and bibliography leads, but CC BY-NC-ND 4.0 prohibits transformed Markdown and the text is sparse. Retain only source and archive metadata. |

### Rejected source

| Source | Score | Reason |
|---|---:|---|
| [Louis Kordexe rope articles](https://louiskordexe.com/resources-and-articles/) | 18 | Ordinary copyright combines with unsupported instructions involving optional boiling, roughly 72-hour tension drying, about 100 friction pulls, multiple torch passes, exact oil/wax treatment, and bacterial, mold, inhalation, or degradation claims. No body, QA, or training facts should be derived from it. |

### Batch 008 rights, evidence, and safety controls

Every batch-008 ledger row records a `training_use` disposition and a
`rights_basis`. The four accepted sources are
`direct_training_bounded`; deferred sources are explicitly reference-only;
the rejected practitioner source is `rejected_no_use`. Project Gutenberg's
license, attribution, marks rules, public-domain-in-the-USA status, and
territorial warning must accompany snapshots. HSE logos, images, video, and
third-party material are excluded from the OGL scope. NIST figures, tables, or
other marked material require a separate rights audit.

Historical and cross-domain texts are not modern bondage instructions. Neither
the two public-domain manuals nor the USDA/HSE/NIST sources can certify knots,
rope, trees, branches, indoor hardpoints, anchors, people, bondage systems, or
human suspension. Numerical ratings, body-weight branch tests, operational
climbing or hoisting steps, and domain-specific equipment claims are excluded.

### Batch 008 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, and discovery tests. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, trainer, or
existing source-corpus artifact was read. Public review used ordinary
rightsholder and official government routes, respected access and reuse terms,
and did not bypass a login, purchase, challenge, crawler restriction, missing
body, or license. Discovery wrote no reviewed page body to a training artifact.

## Discovery batch 009 outcome

`discovery_batch_009` manually reviewed 12 additional, nonduplicate source
candidates against their substantive public pages, primary metadata, current
reuse terms, and safety value. It produced 1 `accept_high_priority`, 2
`accept_targeted_scope`, 7 `defer`, and 2 `reject` decisions. The three accepted
candidates expand the canonical corpus queue from 85 to 88 resources.

This batch adds a kink-specific educator ethics code, a March 2026 federal
clinical reference, and a primary federal rope-yarn flex-fatigue study. It also
records several excellent sources that cannot yet enter transformed Markdown:
two government guides have publication-specific noncommercial language, a new
knot-mechanics preprint is NoDerivatives, and several practitioner or guild
libraries lack an open corpus license.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Kink Education Code of Conduct v1](https://www.thekecc.org/fullVersion.html) | `accept_high_priority` | 38 | CC BY-SA 4.0, kink-specific guidance on consent modeling, student-demo power differentials, competence disclosure, inclusion, accountability, educator-student boundaries, vetting, incident disclosure, and privacy. Preserve the 2019/version-1 status and ShareAlike terms; do not present it as law, certification, current consensus, or proof that a person or venue is safe. |
| [NINDS Peripheral Neuropathy](https://www.ninds.nih.gov/health-information/disorders/peripheral-neuropathy) | `accept_targeted_scope` | 31 | First-party federal clinical overview last reviewed March 13, 2026. Capture only motor/sensory/autonomic distinctions, signaling and symptom categories, compression/prolonged-pressure and vascular mechanisms, heterogeneity, and assessment limits. Exclude treatment, diagnosis, prognosis, disease catalogs, and every rope pressure, placement, timing, or recovery inference. |
| [NBS T300: Standard Bending Test for Rope Yarns](https://nvlpubs.nist.gov/nistpubs/nbstechnologic/nbstechnologicpaperT300.pdf) | `accept_targeted_scope` | 34 | Reusable 1925 federal primary study showing why breaking strength does not represent repeated bending and how test variables, twist, humidity, variability, and replication matter. Preserve its manila rope-yarn and historical-method scope; do not generalize to finished jute, hemp, synthetic rope, care, retirement, bondage, or human suspension. |

### Deferred sources

| Source | Score | Why it is reference-only |
|---|---:|---|
| [Frictional Sliding Strength of Knotted and Capstan Configurations](https://arxiv.org/abs/2604.06962) | 40 | The April 2026 preprint is exceptionally relevant modern mechanics research across a clove hitch, capstan configurations, braided rope, experiments, FEM, DER, and an elastica model, but the arXiv record is CC BY-NC-ND 4.0. Retain versioned metadata and seek derivative/training permission; do not use experimental HTML as a workaround. |
| [Safe Work Australia industrial rope-access guide](https://www.safeworkaustralia.gov.au/doc/guide-managing-risks-industrial-rope-access-systems) | 36 | The 36-page guide has outstanding anchor, inspection-record, rope-protection, planning, exclusion-zone, environment, and rescue coverage. Its specific notice names CC BY 4.0 but describes reuse as noncommercial, conflicting with the site-wide commercial-reuse statement. Clarification is required before transformation. |
| [Transport for NSW fibre-rope guide](https://www.nsw.gov.au/employment/dogging-and-rigging/guide/part-1-general-rigging-principles/lifting-equipment/fibre-rope-and-slings) | 33 | The current 2024 guide covers construction, handling, inspection, knots, slings, and assemblies. Its document-specific notice permits only personal or noncommercial reproduction and overrides the site's CC BY default; linked videos and several components are third-party. |
| [International Guild of Knot Tyers library](https://igkt.net/document-downloads/4-knotting-tools/24-knot-chart) | 32 | Strong knot nomenclature, chart, journal, history, and bibliography lead, but the official chart record is all-rights-reserved and key routes intermittently present a verification page. No challenge was bypassed; request item-level derivative/training permission. |
| [Karada House policies](https://karada-house.de/policies-rules-expectations/) | 29 | Unusually concrete rope-space guidance on affirmative consent, confidentiality, recording, accessibility devices, sobriety, scents, agency, adaptation, support, and accountability, but no open license was found. Venue-specific rules must not be universalized. |
| [NHS neuropathy and compartment-syndrome pages](https://www.nhs.uk/conditions/compartment-syndrome/) | 23 | OGL-compatible clinical pages, but their displayed review deadlines passed in October 2025 and February 2026. Defer until refreshed; England-specific emergency routing, treatment, tests, diagnosis, and rope-specific inference remain excluded. |
| [Consent Academy public resources](https://www.consent.academy/consent-education-resources.html) | 23 | Valuable violation, allegation-response, uncertainty, harm, withdrawal, victim-blaming, pronoun, and privacy topics, but bodies are split across the site and Patreon and no open license was found. Titles are leads, not facts; seek permission for an exact complete-article allowlist. |

### Rejected sources

| Source | Score | Reason |
|---|---:|---|
| [Studio Allegory](https://studio-allegory.com/about/) | 9 | The durable public body is thin, copyrighted, promotional, and dominated by current classes, events, memberships, rental, policies, and shop content. Founder/date/location/URL trivia is not a useful corpus. |
| [Hitchin' Bitches](https://www.hitchinbitch.com/) | 4 | The surviving all-rights-reserved site is a short mission statement plus a 2014–2019 event/link archive, with no substantive technique, history, safety, or evidence-bearing educational body. |

### Batch 009 rights, evidence, and safety controls

The two government-guide deferrals are deliberate. A site-wide open-license
footer does not override a more specific publication notice. Safe Work
Australia requires clarification of contradictory CC BY/noncommercial wording;
Transport for NSW requires broader permission than personal or noncommercial
reproduction. The arXiv paper's NoDerivatives term prevents transformed
Markdown even though its HTML is publicly readable. IGKT verification, Karada
House's resources challenge, Patreon bodies, paid material, member areas,
images, video, diagrams, and third-party standards were not bypassed or copied.

The accepted clinical and mechanics sources retain their evidence domains.
NINDS describes broad peripheral neuropathy, not a rope diagnostic test. NBS
T300 studies one historical manila rope yarn and a laboratory method, not
finished bondage rope or current certification. KECC is a dated collective code
for educators and producers, not an adjudication system or guarantee. None can
certify a person, knot, rope, anchor, hardpoint, bondage system, or human
suspension.

### Batch 009 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, and discovery tests. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, trainer, or
existing source-corpus artifact was read. Public review used ordinary official
or rightsholder routes, respected access and reuse terms, and did not bypass a
login, purchase, member area, Patreon restriction, verification challenge,
crawler restriction, missing body, or license. Discovery wrote no reviewed page
body to a training artifact.

## Discovery batch 010 outcome

`discovery_batch_010` manually reviewed 12 additional, nonduplicate source
candidates against their primary pages, methods and limitations, current reuse
terms, domain transfer, and safety value. It produced 1
`accept_high_priority`, 4 `accept_targeted_scope`, 5 `defer`, and 2 `reject`
decisions. The five accepted candidates expand the canonical corpus queue from
88 to 93 resources.

This batch deliberately separates technically strong evidence from permission
to transform it. It adds a CC BY primary friction-hitch experiment, a federal
aramid-rope fatigue study, a narrowly bounded CC BY nursing assessment source,
and two current OGL maritime system-hazard families. Equally relevant bowline,
injury, healthcare, offshore-lifting, and rope-access sources remain
reference-only where licenses, publication-specific copyright, coauthor rights,
or currency are unresolved.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Tree-climbing rope and friction-hitch study](https://www.mdpi.com/1999-4907/12/11/1457) | `accept_high_priority` | 39 | CC BY 4.0 primary evidence from 36 slow-pull tests, two specified synthetic arborist rope systems, Prusik and Valdotain tresse hitches, and nine replicates per treatment. Preserve load, slip, diameter, temperature, statistics, preliminary status, and call for dynamic and cyclic work; do not generalize the short quasi-static bench tests to natural fiber, bondage rope, falls, safe ratings, human suspension, or a universally best hitch. |
| [NBSIR 76-1159 aramid rope-sling fatigue study](https://www.nist.gov/publications/fatigue-and-weatherability-studies-aramid-fiber-rope-slings) | `accept_targeted_scope` | 36 | Historical federal primary research on 26 prototype aramid sling-leg specimens, two construction families, fatigue-spectrum and simulated-weathering tests, residual strength, permanent elongation, and end fittings. Preserve the report's decisive limitation: inadequate manufacturer end fittings prevented thorough evaluation and left its observations incompletely substantiated. Exclude loads, ratings, operations, vendor design, and transfer from helicopter cargo slings to bondage or human suspension. |
| [Open RN Nursing Skills](https://www.ncbi.nlm.nih.gov/books/NBK593210/) | `accept_targeted_scope` | 27 | CC BY 4.0 clinical vocabulary from only the neurological, cardiovascular, musculoskeletal, and integumentary assessment concept sections. Preserve sensory-versus-motor and expected-versus-unexpected distinctions plus professional-assessment limits; exclude diagnosis, treatment, procedures, self-tests, and every inference about rope-safe placement, pressure, duration, release, or recovery. |
| [MCA work-at-height and lifting guidance](https://www.gov.uk/government/publications/mgn-410-mf-amendment-3-work-at-height-regulations-2010/mgn-410-mf-amendment-3-the-merchant-shipping-and-fishing-vessels-health-and-safety-at-work-work-at-height-regulations-2010) | `accept_targeted_scope` | 33 | Current OGL text from MGN 410 amendment 3 and MGN 332 amendment 4 on planning, competence, supervision, rescue, foreseeable loads, conditional rope access, whole-system design, maintenance, examination, records, and mounts or attachments in the load path. Exclude maritime law, ratings, intervals, operations, certification, and any bondage or structural approval claim. |
| [MCA tensioned-rope hazard guidance](https://www.gov.uk/government/publications/mgn-592-mf-anchoring-mooring-towing-or-hauling-equipment/mgn-592-mf-amendment-2-anchoring-mooring-towing-or-hauling-equipment-on-all-vessels) | `accept_targeted_scope` | 33 | Current OGL whole-system concepts for complete load paths, bights under tension, snap-back, snag release, redirects, changing hazard zones, maintenance, communication, and exclusion. Exclude every maritime operational procedure and numeric value; massive vessel-line behavior must not become bondage, upline, anchor, rescue, or human-suspension instruction. |

### Deferred sources

| Source | Score | Why it is reference-only |
|---|---:|---|
| [Self-locking and Stability of the Bowline Knot](https://arxiv.org/abs/2509.10306) | 40 | Excellent experiments and FEM, DEM, and capstan-inspired theory, but the arXiv record is CC BY-NC-SA 4.0. Retain identifiers and simplified-model limitations and seek commercial derivative/model-training permission; do not use public HTML as a licensing workaround. |
| [Marks and injuries related to BDSM experiences](https://pmc.ncbi.nlm.nih.gov/articles/PMC10236207/) | 32 | The 2023 study is CC BY-NC 4.0. Its 513-person US convenience sample, self-selection, self-report, and broad BDSM or rough-sex scope cannot establish incidence, population prevalence, causality, safety, or rope-specific injury rates. |
| [Injury and healthcare utilization for kink-identified patients](https://academic.oup.com/jsm/article/18/10/1721/6955925) | 32 | The 2021 article is CC BY-NC-ND 4.0. Its 1,398-person US convenience sample is useful for disclosure, stigma, and delayed or avoided care, but not population prevalence, rope injury frequency, severity, or causal claims; transformation requires permission. |
| [HSE HSG221 offshore lifting guidance](https://www.hse.gov.uk/pubns/books/hsg221.htm) | 34 | The 2007 PDF's own all-rights-reserved notice requires prior written permission and overrides the generic current site footer. Its offshore, UK-law, and older context also needs a currency review before any narrowly transferable mechanics digest. |
| [WorkSafe NZ industrial rope-access guidelines](https://www.worksafe.govt.nz/assets/dmsassets/3/3212WKS-2-industrial-rope-access-guidelines.pdf) | 35 | Strong anchor, documentation, inspection, rescue, and system-design leads, but the PDF credits IRAANZ copyright, the general WorkSafe policy does not clearly authorize adaptation, and pre-2015-law material may be replaced or revoked. Resolve coauthor rights and currency first. |

### Rejected sources

| Source | Score | Reason |
|---|---:|---|
| [Geninka and Slavery](https://www.cambridge.org/core/journals/itinerario/article/geninka-and-slavery-jesuit-casuistry-and-tokugawa-legislation-on-japanese-bondage-1590s1620s/37CF82521DE28A9E00EDF9F1ADE915AC) | 17 | High-quality CC BY scholarship, but “bondage” means enslavement and bonded labor, not erotic rope bondage or kinbaku. Reject this search-term false positive to prevent false lineage, etymology, and Tokugawa-slavery claims. |
| [BDSM and the Complexity of Consent](https://www.mdpi.com/2411-5118/6/1/4) | 19 | A non-empirical, non-rope-specific theoretical paper whose detached formulations can blur clear capacity, ongoing agreement, and withdrawal boundaries. Rights are permissive, but operational transfer and safety are inferior to KECC and other clearer ethics sources. |

### Batch 010 rights, evidence, and safety controls

CC BY, OGL, or federal status is necessary but not sufficient for inclusion.
Every accepted source has a component audit and a narrow domain boundary.
Images, trademarks, linked standards, vendor material, and separately credited
components remain excluded. The arXiv and clinical studies are not transformed
under NonCommercial or NoDerivatives terms. HSG221's publication-specific
notice controls over a generic footer, and WorkSafe's accurate-reproduction
policy is not silently expanded into adaptation permission.

The accepted experiments and industrial guidance are not bondage validation.
Short quasi-static arborist tests, prototype aramid helicopter sling tests,
nursing assessment vocabulary, and maritime systems each retain their original
materials, scale, method, and purpose. None supplies a safe rope, knot, hitch,
load, pressure, duration, retirement rule, anchor, hardpoint, upline, body
placement, rescue procedure, or human-suspension guarantee. The two rejected
sources additionally protect against false Japanese lineage and ambiguous
consent-policy transfer.

### Batch 010 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, and discovery tests. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, trainer, or
existing source-corpus artifact was read. Public review used ordinary official,
publisher, repository, DOI, and license routes, respected current access and
reuse terms, and did not bypass a login, paywall, challenge, crawler
restriction, missing body, license, or publication-specific notice. Discovery
wrote no reviewed page body to a training artifact.

## Discovery batch 011 outcome

`discovery_batch_011` manually reviewed 12 additional, nonduplicate source
candidates against their primary body or official failure endpoint, methods,
component rights, evidentiary limits, redundancy, and cross-domain safety. It
produced 1 `accept_high_priority`, 4 `accept_targeted_scope`, 3 `defer`, and 4
`reject` decisions. The five accepted candidates expand the canonical corpus
queue from 93 to 98 resources.

The strongest addition is a rights-clear 2020 primary experiment on eight
loop-knot families, multiple load geometries, dressing, replication, static
break mechanics, infrared imaging, and microscopy. A complementary 2022
statistics paper explains why knot efficiency is a distribution rather than a
portable single percentage. The batch also adds bounded federal evidence on
pleated nylon deformation, OGL inspection-system guidance, and CC BY evidence
that visual or topological confidence does not reliably predict knot strength.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Revision of Commonly Used Loop Knots Efficiencies](https://doi.org/10.12693/APhysPolA.138.404) | `accept_high_priority` | 41 | CC BY 4.0 primary evidence on eight loop-knot families, exact synthetic kernmantle specimens, standard versus cross-load, I versus O geometry, dressing, calibrated static tests, statistical uncertainty, break versus untying, infrared imaging, and microscopy. Preserve source-bound method and limitations; exclude portable rankings, ratings, storage or retirement rules, and transfer to natural fiber, bondage, dynamics, body contact, uplines, or human suspension. |
| [Basic Statistical Properties of the Knot Efficiency](https://www.mdpi.com/2073-8994/14/9/1926) | `accept_targeted_scope` | 37 | CC BY 4.0 methods digest on heterogeneous breaking strength, ratio distributions, replication, dispersion, tolerance intervals, and misleading single-pair or ratio-of-means shortcuts. The empirical illustration comes from earlier work; equations and numeric examples must not become calculators or ratings. |
| [NBSIR 86-3375 pleated nylon-rope tests](https://www.nist.gov/publications/tensile-properties-pleated-synthetic-rope) | `accept_targeted_scope` | 36 | Historical federal evidence from 20 very large pleated nylon ropes, two sizes, untreated and temperature-conditioned groups, quasi-static rupture, deformation, and failure modes. Preserve small samples and unresolved cyclic, connection, moisture, debris, and field questions; exclude every sizing, force, energy, temperature, connection, vehicle-recovery, and safe-rating recipe. |
| [HSE INDG367 inspection guidance](https://www.hse.gov.uk/pubns/indg367.htm) | `accept_targeted_scope` | 32 | OGL inspection-system concepts: no single life boundary, degradation categories, unique identity, whole-length visual and tactile checks, competent judgment, withdrawal authority, records, training, triggers, and uncertainty. Exclude intervals, damage thresholds, law, standards, product rules, and transfer from industrial synthetic fall-arrest equipment to natural bondage rope. |
| [Tangled Physics](https://europepmc.org/article/MED/39439589) | `accept_targeted_scope` | 34 | CC BY 4.0 evidence from five experiments that photographs, renderings, dynamic videos, diagrams, and correct topology discrimination did not yield accurate relative-strength judgments for one controlled four-knot series. Use the EMBL-EBI JATS only; do not generalize to trained riggers, all knots, breaking strength, security, ratings, bondage, or human suspension. |

### Deferred sources

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Static anchor-knot tests](https://www.mdpi.com/2073-8994/16/2/167) | 34 | The CC BY study has a useful controlled static design, but its “safest knot” conclusion exceeds that method. It omits dynamic, cyclic, security, capsizing, other geometry, wet or aged behavior, and untying evidence; the authors withhold data because misinterpretation could cause fall accidents. Require specialist safety adjudication. |
| [HSE RR708 suspension-trauma first-aid review](https://www.hse.gov.uk/research/rrpdf/rr708.pdf) | 18 | The official historical PDF now returns 404. No mirror, cache, or archive was used. Body, publication-specific rights, components, and clinical currency must be recovered first and reconciled with the already accepted 2023 ICAR MEDCOM review. |
| [2021 suspension-trauma systematic review](https://europepmc.org/article/MED/34512820) | 25 | The PMC record states World Journal of Emergency Medicine copyright and no permissive license. Public repository access is not adaptation permission, and the older heterogeneous medical evidence needs current clinical review. |

### Rejected sources

| Source | Score | Reason |
|---|---:|---|
| [Dynamic lanyard prototype study](https://www.mdpi.com/1660-4601/17/3/1107) | 27 | Sixteen total specimens spread across prototype or commercial types and fall arrangements support broad claims about arresting people and possible energy-absorber redundancy. Minimal cell replication, a material-description error, and high-consequence numeric recipes make the incremental value unsafe. |
| [HSE LOLER guidance hub](https://www.hse.gov.uk/work-equipment-machinery/loler.htm) | 15 | Current and authoritative, but mostly UK legal navigation and heavily redundant with accepted MCA, Ontario, Actsafe, FEDEC, USBR, and INDG367 system sources. |
| [Human knot-tying robotics dataset](https://openreview.net/forum?id=t1AKFuIjRS) | 15 | Thirty demonstrations by five people across only overhand and figure-eight knots principally supply RGB-D, trajectory, and topology data for robotics, not verified mechanics, human instruction, bondage knowledge, or safety. |
| [Tokyo Weekender shibari history](https://www.tokyoweekender.com/art_and_culture/history/history-of-shibari-the-evolution-of-japanese-rope-bondage/) | 4 | All rights reserved and weakly sourced. Its rice-bag-to-hojojutsu-to-shunga narrative and unqualified “father of kinbaku” label would reintroduce exactly the neat false-lineage claims the source-critical corpus is designed to prevent. |

### Batch 011 rights, evidence, and safety controls

The APPA journal policy, MDPI article notices, EMBL-EBI JATS, NIST federal
status, and HSE OGL terms were checked independently of technical relevance.
Images, tables, standards, trademarks, vendor material, experiment media, and
separately credited components remain outside every accepted scope. The
ordinary-copyright medical review is not transformed, the missing HSE report is
not recovered through a mirror, and open licensing does not rescue an
underpowered high-consequence source.

Static strength is kept separate from knot security, dynamic response, cyclic
behavior, capsizing, dressing, tail behavior, inspectability, and ease of
untying. Historical vehicle-recovery and industrial fall-protection evidence
does not certify natural-fiber rope, a bondage knot, upline, anchor, hardpoint,
body placement, or human suspension. The cognitive study supports verification
and epistemic humility, not the claim that all practitioners are unable to
judge all knots.

### Batch 011 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, discovery test, and first-class corpus registry work for
an independently completed source artifact. No QA, training projection, manual
review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe, trainer,
or unrelated corpus artifact was read. Public review used ordinary official,
publisher, DOI, licensed API, and live failure routes; no login, paywall,
challenge, crawler restriction, cache, archive, or third-party mirror was
bypassed. Discovery wrote no reviewed source body into a training artifact.

## Discovery batch 012 outcome

`discovery_batch_012` reviewed 12 additional, nonduplicate candidates against
their official record or primary body, publication methods, component rights,
redundancy, conflicts, statistical support, and transfer risk. It produced 1
`accept_high_priority`, 5 `accept_targeted_scope`, 3 `defer`, and 3 `reject`
decisions. The six accepted candidates expand the canonical corpus queue from
98 to 104 resources.

This batch fills four evidence gaps: internal abrasion and tensile-fatigue
mechanisms; historical natural-rope variability; current synthetic- and
hemp-rope test methodology; and bounded primary evidence about structural-point
planning and harness-suspension physiology. None of these sources is treated as
a bondage test, load table, body-placement guide, care recipe, or human-
suspension certification.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [NOAA synthetic-rope deterioration report](https://repository.library.noaa.gov/view/noaa/9887) | `accept_high_priority` | 41 | Public-domain MITSG 90-18 adds internal fiber-to-fiber abrasion, tensile fatigue, creep-related behavior, model assumptions, experiment comparison, and validation limits. Its 41 scanned pages require manual OCR verification; equations, service-life predictions, ratings, field procedures, and bondage transfer are excluded. |
| [NBS statistical analysis of Manila rope](https://nvlpubs.nist.gov/nistpubs/jres/39/jresv39n6p551_A1b.pdf) | `accept_targeted_scope` | 38 | Federal historical evidence covers 863 observations from more than 800 three-strand samples, a nonrandom and heterogeneous acquisition universe, conditioning, measurements, static endpoints, and fiber, fabrication, and treatment variability. Strength values and current-rope claims are excluded. |
| [Weakest-link model for hemp rope](https://www.sciencedirect.com/science/article/pii/S0266892025000360) | `accept_targeted_scope` | 37 | CC BY 4.0 evidence from one rope and supplier at two tested lengths explains first-strand endpoints, independent versus correlated links, calibration, validation, and explicit omissions. Equations, numerical capacity, ratings, and extrapolation to other ropes or uses are excluded. |
| [OSHA Appendix C anchor-planning excerpt](https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926SubpartMAppC) | `accept_targeted_scope` | 29 | Only paragraphs h(1) and h(2) enter a short public-domain digest: plan early, use qualified structural judgment, evaluate existing or makeshift points, prevent connector pull-through, and preserve whole-system strength. Material examples are not DIY approvals, and no structure or human-suspension rig is certified. |
| [Synthetic-rope tensile characterization](https://www.nature.com/articles/s41598-023-44816-x) | `accept_targeted_scope` | 35 | CC BY 4.0 methods evidence covers fiber-yarn-rope hierarchy, nominal versus effective diameter, three synthetic materials, 220-to-196 result inclusion, exclusions, endpoints, and ANN black-box and data-access limits. Models, predictions, ratings, products, and size extrapolation are excluded. |
| [Randomized crossover suspension-syndrome trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC6517360/) | `accept_targeted_scope` | 33 | CC BY 4.0 primary evidence contributes a 20-person, 40-test healthy-male sit-harness crossover design, presyncope interruption, measured physiology, correction, and explicit population and apparatus limits. Timing cannot become a threshold, and the study supplies no rescue, treatment, rope-harness, bondage, or human-suspension advice. |

### Deferred sources

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Tourniquet nerve-compression review](https://link.springer.com/article/10.1186/s42490-020-00041-5) | 28 | CC BY 4.0 does not cure the tourniquet-to-rope scale mismatch, indirect evidence, declared device-company interests and patents, or separately licensed figure. Require independent clinical corroboration before any transformed use; no pressure, duration, cuff-width, treatment, prognosis, or rope-placement inference is permitted. |
| [Water imbibition and swelling of hemp, jute, and flax filaments](https://www.sciencedirect.com/science/article/pii/S0926669025012488) | 26 | The work concerns individual filaments, not rope, and is CC BY-NC-ND 4.0. Retain metadata only and seek commercial derivative and model-training permission if uniquely necessary; never convert filament swelling into washing, conditioning, shrinkage, strength, care, or retirement advice. |
| [NDL digitized works by Seiu Ito](https://ndlsearch.ndl.go.jp/search?cs=bib&from=0&q-author=%2200023065%22&size=20) | 30 | The national-library index identifies potentially valuable primary works, but public reading access is not blanket adaptation or translation permission. Each edition, contributor, illustration, work body, and historical claim needs a separate Japanese-language rights and source-critical audit. |

### Rejected sources

| Source | Score | Reason |
|---|---:|---|
| [PLOS offshore mooring-rope study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0318190) | 25 | The open paper omits clear replication and statistical support while using significance language and broad yarn-to-rope, material-selection, and service-life conclusions. All authors are employees of the company that supplied machines, resources, and data. |
| [Advanced Rigging Principles instructor workbook](https://www.osha.gov/sites/default/files/2021-04/Advanced%20Rigging%20Principles%20-%20Instructor%20Workbook.pdf) | 12 | This 308-page third-party grant workbook is not federal public-domain text and mixes standards, trademarks, images, quizzes, calculations, and high-consequence industrial lifting recipes already covered by cleaner bounded sources. |
| [OSHA natural and synthetic fiber rope slings](https://www.osha.gov/safe-sling-use/nat-synth-fiber) | 20 | Authoritative federal material, but almost entirely redundant legal and numeric industrial material-lifting guidance. Its personnel-platform exclusion reinforces why it cannot be repurposed as human-suspension evidence. |

### Prior-candidate rights update

The previously deferred [Frictional Sliding Strength of Knotted and Capstan
Configurations](https://arxiv.org/abs/2604.06962) now has a publisher version of
record at DOI `10.1016/j.jmps.2026.106628`. The publisher version is CC BY-NC
4.0, while the arXiv copy remains CC BY-NC-ND 4.0. Derivatives are therefore no
longer universally blocked, but commercial and intended model-training use is
still not assumed. The candidate remains deferred: neither the publisher's
NonCommercial route nor the preprint's NonCommercial and NoDerivatives route is
used to populate the corpus.

### Batch 012 rights, evidence, and safety controls

Public domain and CC BY status were verified separately from scientific value.
Open licensing did not rescue underreported statistics, a commercial evidence
chain, or unsafe cross-scale conclusions. NonCommercial and NoDerivatives works
remain metadata-only. National-library access was not mistaken for copyright,
translation, illustration, or model-training permission, and OSHA hosting was
not mistaken for public-domain status when the hosted workbook was authored by
a third party.

Every accepted experiment preserves its exact material, construction, sample,
apparatus, endpoint, loading mode, exclusion, and validation boundary. Static
breaking behavior remains separate from security, cyclic or dynamic response,
moisture, aging, damage, knots, body contact, and field use. Structural-point
planning is preserved as a need for competent evaluation, never as a checklist
that certifies a tree, ceiling, beam, connector, hardpoint, upline, or human-
suspension system. Medical studies retain population and harness limits and do
not generate timing thresholds, first aid, treatment, or prognosis.

### Batch 012 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, discovery tests, and first-class corpus registration work
for an independently completed source artifact. No QA, training projection,
manual review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe,
trainer, or unrelated source-corpus artifact was read. Public review used
ordinary official, publisher, DOI, repository, PubMed, and license routes and
did not bypass a login, paywall, challenge, crawler restriction, or missing
license. Discovery wrote no reviewed source body into a training artifact.

## Discovery batch 013 outcome

`discovery_batch_013` reviewed 12 additional, nonduplicate candidates against
their actual first-party body or official failure endpoint, exact license,
methods, scale, redundancy, and cross-domain transfer risk. It produced 1
`accept_high_priority`, 4 `accept_targeted_scope`, 5 `defer`, and 2 `reject`
decisions. The five accepted candidates expand the canonical corpus queue from
104 to 109 resources.

The batch fills three important gaps. First, a public-domain NOAA report models
where eight-strand construction produces contact pressure and relative movement,
while explicitly limiting its own conclusions to qualitative, assumption-bound
behavior. Second, two CC BY experiments make the fiber-to-yarn-to-rope scale
problem concrete for jute yarn and prototype rayon rope. Third, bounded federal
sources explain why an indoor wood connection or anchorage must be evaluated as
part of its supporting structure and complete load path, rather than approved by
appearance or a generic hardware label.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [NOAA eight-strand rope structural model](https://repository.library.noaa.gov/view/noaa/42461) | `accept_high_priority` | 40 | Public-domain MITSG 89-28 adds eight-strand geometry, strand and plied-yarn hierarchy, contact-pressure regions, relative motion, friction-limit assumptions, and lateral-contraction sensitivity. All 51 scanned pages require manual review; equations, values, design prescriptions, predictions, and all natural-rope or human-suspension transfer are excluded. |
| [Jute-yarn mechanical uncertainty](https://pmc.ncbi.nlm.nih.gov/articles/PMC5459092/) | `accept_targeted_scope` | 33 | CC BY 4.0 evidence from 15 single-ply specimens shows why fiber measurements cannot stand in for yarn and why one natural-material batch needs explicit uncertainty. Numerical properties, possibility-distribution recipes, and transfer from one yarn batch to finished bondage rope are excluded. |
| [Prototype rayon-rope marine degradation](https://www.sciencedirect.com/science/article/pii/S0025326X25003923) | `accept_targeted_scope` | 33 | CC BY 4.0 multiscale methods compare fiber, yarn, and eight-yarn braided prototype rope under biotic and abiotic seawater conditions. Supplier and coating caveats, five-replicate rope or yarn conditions, and request-only data remain visible; degradation values and care or retirement advice do not enter. |
| [USFS Wood Handbook Chapter 8: Fastenings](https://research.fs.usda.gov/treesearch/62253) | `accept_targeted_scope` | 35 | A public-domain concepts digest can distinguish member, fastener, connection, and load path and preserve grain, moisture, geometry, condition, and failure-mode dependence. Every table, equation, load, sizing or installation rule, current-code claim, and DIY ceiling or hardpoint recipe is excluded. |
| [OSHA rope-descent anchorage evaluation factors](https://www.osha.gov/laws-regs/standardinterpretations/2017-11-20) | `accept_targeted_scope` | 29 | Only the qualified-person factor list and separation of inspection, test, certification, maintenance, supervision, communication, and records enter a short public-domain digest. The historical enforcement memo is not current legal advice, an inspection checklist, or certification of any tree, ceiling, anchor, bondage rig, or human-suspension system. |

### Deferred sources

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [NOAA marine rope design brief 42](https://repository.library.noaa.gov/view/noaa/46899) | 18 | The record supplies a public-domain label, MIT-T-86-008, and checksum, but its exact first-party PDF route returns 404 and no abstract exists. No mirror, cache, archive, title inference, or search snippet substitutes for body review. |
| [Climbing-rope wear, fatigue, and heat study](https://www.sciencedirect.com/science/article/pii/S259012302502540X) | 37 | The directly relevant paper is CC BY-NC-ND 4.0 and makes high-consequence claims from one dynamic polyamide kernmantle rope type and bounded artificial exposures. Require permission and independent rope-engineering review; no results, thresholds, or visual-retirement rules enter. |
| [Peripheral-nerve microcirculation review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4145887/) | 24 | This older heterogeneous review is CC BY-NC-SA 3.0, heavily numeric, mixes nerve and experimental contexts, and overlaps stronger accepted clinical sources. Commercial and ShareAlike compatibility plus qualified clinical transfer review are unresolved. |
| [Canadian Conservation Institute textile guide](https://www.canada.ca/en/conservation-institute/services/preventive-conservation/guidelines-collections/textiles-costumes.html) | 31 | The museum guide has useful fiber and deterioration vocabulary, but Canada.ca grants general reproduction only for noncommercial purposes and components have separate rights. Museum textile preservation cannot become bondage-rope cleaning, care, strength, or retirement guidance. |
| [Synthetic-rope fatigue experimental methodology](https://www.sciencedirect.com/science/article/pii/S2452321624002543) | 38 | The full-scale bending-over-sheave or drum methodology could fill a cyclic-testing gap, but the exact publisher license is CC BY-NC-ND 4.0. Permission and qualified textile-rope review are required before any transformed method digest. |

### Rejected sources

| Source | Score | Reason |
|---|---:|---|
| [CDC wire-rope termination report](https://stacks.cdc.gov/view/cdc/206576) | 27 | The 152-page public-domain contractor report is non-peer-reviewed, metallic, mining-specific, numeric, and centered on industrial sockets, swaging, zinc or resin, and steel-wire fatigue. Its material system cannot inform textile knots, splices, bondage rope, or human suspension. |
| [Dea Nexa personal rope advice](https://www.deanexa.com/jute-rope-maintenance-tips/) | 12 | Ordinary-copyright personal posts combine unsupported hygiene and material claims, flame and chemical hazards, subjective retirement advice, and a replicable DIY self-suspension setup framed as accessibility. Specificity and personal success are not reliable evidence. |

### Batch 013 rights, evidence, and safety controls

Exact article-level license links were followed rather than treating “open
access” as permission. Two otherwise valuable fatigue studies are metadata-only
because their publisher links resolve to CC BY-NC-ND, and the Canadian source is
deferred because commercial transformation is not authorized. Public-domain
labels did not rescue a missing body or an out-of-material mining report.

Material hierarchy remains explicit: an elementary fiber is not a yarn, a yarn
is not a laid or braided rope, a prototype rayon braid is not jute or hemp, an
eight-strand marine model is not a three-strand natural rope, and steel wire is
not textile fiber. Static endpoints remain separate from cyclic or dynamic
behavior, abrasion, heat, moisture, aging, security, and service life.

Structural material is similarly bounded. Wood fastener research and
occupational rope-descent guidance explain variables and professional process;
they do not tell a reader how to build, inspect, rate, or certify a ceiling,
beam, joist, truss, tree, connector, anchor, hardpoint, upline, or human-
suspension setup. The rejected personal blog is retained as a contamination
quarantine, not as instructional evidence.

### Batch 013 access and clean-room statement

Repository inspection remained restricted to the governing discovery ledger,
corpus queue, report, discovery tests, and first-class corpus registration work
for independently completed source artifacts. No QA, training projection,
manual review, evaluation, heldout, holdout, OOD, shadow, benchmark, probe,
trainer, or unrelated source-corpus artifact was read. Public review used
ordinary official, publisher, DOI, repository, PubMed Central, and license
routes. No login, paywall, challenge, crawler restriction, cache, archive, or
third-party mirror was bypassed, and no reviewed page body was written to a
training artifact during discovery.

## Discovery batch 016 outcome

`discovery_batch_016` reviewed 12 exact-new source identities after normalized
title, DOI, PMID, PMCID, document and source-identity comparison with the
existing ledger and queue. It produced 0 `accept_high_priority`, 8
`accept_targeted_scope`, 4 `defer`, and 0 `reject` decisions. The eight accepted
sources expand the canonical Markdown queue from 122 to 130 resources.

This batch strengthens two underrepresented layers without inventing rope
facts. Four controlled or observational knot-learning studies support bounded
curriculum design: alternate concise egocentric demonstration with immediate
practice, assess intermediate topology as well as final form, demonstrate
recognizable errors and recovery, provide process feedback, and retain an
expert reference to detect learner-to-learner drift. A systematic review adds
the testing principle that knot-security results depend on material,
configuration, environment and loading method. Cotton-yarn research adds a
similarly bounded construction-and-contact-geometry lesson. None establishes a
secure bondage knot, load rating or suspension method.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [Action observation plus execution for knot learning](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.793849/full) | 40 | CC BY controlled study of 54 knot-naive participants supports interleaving concise egocentric demonstration and immediate execution and checking both intermediate steps and final form. Preserve its short-term, no-retention and no-safety-outcome limits; exclude recipes, media, effects, neuroscience overclaim and rope-suspension transfer. |
| [Repeated knot-skill transmission chains](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1545120/full) | 38 | CC BY exploratory pilot found accumulated learner-to-learner form changes. Retaining an expert reference and checking topology or function is an inference, not a tested prescription. Preserve the two-chain, 16-analyzed-participant, constrained fMRI and no-functional-test limits; exclude the noose stimulus and all safety inference. |
| [Coping models and process feedback](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.01171/full) | 36 | CC BY simulation supports showing common errors and recovery and giving feedback on process plus the next adjustment. Preserve baseline imbalance, multiple-teacher and artificial-material limits and the distinction between self-efficacy and competence; exclude surgical steps and transfer. |
| [Physical and observational knot learning](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0185044) | 34 | CC BY 4.0 study can support combining egocentric observation with physical rehearsal and a surprise reconstruction check. Preserve its 22-completer scale, condition confound, no familiarity control, restricted data and exploratory neuroscience; exclude recipes and safety claims. |
| [Surgical knot-security systematic review](https://doi.org/10.1097/JS9.0000000000000298) | 34 | CC BY 4.0 EMBL-EBI JATS supports only the principle that configuration, material, size, environment and loading protocol affect measured security and that static and cyclic tests differ. Preserve the single-extractor and review-protocol limits; exclude all surgical recipes, rankings, numbers and rope transfer. |
| [Egyptian cotton-yarn dynamic-friction experiment](https://mej.researchcommons.org/home/vol11/iss2/5/) | 30 | CC BY 4.0 methods digest may state that measured friction depended on tested yarn construction, processing, speed, wraps and guides. Preserve the conflicting 1986, 2020 and 2021 date signals and noarchive header; yarn is not finished jute, hemp or bondage rope. |
| [NBSIR 76-1146 historical fall-safety study](https://www.nist.gov/publications/study-personal-fall-safety-equipment) | 31 | Eligible federal text adds only historical experimental-design, environmental-conditioning and component-versus-system evaluation cautions. Preserve the scan-versus-metadata date conflict and obsolete fall-arrest context; exclude every number, configuration, standard, physiology claim and human-suspension transfer. |
| [Hasluck, *Knotting and Splicing Ropes and Cordage*](https://www.gutenberg.org/ebooks/70874) | 24 | Public-domain supplement may retain exact-new text-supported topology and vocabulary for meshing, netting, mats, lashings, whipping, seizing and decorative patterns after comparison with Verrill and Dana. Exclude nooses, restraint, scaffold, wire, figures, security claims and obsolete practices. |

### Permission-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [NPS Conserve O Gram on fatty-acid spew](https://www.nps.gov/articles/conserve-o-gram-9-1-leather-dressings.htm) | 35 | General NPS public-domain language does not resolve possible joint authorship with a nonfederal conservator. Seek page-specific permission; photos, quoted reports, diagnosis and treatment instructions remain excluded even if the general material-aging caution is cleared. |
| [Nayland Blake oral history](https://www.aaa.si.edu/collections/interviews/oral-history-interview-nayland-blake-17406) | 35 | Rare first-person leather-scene, performance-art and restraint-sculpture history remains ordinary copyright, and the Smithsonian host signals `ai-train=no`. Retain metadata only pending explicit permission; never generalize one recollection into universal history or instruction. |
| [*LGBTQ America* theme study](https://www.nps.gov/subjects/lgbtqheritage/upload/lgbtqtheme-index.pdf) | 32 | The National Park Foundation PDFs are expressly all rights reserved despite federal hosting. Relevant Western leather and S/M chapter/page leads remain metadata-only pending permission and must not be conflated with Japanese rope lineage or technical evidence. |
| [Disability and BDSM communication study](https://doi.org/10.1007/s12119-022-10058-8) | 27 | Springer holds an exclusive license, and PMC's special COVID-era reuse term ended with the WHO public-health emergency in 2023. Sensitive participant narratives, mixed self-disclosed disabilities and the seven-person kink subset remain body-excluded pending current permission. |

### Batch 016 evidence, transfer, and access controls

Pedagogy findings remain tied to each study's task, comparison condition,
sample, timing and outcome. Correctness, self-efficacy, recall, shorter
demonstrations and fMRI patterns are not interchangeable with functional knot
security or safe rope practice. Surgical suture, cotton yarn, historical
fall-arrest equipment and century-old ropecraft are each distinct from modern
finished bondage rope. Every accepted extraction scope blocks ratings, recipes,
body contact, anchors, uplines and human-suspension transfer.

Open licenses were verified at the article level. Project Gutenberg's license
and territorial notice remain attached to the public-domain work; the Mansoura
`noarchive` signal permits transformed Markdown but bars raw-PDF retention.
Possible nonfederal joint authorship, all-rights-reserved third-party work on a
federal host, ordinary oral-history copyright, an explicit `ai-train=no`
signal, an exclusive publisher license and expired temporary reuse language all
remain hard body-ingestion gates.

### Batch 016 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests
plus ordinary first-party public sources, official preservation copies and
their explicit license, terms or robots routes. No QA, trainer, projection,
evaluation, heldout, holdout, OOD, shadow, benchmark, probe, unrelated corpus
body or unrelated repository artifact was read. No login, paywall, challenge,
robots restriction, mirror, cache, archive or snippet reconstruction was
bypassed, and no reviewed source body entered a training artifact during
discovery.

## Discovery batch 017 outcome

`discovery_batch_017` reviewed 12 exact-new source identities after normalized
title, DOI, PMID, PMCID, report, document and source-identity comparison with
the existing ledger and queue. After a contractor-rights correction, it
produced 0 `accept_high_priority`, 11 `accept_targeted_scope`, 1 `defer`, and 0
`reject` decisions. The eleven accepted sources expand the canonical Markdown
queue from 130 to 141 resources.

This batch fills concrete and tree-anchor evaluation, anchor governance,
natural-fiber mechanisms, capstan-model limits, cross-cultural knot methods,
bottoming accessibility and skill pedagogy. It deliberately does not produce
an anchor rating, hardpoint design, proof-test recipe, natural-rope care
schedule, secure-knot ranking, pain treatment or Japanese kinbaku lineage
claim. Each accepted source remains in its original civil-infrastructure,
workplace, rope-access, forestry, arboriculture, materials-science,
mathematical, archaeological, health-survey or surgical-education domain.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [FHWA adhesive-anchor advisory T5140.34](https://www.fhwa.dot.gov/Bridge/t514034.pdf) | 41 | Eligible federal text preserves an investigated Boston tunnel suspended-ceiling and ceiling-panel collapse context, with no casualty detail in the advisory, plus sustained-tension and overhead adhesive-anchor creep, installation sensitivity, system qualification, trained installation and continuous inspection. Exclude standards, loads, recipes and products; never treat it as a DIY hardpoint or human-suspension qualification. |
| [USBR testing and verifying rope-access anchors](https://usbr.gov/research/projects/detail.cfm?id=6390) | 39 | Eligible federal project-page and two-page bulletin text adds simulated installation defects, substrate variation, failure-mode diversity and angle-sensitive non-additive load sharing. Protected final report Document 1018 must not be requested, inferred or reconstructed; exclude every capacity, proof-load and procedure. |
| [USACE EP 385-1-101 fall prevention and protection](https://www.publications.usace.army.mil/USACE-Publications/Engineer-Pamphlets/) | 35 | Current federal governance supports hierarchy of controls, documented authorized/competent/qualified roles, qualified-person anchorage decisions, inspection, site-specific rescue plans and drills. Use only the targeted public-library PDF; exclude incorporated standards, numbers, forms and operational procedures. |
| [Trees as supports and anchors in forest operations](https://www.mdpi.com/1999-4907/9/8/467) | 37 | CC BY 4.0 review preserves interacting root plate, stem, decay, soil, root architecture, slope, wind, species and dynamic-load factors plus sparse-evidence and model-calibration limits. It cannot identify a safe tree, branch or human-suspension anchor. |
| [Nondestructive tree-anchorage assessment experiment](https://www.mdpi.com/1999-4907/14/3/533) | 37 | CC BY 4.0 comparison across more than 280 trees adds small-tree errors, algorithm failures and soil/species inference limits. Preserve the relevant Arbosafe hardware/software conflict prominently; exclude pulling procedures, factors, thresholds and DIY assessment. |
| [Hemp-rope constituent environmental-aging experiment](https://www.sciencedirect.com/science/article/pii/S2352710224011392) | 41 | Article-level CC BY 4.0 supports only the bounded result that combined accelerated ultraviolet and hygrothermal exposure reduced measured properties in one commercial hemp-rope constituent context. No calendar-life, jute equivalence, care, retirement, remaining-strength or suspension inference is allowed. |
| [Hemp fibre properties and textile processing review](https://www.mdpi.com/1996-1944/15/5/1901) | 32 | CC BY 4.0 narrative review adds bast-fibre heterogeneity, plant/process/batch variability, moisture sorption, swelling, hysteresis and retting as controlled biological degradation. Raw-fibre/textile evidence is not finished-rope testing; recipes and schedules are excluded. |
| [Cross-cultural computational knot study](https://www.cambridge.org/core/journals/cambridge-archaeological-journal/article/ties-that-bind-computational-crosscultural-analyses-of-knots-reveal-their-cultural-evolutionary-history-and-significance/B35E9C2DB89FA3C81F58F309B8F754FA) | 41 | CC BY 4.0 article covers 338 visually evidenced knots from 86 traditions and distinguishes visual resemblance, topology, orientation/material/tightness and historical transmission. Preserve source and preservation biases; exclude images, HRAF/ABoK bodies and unaudited OSF data. It is not security evidence or kinbaku-lineage proof. |
| [Elastic-rod capstan equilibria](https://infoscience.epfl.ch/entities/publication/b8b8fb31-a6b4-43b3-a4a0-1e21b633fe6c) | 33 | CC BY 4.0 conceptual digest may contrast the classical perfectly flexible filament with finite bending stiffness, thickness, length, contact and boundary geometry. Idealized planar rod equilibria are not an operational textile-rope, redirect, anchor or safety calculator. |
| [Chronic pain and BDSM exploratory study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12435486/) | 37 | CC BY 4.0 methods-first digest may retain non-universal participant-reported benefits and challenges while preserving the 525-person convenience sample, 201 self-reported pain subset, demographic skew, missing pain type/severity and selection bias. BDSM or rope is not pain treatment or clinical advice. |
| [Instructional video and knot proficiency RCT](https://mededu.jmir.org/2018/1/e9/) | 33 | Final CC BY 4.0 article supports randomized pre/post, physical practice, blinded structured assessment and the negative result that video was not established as superior. Exclude the all-rights-reserved preprint, surgical steps and media; no retention or rope outcome was tested. |

### Permission-gated reference

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [FHWA-HRT-14-071 concrete-barrier NDE TechBrief](https://www.fhwa.dot.gov/publications/research/infrastructure/structures/14071/index.cfm) | 37 | The useful negative NDE results are in research funded by FHWA but performed under contract by Engineering Software Consultants. The first-party page supplies no federal authorship, copyright assignment or reuse grant. Federal funding and hosting are insufficient, so body transformation waits for written FHWA or contractor confirmation. |

### Batch 017 rights, evidence, and safety controls

The initial federal-public-domain classification of FHWA-HRT-14-071 was
reversed after its contractor credit was checked. Distribution, availability,
government sponsorship, disclaimer and quality-assurance notices do not prove
government authorship or a reuse grant. The USBR public project page and
bulletin remain distinct from its explicitly protected detailed report. USACE
use is limited to the targeted official public-library document because its
index access route is crawler-blocked.

Article-level CC BY licenses apply only to the named open works. Publisher-wide
TDM reservations, robots failures or rate limits were honored by allowing only
targeted official acquisition, never a site crawl. Figures, tables, raw data,
preprints, incorporated standards, ethnographic databases, commercial
products, participant narratives and third-party components remain excluded or
separately gated.

Anchor and tree sources preserve uncertainty, qualified roles and evaluation
process—not capacities or selection rules. Materials evidence preserves exact
fiber, construction, specimen and exposure boundaries. Mathematical and
topological studies do not prove friction security, load behavior, safe knots
or lineage. The chronic-pain study supplies participant perceptions rather
than treatment evidence, and the teaching trial supplies no retention or rope
validation.

### Batch 017 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests
plus ordinary first-party public sources, targeted official documents and
their explicit license, terms or robots routes. No QA, trainer, projection,
evaluation, heldout, holdout, OOD, shadow, benchmark, probe, unrelated corpus
body or unrelated repository artifact was read. No login, protected report,
paywall, challenge, robots restriction, mirror, cache, archive or snippet
reconstruction was bypassed, and no reviewed source body entered a training
artifact during discovery.

## Queue completion synchronization

The extraction queue now treats the sealed direct-training Markdown registry
as the completion authority. Twenty discovery candidates that already had a
registered canonical Markdown artifact were changed from `pending` to
`complete` and bound to their exact `canonical-markdown:<resource_id>` artifact
IDs. A contract test rejects both stale pending entries for registered sources
and unregistered sources falsely marked complete. This prevents conversion
workers from repeatedly selecting already finished high-priority sources.

## Discovery batch 018 outcome

`discovery_batch_018` reviewed 15 exact-new source identities after normalized
title, DOI, PMID, PMCID, report, document and source-identity comparison with
the existing ledger and queue. It produced 0 `accept_high_priority`, 5
`accept_targeted_scope`, 9 `defer`, and 1 `reject` decisions. The five accepted
sources expand the canonical Markdown queue from 141 to 146 resources.

This batch adds historical self-evacuation governance, structured knot-skill
pedagogy, a clearly attributed practitioner account of Japanese-rope lineage,
person-centered accessible communication, and historical concrete-anchor
failure-mode vocabulary. It does not turn rescue, dental education, ADA legal
guidance, autobiography, or legacy anchor research into bondage technique or a
human-suspension qualification.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [HSE Operational Circular 282/31](https://www.hse.gov.uk/foi/internalops/ocs/200-299/282_31.htm) | 36 | Eligible OGL text supports a historical self-evacuation control hierarchy, compatibility, competence, practical assessment, checks and error-under-stress concepts. Preserve its 2003 version and 2013 review date; exclude old intervals, devices, standards and operational recipes, and never present it as current rescue or bondage procedure. |
| [Structured knot-tying training randomized study](https://www.egms.de/static/en/journals/zma/2023-40/zma001630.shtml) | 35 | CC BY 4.0 HTML supports Peyton's four-step structure, feedback, deliberate practice and immediate structured assessment in 124 dental students. Preserve prior-experience, subjective-score, post-hoc-threshold and no-retention limits; exclude surgical steps, recipes, effects and transfer to rope security or suspension. |
| [Andy Buru, “History of Japanese Rope Bondage”](https://andyburu.se/book/first-paradox/chapter-3/history-of-japanese-rope-bondage/) | 19 | The written text alone is explicitly CC BY 4.0. Retain a quotation-light, attributed practitioner-primary account with its commercial setting and historical uncertainty; exclude images, media, commerce and technique, and never present testimony as consensus or independent proof of lineage. |
| [DOJ ADA effective-communication guidance](https://www.ada.gov/resources/effective-communication/) | 35 | Eligible public-domain text supports choosing communication methods for the person and context, consulting the person, allowing time, clarifying instead of guessing, and considering aids and training. Preserve its covered-entity legal context; do not universalize legal duties or imply that accommodation makes a rope technique safe. |
| [NISTIR 6096 post-installed-anchor literature review](https://www.nist.gov/publications/post-installed-anchors-literature-review) | 36 | Eligible federal text adds historical failure-mode vocabulary, test-variable distinctions and evidence gaps. Exclude every capacity, load, product, standard, table, figure, installation or test recipe; it cannot approve, size, proof-test or certify a ceiling, hardpoint or human-suspension system. |

### Permission- or access-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Jutsum, *Knots, Bends, Splices*](https://www.gutenberg.org/ebooks/30983) | 26 | The underlying United States work is public domain, but current Project Gutenberg terms prohibit automated acquisition. Retain metadata only unless a compliant human single-item process supplies a lawful copy; then deduplicate topology against existing historical manuals and exclude obsolete safety and strength claims. |
| [Aldridge, *A Handbook of Knots and Marlinespike Seamanship*](https://www.gutenberg.org/ebooks/78376) | 22 | The public-domain work has the same automation gate and substantial overlap. No mirroring, deep-fetching or bypass is allowed, and Project Gutenberg marks remain excluded. |
| [Ontario firefighter rope-rescue guidance](https://www.ontario.ca/document/guidance-notes-for-firefighters/rope-rescue) | 33 | The current guidance has no page-level Open Government Licence and default Crown terms do not authorize this commercial transformed corpus. Written permission is required; operational rescue steps and bondage transfer remain excluded. |
| [NZQA rope-rescue skill standards 40866 and 40867](https://www.nzqa.govt.nz/nqfdocs/units/pdf/40866.pdf) | 37 | Current skill-standard material is restricted to citation or reference. Keep only competency-map metadata pending explicit commercial derivative permission; do not reproduce assessment bodies or convert them into procedure. |
| [Finished hemp and sisal rope temperature study](https://library.csbe-scgab.ca/all-publications/2712%3Atensile-strength-and-elongation-of-hemp-and-sisal-ropes-at-different-temperatures) | 35 | No open license authorizes transformation. Its small supplier-confounded design is metadata-only pending permission and cannot establish jute equivalence, care, service, retirement or human-suspension advice. |
| [Disability justice in BDSM and sexual consent](https://cjds.uwaterloo.ca/index.php/cjds/article/view/427) | 31 | CC BY-NC-ND does not permit the intended transformed commercial corpus. Seek permission before body use, preserve critical-essay and jurisdiction limits, and exclude technique, medical or individualized legal advice. |
| [FEMA E-74 nonstructural earthquake guide](https://www.fema.gov/emergency-managers/risk-management/earthquake/training/fema-e-74) | 39 | Contractor authorship, extensive third-party components and a 403 on the official PDF jointly block extraction. Federal sponsorship and hosting do not establish public-domain rights; use metadata only and never equate ceiling bracing with a human-suspension hardpoint. |
| [Japanese Rope Bondage under the Orientalist Gaze](https://iccs.chss.nycu.edu.tw/en/activity.php?USN=729) | 22 | The university event and report have no reuse license. Seek speaker, reporter and university permission; if cleared, retain only attributed interpretations and cross-check them rather than treating them as consensus history. |
| [Western Canada Mine Rescue Manual](https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/mineral-exploration-mining/documents/health-and-safety/emergency-preparedness/western_canada_mine_rescue_manual_october2017.pdf) | 37 | British Columbia terms are all rights reserved, and the manual is a mixed government-industry compilation with third-party material. Written component-level permission is required; rescue mechanics cannot be transferred directly to bondage or human suspension. |

### Rejected source

| Source | Score | Reason |
|---|---:|---|
| [Hojōjutsu Research Society](https://www.hojojutsu.org/) | 19 | The copyrighted commercial site explicitly disallows named AI and data bots across the site and includes high-risk combative-restraint material. Do not crawl or transform it; retain only organization, URL, copyright and permission-contact metadata. |

### Batch 018 rights, evidence, and safety controls

Rights were resolved at article, page and component level rather than inferred
from public access, government hosting or a generic site label. A text-only CC
BY grant does not cover images or media. OGL and federal-public-domain status do
not cover incorporated standards, logos or third-party components. CC
BY-NC-ND, all-rights-reserved Crown material, contractor work, missing licenses,
and explicit AI-bot disallows remain hard transformation gates.

The accepted materials stay inside their evidence domains. A historical
self-evacuation circular is not a current rescue plan; a dental-student
immediate post-test is not rope-safety evidence; a practitioner chapter is not
consensus history; ADA communication guidance is not universal scene law; and
a 1998 concrete-anchor review is not a hardpoint evaluation. Deferred natural-
rope, ceiling, rescue, disability and lineage sources remain useful discovery
leads without leaking unsupported body claims into training.

### Batch 018 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests
plus ordinary first-party public sources and their explicit license, terms or
robots routes. No QA, trainer, projection, evaluation, heldout, holdout, OOD,
shadow, benchmark, probe, unrelated corpus body or unrelated repository
artifact was read. No login, paywall, challenge, crawler restriction, mirror,
cache, archive or snippet reconstruction was bypassed, and no reviewed source
body entered a training artifact during discovery.

## Discovery batch 019 outcome

`discovery_batch_019` manually reviewed 10 exact-new source identities after
title, DOI, PMID, PMCID, document and canonical-URL deduplication. It records 3
`accept_targeted_scope`, 7 `defer`, and no rejection. The 3 accepted sources
expand the canonical Markdown queue from 146 to 149 resources.

This batch adds temporary-support governance, empirical evidence that knot
configuration reasoning is a distinct non-rigid spatial task, and a randomized
error-coping teaching model. It also preserves unusually valuable wood
inspection, natural-rope aging, accessible consent, component-skill,
multi-angle feedback, terminology and first-aid leads without admitting bodies
whose rights, official access route or clinical currency remain unresolved.

Repeated search leads for Masakazu Tanaka's 2014 kinbaku-globalization abstract
and the 2025 PLOS offshore-mooring study were exact identity matches for
existing batch 005 and batch 012 records and therefore did not create duplicate
candidates or queue jobs.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [HSE temporary-works FAQs](https://www.hse.gov.uk/construction/faq-temporary-works.htm) | 33 | Current OGL text supports short-duration-is-not-less-critical, end-to-end responsibility, site-assumption validation, load-path completeness, relevant competence and separation of design from coordination. Exclude construction recipes, calculations, standards and universalized UK law; it cannot design or approve a ceiling-tension frame, hardpoint, upline or human suspension. |
| [Knot-based non-rigid spatial-skill study](https://link.springer.com/article/10.1186/s41235-025-00621-w) | 34 | CC BY 4.0 methods support the bounded fact that reasoning about deformable string configurations is a measurable spatial challenge. Recognition under material or orientation changes and backward reasoning are teaching hypotheses only: the study did not train physical knot tying or test identity, security, loading, body contact or suspension. |
| [Coping-error video knot-training randomized study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10156798/) | 34 | CC BY 4.0 evidence adds active error diagnosis, selectable error-and-recovery examples, peer discussion and blinded verification to mastery demonstration. Preserve the null attempts-and-time primary outcome and all single-center, handedness, partner, questionnaire and no-retention limits; exclude procedures, checklists, videos, effects and rope-safety transfer. |

### Permission-, access-, or currency-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [USFS *Wood and Timber Condition Assessment Manual*, third edition](https://research.fs.usda.gov/treesearch/69136) | 38 | The report mixes federal, retired-federal, university and private authors plus a nonfederal foreword and credited artwork. Federal hosting does not place the whole manual in the public domain. Keep the contributor map and topic inventory only until chapter and component rights are cleared. |
| [Finished natural-rope marine-degradation study](https://www.sciencedirect.com/science/article/pii/S2468550X20300824) | 31 | The publisher marks the article CC BY 4.0, but its ordinary HTML route exposed only metadata and abstract and the official PDF route returned 403. No mirror or search-text reconstruction is allowed; a compliant official body route is needed before methods and limitations can be audited. |
| [WIRE accessible research-consent guidelines](https://journals.sagepub.com/doi/10.1089/aut.2024.0263) | 36 | The participatory guidance is highly relevant, but the publisher labels it restricted access, credits Mary Ann Liebert copyright and requires permission. NIH funding and a public manuscript route do not grant derivative commercial model-training rights. |
| [Operant-learning knot-skill study](https://link.springer.com/article/10.1007/s11999-015-4555-8) | 26 | Ordinary copyright blocks transformation. The small non-equivalent groups, single method-skilled assessor, weak demonstration control, TAGteach or clicker-training employment and institutional funding links also require careful bias review. |
| [Self-directed multi-angle video-feedback trial](https://link.springer.com/article/10.1186/s12909-026-09104-y) | 29 | CC BY-NC-ND does not authorize a transformed commercial corpus. Keep the 62-person single-center design, tiny feedback subgroups, no detected feedback-mode difference, company co-development, funding and AI-language-editing disclosure as metadata while seeking permission. |
| [Veterinary friction-knot terminology survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC11665728/) | 32 | The journal and PMC record preserve a useful finding that specialists may assign inconsistent names to configurations, but the association holds copyright and the official issue says personal use only. Permission is required; veterinary suture naming does not control shibari terminology or prove knot security. |
| [NHS fainting and first-aid pages](https://www.nhs.uk/symptoms/fainting/) | 28 | The text is generally OGL, but the fainting page's February 2026 review date and the first-aid overview's March 2025 review date have passed. Rights compatibility cannot cure overdue high-stakes medical guidance; retain dates and terms only pending current clinical reconciliation. |

### Batch 019 rights, evidence, and safety controls

Rights were resolved at page, article and component level. Public access, PMC
deposit, federal funding or government hosting was never treated as a reuse
license. OGL and CC BY bodies were accepted only with attribution, date,
change, component and evidence-domain controls. Mixed authorship, ordinary
copyright, personal-use-only terms, CC BY-NC-ND, an official-route 403 and
overdue clinical review dates remain hard transformation gates.

The accepted sources do not become operational rope instructions. Temporary-
works governance cannot approve a body-support structure; a visual spatial-
reasoning measure is not physical tying competence; and a laparoscopic coping-
video trial is not evidence of rope-knot security or human-suspension safety.
All teaching implications are explicitly marked for direct validation in rope
education, and no source supplies a capacity, service-life, medical-treatment,
anchor, hardpoint or upline rule.

### Batch 019 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests,
the metadata package for an official report already under rights audit, and
ordinary first-party public sources plus their explicit licenses, terms and
robots routes. No QA, trainer, projection, evaluation, heldout, holdout, OOD,
shadow, benchmark, probe, unrelated corpus body or unrelated repository
artifact was read. No login, paywall, challenge, crawler restriction, mirror,
cache, archive, search-text reconstruction or official-route 403 was bypassed,
and no reviewed source body entered a training artifact during discovery.

## Discovery batch 020 outcome

`discovery_batch_020` manually reviewed 10 exact-new source identities after
title, DOI, document and canonical-URL deduplication. It records 4
`accept_targeted_scope`, 5 `defer`, and 1 `reject`. The accepted sources expand
the canonical Markdown queue from 149 to 153 resources.

The batch adds two complementary video-and-practice studies, a natural-rope
review whose main value is its explicit lack of finished-rope evidence, and a
physical-knot motor-learning study that tests whether precise laboratory
measures generalize to ecological skill. Restricted consent and feedback
sources remain permission leads. A public-domain historical manual is also
quarantined because its rope lesson incorporates unidentified Bell System
material, despite GovInfo's document-level federal rights statement.

Repeated search leads for the 2024 BDSM consent-norms article, the 2024 autism
and BDSM article, the PubPub kinbaku ritual article and a SEDICI stage-shibari
paper were existing identities and did not create duplicate candidates.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [Household-item online knot-training study](https://link.springer.com/article/10.1186/s12909-024-05549-1) | 31 | CC BY mixed-methods evidence adds low-resource, self-paced video practice plus the learner-reported need for usable camera views, supervision, formal feedback and blended physical practice. Preserve 304 eligible and 71 voluntary participants, lack of control, clinical practice and retention, and the researcher relationship; exclude the procedure, media, materials recipe and effects. |
| [Video-adjunct knot-training randomized trial](https://link.springer.com/article/10.1007/s44217-024-00101-1) | 31 | CC BY evidence supports video only as an adjunct to a common coached workshop. Preserve the 29-person randomization, blinded assessment, unmonitored external learning and four-week attrition to 6 versus 8; the follow-up efficiency result was weak and nonsignificant. It is not evidence of independent rope competence or safety. |
| [Wool-rope aquaculture systematic review](https://www.mdpi.com/2071-1050/14/15/9011) | 32 | The CC BY review's useful result is negative: proposed wool-rope strength and durability were untested, marine-wool evidence was sparse and not finished rope, and fibre or carpet properties are not finished-rope performance. Do not transfer the review to hemp, jute, care, service life, body support or suspension. |
| [Ecological knot motor-learning study](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0319715) | 32 | CC BY evidence found only low-to-moderate correspondence between laboratory and ecological motor-learning tasks. Preserve the 42-person, one-session design, measurement mismatch and lack of long-term retention; exclude the knot identity and media. Laboratory sequence scores cannot replace observed physical rope skill. |

### Permission- or component-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Feedback-valence randomized trial](https://link.springer.com/article/10.1007/s00464-025-12285-0) | 34 | CC BY-NC-ND does not authorize a transformed commercial digest. Keep the 42-person three-arm simulation, broad outcome direction and limits as metadata while seeking permission; surgical feedback effects cannot be presumed to transfer to rope teaching. |
| [Video peer-feedback randomized trial](https://link.springer.com/article/10.1186/s12909-025-08029-2) | 31 | CC BY-NC-ND blocks transformation. The no-significant-difference suturing result, task-dependent findings, no power analysis, possible peer leniency and no retention are useful metadata, not proof that peers replace expert rope feedback. |
| [Association EDO four-Cs rope-prevention leaflet](https://www.asso-edo.fr/telechargement) | 24 | This direct French rope-consent lead is licensed CC BY-NC-ND for unmodified noncommercial distribution. Translation, condensation and commercial model-training use require permission from the relevant rightsholders. |
| [*Running the Yellow Light* consent thesis](https://digitalcollections.trentu.ca/node/23537) | 38 | Trent's record says the author holds all rights. The abstract's 1,118-person result includes a sensitive role-identity comparison that must not be taught without permission and full recruitment, instrument, category, subgroup, multiplicity and ethics review. |
| [War Department *Shop Work*, Lesson 8](https://www.govinfo.gov/app/details/GOVPUB-W-c7e159acf398602aa98d8a1ab3ff35d5) | 26 | GovInfo labels the 1942 manual a federal public-domain work, but the preface acknowledges Bell System Practices sections inside the rope lesson on printed pages 99–133. A page-and-paragraph component audit must separate third-party material before transformation; all dated operational and load advice remains excluded. |

### Rejected source

| Source | Score | Why it was rejected |
|---|---:|---|
| [Prorope 8 mm sisal in-house test](https://prorope.eu/blogs/news/8-mm-sisal-breaking-strength-test) | 10 | An ordinary-copyright seller article reports five in-house tests but lacks complete specimen, setup and raw-data provenance, links directly to products, compares the seller's alternatives and discloses AI-assisted drafting. It cannot establish general sisal, jute, bondage-rope or human-support performance. |

### Batch 020 rights, evidence, and safety controls

CC BY bodies are queued only as bounded methods, findings and limitations
digests. NoDerivatives, NonCommercial and all-rights-reserved sources remain
metadata-only permission leads. A government-hosted document-level rights
statement did not override an internal third-party acknowledgment, and public
readability did not become reuse permission. The commercial in-house test was
rejected independently on rights, provenance and commerce grounds.

The accepted teaching studies do not become tying recipes, and their surgical
or generic motor tasks do not establish rope-bondage knot identity, competence,
security, body-contact or suspension safety. The material review establishes
an evidence gap rather than a strength, hygiene, care or retirement rule. No
candidate supplies a load, hardpoint, upline, medical-treatment or human-
suspension rule.

### Batch 020 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests
plus ordinary first-party publisher, repository, association, vendor and
government sources and their visible rights statements. No QA, trainer,
projection, evaluation, heldout, holdout, OOD, shadow, benchmark, probe,
unrelated corpus body or unrelated repository artifact was read. No login,
paywall, challenge, crawler restriction, mirror, cache, archive or snippet
reconstruction was bypassed, and no reviewed source body entered a training
artifact during discovery.

## Discovery batch 021 outcome

`discovery_batch_021` manually reviewed 12 exact-new source identities after
title, DOI, document, repository-handle, clipping-ID, eBook-number and
canonical-URL deduplication. It records 4 `accept_targeted_scope`, 7 `defer`,
and 1 `reject`. The accepted sources expand the canonical Markdown queue from
153 to 157 resources; the discovery ledger grows from 255 to 267 records.

The batch adds one historical federal finished-Manila-rope test, two public-
domain historical seamanship witnesses, and a CC BY industrial inspection-
observability article. It also preserves unusually strong natural-fibre
failure, occupational inspection, accessibility, consent, sexual-health,
Latin American genealogy, and Itō Seiu archival leads without crossing
copyright, AI-access, component, license-compatibility or evidence gates.

The SEDICI handle and Waseda clipping are exact-new identities despite thematic
overlap with earlier lineage records; their duplication penalties preserve
that overlap. No candidate body was compared with an existing corpus body.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [NBS T198, *Results of Some Tests of Manila Rope*](https://doi.org/10.6028/nbst.6078) | 34 | NIST's rights-clean 1921 primary report adds yarn, strand and lay vocabulary, purchase-order sampling, specimen preparation, test-design distinctions and manufacturer/lot variability. Exclude all numbers, formulas, tables, specifications and modern recommendations. Historical three-strand regular-lay commercial Manila rope is not evidence for modern products, 6 mm jute/hemp, care, working loads, body contact or suspension. |
| [Brady, *The Kedge-Anchor*, eBook 77729](https://www.gutenberg.org/ebooks/77729) | 29 | U.S. public-domain Part I can add historical knots, hitches, bends, splices, seizings, serving and teaching sequence. The main Gutenberg site is for human users only: conversion must use a sanctioned mirror or documented manual single-item route. Exclude images, endorsements, automatic summary, wrapper, naval operations, numbers and obsolete safety. |
| [Dana, *The Seaman's Friend*, eBook 40958](https://www.gutenberg.org/ebooks/40958) | 25 | A second U.S. public-domain historical terminology witness can corroborate cordage, marlinespike, knot and splice vocabulary after aggressive semantic deduplication. Use the same sanctioned acquisition rule and exclude maritime law, operations, numerical proportions, load claims, media and near-duplicate prose. |
| [Camera-Based Visual Rope Inspection](https://www.bibliothek.tu-chemnitz.de/ojs/index.php/innoTRAC/article/view/462) | 33 | CC BY industrial evidence adds one-sided-view, inspector-fatigue and absent-record limitations plus multi-angle surface imaging and the need to combine indicators. Exclude thresholds, figures, equipment recipes and remaining-life or discard rules. High-modulus synthetic running-rope inspection does not reveal internal condition or validate natural-fibre shibari rope or human support. |

### Permission-, access-, or license-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [IMCA natural-fibre embarkation-ladder failure](https://www.imca-int.com/resources/safety/safety-flashes/0721-failure-of-natural-fibre-rope-in-embarkation-ladder/) | 30 | The anonymized incident may illustrate hidden abrasion near a securing thimble, but IMCA requires express reproduction permission and member-supplied text/images may have separate ownership. One marine incident is not a controlled materials study or a jute-care rule. |
| [CCOHS Fibre Rope Slings](https://www.ccohs.ca/oshanswers/safety_haz/materials_handling/sling_fibre.html) | 28 | CCOHS's default terms permit internal copies but require written authorization for modification and redistribution. Industrial inspection categories may be useful if cleared; numerical retirement, temperature, care and working-load guidance cannot transfer to body support. |
| [Quartz Rope accessibility curriculum](https://www.quartzrope.com/class-descriptions) | 23 | First-party neurodivergence, mobility, sensory, dysphoria/euphoria and tied-person co-teaching topics are valuable, but the site is ordinary copyright and promotional lived experience. Permission, component clearance and independent clinical review are required. |
| [Rope Office Hours](https://ropeofficehours.com/) | 28 | Ordinary copyright plus explicit robots blocks for GPTBot, ChatGPT-User, CCBot and other AI agents make this metadata-only. No body review or conversion may occur without written access and derivative commercial training authorization. |
| [SafeLink Alberta *Kinky Sex* toolkit](https://safelinkalberta.ca/toolkits/) | 36 | The strongest BDSM hygiene and consent lead in this batch is still blocked: terms prohibit modification/redistribution, robots advertises `ai-train=no` and blocks GPTBot, and adapted AIDS Committee/community components require separate permission. The PDF body was not ingested. |
| [Latin American shibari genealogy, SEDICI 10915/187026](https://sedici.unlp.edu.ar/handle/10915/187026) | 30 | The situated Spanish exploratory paper fills a major regional-history gap, but CC BY-NC-SA compatibility with intended commercial training is not assumed and robots restrict named/bulk agents. Permission and converter-specific access review are required. |
| [Waseda Itō Seiu semeba manuscript clipping](https://archive.waseda.jp/archive/vm-view.html?arg=%7B%22clipping_id%22%3A%22242eb88de504d971d879546e9c0418e1%22%7D) | 21 | Official catalog context is valuable, but the image-viewer route is robots-disallowed and image/manuscript/digitization/transcription rights require permission. It is evidence about staged imagery, not a tying manual or self-proving kinbaku lineage. |

### Rejected source

| Source | Score | Why it was rejected |
|---|---:|---|
| [RACKWiki Bondage](https://rackwiki.org/wiki/Bondage) | 14 | The footer exposes no usable contributor license, while the article mixes unsafe or weakly sourced duration, circulation, nerve, capillary-check and emergency implications. Rights ambiguity and medical/procedural contamination each independently fail. |

### Batch 021 rights, evidence, and safety controls

NIST employee-authored technical-series text and the two United States public-
domain books were accepted with attribution, component and historical-domain
controls. Gutenberg's public-domain determination does not override its server
access terms: automated main-host retrieval is forbidden, sanctioned mirrors
or a manual single-item route are mandatory, trademarks are excluded and non-
U.S. status remains jurisdiction-specific. The innoTRAC article is queued only
under its article-level CC BY 4.0 terms with figures, products and numerical
rules excluded.

Ordinary copyright, internal-copy-only terms, CC BY-NC-SA uncertainty,
component ownership, named-agent blocks, `ai-train=no`, and robots-disallowed
image routes remain hard gates. Robots permission was never treated as a
license, and open licensing was never treated as evidence that a source's
domain transfers to bondage or human suspension.

Historical seamanship and 1921 materials evidence do not become current knot-
security, rope-care, load or body-support rules. Industrial surface inspection
does not establish internal condition, retirement, remaining life or safe use.
No accepted source supplies a modern load, capacity, hardpoint, upline,
medical-treatment or human-suspension rule.

### Batch 021 access and clean-room statement

The read-only scout inspected only the four authorized discovery metadata
ledgers and ordinary first-party source, repository, publisher, terms, license
and robots routes. No QA, trainer, projection, evaluation, heldout, holdout,
OOD, shadow, benchmark, probe, experiment, existing corpus body or unrelated
repository artifact was read. The scout stopped at explicit AI-agent and image-
viewer prohibitions, did not retrieve the SafeLink toolkit body or Waseda
images, and did not use a mirror during discovery. No login, paywall,
challenge, access block, cache, archive or snippet reconstruction was bypassed,
and no reviewed source body entered a training artifact during discovery.

## Discovery batch 022 outcome

`discovery_batch_022` manually reviewed 10 exact-new source identities after
title, DOI, document, article, wiki and canonical-URL deduplication. It records
2 `accept_targeted_scope`, 7 `defer`, and 1 `reject`. The accepted sources
expand the canonical Markdown queue from 157 to 159 resources; the discovery
ledger grows from 267 to 277 records.

The batch adds two unusually useful category boundaries. A 1933 NIST paper
explicitly says its Manila-rope fiber-color study did not test serviceability
and documents several reasons appearance can mislead. A 2024 CC BY primary
study establishes microbial retting as a jute-fiber production stage, not
evidence about the hygiene of finished rope. Both are queued as bounded
evidence-and-limitations digests, not procedures or safety rules.

The batch also records strong edge-failure, industrial incident, climbing-rope
care, anatomy teaching, synthetic degradation, school-lineage and Japanese
knot-history leads without crossing copyright, `ai-train=no`, NoDerivatives,
clinical-provenance, PDF-robots or contamination gates.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [NIST, *Evaluation of Manila-Rope Fiber for Color*](https://doi.org/10.6028/jres.011.057) | 38 | The public-domain 1933 primary paper explicitly did not test serviceability. It adds the bounded lesson that fiber color or finished-rope appearance alone cannot establish strength, deterioration, remaining life or fitness, because lubricant, dust, exposure, bleaching and construction can change appearance. Exclude every measurement recipe, value, figure, table, specification and color-based discard rule; do not transfer it to modern jute or hemp, care, hygiene, body contact or suspension. |
| [Datta et al., jute water-retting study](https://doi.org/10.1186/s12870-024-04970-4) | 35 | CC BY primary evidence defines retting as a microbial production stage that degrades pectin and other cementing material so bast-fiber bundles separate from stem tissue. Preserve the one-cultivar laboratory design, commercial-consortium and funding disclosures, and under- or over-retting quality boundary. It did not study spun yarn, finished rope, cleaning, mold, hygiene, care, body contact or suspension. |

### Permission-, access-, license-, or evidence-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [UIAA and Norwegian Climbing Federation sharp-edge report](https://www.theuiaa.org/special-report-sharp-edges-and-rope-cuts/) | 34 | The June 2025 revised report is a valuable edge-cut mechanism, incident and uncertainty lead, but UIAA copyright and translation or component rights require express derivative commercial training permission. Dynamic climbing-rope incidents cannot set a natural-fiber padding, edge-tolerance or human-suspension rule. |
| [IRATA historical safety bulletins](https://irata.org/safety-bulletins) | 30 | Current robots expressly declares `ai-train=no`, `use=reference` and named-agent blocks. IRATA also says historical recommendations may not reflect current best practice and should be disregarded in favor of current guidance. Do not retrieve or transform bulletin bodies without written access, training and component permission plus currentness review. |
| [BMC climbing-rope guide](https://www.thebmc.co.uk/en/ropes-a-guide-for-climbers-and-mountaineers) | 20 | BMC terms restrict copying, adaptation and derivatives to personal noncommercial use and require written permission otherwise. Most rope-type, care, retirement and fall-factor coverage also overlaps stronger sources and cannot transfer to natural-fiber body support. |
| [BDSM Dojo anatomy and safety resources](https://www.bdsmdojo.com/links/class-resources) | 24 | Public Google Drive viewing is not a reuse license. The handouts, slides, nerve card and videos need rightsholder permission, component clearance and qualified independent clinical review; an unspecified medical-professional role does not validate medical claims. |
| [Oman et al. climbing-rope degradation study](https://doi.org/10.1016/j.rineng.2025.106471) | 28 | CC BY-NC-ND does not authorize a transformed commercial digest. Its single dynamic polyamide kernmantle product and controlled abrasion, cyclic-load and high-temperature exposures cannot establish natural-fiber care, ordinary heat effects, retirement or human-suspension safety. |
| [Harukumo-Juku encyclopedia](https://harukumo.com/en/enzyklopaedie/) | 17 | First-party school lineage, philosophy and terminology could be useful if permission is granted, but the ordinary-copyright site also mixes commerce with durations, body configurations, relative-safety claims, suspension directions and a rope-collar exercise. Those operational claims remain quarantined even if lineage text is cleared. |
| [Nukada knotting studies, 1956 and 1958](https://doi.org/10.14890/minkennewseries.19.3-4_179) | 25 | The two J-STAGE records have society copyright, no CC license and PDF routes disallowed by robots. Publisher permission, an authorized converter route, Japanese specialist review and contextualization of obsolete or offensive anthropology are required; harmful restraint and strangulation material remains excluded. |

### Rejected source

| Source | Score | Why it was rejected |
|---|---:|---|
| [Kinbakuwiki](https://kinbaku.wiki/index.php?title=Hauptseite) | 8 | No usable contributor-text license was found. Sampled content is weakly and circularly sourced, heavily duplicates stronger lineage references and generalizes that role framing can replace explicit verbal negotiation in Japan. That cultural consent claim is unsafe; status, role and implied expectations are not consent. |

### Batch 022 rights, evidence, and safety controls

The two accepted bodies have affirmative transformation routes: eligible NIST
federal public information and article-level CC BY 4.0. Attribution, edition,
checksum, component and modification controls remain mandatory. Public access,
a free PDF, a Google Drive share, an institutional platform or robots allowance
was never treated as reuse permission. Ordinary copyright, personal-
noncommercial-only terms, CC BY-NC-ND, absent contributor licenses, explicit
`ai-train=no` and robots-disallowed PDF routes remain hard gates.

Neither accepted source produces an operational rule. The NIST paper is
negative evidence against visual overconfidence, not a color-based inspection
method. The jute study describes plant-stem processing, not finished-rope
contamination. Deferred synthetic climbing, industrial rope-access and
mountaineering evidence cannot be silently transferred to natural-fiber
bondage rope. Practitioner status, school identity or role framing does not
establish clinical authority, safety or consent. No candidate supplies a load,
capacity, hardpoint, upline, medical-treatment or human-suspension rule.

### Batch 022 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests,
ordinary first-party source and rights pages, the complete two accepted source
bodies, and reference-only metadata needed to resolve gates. No QA, trainer,
projection, evaluation, heldout, holdout, OOD, shadow, benchmark, probe,
experiment, existing corpus body or unrelated repository artifact was read.
The scout stopped at IRATA's `ai-train=no` signal and J-STAGE's PDF prohibition,
did not retrieve the gated bodies for corpus use, and did not use a mirror,
cache, archive or snippet reconstruction. No login, paywall, challenge or
access block was bypassed, and no reviewed source body entered a training
artifact during discovery.

## Discovery batch 023 outcome

`discovery_batch_023` manually reviewed 10 exact-new source identities after
title, DOI, document, article, repository and canonical-URL deduplication. It
records 3 `accept_targeted_scope`, 6 `defer`, and 1 `reject`. The accepted
sources expand the canonical Markdown queue from 159 to 162 resources; the
discovery ledger grows from 277 to 287 records.

This gap-focused batch adds three open primary studies that isolate different
levels of rope and knot behavior: finished coir rope, a defined fishing-knot
and net system, and two slip-knot topologies in single silk fibers. Their value
is not a new strength, care, knot-security or suspension rule. Together they
show why fiber, material, knot, construction and assembled-system behavior
must be distinguished and why friction or water exposure cannot be assigned a
single context-free effect.

The batch also preserves a technically and culturally valuable Q'eswachaka
lead, a federal naval manual identity, alternative-fiber rope tests, a natural-
fiber antibacterial review, rice-straw rope mechanics and cotton-net aging
without crossing publisher, community, component, currentness, chemical-
safety, method-quality or unsupported-transfer gates.

### Newly accepted sources

| Source | Score | Exact useful scope and boundary |
|---|---:|---|
| [Coir ropes in French Polynesia pearl farming](https://doi.org/10.1016/j.clcb.2024.100111) | 39 | CC BY primary evidence compares individual coir fiber with machine-twisted and hand-braided finished ropes, including dry and water-saturated conditions. Preserve construction-quality effects, the fiber-to-rope prediction limit, conflicts and in-situ-aging gap. Exclude all values, recipes, ratings and life-cycle numbers; coir pearl-farming rope is not jute or hemp body-support rope. |
| [Knot strength in a biodegradable gillnet](https://www.nature.com/articles/s41598-024-66474-3) | 36 | CC BY tests PA6 and PBSAT fishing monofilaments at monofilament, double-weaver's-knot and net-panel scales. It supports only the bounded lesson that tensile behavior alone missed geometry, tightening, sliding, friction, curvature and prior degradation effects. Preserve the degraded-material and small-sample limits; exclude values, procedures, code and transfer to other knots or rope bondage. |
| [Slip knots in raw and degummed silk](https://www.nature.com/articles/srep18222) | 29 | CC BY single-fiber evidence shows that two superficially related slip-knot topologies loosened or tightened differently under pull and that added surface friction could help or harm sliding behavior depending on topology. Exclude optimization, values, recipes and operational instructions; a silkworm-silk fiber is not finished rope and supplies no security or human-support rule. |

### Permission-, provenance-, or specialist-review references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Q'eswachaka natural-fiber rope characterization](https://publicacoes.softaliza.com.br/cilamce/article/view/8120) | 35 | Rare Festuca, Q'eswa, Maqui and Duro vocabulary and bridge-renewal context merit preservation, but ordinary copyright plus community, cultural and component authority require explicit permission and appropriate community-centered review. |
| [NAVSEA NSTM Chapter 613](https://maritime.org/doc/nstm/ch613.pdf) | 32 | A third-party 1999 mirror and current official cross-reference establish a useful document lead, not current authenticated text, federal authorship or component rights. Find an official current body and audit every component before conversion; Navy values and policy do not transfer to bondage. |
| [Cotton gillnet degradation](https://doi.org/10.1111/j.1444-2906.2008.01517.x) | 26 | Intermittent fishing-and-drying versus continuous seawater immersion could inform study design, but society copyright requires permission and treated cotton mesh cannot establish jute or hemp care, hygiene or retirement. |
| [Rice-straw rope mechanics](https://www.sciencedirect.com/science/article/pii/S0926669022002126) | 25 | The fiber-to-rope hierarchy is a potential methods lead, but ordinary copyright, partial body access and one rice-variety context make this metadata-only pending permission and full review. |
| [Antibacterial properties of hemp and other natural-fiber plants](https://bioresources.cnr.ncsu.edu/resources/antibacterial-properties-of-hemp-and-other-natural-fibre-plants-a-review/) | 20 | BioResources permits attributed noncommercial reuse only. Plant, extract, compound, isolated-fiber or composite observations do not prove finished-rope hygiene, mold resistance, cleaning or safe body contact; permission and claim-level primary-source review are required. |
| [Banana-peduncle fiber rope characterization](https://ejournal.brin.go.id/jsmi/article/view/9439) | 18 | CC BY-SA availability does not cure small samples, apparatus limits, specimen-code inconsistencies, manufacture and treatment confounding or hazardous chemical-treatment contamination. Textile-methods and chemical-safety review are required before any body use. |

### Rejected source

| Source | Score | Why it was rejected |
|---|---:|---|
| [Hibiscus bark fiber promoted as rope material](https://www.e3s-conferences.org/articles/e3sconf/ref/2020/18/e3sconf_icepp2020_04007/e3sconf_icepp2020_04007.html) | 8 | Despite CC BY and a rope-focused title, the paper makes and tests no finished rope: it tests seven chemically treated single fibers, speculates about a marine application and has conflicting DOI suffixes. Open licensing does not cure those evidence failures. |

### Batch 023 rights, evidence, and safety controls

The three accepted articles carry article-level CC BY 4.0 and will require
attribution, checksums, version identity, modification notices, disclosure
retention and component exclusions. Separately linked code, supplements,
figures and cited-source bodies are not swept into the license assumption.
Ordinary copyright, noncommercial terms, uncertain ShareAlike compatibility,
third-party mirrors and unresolved community or component authority remain
hard gates even when a document is public or marked for public release.

No result is treated as a recipe or rating. Coir, cotton, rice straw, banana
peduncle, Hibiscus bark, silk, PA6 and PBSAT remain distinct materials and
forms. A single fiber is not a yarn or rope; a fishing knot is not every knot;
a net panel is not a bondage harness; pearl farming, naval service and marine
immersion are not human suspension. None of these candidates supplies a body-
contact, care, cleaning, hygiene, retirement, working-load, hardpoint, upline,
medical-treatment or human-suspension rule.

### Batch 023 access and clean-room statement

Discovery read only the governing candidate ledger, queue, report and tests,
ordinary first-party publisher, repository, government and rights pages, the
complete bodies needed for accepted-source and rejection audits, and limited
reference-only bodies or metadata needed to resolve gates. No QA, trainer,
projection, evaluation, heldout, holdout, OOD, shadow, benchmark, probe,
experiment, existing corpus body or unrelated repository artifact was read.
No login, paywall, challenge, crawler restriction, cache, archive or snippet
reconstruction was bypassed, and no reviewed source body entered a training
artifact during discovery.

## Discovery batch 024 outcome

`discovery_batch_024` manually reviewed 10 exact-new source identities after
title, DOI, document, record, organization and canonical-URL deduplication. It
records 1 `accept_high_priority`, 4 `accept_targeted_scope`, 5 `defer`, and 0
`reject` decisions. The five accepted sources expand the canonical Markdown
queue from 162 to 167 resources; the discovery ledger grows from 287 to 297
records.

This batch follows mechanics and incident evidence back to first-party or
primary records. It adds a CC BY analytical rope–sheave model, an explicitly
unverified ATSB Manila-rope occurrence brief, a federal structural-support
concept page, and tightly bounded DOE and NASA lifecycle-governance material.
It rejects the tempting but wrong inference that public federal hosting makes
incorporated consensus standards reusable government text.

The batch also preserves a high-value clove-hitch paper, a disability-led
sexual-health toolkit, a physical-knot mechanics paper, a natural-fiber
climbing-rope incident alert and a recent first-person Naka-associated lineage
record without crossing `ai-train=no`, NoDerivatives, NonCommercial, AAAS,
ordinary-copyright, currentness, component or corroboration gates.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Static rope–sheave contact model](https://doi.org/10.1016/j.mechmachtheory.2023.105334) | `accept_high_priority` | 35 | CC BY primary mechanics distinguishes fully stuck and saturated-friction contact regions under a defined idealized static model. Preserve assumptions and why a dynamic or impending-slip capstan relation is not universal; exclude equations, numbers, designs and every transfer from an ideal rod and elevator sheave to natural rope, body contact or suspension. |
| [ATSB occurrence MB-2025-001](https://www.atsb.gov.au/occurrence-briefs/mb-2025-001) | `accept_targeted_scope` | 39 | CC BY official record adds a finished-Manila-side-rope failure and several operator-reported possible contributors. Front-load that ATSB did not investigate and did not verify the supplied information; exclude third-party media, ISO material, values, instructions and causal or bondage-rope generalization. |
| [DOE-STD-1090-2020](https://www.energy.gov/ehss/articles/doe-std-1090-2020) | `accept_targeted_scope` | 35 | Current federal material can add DOE-original high-level accountability, classification, role, competence, change-control and record concepts only. Paragraph-level review must exclude every ASME, ANSI, consensus-standard, OSHA, regulatory, visual, value and operational component. |
| [OSHA Anchorages and Attachments eTool](https://www.osha.gov/etools/shipyard/general-requirements/scaffolds/hanging-staging/key-components1) | `accept_targeted_scope` | 35 | Federal-authored concepts add the useful boundary that support condition and connection geometry jointly determine the force path and a visible point is not a qualified hardpoint. Exclude VSRA media, regulations, numbers, attachment rankings and any conclusion about a tree, ceiling, beam or human support. |
| [NASA-STD-8719.9C](https://standards.nasa.gov/node/278) | `accept_targeted_scope` | 35 | NASA-specific lifecycle framing connects design, hazard analysis, testing, inspection, maintenance, operations, records, personnel and mishap-driven revision. Exclude all OSHA and voluntary-consensus material, Appendix A, values, equipment procedures and transfer to bondage or human suspension. |

### Permission-, access-, currentness-, or corroboration-gated sources

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [The mechanics of a clove hitch](https://doi.org/10.1016/j.eml.2022.101788) | 36 | Valuable cylinder-contact, self-contact-nip and localized-tension mechanics are blocked by CC BY-NC-ND plus EPFL's explicit `ai-train=no`. Do not retrieve or transform the body without written authorization. |
| [Topological mechanics of knots and tangles](https://doi.org/10.1126/science.aaz0135) | 35 | The specialized elastic-fiber study could add crossing, handedness, writhe, stress, stability and untying concepts, but the accessible PDF says some rights reserved under an exclusive AAAS license. |
| [ASAN/NCIL Real Talk](https://autisticadvocacy.org/projects/completed/realtalk/) | 30 | Disability-led autonomy, accommodation, intersectionality and direct-communication principles are valuable, but the site is ordinary copyright. Permission is required and any rope-scene application must be labeled as an inference. |
| [DIO Safety Alert SA 2021/08](https://assets.publishing.service.gov.uk/media/61c1a4558fa8f5037ac74533/SA_2021-08_Failure_of_assault_course_climbing_ropes.pdf) | 25 | The natural-fiber climbing-rope failure is a useful lead, but no canonical status or rescission record was found and Defence, JSP and British Standards components require reconciliation. |
| [KinbakuOdyssey lineage profile](https://www.kinbakuodyssey.com/index.php/about/) | 18 | Self-reported Odys, Akira Naka, Nawa Naka Kai and Semenawa Research Meetings chronology may document a recent lineage branch, but 2026 site copyright requires permission and broader lineage claims need independent corroboration. |

### Batch 024 rights, evidence, and safety controls

The Escalona article has an explicit repository-level CC BY 4.0 route. ATSB
and eligible federal text have affirmative reuse routes only after their own
component exclusions. DOE and NASA standards are mixed works: eligible agency-
original prose does not carry ASME, ANSI, OSHA, regulatory or voluntary-
consensus material into the corpus. DOL federal status likewise does not cover
VSRA photographs. ATSB's license excludes third-party graphics, logos and the
Coat of Arms, and unresolved robots retrieval requires manual low-rate capture.

No source becomes an operational recipe or rating. An analytical rod–sheave
model is not a natural-fiber redirect; DOE, NASA and shipyard governance do not
qualify a domestic hardpoint; and an operator-supplied occurrence narrative is
not an investigated causal finding. None of the accepted sources establishes a
rope, knot, load, capacity, attachment, inspection threshold, care method,
body-contact practice, upline, anchor, bondage system or human suspension.

### Batch 024 access and clean-room statement

The delegated scout and foreground reviewer read only the governing discovery
ledger, queue, report and contract, ordinary first-party publisher,
institutional-repository, government, practitioner, rights and robots pages,
and source bodies needed to resolve scope and component gates. No QA, trainer,
projection, evaluation, heldout, holdout, OOD, shadow, benchmark, probe,
experiment, existing corpus body or unrelated repository artifact was read.
The scout stopped at EPFL's `ai-train=no`, did not treat author-hosted or public
PDFs as permission, and used no login, paywall, challenge, cache, archive or
access-control bypass. No reviewed source body entered a training artifact
during discovery.

## Discovery batch 025 outcome

`discovery_batch_025` manually reviewed 10 exact-new source identities after
title, record, organization and canonical-URL deduplication. It records 5
`accept_high_priority`, 2 `accept_targeted_scope`, 3 `defer`, and 0 `reject`
decisions. The seven accepted sources expand the canonical Markdown queue from
167 to 174 resources; the discovery ledger grows from 297 to 307 records.

This batch fills consent, disability-autonomy, accessible-communication,
trauma-aware teaching and primary Japanese historical-source gaps. It adds
three Australian government resources, three OGL government resources and one
exact public-domain National Diet Library work. It also records why a
page-specific restriction overrides a general government license, why public
health pages whose terms prohibit storage cannot become corpus text, and why
an NDL work published by copyright arbitration remains protected.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Consent.gov.au resources](https://www.consent.gov.au/) | `accept_high_priority` | 35 | CC BY text can cover free, informed, active, specific and ongoing consent, check-ins, withdrawal, privacy and separate image permissions. Images, third-party resources, legal claims and culturally or linguistically separate components require exclusion or their own audit. |
| [NDIS supported decision-making and intimacy](https://www.ndiscommission.gov.au/rules-and-standards/ndis-practice-standards/core-module-rights-and-responsibilities) | `accept_high_priority` | 39 | CC BY material adds understandable communication, sufficient time, dignity of risk, intimacy, will and preferences, and decisions made by the person rather than for them. Disability never establishes incapacity, and NDIS duties do not become general rope rules. |
| [eSafety consent and image privacy](https://www.esafety.gov.au/key-topics/staying-safe/consent) | `accept_high_priority` | 39 | CC BY text separates consent to activity, recording, screenshots and each sharing action; permission for one recipient is not onward-sharing permission. Images, videos, songs, third-party research and legal or platform details are excluded. |
| [Trauma-informed practice working definition](https://www.gov.uk/government/publications/working-definition-of-trauma-informed-practice/working-definition-of-trauma-informed-practice) | `accept_targeted_scope` | 27 | OGL text can preserve safety, trust, choice, collaboration, empowerment and cultural consideration, while clearly distinguishing trauma-informed conduct from specialist treatment. Rope application is an inference, not diagnosis, therapy or certification. |
| [Disability, loneliness and relationships thematic report](https://www.gov.uk/government/publications/disability-loneliness-and-relationships-a-thematic-report/disability-loneliness-and-relationships-a-thematic-report) | `accept_high_priority` | 39 | OGL government synthesis adds assumed asexuality, infantilization, overprotection, privacy, support without substitution, communication exclusion and intersecting stigma. Preserve the qualitative sample and underrepresentation limits; exclude third-party quotations and generalization to individuals. |
| [Accessible communication formats](https://www.gov.uk/government/publications/inclusive-communication/accessible-communication-formats) | `accept_targeted_scope` | 33 | OGL guidance can improve Markdown, education and consent communication through co-design, HTML, plain language, Easy Read and alternative formats. A format never proves understanding or consent; third-party standards, symbols, examples and vendor content are excluded. |
| [伊藤晴雨『責の話』, 1929](https://ndlsearch.ndl.go.jp/books/R100000002-I000000607148) | `accept_high_priority` | 31 | The exact NDL work is marked PDM and may support source-critical first-person history and terminology after Japanese expert, OCR, translation, territorial and frame-level review. Exclude every image and operational coercive technique; infer neither direct hojojutsu-to-kinbaku lineage nor an inventor claim. |

### Permission- or rights-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [NSW Make No Doubt](https://www.nsw.gov.au/family-and-relationships/make-no-doubt) | 21 | A page-specific notice says third-party copyright limitations permit social-media sharing but require permission for other uses. That specific restriction overrides the site's general CC BY policy; metadata and permission contact only. |
| [Better Health disability and sexuality](https://www.betterhealth.vic.gov.au/health/servicesandsupport/disability-and-sexuality) | 22 | Official terms prohibit adaptation, reproduction, storage, derivative works and commercialisation without permission. Preserve only metadata and a permission lead; any later use also needs disability, medical and language review. |
| [野村昌靖『捕縄術押送監視研究之栞』, 1921](https://ndlsearch.ndl.go.jp/books/R100000039-I964857) | 34 | NDL records undetermined rights and internet publication by Commissioner for Cultural Affairs arbitration. NDL says this status remains protected; metadata only, with permission and exceptional Japanese historical and harm review required before even non-operational use. |

### Batch 025 rights, evidence, and safety controls

The Australian sources use only eligible government-authored text under their
respective CC BY policies; Coat of Arms, logos, trademarks, images, media,
third-party bodies and separately identified components remain excluded.
Consent.gov.au, NDIS and eSafety robots retrieval did not resolve during
review, so their jobs require manual low-rate acquisition only after policy is
resolved. The GOV.UK pages carry page-level OGL notices and permit their public
content paths, but cited and other third-party components are not swept into
that license.

The Ito Seiu acceptance applies to the exact 1929 edition and its NDL PDM
record, not to an author's collected output or every scan component. It needs
Japanese-language historical, OCR, translation, territorial, moral-rights,
human-rights, privacy and frame-level review. Historical description is not
modern consent or safety guidance, and operational restraint, torture and
unsafe technique remain excluded. The separate 1921 police-restraint title is
not public-domain corpus material merely because NDL makes it internet-public.

Nothing in this batch adjudicates an individual's capacity, diagnoses or
treats trauma, certifies consent, or establishes a rope, knot, load, tie,
hardpoint, upline, body-contact practice or suspension rule. Accessible
presentation supports communication but does not prove comprehension or
agreement.

### Batch 025 access and clean-room statement

The source worker read only the governing discovery ledger, queue, report and
tests plus first-party government, library, rights, robots and source pages
needed to resolve identity, scope and reuse gates. No QA, trainer, projection,
evaluation, heldout, holdout, OOD, shadow, benchmark, probe, experiment,
existing corpus body or unrelated repository artifact was read. No login,
paywall, challenge, cache, archive or access-control bypass was used, and no
reviewed source body entered a training artifact during discovery.

## Discovery batch 026 outcome

`discovery_batch_026` manually reviewed 11 exact-new source identities after
title, object, accession, organization and canonical-URL deduplication. It
records 0 `accept_high_priority`, 2 `accept_targeted_scope`, 8 `defer`, and 1
`reject` decisions. The two accepted museum-data records expand the canonical
Markdown queue from 174 to 176 resources; the discovery ledger grows from 307
to 318 records.

This batch rotates toward the thinnest cultural-knowledge categories:
`aesthetics_performance`, `lineage_history_people`, `pattern_architecture` and
`terminology_cultural_context`. It also records four unusually strong
accessible-consent and disability-safety permission leads. Exact-object and
page-level rights were decisive: clear public access, a free PDF, a free cart,
an underlying public-domain artwork or a site-wide open-data program did not
override restrictive terms or an exact `Usage Conditions Apply` label.

### Newly accepted sources

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Rijksmuseum Yoshitoshi Adachi Moor records](https://www.rijksmuseum.nl/en/collection/object/Lonely-house-on-Adachi-Moor--294bdb485e29697b36fd74cae97600a8) | `accept_targeted_scope` | 26 | PDM/CC0 collection data can preserve the two exact object identities, maker, dates, media, persistent identifiers and only museum-stated theatrical context. Images and any claim that Yoshitoshi inspired Ito Seiu or modern kinbaku are excluded. |
| [Art Institute Oshu Adachi ga Hara record](https://www.artic.edu/artworks/31601/the-actor-ichikawa-danjuro-v-as-abe-no-sadato-in-the-play-oshu-adachi-ga-hara-performed-at-the-ichimura-theater-in-the-fifth-month-1777) | `accept_targeted_scope` | 21 | CC0 non-description API fields anchor a named actor, role, play, theatre and circa-1777 print. The record does not establish that it depicts the same narrative as a later object or supply modern lineage, tying, consent or safety evidence. |

### Permission- or rights-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Smithsonian Ito Seiu Art of Knots and Ropes](https://www.si.edu/object/art-knots-and-ropes-series-torture-pictures-seme-e%3Afsg_S2021.5.93) | 27 | The first-party archive identifies an exact circa-1930s Ito Seiu object but displays conflicting CC0 boilerplate and `Usage Conditions Apply`; the exact object could not be verified in open API data. Metadata and permission lead only. |
| [Japan Arts Council Adachigahara performance context](https://www2.ntj.jac.go.jp/unesco/kabuki/en/production/scene2.html) | 25 | Official Kabuki and Noh records help disambiguate titles, but current terms prohibit unauthorized reuse, distribution and non-personal storage. No page, image, audio or performance body enters training without permission. |
| [National Gallery of Australia Yoshitoshi inventory record](https://nga.gov.au/media/dd/documents/annualreport_08-09.2c14bdd.pdf) | 7 | The all-rights-reserved inventory line corroborates accession 2008.774.A-B but duplicates the rights-clear Rijksmuseum evidence. An underlying public-domain artwork does not license Gallery-authored metadata or images. |
| [SECCA Introduction to Consent](https://www.secca.org.au/resources/introduction-to-consent/) | 34 | The co-designed Easy English books are a strong accessibility target, but their copyright page says all rights reserved and prohibits reproduction without prior permission. Free download is not corpus permission. |
| [Safer Me Safer You National Guidelines](https://safermesaferyou.org.au/projects/guidelines) | 35 | The 68-page co-designed social and sexual safety guideline and Easy Read companion are limited by governing terms to personal, non-commercial use; lived-experience quotations and case studies also need component clearance. |
| [WWDA Neve intellectual-disability consent guide](https://www.neve-plainenglish.wwda.org.au/level-4-page/teaching-people-with-intellectual-disabilities-about-sexual-consent) | 35 | Detailed paced, trauma-aware and communication-specific teaching guidance is valuable, but WWDA terms prohibit copying, adaptation, translation, distribution and commercialisation without express approval. |
| [SACID Relationship Wise bundle](https://sacid.org.au/product/relationship-wise-bundle/) | 24 | Eleven Easy Read booklets cover relationships, boundaries, safety and consent, but access requires a free cart and no open reuse license was found. Checkout was not entered; Photosymbols components need separate clearance. |
| [Japan House London kumihimo and wrapping context](https://www.japanhouselondon.uk/read-and-watch/kumihimo-a-thread-through-japanese-history/) | 28 | The cultural articles are dense on braiding, historical use, named patterns and mizuhiki wrapping, but commercial reuse needs permission. Decorative-cord knowledge cannot be presented as direct kinbaku lineage, body-restraint technique or suspension safety. |

### Rejected false-relevance source

| Source | Score | Why it was rejected |
|---|---:|---|
| [Cleveland Museum Trefoil Knots object](https://www.clevelandart.org/art/1961.202.q) | 8 | CMA metadata is exemplary CC0 open data, but “Trefoil Knots” is a literary chapter title and the record describes a Tale of Genji painting rather than a rope structure. Open licensing does not turn keyword coincidence into useful training knowledge. |

### Batch 026 rights, evidence, and safety controls

The two accepted jobs are text-only exact-record conversions. Rijksmuseum
objects are individually designated public domain and their eligible open
collection metadata follows PDM/CC0 policy. The Art Institute job uses only
CC0 API fields; its separately CC BY description field is empty for this
object. No museum image, visual inference, linked essay, bibliography body,
provenance narrative or generic navigation enters either job.

Adachi, Adachigahara and Oshu Adachi ga Hara titles were not flattened into a
single narrative. The reviewed records do not demonstrate a causal chain from
Kabuki, Noh or Yoshitoshi to Ito Seiu, hojojutsu or modern kinbaku. Museum
catalog titles also do not establish consent, technique, anatomy, load,
hardpoint, upline, pattern or safety facts.

The accessibility leads remain permission metadata only. Disability does not
establish incapacity; a communication format does not prove understanding or
agreement; a population guideline does not adjudicate an individual; and
cross-domain consent education does not certify a rope scene. If permissions
are obtained later, images, symbols, lived-experience quotations, case studies,
third-party models, medical or legal claims and problematic prescriptive
heuristics require separate review.

### Batch 026 access and clean-room statement

The source worker read only the governing discovery ledger, queue, report and
tests plus first-party museum, performing-arts, disability-organization,
rights, policy, API-documentation and exact source pages needed to resolve
identity, scope and reuse gates. No QA, trainer, projection, evaluation,
heldout, holdout, OOD, shadow, benchmark, probe, experiment, existing corpus
body or unrelated repository artifact was read. No login, checkout, paywall,
challenge, cache, archive-service or access-control bypass was used. The worker
did not inspect museum images or retrieve gated bundle bodies, and no reviewed
source body entered a training artifact during discovery.

## Discovery batch 027 outcome

`discovery_batch_027` manually reviewed 10 exact-new source identities after
canonical-URL, title, author, organization and document deduplication. It
records 0 `accept_high_priority`, 1 `accept_targeted_scope`, 9 `defer`, and 0
`reject` decisions. The accepted rights-clear guidelines expand the canonical
Markdown queue from 176 to 177 resources; the discovery ledger grows from 318
to 328 records.

This batch focuses on the relationship among rope performance, declared
lineage, audience or performer limits, consent-based choreography, production
privacy, access needs and process evaluation. It deliberately separates three
evidence domains: first-party descriptions of a particular rope style or
artwork, performance-art case studies, and professional stage or screen
practice. None is silently converted into tying, anatomy, load, hardpoint,
upline, emergency or suspension guidance.

### Newly accepted source

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Ita O'Brien's Intimacy On Set Guidelines](https://www.intimacyonset.com/intimacy-on-set-guidelines/) | `accept_targeted_scope` | 38 | The exact English guidelines body is explicitly CC BY-SA 4.0 and supports advance disclosure, consent each time, explicit boundaries, a halt strategy, third-party presence, blocking, plain-language choreography, closed rehearsals or sets, pre-performance check-ins and change control. Attribution, change indication and ShareAlike are mandatory. Images, translations, acknowledgements, services and surrounding site content are excluded. Any rope application must be labeled as an inference; the source does not certify a rope scene or teach rope safety. |

### Permission- or rights-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [LA Rope Yukimura-style concepts](https://www.lashibaridojo.com/core-concepts-in-yukimura-style-shibari/) | 24 | The pages provide unusually specific first-party declared-lineage and terminology evidence, but are all rights reserved. Permission, Japanese review and independent historical review are required. Passages treating resistance, hesitation or bodily response as communication need explicit-consent correction and cannot enter training as written. |
| [being surface](https://www.performancephilosophy.org/journal/article/view/360) | 32 | The peer-reviewed article documents durational rope performance, audience relation, corporeal trace and care, but is CC BY-NC-SA rather than commercially reusable. Credited images and violent-performance components are separately excluded; the work cannot become actionable body-marking or rope instruction. |
| [Tender Container: How to Self-Suspend](https://www.tendercontainer.com/how-to-self-suspend) | 28 | This first-party production record identifies a collaborative interactive rope performance, its creative roles and its audience-consent question. No open text license was found, photographs have separate rights and sensitive autobiography or trauma remains outside scope. |
| [Kinbaku Society of Berlin magazine archive](https://www.kinbaku-society.org/) | 6 | Public issue pages preserve a small amount of project history and aesthetics, but the knowledge-rich multi-contributor magazines are unavailable or purchase-gated, the site is copyrighted and commerce noise is high. Named influence and origin claims need independent corroboration. No cart or issue body was accessed. |
| [Caring for Limits in Sex Worker's Opera](https://www.royalholloway.ac.uk/media/20087/1-full%20issue.pdf.pdf) | 34 | The peer-reviewed case study is strong on member-set limits, anonymity, access needs, collaborative support and creating an alternative rather than overriding a performer's boundary. The issue explicitly says all rights reserved; participant quotations, trauma material and photographs need separate clearance. |
| [UK National Occupational Standards for Intimacy Coordination](https://www.ukstandards.org.uk/en/nos-finder/SKSIC2/liaise-with-performers-during-intimacy-planning) | 35 | The current 2025 suite is a dense professional competency map for disclosure, power, ongoing consent, choreography, access support, documentation and evaluation. The database is freely accessible, but no clear open derivative-reuse license was found; permission from the standards and database owners is required. |
| [EDIT Media Teaching Intimacy guide](https://editmedia.org/intimacy-coordination-guide/) | 33 | The multi-author classroom guide is rich and explicitly non-prescriptive, but the site is all rights reserved and much of the text summarizes or quotes separately protected professional sources. Author permission and sentence-level component provenance are required. |
| [Intimacy Practitioners South Africa protocols](https://www.intimacysouthafrica.com/protocols/) | 33 | The first-party framework adds a South African film-production perspective across planning, production and post-production, but it is explicitly all rights reserved. Its labor, legal and film context cannot be generalized into rope rules. |
| [UT Austin staging-intimacy policies](https://theatredance.utexas.edu/policies-staging-intimacy) | 31 | The operational policy covers disclosure, confidential communication, boundaries, exit cues, written choreography, recording restrictions, alternatives and revocable consent. University copyright applies, and the document states that it is based on Theatrical Intimacy Education work, so both permission and component resolution are needed. |

### Batch 027 rights, evidence, and safety controls

The accepted job is confined to one exactly licensed English work. CC BY-SA
attribution, license linking, change indication and ShareAlike obligations must
travel with the transformed Markdown. The surrounding Intimacy On Set site,
its photographs, translations, biographies, courses, services,
acknowledgements and related articles do not inherit the guideline's license.

Open access, free download, free occupational-standard access and public
product descriptions were not treated as training permission. The remaining
nine candidates are metadata and permission leads only. The Performance
Philosophy noncommercial license, Platform's explicit no-reproduction notice,
institutional copyright, multi-author or participant components and
third-party foundations were recorded at page or document level.

Rope aesthetics and performance evidence remains source-attributed. A
self-reported teaching license is not independent lineage verification; a
project's named influences do not establish a universal genealogy; and an
artist's use of rope does not establish ordinary practice, therapeutic value,
consent or safety. Hesitation, resistance, facial expression, stillness,
embarrassment and other nonverbal responses are never substitutes for prior
explicit agreement, ongoing check-ins or an unambiguous stop mechanism.

### Batch 027 access and clean-room statement

The source worker read only the governing discovery ledger, queue, report and
tests plus the exact first-party articles, policies, standards, production
records, official journal documents, rights pages and robots policies needed
to resolve identity, evidence and reuse gates. No QA, trainer, projection,
evaluation, heldout, holdout, OOD, shadow, benchmark, probe, experiment,
existing corpus body or unrelated repository artifact was read. No login,
cart, purchase, paywall, challenge, cache, archive-service or access-control
bypass was used; no source body or image entered a training artifact during
discovery.

## Discovery batch 028 outcome

`discovery_batch_028` manually reviewed 12 exact-new source identities after
canonical-URL, title, DOI, author, organization and document deduplication. It
records 0 `accept_high_priority`, 2 `accept_targeted_scope`, 9 `defer`, and 1
`reject` decision. The two accepted rights-clear transcript sources expand the
canonical Markdown queue from 177 to 179 resources; the discovery ledger grows
from 328 to 340 records.

This batch targets underrepresented first-person bottoming perspectives,
rope-specific negotiation tools, qualitative incident reporting, queer and
photographic history, Latin American practice research, and a transferable
risk-assessment framework. Public access, an adaptable template, an informal
educational-distribution statement, a repository deposit agreement and a
noncommercial license were not treated as permission for commercial
derivative model training. Visual-only technique remains metadata-only until
a qualified person performs a current, timestamped safety review.

### Newly accepted source

| Source | Decision | Score | Exact useful scope and boundary |
|---|---|---:|---|
| [Midori and Zoë Ligon: Japanese Rope Bondage](https://commons.wikimedia.org/wiki/File:Japanese_Rope_Bondage_-_Sex_Stuff.webm) | `accept_targeted_scope` | 33 | The exact 2018 Commons video is CC BY 3.0 with human-reviewed source-license provenance and has an English TimedText route. A text-only, speaker- and time-attributed record may retain negotiation of purpose and desired or undesired responses, day-to-day variability, a first-time bottom's sensory and communication experience, transition, cuddling aftercare and debriefing. It must remove the false claim that nobody discussed Japanese rope bondage before Midori; keep Midori's custody-origin comment limited to her particular style; and exclude all media, visual technique, material generalizations, neck-adjacent content and present-day safety inference. Nonverbal negotiation never replaces explicit agreement, direct check-ins or a stop mechanism. Component-level CC BY, Commons CC BY-SA and CC0 obligations remain attached. |
| [The Endless Knot Episode 94 with Isobel Williams](https://www.alliterative.net/episode-94-transcript) | `accept_targeted_scope` | 30 | The exact official transcript and podcast are CC BY-SA 4.0. A manually audio-proofed, speaker- and time-attributed record may retain first-person observations about a London Japanese-rope performance community, Gorgone, rehearsed performance, expert-peer audiences, bottom/top/rigger vocabulary, power moving between roles, women tying heavier partners, rope as live art and its influence on translation. The mostly automatic transcript must be corrected; quoted poems, translations, book text, third-party criticism, tie-name strings and technical instructions are excluded. Claims that the community is safer than a pub remain attributed testimony, not a safety protocol. |

### Permission-, access-, methods-, or review-gated references

| Source | Score | Why it remains reference-only |
|---|---:|---|
| [Escola de Shibari incident synthesis](https://escoladeshibari.com/seguranca-em-shibari-o-que-137-incidentes-nos-ensinaram/) | 30 | The Portuguese article can support later incident-theme education, but the private volunteer reports cannot be audited and the page's free educational-distribution language is not a clear commercial derivative license. Permission, methods, provenance, privacy and ethics review are required. No narrative, prevalence, causality, threshold, treatment or operational instruction is approved. |
| [Kinbaku Japan Voices](https://note.com/yuna_zombie/m/m44712c56283e) | 29 | Yuna's three English transcripts offer rare Japanese ukete-centered testimony on agency, co-created timing and whose voices lineage stories center. They are ordinary-copyright creator works without a model-training grant. Direct permission is required, and the surrounding account's secondary medical summaries are excluded. |
| [Rope Radar](https://roperadar.com/) | 34 | This bounded framework usefully distinguishes desired relational qualities from agreed practices and boundaries and asks people to define terms together for each context. Private, personal-modification and workshop permissions do not authorize publication in a training corpus. The HOW may be narrowed or stopped at any time and never expanded without renewed agreement; a completed radar does not prove consent or safety. |
| [AYCO and ACE basic risk assessment](https://www.americanyouthcircus.org/safety-in-circus) | 32 | Jonathan Deull's 2020 circus guide offers qualitative hazard identification, control hierarchy, emergency planning, reassessment and documentation. Permission to adapt its template does not license all prose or quoted standards components. No matrix, legal threshold or claim that the guide certifies a bondage hardpoint, load system, rigger or scene is approved. |
| [4toscuro and the view of people under the ropes](http://eventosacademicos.filo.uba.ar/index.php/FIICT/JPAMA/paper/view/8392) | 25 | This self-reflective three-author conference paper may illuminate photographic exposure, image selection, later audiences and ambivalence despite consent. Its item-level CC BY 3.0 signal conflicts with the site's CC BY-NC footer. Editor, author, participant and component resolution is required; photographs, quotations, intimate disclosures and participant-identifying narrative are excluded. |
| [Shibari and the construction of communication](https://repository.udistrital.edu.co/items/29734187-decf-4b21-ab2b-e787f27e131d) | 24 | The 2026 Universidad Distrital thesis may add a Latin American practice-research perspective, but a repository deposit agreement is not a public derivative license. Author permission, methods, privacy and evidence review are required, especially before any communication, emotional-connection or transformative claim. |
| [Between Erotic Representation and Minority Identity](https://doi.org/10.3390/h15010001) | 29 | Crossref identifies Soojung Park's article and an immediate CC BY 4.0 version-of-record license, but MDPI's official body is currently blocked by Akamai in this environment. No snippet, cache or ResearchGate substitute was used. When the official work becomes accessible, article and third-party magazine-component review must precede any queue decision. |
| [Shay Tiziano beginner column-tie videos](https://commons.wikimedia.org/wiki/File:Single_column_tie_for_bondage_beginners.webm) | 20 | Both Commons files are human-reviewed CC BY-SA 3.0 works, but they have no TimedText and teach visually. They remain metadata-only pending qualified manual, timestamped and current safety review. No automated vision reconstruction, inferred step or claim of suspension suitability is permitted. |
| [Rope makes me fuzzy inside](https://kohljournal.press/rope-makes-me-fuzzy) | 32 | Tiffany Kagure Mugo's article offers unusually valuable African queer and gender-nonconforming perspectives on rope, bodily autonomy, vulnerability, agency, trust and co-created safety. Its CC BY-NC-SA 4.0 license does not authorize unrestricted or commercial training reuse. Author, publisher and sensitive-component permission are required; every voice must remain attributed lived experience rather than a universal fact. |

### Rejected source

| Source | Score | Why it was rejected |
|---|---:|---|
| [On the Art of Shibari as a Form of Writing](https://soctech.spbstu.ru/userfiles/files/articles/2023/3/49-58.pdf) | 10 | The 2023 practitioner essay is CC BY-NC 4.0, so its license is incompatible with the intended use. Independently, its philosophical metaphor does not compensate for weak source-critical history and unsupported categorical circulation, timing and body-placement claims. It is not a history, anatomy, technique or safety source. |

### Batch 028 rights, evidence, and safety controls

The accepted jobs are confined to the exact licensed Commons interview and
the exact licensed Endless Knot transcript. Extraction workers must retain
speaker, date, source, timestamp, testimonial framing, attribution, change
indication, license links and ShareAlike where applicable. Video, audio, still
images, music, interface material, promotion and inferred visual steps are not
part of the text corpus. License clearance does not validate a claim: a
demonstrably false historical statement is removed, other historical remarks
remain explicitly attributed and limited to the speaker's described style,
and automatic transcription must be manually checked against official audio.
Separately authored poems, translations, books and criticism do not inherit a
podcast's blanket license.

The incident collection is self-selected and privately hosted, with no public
denominator or complete audit trail, so its counts cannot become prevalence or
causal estimates. Three self-reflective paper authors are not an external
sample; a thesis abstract is not evidence of transformation; first-person
testimony is not universal Japanese practice; and a philosophical practitioner
essay is not medical evidence. Photographs, sensitive narratives, participant
quotations and identifying detail have independent privacy and component
boundaries even when a surrounding work appears openly licensed.

Risk tools and negotiation aids preserve their original domains. Circus risk
assessment can contribute a method after permission, but cannot certify a
rope-bondage anchor or scene. Rope Radar may structure conversation, but cannot
replace vetting, present-state and physical-limit discussion, direct check-ins,
revocation or emergency planning. Desired or undesired observable responses
are pre-negotiated preferences, never proof of continuing consent.

### Batch 028 access and clean-room statement

The source worker read only the governing discovery ledger, corpus queue,
report, registry and tests plus exact first-party pages, official repository or
publisher records, rights notices, license pages, Crossref metadata, robots
policies and source-owned files needed to resolve identity, access, evidence
and component boundaries. No QA, trainer, projection, evaluation, heldout,
holdout, OOD, shadow, benchmark, probe, experiment, existing corpus body or
unrelated repository artifact was read. No login, cart, purchase, paywall,
challenge, cache, archive-service or access-control bypass was used. An Akamai
403 was respected, private incident reports were not accessed, visual-only
technique was not reconstructed and no source body or image entered a training
artifact during discovery.
