#!/usr/bin/env python3
"""Build the ICAR suspension-syndrome evidence corpus offline."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
XML = SNAPSHOT / "fullTextXML.xml"
PROVENANCE = SNAPSHOT / "provenance.json"
MARKDOWN = ROOT / "europepmc_icar_suspension_syndrome.md"
MANIFEST = ROOT / "manifest.json"
REPORT = ROOT / "report.json"
SOURCE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10710713/fullTextXML"
SOURCE_SHA256 = "ffffb650be036b8cb55ef0bb85528ca0c9bb3fdc66bb6038205905b9e6d54682"
SOURCE_BYTES = 105441
TITLE = "Suspension syndrome: a scoping review and recommendations from the International Commission for Mountain Emergency Medicine (ICAR MEDCOM)"
AUTHORS = [
    "Simon Rauch",
    "Raimund Lechner",
    "Giacomo Strapazzon",
    "Roger B Mortimer",
    "John Ellerton",
    "Sven Christjar Skaiaa",
    "Tobias Huber",
    "Hermann Brugger",
    "Mathieu Pasquier",
    "Peter Paal",
]
DOI = "10.1186/s13049-023-01164-z"
PMCID = "PMC10710713"
PMID = "38071341"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
TAXONOMY = {
    "anatomy_injury_prevention",
    "uplines_suspension_hardpoints",
    "emergency_procedures",
    "teaching_evaluation",
}


SECTIONS = [
    {
        "heading": "Source scope and transfer boundary",
        "categories": ["anatomy_injury_prevention", "emergency_procedures", "teaching_evaluation"],
        "text": (
            "Rauch and colleagues reviewed suspension syndrome in occupational rope access, climbing, mountain rescue, experimental harness suspension, and reported accidents, then developed International Commission for Alpine Rescue Medical Commission recommendations. The source defines the syndrome around passive hanging in a vertical or near-vertical position, usually in a harness. Its evidence is not a study of rope bondage, decorative rope, bondage-specific load paths, or bondage participants. Any relevance to bondage suspension is therefore a cautious hazard analogy, not validated transfer of incidence, equipment findings, rescue technique, or treatment thresholds."
        ),
    },
    {
        "heading": "Review question and two-part evidence process",
        "categories": ["teaching_evaluation"],
        "text": (
            "The work has two distinct components. First, it is a scoping review seeking original epidemiological or medical data about suspension-syndrome mechanisms. Second, the author group drafted recommendations from the review, prior recommendations, discussion within ICAR MEDCOM, review by all four ICAR commissions at the 2019 meeting, and final circulation through the ICAR MEDCOM list server. Consensus recommendations should not be mistaken for effects directly tested in controlled treatment trials."
        ),
    },
    {
        "heading": "Search, eligibility, extraction, and appraisal",
        "categories": ["teaching_evaluation"],
        "text": (
            "The authors searched PubMed, Embase, Web of Science, and the Cochrane Library through 17 September 2023 using seven suspension-syndrome and harness-hanging terms joined with OR. They also screened bibliographies and personal article databases. Eligible material needed original epidemiological or pathophysiological data such as vital signs, laboratory values, or objective symptoms. They excluded duplicates, unrelated topics, unavailable full text, registrations, abstracts, reviews without original data, case reports without original data, letters, editorials, short communications, and languages outside the authors' English, German, or Italian. Two authors independently extracted and appraised data with National Heart, Lung and Blood Institute study-quality tools, resolving disagreements by discussion; the authors report no inter-rater reliability calculation."
        ),
    },
    {
        "heading": "Included evidence and quality profile",
        "categories": ["teaching_evaluation"],
        "text": (
            "The body reports that 121 records were returned, 70 remained after title and abstract screening, 47 were excluded at full text, and 23 articles were included. Of those 23, two supplied epidemiological survey data and 21 addressed pathophysiology. Designs ranged from autopsy reports and case reports or series to manikin work and small observational or interventional human physiology studies. From the review's own table, 9 studies were rated good, 8 fair, and 6 poor. No randomized controlled interventional study was found. The review also found no experimental studies directly testing prevention, diagnosis, or treatment."
        ),
    },
    {
        "heading": "Epidemiology remains uncertain",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "The review calls suspension syndrome rare but says exact epidemiology is unavailable. One 2002 enquiry reported no suspension syndrome during 5.8 million on-rope hours among technicians qualified by the International Rope Access Trade Association. A small survey of height-rescue organizations identified three cases but did not provide an observation period or adequate symptom and severity detail. These observations come from trained industrial or rescue systems and cannot be converted into a rate for climbing generally, for other harness systems, or for bondage suspension. Absence of reported events is not proof of zero risk."
        ),
    },
    {
        "heading": "Venous pooling is observed, but the traditional collapse story is unproven",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "Experimental studies show venous pooling in the legs during motionless harness suspension. The review nevertheless finds no proof that pooling alone produces a progressive loss of cardiac preload that directly causes cardiac arrest. In the cited experiments, pooling was not accompanied by the expected major systemic haemodynamic deterioration before presyncope; mild tachycardia and hypertension were sometimes attributed to discomfort and sympathetic activation. The authors therefore reject a simple proven blood-pooling-to-arrest chain while retaining an uncertain contributory role for orthostatic stress."
        ),
    },
    {
        "heading": "Neurocardiogenic mechanism is favored, not fully resolved",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "More recent experimental evidence favored a sudden neurocardiogenic response: heart rate, blood pressure, and stroke volume dropped around presyncope, with reduced cerebral oxygenation and a pattern resembling vasodilatory and cardio-inhibitory syncope. The exact trigger remains unknown. The review discusses pain, altered lower-limb sensation, inability to move, and orthostatic stress as possible contributors. It notes that proposed baroreceptor and Bezold–Jarisch explanations were not demonstrated in the cited experiments. This is a favored mechanism assembled from small physiology studies, not a complete causal model of every collapse during suspension."
        ),
    },
    {
        "heading": "Symptoms and timing are variable and sometimes abrupt",
        "categories": ["anatomy_injury_prevention", "emergency_procedures"],
        "text": (
            "Reported presyncopal features include light-headedness, dizziness, confusion, pallor, cold sweating, warmth or hot flashes, blurred vision, nausea, and bradycardia. Experimental tolerance varied markedly between people: symptoms could begin within minutes or much later, and one study found abrupt presyncope without a warning phase. Increased body weight and prior exercise were associated with shorter tolerance in some studies, while sex was not. These heterogeneous observations do not provide a safe waiting time, a reliable warning sequence, or a person-level prediction."
        ),
    },
    {
        "heading": "Why loss of consciousness while still suspended is dangerous",
        "categories": ["anatomy_injury_prevention", "emergency_procedures"],
        "text": (
            "Ordinary fainting allows a fall toward horizontal, which can restore cerebral perfusion. A person who loses consciousness while attached in a near-vertical suspension may remain upright and cannot make that compensatory position change. The review also identifies possible airway obstruction from neck flexion or extension. With prolonged suspension, reported concerns include tissue hypoperfusion, rhabdomyolysis, hyperkalaemia, acute kidney injury, hypothermia, thromboembolic events, and neurological compression injury. Evidence for these later stages is sparse because human experiments ethically stop at early symptoms."
        ),
    },
    {
        "heading": "Harness and attachment findings are context-specific",
        "categories": ["anatomy_injury_prevention", "uplines_suspension_hardpoints", "teaching_evaluation"],
        "text": (
            "Older belt-only and chest-harness systems were associated with rapid pain, breathing restriction, and nerve pressure and are treated by the review as obsolete relative to modern sit harnesses. Some studies found lower tolerance with dorsal industrial attachment, while findings were not uniform. Fit, sizing, clothing, body dimensions, attachment position, and hanging angle can affect comfort or tolerance. The authors explicitly warn that anthropometry and tolerance from industrial dorsal systems cannot simply be transferred to ventral sit harnesses used in mountain sports. The same restriction applies more strongly to bondage rope, which is not a certified sit harness and was not evaluated in this review."
        ),
    },
    {
        "heading": "Proposed definition and classification",
        "categories": ["anatomy_injury_prevention", "emergency_procedures", "teaching_evaluation"],
        "text": (
            "Because terminology was inconsistent, the authors prefer “suspension syndrome” over names that require a harness or imply trauma. Their proposed acute category includes near-suspension syncope, suspension syncope, cardiac arrest while suspended, and cardiac arrest within 60 minutes after rescue. Their proposed subacute category includes lower-limb sensory or motor deficit lasting more than 24 hours, end-organ dysfunction such as rhabdomyolysis-associated acute kidney injury, and cardiac arrest more than 60 minutes after rescue. The framework excludes signs better explained by trauma, hypothermia, hypoglycaemia, or another condition. It is a proposed review classification, not a prospectively validated diagnostic rule."
        ),
    },
    {
        "heading": "ICAR prevention and rescue recommendations, with source grades",
        "categories": ["emergency_procedures", "uplines_suspension_hardpoints", "teaching_evaluation"],
        "text": (
            "The article reports seven ICAR recommendations using American College of Chest Physicians grades. The source recommends proper equipment and knowledge and not working alone (1C); rescue as soon as possible even when the suspended person has no symptoms because time to syncope is unpredictable (1B); leg movement while awaiting rescue to reduce pooling (2B); using foot support when no structure is reachable (2B); and, when the person cannot act and rescue conditions permit, having the first rescuer raise the legs toward a more horizontal position while lowering proceeds (2C). These are harness and rope-rescue recommendations. They are not instructions for creating or continuing a bondage suspension."
        ),
    },
    {
        "heading": "Supine recovery and standard resuscitation replace the seated-recovery myth",
        "categories": ["emergency_procedures", "teaching_evaluation"],
        "text": (
            "The review finds no evidence for older advice to keep a rescued casualty seated or to avoid laying the person down abruptly because of a presumed dangerous return of pooled blood. Its recommendation is to position the casualty supine once on the ground, assess and treat using standard advanced life-support algorithms, and consider reversible causes including hyperkalaemia and pulmonary embolism (1A). This recommendation applies after extrication in the source's rescue context; it does not replace local emergency protocols or qualified medical direction."
        ),
    },
    {
        "heading": "Assessment after rescue and prolonged suspension",
        "categories": ["emergency_procedures", "anatomy_injury_prevention"],
        "text": (
            "The review recommends monitoring pulse, respiration, blood pressure, oxygen saturation, electrocardiography, and core temperature after rescue and during transport. It discusses hospital evaluation of kidney and liver function, creatine kinase, myoglobin, blood gases, electrolytes, acid–base disturbance, urine output, and associated trauma when clinically appropriate. Its recommendation says that after passive hanging longer than two hours, patients are at risk of hyperkalaemia and acute kidney injury and should be transported to a hospital capable of emergency renal replacement therapy (2C). The two-hour value is a consensus trigger in this rescue guideline, not a safe exposure limit and not a bondage-suspension threshold."
        ),
    },
    {
        "heading": "Evidence supporting recommendations is indirect",
        "categories": ["teaching_evaluation", "emergency_procedures"],
        "text": (
            "The authors state that the review found no experimental studies of suspension-syndrome prevention, diagnosis, or treatment. Recommendations therefore combine early-stage human physiology, case evidence, older recommendations, standard care for conditions such as cardiac arrest, and ICAR expert consensus. A strong source grade on an individual recommendation does not turn the complete management pathway into a randomized suspension-syndrome trial. Users of the corpus should preserve the stated grades, context, and evidentiary gap together."
        ),
    },
    {
        "heading": "Limitations and internal reporting discrepancy",
        "categories": ["teaching_evaluation"],
        "text": (
            "The review highlights very low incidence, especially rare severe cases, few and incomplete case reports, inaccessible early literature, no randomized controlled intervention studies, small samples, and difficulty mapping heterogeneous studies to the NHLBI quality categories. It also reports no inter-rater reliability statistic. Experimental studies focus on healthy volunteers and early presyncope, leaving later collapse and cardiac arrest uncertain. The captured abstract says the online search yielded 210 articles and reference screening another 30, whereas the Results body says 121 records were returned before screening. The included total of 23 is consistent, but the initial-search-count discrepancy should not be silently harmonized."
        ),
    },
    {
        "heading": "Research needs and domain-qualified use",
        "categories": ["teaching_evaluation", "anatomy_injury_prevention"],
        "text": (
            "The authors call for an international registry to estimate incidence and outcomes and for better pathophysiological classification. Important gaps include representative epidemiology, later-stage mechanisms, validated diagnostic criteria, and direct evaluation of preventive and treatment measures. Separate evidence would be needed before applying harness-specific timing, fit, attachment, leg-support, or rescue findings to bondage suspension. The durable transferable lesson is uncertainty-aware emergency recognition and rapid extrication—not a claim that any suspension system is safe."
        ),
    },
    {
        "heading": "Publication, funding, and disclosure record",
        "categories": ["teaching_evaluation"],
        "text": (
            "The article was published on 9 December 2023 in the Scandinavian Journal of Trauma, Resuscitation and Emergency Medicine, volume 31, article 95. The captured JATS identifies DOI 10.1186/s13049-023-01164-z, PMCID PMC10710713, and PMID 38071341. It reports no study funding, says ethics approval and consent were not applicable, and declares no competing interests. Open-access publication costs were covered by the Department of Innovation, Research, University and Museums of the Autonomous Province of Bozen/Bolzano."
        ),
    },
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_digest(text: str) -> str:
    return digest(text.encode("utf-8"))


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def normalize(value: str) -> str:
    return " ".join(value.split())


def word_count(value: str) -> int:
    return len(re.findall(r"[\w’'-]+", value, flags=re.UNICODE))


def load_and_verify() -> tuple[ET.Element, dict]:
    xml_bytes = XML.read_bytes()
    if len(xml_bytes) != SOURCE_BYTES or digest(xml_bytes) != SOURCE_SHA256:
        raise RuntimeError("authorized JATS snapshot hash or length drift")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if provenance["request_count"] != 1 or provenance["requested_url"] != SOURCE_URL:
        raise RuntimeError("capture route or request-count drift")
    if provenance["capture_scope"] != "one_authorized_embl_ebi_europepmc_fullTextXML_get":
        raise RuntimeError("capture scope drift")
    if provenance["body_sha256"] != SOURCE_SHA256 or provenance["body_byte_length"] != SOURCE_BYTES:
        raise RuntimeError("provenance body contract drift")
    if provenance["excluded_routes"] != ["Springer publisher endpoint", "United States PMC endpoint"]:
        raise RuntimeError("excluded-route boundary drift")

    root = ET.fromstring(xml_bytes)
    meta = root.find("./front/article-meta")
    if meta is None:
        raise RuntimeError("JATS article metadata missing")
    title = normalize("".join(meta.find("./title-group/article-title").itertext()))
    if title != TITLE:
        raise RuntimeError("article title drift")
    ids = {
        node.attrib.get("pub-id-type"): normalize("".join(node.itertext()))
        for node in meta.findall("article-id")
    }
    if (ids.get("doi"), ids.get("pmcid"), ids.get("pmid")) != (DOI, PMCID, PMID):
        raise RuntimeError("article identifier drift")
    authors = [
        normalize((node.findtext("name/given-names", "") + " " + node.findtext("name/surname", "")))
        for node in meta.findall("contrib-group/contrib")
    ]
    if authors != AUTHORS:
        raise RuntimeError("author-list drift")
    license_text = normalize(" ".join(meta.find("permissions/license").itertext()))
    if "creativecommons.org/licenses/by/4.0" not in license_text or "Creative Commons Attribution 4.0" not in license_text:
        raise RuntimeError("CC BY 4.0 license drift")
    section_titles = [normalize(node.findtext("title", "")) for node in root.findall("./body/sec")[:8]]
    if section_titles != ["Background", "Methods", "Results", "Discussion", "Research implications", "Limitations", "Conclusions", "Recommendations"]:
        raise RuntimeError("article section inventory drift")
    tables = root.findall(".//table-wrap")
    if len(tables) != 3 or len(tables[0].findall(".//tr")) - 1 != 23:
        raise RuntimeError("evidence-table inventory drift")
    quality = Counter(normalize("".join(list(row)[-1].itertext())) for row in tables[0].findall(".//tr")[1:])
    if quality != Counter({"Good": 9, "Fair": 8, "Poor": 6}):
        raise RuntimeError("study-quality profile drift")
    source_text = normalize(" ".join(root.itertext()))
    for required in (
        "no suspension syndrome within 5.8 million on-rope hours",
        "no relevant effects on systemic hemodynamic parameters",
        "no evidence to support this",
        "positioned supine",
        "Randomized controlled interventional studies on suspension syndrome are absent",
        "The online literature search yielded 210 articles",
        "Of the 121 studies returned from search",
    ):
        if required not in source_text:
            raise RuntimeError(f"required JATS evidence missing: {required}")
    return root, provenance


def render() -> tuple[str, str, str]:
    _, provenance = load_and_verify()
    lines = [
        "# Suspension syndrome — ICAR scoping-review evidence corpus",
        "",
        f"Captured: {provenance['captured_at']}",
        "",
        f"Source article: {TITLE}",
        "",
        f"Authors: {', '.join(AUTHORS)}",
        "",
        "Journal record: Scandinavian Journal of Trauma, Resuscitation and Emergency Medicine 31, article 95 (9 December 2023)",
        "",
        f"Identifiers: DOI {DOI}; PMCID {PMCID}; PMID {PMID}",
        "",
        f"Authorized source route: {SOURCE_URL}",
        "",
        f"License: Creative Commons Attribution 4.0 ({LICENSE_URL})",
        "",
        "This non-Q&A document is an original, attributed evidence summary of the captured JATS article. It preserves review methods, denominators, source grades, consensus recommendations, contradictions, and limitations. It does not provide a bondage suspension method; harness-rescue evidence is not validated for bondage suspension.",
        "",
    ]
    category_counts: Counter[str] = Counter()
    for section in SECTIONS:
        if not set(section["categories"]) <= TAXONOMY:
            raise RuntimeError(f"unsupported taxonomy label: {section['categories']}")
        category_counts.update(section["categories"])
        lines.extend([
            f"## {section['heading']}",
            "",
            "Categories: " + ", ".join(section["categories"]),
            "",
            section["text"],
            "",
        ])
    markdown = "\n".join(lines).rstrip() + "\n"
    manifest_value = {
        "schema": "site-corpus-manifest-v1",
        "resource_id": "europepmc_icar_suspension_syndrome",
        "source_type": "peer_reviewed_scoping_review_and_guideline",
        "artifact_role": "canonical_markdown_direct_training_source",
        "direct_training_ready": True,
        "non_qa": True,
        "retrieved_at": provenance["captured_at"],
        "license": {"name": "CC BY 4.0", "url": LICENSE_URL},
        "article": {
            "title": TITLE,
            "authors": AUTHORS,
            "journal": "Scandinavian Journal of Trauma, Resuscitation and Emergency Medicine",
            "publication_date": "2023-12-09",
            "doi": DOI,
            "pmcid": PMCID,
            "pmid": PMID,
        },
        "entries": [{
            "url": SOURCE_URL,
            "disposition": "included",
            "reason": "Authorized CC BY 4.0 JATS preserved as the sole source for an attributed evidence summary.",
            "content_type": provenance["content_type"],
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "snapshot_path": "source_snapshot/fullTextXML.xml",
        }],
        "supported_categories": sorted(category_counts),
        "domain_transfer": "Harness, occupational rope-access, climbing, and mountain-rescue evidence; not validated as a bondage-suspension protocol.",
        "document_disjoint_requirement": "assign this PMCID and full-document hash to one split before Markdown chunking or QA derivation",
        "protected_split_requirement": "exclude validation, OOD, shadow, and sealed-holdout documents from every training layer",
        "excluded_routes": provenance["excluded_routes"],
    }
    manifest = stable_json(manifest_value)
    report_value = {
        "schema": "site-corpus-report-v1",
        "resource_id": "europepmc_icar_suspension_syndrome",
        "role": "canonical_markdown_direct_training_source",
        "direct_training_ready": True,
        "non_qa": True,
        "source_claim_policy": "Review findings and consensus recommendations remain attributed and graded; harness-rescue evidence is not presented as validated for bondage suspension.",
        "counts": {
            "documents": 1,
            "sections": len(SECTIONS),
            "words": word_count(markdown),
            "source_tables_reviewed": 3,
            "included_studies": 23,
        },
        "evidence_level": {
            "design": "scoping review plus ICAR MEDCOM consensus recommendations",
            "epidemiology_studies": 2,
            "pathophysiology_studies": 21,
            "quality_good": 9,
            "quality_fair": 8,
            "quality_poor": 6,
            "randomized_intervention_studies": 0,
            "experimental_prevention_diagnosis_treatment_studies": 0,
            "exact_incidence_known": False,
            "bondage_transfer_validated": False,
        },
        "recommendation_grades_preserved": ["1C", "1B", "2B", "2B", "2C", "1A", "2C"],
        "genuine_gaps": [
            "current representative incidence and outcome registry",
            "later-stage pathophysiology after presyncope",
            "validated diagnostic classification",
            "experimental prevention, diagnosis, and treatment evidence",
            "randomized intervention evidence",
            "direct evidence for bondage-suspension transfer",
        ],
        "capture_boundary": {
            "authorized_get_routes": [SOURCE_URL],
            "request_count": provenance["request_count"],
            "excluded_routes": provenance["excluded_routes"],
        },
        "repository_provenance_discrepancy": provenance["discovery_metadata_discrepancy"],
        "hashes": {
            "markdown_sha256": text_digest(markdown),
            "manifest_sha256": text_digest(manifest),
            "source_snapshot_sha256": SOURCE_SHA256,
            "provenance_sha256": digest(PROVENANCE.read_bytes()),
        },
    }
    return markdown, manifest, stable_json(report_value)


def build() -> None:
    markdown, manifest, report = render()
    MARKDOWN.write_text(markdown, encoding="utf-8", newline="\n")
    MANIFEST.write_text(manifest, encoding="utf-8", newline="\n")
    REPORT.write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    build()
