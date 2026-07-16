#!/usr/bin/env python3
"""Build the Europe PMC rope-neuropathy evidence corpus offline."""

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
MARKDOWN = ROOT / "europepmc_rope_neuropathy_study.md"
MANIFEST = ROOT / "manifest.json"
REPORT = ROOT / "report.json"
SOURCE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML"
SOURCE_SHA256 = "2419fe0e0757e6fcaf3b082e481b5a9688c41b322441f2bcc8a509a7772b4a85"
SOURCE_BYTES = 75939
TITLE = "Acute Radial Compressive Neuropathy: The Most Common Injury Induced by Japanese Rope Bondage"
AUTHORS = ["Vasily Khodulev", "Artsiom Klimko", "Nataliya Charnenka", "Marina Zharko", "Hanna Khoduleva"]
DOI = "10.7759/cureus.39588"
PMCID = "PMC10294117"
PMID = "37384078"
LICENSE_URL = "https://creativecommons.org/licenses/by/3.0/"
TAXONOMY = {
    "anatomy_injury_prevention",
    "bottoming_skills",
    "emergency_procedures",
    "teaching_evaluation",
}


SECTIONS = [
    {
        "heading": "Study question and evidence design",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "Khodulev and colleagues investigated which peripheral-nerve injuries appeared in voluntarily shared rope-bondage injury accounts and described one person with recurrent, clinically evaluated radial neuropathy. The evidence combines a retrospective, injury-enriched survey or case series with a detailed case report. It is not prospective surveillance, a population cohort, a case-control comparison, or an experiment. It therefore cannot estimate the incidence or prevalence of nerve injury among rope-bondage participants, compare the risk of different suspension forms, or establish that one setup or duration caused an observed outcome."
        ),
    },
    {
        "heading": "Recruitment and sample boundary",
        "categories": ["teaching_evaluation"],
        "text": (
            "The authors recruited through rope-bondage online forums, social media, and the personal networks of four experienced practitioners, along with participants willing to share injury experiences. The four practitioners reported three to five years of practice and approximately 100 binding cases per year. Those activity estimates were not linked to a defined number of unique people, sessions, exposure-hours, or systematically observed injuries, so they are not a denominator for an injury rate. The exploratory sample size was not chosen in advance, recruitment was non-random, and the main survey included people who had experienced injuries rather than a representative sample of all participants."
        ),
    },
    {
        "heading": "Methods and information collected",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "Semi-structured interviews and online survey responses covered demographics, anthropometric information, medical history, rope-bondage experience, symptoms, the injury context, reported suspension or compression duration, recovery, medical attention, practice changes, and residual effects. The instrument used open-ended, multiple-choice, and Likert-style questions. The authors classified injuries by the reported nerve type and location and looked for recurring themes. Most observations were subjective retrospective reports. All participants provided informed consent, according to the article's ethics statement."
        ),
    },
    {
        "heading": "Case accounting and exclusions",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "The main cohort contained 10 identified individuals and 16 counted peripheral-nerve injury instances associated with motor impairment. The article reports that, with one exception, these injuries appeared acutely and promptly after full-body suspension. Two people experienced multiple episodes; one person had simultaneous upper- and lower-limb injuries; and two people had bilateral injury to the same nerve structure. Repeated and bilateral events mean that injury instances are not independent observations. An additional person with bilateral proximal brachial-plexus symptoms beginning days after suspension was excluded from the main cohort because the cause was unclear. Fleeting hand or finger paresthesia and numbness were also reported but excluded when they did not meet the study's inclusion boundary."
        ),
    },
    {
        "heading": "Observed nerve pattern in the identified cases",
        "categories": ["anatomy_injury_prevention"],
        "text": (
            "Within the selected main cohort, radial-nerve injury was reported for 9 of 10 individuals and accounted for 13 of 16 counted injury instances. The remaining counted instances were one axillary-nerve injury and bilateral femoral-nerve injury in one participant. These fractions describe the authors' small, injury-selected case set only. They do not show that 90% of rope-bondage participants are injured, that 90% of sessions carry a radial-nerve injury, or that radial injury is necessarily the most common outcome in the wider community. Most nerve assignments were based on recalled symptoms and context rather than a uniform clinical examination."
        ),
    },
    {
        "heading": "Symptoms, medical attention, and diagnostic support",
        "categories": ["anatomy_injury_prevention", "bottoming_skills", "emergency_procedures"],
        "text": (
            "Radial-pattern accounts commonly included weakness of wrist or finger extension together with numbness over the back or outer back of the hand. The axillary-pattern account included an inability to abduct one arm after an acute event. The femoral-pattern account included altered sensation over the front of the thigh and a sense of knee instability. Five of the 10 people sought medical attention, and nerve-conduction studies were performed in two cases. Consequently, the survey supports a clinically important signal and a detailed electrodiagnostic case, but not uniform diagnostic confirmation across the cohort. The article's excluded brachial-plexus account developed later and was explicitly treated as causally uncertain."
        ),
    },
    {
        "heading": "Reported recovery range is descriptive, not a safety threshold",
        "categories": ["anatomy_injury_prevention", "bottoming_skills", "teaching_evaluation"],
        "text": (
            "Table 1 reports recovery of sensation and movement ranging from two minutes to five months across recorded episodes, with several outcomes measured in days or weeks and some lasting one to three months. The observed suspension times ranged from about five minutes to 30 minutes when a time was available. Because the sample contains only selected injury accounts, has no exposure comparison, and includes repeated events in the same people, these durations cannot define a safe time, a dose-response relationship, or an expected recovery time for another person. A short reported exposure and rapid recovery in one episode do not make a similar exposure safe."
        ),
    },
    {
        "heading": "Detailed recurrent radial-neuropathy case",
        "categories": ["anatomy_injury_prevention", "emergency_procedures", "teaching_evaluation"],
        "text": (
            "The clinically detailed participant was a 29-year-old woman at the first recorded episode. After a reported 25-minute full-body suspension, she developed left wrist and finger drop and reduced sensation in a radial distribution. She first attended the authors' department 48 days later. Motor nerve-conduction testing showed a 77.3% conduction block across the upper-arm segment, a small reduction in distal response amplitude, and no reported reduction in conduction velocity. Clinical improvement began after three months and was complete after five months. The authors interpreted the combination of conduction block, reduced motor response, and prolonged recovery as evidence compatible with focal demyelination plus axonal injury in this case."
        ),
    },
    {
        "heading": "Recurrence, follow-up, and ultrasound findings",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "Seventeen months after recovery, the same participant reported bilateral radial symptoms after a similar suspension lasting about eight to 10 minutes; improvement began after one week and was complete after four weeks. A third episode, three years after the first and reported after five minutes of suspension, resolved clinically within two minutes. Nerve-conduction testing one month after that third episode was normal in the tested radial, median, ulnar, fibular, tibial, and sural nerves. Ultrasound described the initially affected radial nerve as having a normal-range cross-sectional area of 0.10 square centimetres, while also being larger than the opposite side's 0.081 square centimetres and the cited mean of 0.087 ± 0.009 square centimetres. These within-person observations show variable recurrence and recovery; they do not establish an exposure limit or predict prognosis for other people."
        ),
    },
    {
        "heading": "Interpretation and causal restraint",
        "categories": ["anatomy_injury_prevention", "teaching_evaluation"],
        "text": (
            "The authors relate the detailed radial case to established compression-neuropathy mechanisms, including focal demyelination, conduction block, and possible axonal degeneration. Mechanical compression is biologically plausible, and the timing, symptom distribution, and electrodiagnostic findings strengthen that interpretation for the evaluated case. For the wider survey, however, the design cannot independently verify every reported nerve, separate compression from traction or other contributors, or establish the effect of a particular technique, body position, rope property, or duration. The article's causal wording and anatomical proposals should therefore be read as source conclusions from exploratory evidence, not as controlled proof or a validated protocol."
        ),
    },
    {
        "heading": "Limitations and uncertainty",
        "categories": ["teaching_evaluation"],
        "text": (
            "The authors acknowledge the small sample, retrospective recall, selection bias from online and personal-network recruitment, non-random sampling, and limited generalizability. Additional constraints follow from the reported methods: the sample was enriched for known injuries; transient sensory events were excluded; there was no uninjured comparison group or prospective exposure denominator; only half the main cohort sought medical attention; only two people received nerve-conduction studies; nerve classification and recovery were otherwise largely self-reported; repeated and bilateral events were counted within 16 injury instances; and the delayed brachial-plexus case had uncertain attribution. The study can identify hypotheses and clinically important case patterns, but it cannot quantify community risk or rank preventive interventions."
        ),
    },
    {
        "heading": "Safety meaning without tying instructions",
        "categories": ["anatomy_injury_prevention", "bottoming_skills", "emergency_procedures", "teaching_evaluation"],
        "text": (
            "The study shows that reported neurological symptoms can include motor loss as well as altered sensation, can arise promptly or—in the excluded uncertain case—later, and can resolve quickly or persist for months. The authors advocate recognition of unusual sensations or weakness, communication, anatomy education, avoidance of excessive focal pressure or extreme strain, and timely medical attention. They also state that further research and expert consensus are needed before comprehensive, standardized safety guidance can be developed. This corpus deliberately omits knot, wrap, placement, suspension, landmark, and time-limit instructions: the study does not validate a universally safe configuration, location, or duration."
        ),
    },
    {
        "heading": "Ethics, disclosure, and publication record",
        "categories": ["teaching_evaluation"],
        "text": (
            "The captured JATS states that participant consent was obtained or waived, that the work involved no animal subjects or tissue, and that the authors declared no competing interests. The article was published electronically on 28 May 2023 in Cureus, volume 15, issue 5, as e39588. The bibliographic identity in the JATS is DOI 10.7759/cureus.39588, PMCID PMC10294117, and PMID 37384078."
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
    if provenance["excluded_routes"] != ["Cureus publisher endpoint", "United States PMC endpoint"]:
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
        for node in meta.findall('./contrib-group/contrib[@contrib-type="author"]')
    ]
    if authors != AUTHORS:
        raise RuntimeError("author-list drift")
    license_text = normalize(" ".join(meta.find("permissions/license").itertext()))
    if LICENSE_URL not in license_text or "Creative Commons Attribution License" not in license_text:
        raise RuntimeError("CC BY 3.0 license drift")
    section_titles = [normalize(node.findtext("title", "")) for node in root.findall("./body/sec")]
    if section_titles != ["Introduction", "Materials and methods", "Results", "Discussion", "Conclusions"]:
        raise RuntimeError("article section inventory drift")
    if len(root.findall("./body/sec[3]/table-wrap")) != 2:
        raise RuntimeError("results-table inventory drift")
    source_text = normalize(" ".join(root.itertext()))
    for required in (
        "10 individuals (16 injuries)",
        "90.0% of patients and 81.3% of injuries",
        "CB of 77.3%",
        "small sample size and non-randomized recruitment methods",
        "further research and expert consensus is needed",
        "no competing interests exist",
    ):
        if required not in source_text:
            raise RuntimeError(f"required JATS evidence missing: {required}")
    return root, provenance


def render() -> tuple[str, str, str]:
    _, provenance = load_and_verify()
    lines = [
        "# Self-reported rope-bondage nerve injuries — clinical evidence corpus",
        "",
        f"Captured: {provenance['captured_at']}",
        "",
        f"Source article: {TITLE}",
        "",
        f"Authors: {', '.join(AUTHORS)}",
        "",
        "Journal record: Cureus 15(5), e39588 (28 May 2023)",
        "",
        f"Identifiers: DOI {DOI}; PMCID {PMCID}; PMID {PMID}",
        "",
        f"Authorized source route: {SOURCE_URL}",
        "",
        f"License: Creative Commons Attribution 3.0 ({LICENSE_URL})",
        "",
        "This non-Q&A document is an original, attributed evidence summary of the captured JATS article. The title and source conclusions are preserved as bibliographic claims, but the corpus distinguishes self-report, clinical observation, and inference. It does not reproduce a tying method and does not convert selected injury cases into causal rules, population rates, safe locations, or safe durations.",
        "",
    ]
    category_counts: Counter[str] = Counter()
    for section in SECTIONS:
        categories = section["categories"]
        if not set(categories) <= TAXONOMY:
            raise RuntimeError(f"unsupported taxonomy label: {categories}")
        category_counts.update(categories)
        lines.extend(
            [
                f"## {section['heading']}",
                "",
                "Categories: " + ", ".join(categories),
                "",
                section["text"],
                "",
            ]
        )
    markdown = "\n".join(lines).rstrip() + "\n"
    manifest_value = {
        "schema": "site-corpus-manifest-v1",
        "resource_id": "europepmc_rope_neuropathy_study",
        "source_type": "peer_reviewed_clinical_study",
        "artifact_role": "canonical_markdown_direct_training_source",
        "direct_training_ready": True,
        "non_qa": True,
        "retrieved_at": provenance["captured_at"],
        "license": {"name": "CC BY 3.0", "url": LICENSE_URL},
        "article": {
            "title": TITLE,
            "authors": AUTHORS,
            "journal": "Cureus",
            "publication_date": "2023-05-28",
            "doi": DOI,
            "pmcid": PMCID,
            "pmid": PMID,
        },
        "entries": [
            {
                "url": SOURCE_URL,
                "disposition": "included",
                "reason": "Authorized CC BY 3.0 JATS preserved as the sole source for an attributed evidence summary.",
                "content_type": provenance["content_type"],
                "bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
                "snapshot_path": "source_snapshot/fullTextXML.xml",
            }
        ],
        "supported_categories": sorted(category_counts),
        "document_disjoint_requirement": "assign this PMCID and full-document hash to one split before Markdown chunking or QA derivation",
        "protected_split_requirement": "exclude validation, OOD, shadow, and sealed-holdout documents from every training layer",
        "excluded_routes": provenance["excluded_routes"],
    }
    manifest = stable_json(manifest_value)
    report_value = {
        "schema": "site-corpus-report-v1",
        "resource_id": "europepmc_rope_neuropathy_study",
        "role": "canonical_markdown_direct_training_source",
        "direct_training_ready": True,
        "non_qa": True,
        "source_claim_policy": "Self-reported and clinical observations remain attributed; no prevalence, causal, safe-duration, or tying-protocol inference is added.",
        "counts": {
            "documents": 1,
            "sections": len(SECTIONS),
            "words": word_count(markdown),
            "source_tables_reviewed": 2,
            "source_body_sections_reviewed": 5,
        },
        "category_section_counts": dict(sorted(category_counts.items())),
        "supported_categories": sorted(category_counts),
        "evidence_level": {
            "design": "retrospective injury-enriched survey/case series plus one detailed recurrent case report",
            "main_cohort_people": 10,
            "counted_injury_instances": 16,
            "medical_attention_people": 5,
            "nerve_conduction_study_people": 2,
            "population_incidence_estimable": False,
            "population_prevalence_estimable": False,
            "causal_protocol_validated": False,
        },
        "genuine_gaps": [
            "representative exposure denominator and population injury rate",
            "prospective symptom and outcome ascertainment",
            "uniform clinical and electrodiagnostic confirmation",
            "uninjured or alternative-exposure comparison groups",
            "independent observations without repeated or bilateral counting",
            "validated preventive interventions and standardized safety guidance",
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
