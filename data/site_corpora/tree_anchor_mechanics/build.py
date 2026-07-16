#!/usr/bin/env python3
"""Build the TreeConsult rights-deferred inventory artifact offline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
REASON = "treeconsult_copyright_notice_requires_written_permission_for_text_reuse"
INTERNAL_REASON = "treeconsult_written_permission_required_for_text_reuse"
OUTSIDE_REASON = "outside_requested_tree_anchor_mechanics_and_written_permission_required"
EXTERNAL_REASON = "external_landing_404_and_separate_rights_review_required"
ROBOTS_URL = "https://www.tree-consult.org/robots.txt"
LEGAL_URL = "https://www.tree-consult.org/legal-notice.htm"
DOWNLOADS_URL = "https://www.tree-consult.org/downloads.htm"
HSE_URL = "https://www.hse.gov.uk/research/rrhtm/rr668.htm"
RIGHTS_FRAGMENT = (
    "no texts, excerpts of texts, images or parts of images may otherwise be used "
    "without written permission"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def notice_word_count(text: str) -> int:
    return len(re.findall(r"[\w’'-]+", text, flags=re.UNICODE))


def load_and_verify() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    policy = json.loads((ROOT / "policy_decision.json").read_text(encoding="utf-8"))
    provenance = json.loads((SNAPSHOT / "provenance.json").read_text(encoding="utf-8"))
    documents = json.loads((SNAPSHOT / "document_inventory.json").read_text(encoding="utf-8"))

    if policy["decision_reason"] != REASON or policy["direct_training_ready"]:
        raise RuntimeError("policy decision is not the reviewed rights deferral")
    if policy["rights_evidence_fragment"] != RIGHTS_FRAGMENT:
        raise RuntimeError("rights evidence does not match the reviewed notice")
    if policy["rights_evidence_fragment_word_count"] != 17:
        raise RuntimeError("rights evidence fragment unexpectedly changed")
    if policy["legal_notice_body_snapshot_retained"]:
        raise RuntimeError("legal-notice body must not be retained")

    expected_scope = "robots_and_page_metadata_plus_research_resource_head_only"
    if provenance["capture_scope"] != expected_scope:
        raise RuntimeError("capture scope is not the reviewed metadata-only scope")
    if provenance["research_document_get_requests"] != 0:
        raise RuntimeError("research document bodies were requested")
    if provenance["canonical_instructional_body_snapshots_retained"] != 0:
        raise RuntimeError("instructional body snapshots were retained")
    if provenance["inspected_html_body_snapshots_retained"] != 0:
        raise RuntimeError("inspected HTML body snapshots were retained")
    if provenance["research_resource_head_requests"] != 19:
        raise RuntimeError("HEAD inventory is incomplete")
    if policy["audited_at"] != provenance["completed_at"]:
        raise RuntimeError("policy and capture completion timestamps differ")
    if policy["legal_notice_url"] != provenance["legal_notice"]["requested_url"]:
        raise RuntimeError("policy and provenance legal-notice URLs differ")
    if policy["legal_notice_body_sha256"] != provenance["legal_notice"]["body_sha256"]:
        raise RuntimeError("policy and provenance legal-notice hashes differ")
    if policy["legal_notice_body_byte_length"] != provenance["legal_notice"]["body_byte_length"]:
        raise RuntimeError("policy and provenance legal-notice lengths differ")
    for key in ("legal_notice", "downloads_index"):
        if provenance[key]["body_snapshot_retained"]:
            raise RuntimeError(f"{key} body was retained")

    robots_path = ROOT / provenance["robots"]["path"]
    robots = robots_path.read_bytes()
    if digest(robots) != provenance["robots"]["sha256"] or len(robots) != provenance["robots"]["byte_length"]:
        raise RuntimeError("robots snapshot hash or length mismatch")
    robots_text = robots.decode("utf-8")
    for required in ("User-agent: *", "Disallow: /cms/", "Disallow: /includes/"):
        if required not in robots_text:
            raise RuntimeError(f"required robots directive missing: {required}")

    inventory_path = ROOT / provenance["document_inventory"]["path"]
    inventory_bytes = inventory_path.read_bytes()
    if digest(inventory_bytes) != provenance["document_inventory"]["sha256"]:
        raise RuntimeError("document inventory hash mismatch")
    if len(inventory_bytes) != provenance["document_inventory"]["byte_length"]:
        raise RuntimeError("document inventory length mismatch")
    if len(documents) != provenance["document_inventory"]["record_count"] or len(documents) != 19:
        raise RuntimeError("document inventory record count mismatch")
    if [item["source_order"] for item in documents] != list(range(1, 20)):
        raise RuntimeError("document source order is incomplete")
    if len({item["inventory_id"] for item in documents}) != 19 or len({item["url"] for item in documents}) != 19:
        raise RuntimeError("document identifiers or URLs are not unique")

    internal = [item for item in documents if item["hosted_by_treeconsult"]]
    external = [item for item in documents if not item["hosted_by_treeconsult"]]
    if len(internal) != 18 or len(external) != 1 or external[0]["url"] != HSE_URL:
        raise RuntimeError("internal/external inventory boundary changed")
    for item in documents:
        if item["body_retrieved"] or item["body_snapshot_retained"]:
            raise RuntimeError(f"research body capture recorded for {item['url']}")
        if item["access_audit"]["method"] != "HEAD":
            raise RuntimeError(f"non-HEAD research access recorded for {item['url']}")
        if not item["title_as_displayed"] or not item["citation_as_displayed"]:
            raise RuntimeError(f"incomplete displayed provenance for {item['url']}")
        publication = item["publication_metadata"]
        if not publication["date_as_displayed"] or not publication["status"] or not publication["year_claims"]:
            raise RuntimeError(f"incomplete date/version provenance for {item['url']}")
    if any(item["access_audit"]["http_status"] != 200 for item in internal):
        raise RuntimeError("a same-domain PDF was not publicly HEAD-accessible at capture time")
    if any(item["access_audit"]["content_type"] != "application/pdf" for item in internal):
        raise RuntimeError("a same-domain inventoried resource was not a PDF")
    if external[0]["access_audit"]["http_status"] != 404:
        raise RuntimeError("external HSE landing disposition changed")
    return policy, provenance, documents


def disposition(item: dict[str, object]) -> str:
    if not item["hosted_by_treeconsult"]:
        return EXTERNAL_REASON
    if item["scope_class"] == "outside_requested_tree_anchor_mechanics":
        return OUTSIDE_REASON
    return INTERNAL_REASON


def build() -> None:
    policy, provenance, documents = load_and_verify()
    captured = provenance["completed_at"]

    inventory: list[dict[str, object]] = []
    for item in documents:
        inventory.append(
            {
                **item,
                "disposition": "excluded_rights_deferred",
                "reason": disposition(item),
                "content_retrieved_for_corpus": False,
                "direct_training_included": False,
            }
        )
    inventory_text = "".join(stable_json(item) + "\n" for item in inventory)
    write(ROOT / "inventory.jsonl", inventory_text)

    dispositions = [
        {
            "url": ROBOTS_URL,
            "role": "robots_policy",
            "disposition": "retained_metadata",
            "body_snapshot_retained": True,
            "sha256": provenance["robots"]["sha256"],
        },
        {
            "url": LEGAL_URL,
            "role": "rights_policy",
            "disposition": "inspected_for_policy_body_not_retained",
            "body_snapshot_retained": False,
            "body_sha256": provenance["legal_notice"]["body_sha256"],
        },
        {
            "url": DOWNLOADS_URL,
            "role": "document_inventory_index",
            "disposition": "parsed_metadata_body_not_retained",
            "body_snapshot_retained": False,
            "body_sha256": provenance["downloads_index"]["body_sha256"],
        },
    ] + [
        {
            "url": item["url"],
            "inventory_id": item["inventory_id"],
            "role": item["resource_type"],
            "disposition": "excluded_rights_deferred",
            "reason": disposition(item),
            "access_method": "HEAD",
            "http_status": item["access_audit"]["http_status"],
            "body_retrieved": False,
            "direct_training_included": False,
        }
        for item in documents
    ]
    write(ROOT / "url_dispositions.jsonl", "".join(stable_json(item) + "\n" for item in dispositions))

    # This empty file is the complete machine-readable training surface.
    write(ROOT / "content_records.jsonl", "")

    corpus = f"""# TreeConsult tree-anchor mechanics access notice

**Status:** rights-deferred; excluded from direct training. **Content records:** 0. **Training sections:** 0. **Direct-training ready:** false. **Non-QA:** true.

The source audit completed at `{captured}`. TreeConsult's [legal notice]({LEGAL_URL}) says site content is copyright-protected and requires written permission before text or excerpts are otherwise used. Public availability and the [robots policy]({ROBOTS_URL}) are therefore not treated as permission to construct the requested dense paraphrase corpus. The decision is `{REASON}`.

The [downloads index]({DOWNLOADS_URL}) exposed a metadata-only inventory of nineteen resources in its arboriculture climbing-and-rigging category. Eighteen same-domain PDF endpoints accepted a public `HEAD` request; no PDF body was requested. One external HSE landing URL returned `404` and requires a separate source-specific rights review. The inspected HTML page bodies were not retained. `content_records.jsonl` is intentionally empty, while titles, displayed authorship, date/version claims, scope review, access status, and dispositions remain isolated in compliance inventory files and are not training prose.

No force value, unit, setup, procedure, figure, conclusion, anchor criterion, rejection criterion, or safety factor was copied, paraphrased, inferred, or mapped into a training section. The inventory remains explicitly in its original arboriculture context. It neither certifies a tree or hardpoint nor supports a claim that any anchor is safe for bondage or human suspension.

There are consequently no section-level taxonomy mappings. If written permission is documented later, the candidate scope would be reviewed for `rigging_mechanics` and `uplines_suspension_hardpoints`; unsupported categories must remain absent. A new capture and technical review would still be required.

Any future authorized corpus must assign train or validation membership to each canonical source document before Markdown chunking or QA derivation. Every Markdown section and all derived QA from a document must remain in that same split, and protected OOD, shadow, validation, and sealed-holdout documents must remain excluded from training.
"""
    write(ROOT / "CORPUS.md", corpus)

    internal = [item for item in documents if item["hosted_by_treeconsult"]]
    external = [item for item in documents if not item["hosted_by_treeconsult"]]
    candidate = [item for item in documents if str(item["scope_class"]).startswith("candidate_")]
    out_of_scope = [item for item in documents if item["scope_class"] == "outside_requested_tree_anchor_mechanics"]
    report = f"""# TreeConsult tree-anchor mechanics rights-deferred report

- Audit completed: `{captured}`
- Decision: `{REASON}`
- Direct-training ready: `false`
- Canonical training documents: `0`
- Training sections: `0`
- Direct-training words: `0`
- Access-notice words: `{notice_word_count(corpus)}`
- Listed climbing-and-rigging resources: `{len(documents)}`
- Same-domain PDF resources: `{len(internal)}`
- Same-domain PDFs returning `200 application/pdf` to `HEAD`: `{sum(item['access_audit']['http_status'] == 200 and item['access_audit']['content_type'] == 'application/pdf' for item in internal)}`
- External landing resources: `{len(external)}`
- External landing resources returning `404` to `HEAD`: `{sum(item['access_audit']['http_status'] == 404 for item in external)}`
- Metadata candidates within requested mechanics scope: `{len(candidate)}`
- Metadata resources outside requested mechanics scope: `{len(out_of_scope)}`
- Research document `GET` requests: `{provenance['research_document_get_requests']}`
- Retained instructional or research bodies: `0`
- Retained inspected HTML bodies: `0`
- Retained robots-policy files: `1`

## Rights and access disposition

The legal notice was publicly readable and contained a written-permission requirement for reuse of text and excerpts. The ordinary crawl directives did not prohibit the public downloads route, and no `Content-Signal` or `X-Robots-Tag` header was present on the three audited metadata pages. Those facts establish access behavior, not reuse permission. All same-domain research resources are deferred pending written permission. The single HSE landing is both stale at the captured URL and outside TreeConsult's rights boundary, so it needs an independent audit rather than silent substitution.

## Provenance and inventory boundary

The downloads category listed nineteen unique resources in source order: eighteen TreeConsult-hosted PDFs and one external HSE landing. The compliance inventory preserves displayed title, citation, authorship, date/version claims, conflicting year claims, file-size label, endpoint status, and manual topical disposition. No research body was opened. The metadata identifies seventeen potential mechanics candidates, one same-domain friction-hitch item outside this worker's tree-anchor scope, and one external report landing requiring separate review.

## Taxonomy and genuine gaps

There are no included sections, so `taxonomy_mappings` is empty. The permission-planning scope is limited to `rigging_mechanics` and `uplines_suspension_hardpoints`; it is not an assertion that either category has been captured. Missing by design are measured setups and units, anchor and redirect forces, friction results, rope-angle relationships, dynamic-event and peak-load behavior, branch or stem response, safety-factor and uncertainty reasoning, inspection and rejection criteria, experimental limitations, and any text-supported figure interpretation. No claim was transferred from arboriculture to bondage or human-suspension safety.

## Promotion boundary

The access notice and metadata inventory are compliance artifacts, not direct-training data and not QA source material. Permission alone would not promote them. A future authorized body capture must be rebuilt, technically reviewed, mapped section by section, and split by canonical source document before Markdown chunking or QA derivation. Existing validation, OOD, shadow, and sealed-holdout boundaries remain untouched.
"""
    write(ROOT / "REPORT.md", report)

    output_names = ("CORPUS.md", "REPORT.md", "content_records.jsonl", "inventory.jsonl", "url_dispositions.jsonl")
    manifest = {
        "schema_version": 1,
        "resource_id": "tree_anchor_mechanics",
        "source_id": "treeconsult_rigging_research",
        "artifact_role": "rights_deferred_metadata_inventory",
        "generated_at": captured,
        "policy_reason": REASON,
        "direct_training_ready": False,
        "non_qa": True,
        "content_record_count": 0,
        "training_document_count": 0,
        "training_section_count": 0,
        "direct_training_word_count": 0,
        "access_notice_word_count": notice_word_count(corpus),
        "word_count_method": "Unicode regex [\\w’'-]+ applied only to the non-training access notice",
        "inventory_record_count": len(documents),
        "same_domain_pdf_count": len(internal),
        "same_domain_pdf_head_200_count": sum(item["access_audit"]["http_status"] == 200 for item in internal),
        "external_resource_count": len(external),
        "candidate_scope_record_count": len(candidate),
        "outside_scope_record_count": len(out_of_scope),
        "research_document_get_requests": provenance["research_document_get_requests"],
        "canonical_instructional_body_snapshots_retained": provenance["canonical_instructional_body_snapshots_retained"],
        "taxonomy_mappings": [],
        "taxonomy_mapping_status": "no_training_sections_due_to_rights_deferral",
        "candidate_taxonomy_scope_for_permission_planning_only": ["rigging_mechanics", "uplines_suspension_hardpoints"],
        "genuine_gaps": [
            "measured test setups and units",
            "anchor and redirect loads",
            "friction and rope-angle relationships",
            "dynamic-event and peak-load behavior",
            "branch and stem response",
            "safety-factor and uncertainty reasoning",
            "inspection, selection, and rejection criteria",
            "experimental limitations and text-supported figure interpretation",
        ],
        "human_suspension_transfer_status": "no_tree_or_hardpoint_certification_or_safety_claim",
        "document_disjoint_requirement": "assign canonical source document before Markdown chunking or QA derivation; keep all Markdown and QA from that document in one split",
        "protected_split_requirement": "exclude validation, OOD, shadow, and sealed-holdout documents from every training layer",
        "policy_decision": {
            "path": "policy_decision.json",
            "sha256": digest((ROOT / "policy_decision.json").read_bytes()),
            "byte_length": (ROOT / "policy_decision.json").stat().st_size,
        },
        "source_snapshot": {
            "provenance": {
                "path": "source_snapshot/provenance.json",
                "sha256": digest((SNAPSHOT / "provenance.json").read_bytes()),
                "byte_length": (SNAPSHOT / "provenance.json").stat().st_size,
            },
            "robots": {
                "path": "source_snapshot/robots.txt",
                "sha256": digest((SNAPSHOT / "robots.txt").read_bytes()),
                "byte_length": (SNAPSHOT / "robots.txt").stat().st_size,
            },
            "document_inventory": {
                "path": "source_snapshot/document_inventory.json",
                "sha256": digest((SNAPSHOT / "document_inventory.json").read_bytes()),
                "byte_length": (SNAPSHOT / "document_inventory.json").stat().st_size,
                "record_count": len(documents),
            },
        },
        "outputs": {
            name: {"sha256": digest((ROOT / name).read_bytes()), "byte_length": (ROOT / name).stat().st_size}
            for name in output_names
        },
    }
    write(ROOT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    build()
