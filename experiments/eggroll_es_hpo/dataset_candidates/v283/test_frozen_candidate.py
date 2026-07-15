#!/usr/bin/env python3
"""Rebuild and validate the immutable v283 train-only dataset candidate."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MANIFEST = HERE / "manifest.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip()
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"si", "fbclid", "gclid"}
    ]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(sorted(query)), ""))


def row_urls(row: dict) -> set[str]:
    scalar = (
        "url",
        "evidence_url",
        "canonical_url",
        "supplied_url",
        "original_ropetopia_url",
        "title_evidence_url",
        "url_evidence_url",
        "canonical_resource_url",
    )
    arrays = ("urls", "canonical_urls", "supplied_urls")
    values = {row[key] for key in scalar if row.get(key)}
    values.update(url for key in arrays for url in row.get(key, ()) if url)
    return {normalize_url(value) for value in values}


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def duplicate_groups(rows: list[dict]) -> dict[str, int]:
    return {
        key: sum(count > 1 for count in Counter(normalize(row[key]) for row in rows).values())
        for key in ("fact_id", "question", "answer")
    }


class FrozenV283Candidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.artifact = ROOT / cls.manifest["artifact"]["path"]
        builder_path = ROOT / cls.manifest["lineage"]["canonical_builder"]["path"]
        sys.path.insert(0, str(builder_path.parent))
        import build_context_merit_audit_v283 as builder

        cls.builder = builder
        cls.builder_path = builder_path
        cls.rows = read_jsonl(cls.artifact)
        cls.production = read_jsonl(ROOT / cls.manifest["lineage"]["production_input"]["path"])

        sampler_path = ROOT / cls.manifest["train_panel_capacity"]["lexical_semantic_rule"]["implementation_path"]
        sys.path.insert(0, str(ROOT))
        import eggroll_es_train_panel_sampler_v13 as sampler

        cls.sampler = sampler
        cls.sampler_path = sampler_path

    def test_01_manifest_and_lineage_bindings(self) -> None:
        artifact = self.manifest["artifact"]
        lineage = self.manifest["lineage"]
        self.assertEqual(self.manifest["schema"], "specialist-train-only-candidate-freeze-v1")
        self.assertEqual((len(self.rows), file_sha256(self.artifact)), (artifact["rows"], artifact["sha256"]))
        self.assertEqual(file_sha256(self.builder_path), lineage["canonical_builder"]["sha256"])
        self.assertEqual(self.builder.EXPECTED_OUTPUT_SHA256, artifact["sha256"])
        production = ROOT / lineage["production_input"]["path"]
        self.assertEqual((len(read_jsonl(production)), file_sha256(production)), (784, lineage["production_input"]["sha256"]))
        production_report = ROOT / lineage["production_report"]["path"]
        self.assertEqual(file_sha256(production_report), lineage["production_report"]["sha256"])

    def test_02_curation_chain_binding(self) -> None:
        records = []
        for path in self.builder.OUTPUT_PROJECTION_CURATIONS:
            path = Path(path)
            records.append(f"{path.relative_to(ROOT)}\t{file_sha256(path)}")
        root = hashlib.sha256(("\n".join(records) + "\n").encode()).hexdigest()
        chain = self.manifest["lineage"]["curation_chain"]
        self.assertEqual((len(records), root), (chain["entries"], chain["root_sha256"]))
        self.assertEqual(records[-1], f"{chain['terminal_path']}\t{chain['terminal_sha256']}")

    def test_03_content_guards(self) -> None:
        validation = self.manifest["validation"]
        for key in ("fact_id", "question", "answer"):
            counts = Counter(normalize(row[key]) for row in self.rows)
            self.assertFalse([value for value, count in counts.items() if count > 1], key)
        blob = self.artifact.read_text()
        self.assertEqual({token: blob.count(token) for token in validation["leak_token_matches"]}, validation["leak_token_matches"])
        resource_manifest = json.loads(self.builder.RESOURCE_MANIFEST.read_text())
        urls = {
            url
            for resource in resource_manifest["resources"]
            for url in (resource["canonical_url"], resource.get("recommendation_url"))
            if url
        }
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))

    def test_04_complete_candidate_and_capacity_manifest(self) -> None:
        capacity = self.manifest["train_panel_capacity"]
        self.assertFalse(self.manifest["artifact"]["is_overlay"])
        self.assertEqual(
            self.manifest["artifact"]["candidate_role"],
            "complete_training_candidate_replacing_production_if_promoted",
        )
        semantic_ids = self.sampler.build_semantic_clusters(self.rows)
        self.assertEqual(file_sha256(self.sampler_path), capacity["lexical_semantic_rule"]["implementation_sha256"])
        self.assertEqual(len(set(semantic_ids)), capacity["lexical_semantic_rule"]["clusters"])

        disjoint = DisjointSet(len(self.rows))
        first: dict[str, int] = {}

        def connect(identifier: str, index: int) -> None:
            if identifier in first:
                disjoint.union(index, first[identifier])
            else:
                first[identifier] = index

        for index, (row, semantic_id) in enumerate(zip(self.rows, semantic_ids)):
            connect(f"document:{row['document_sha256']}", index)
            for url in row_urls(row):
                connect(f"url:{url}", index)
            lineage = row.get("source_lineage") or {}
            for key in ("raw", "raw_document", "raw_successor_document"):
                if lineage.get(key):
                    connect(f"lineage:{key}:{json.dumps(lineage[key], sort_keys=True)}", index)
            connect(f"semantic:{semantic_id}", index)

        components: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.rows)):
            components[disjoint.find(index)].append(index)
        assignments = []
        unit_ids = []
        unit_sizes = Counter()
        strata = Counter()
        for indices in components.values():
            member_ids = sorted(self.sampler.row_sha256(self.rows[index]) for index in indices)
            unit_id = self.sampler.canonical_sha256(
                {"schema": "v283-conservative-conflict-unit-v1", "members": member_ids}
            )
            unit_ids.append(unit_id)
            assignments.extend(f"{row_id}\t{unit_id}" for row_id in member_ids)
            unit_sizes[len(indices)] += 1
            row_strata = Counter(self.sampler.classify_stratum(self.rows[index]) for index in indices)
            dominant = max(
                self.sampler.STRATA,
                key=lambda name: (row_strata[name], self.sampler._TIE_PRIORITY[name]),
            )
            strata[dominant] += 1

        conflict = capacity["conflict_units"]
        assignment_root = hashlib.sha256(("\n".join(sorted(assignments)) + "\n").encode()).hexdigest()
        unit_root = hashlib.sha256(("\n".join(sorted(unit_ids)) + "\n").encode()).hexdigest()
        self.assertEqual(len({row["document_sha256"] for row in self.rows}), capacity["unique_document_sha256s"])
        self.assertEqual((len(components), assignment_root, unit_root), (conflict["count"], conflict["assignment_root_sha256"], conflict["unit_id_set_root_sha256"]))
        self.assertEqual({str(key): value for key, value in sorted(unit_sizes.items())}, conflict["unit_size_histogram"])
        self.assertEqual(dict(strata), conflict["per_stratum"])

        primary_urls = [normalize_url(row["url"]) for row in self.rows]
        primary_counts = Counter(primary_urls)
        url_assignments = sorted(
            f"{self.sampler.row_sha256(row)}\t{hashlib.sha256(url.encode()).hexdigest()}"
            for row, url in zip(self.rows, primary_urls)
        )
        url_root = hashlib.sha256(("\n".join(url_assignments) + "\n").encode()).hexdigest()
        urls = capacity["url_groups"]
        self.assertEqual(len({url for row in self.rows for url in row_urls(row)}), urls["all_normalized_identifiers"])
        self.assertEqual((len(primary_counts), url_root), (urls["primary_group_count"], urls["primary_assignment_root_sha256"]))
        self.assertEqual(
            {str(key): value for key, value in sorted(Counter(primary_counts.values()).items())},
            urls["primary_group_size_histogram"],
        )

        comparison = capacity["duplicate_and_production_conflicts"]
        self.assertEqual(duplicate_groups(self.rows), comparison["candidate_normalized_duplicate_groups"])
        self.assertEqual(duplicate_groups(self.production), comparison["production_normalized_duplicate_groups"])
        candidate_by_fact = {row["fact_id"]: row for row in self.rows}
        production_by_fact = {row["fact_id"]: row for row in self.production}
        shared_facts = set(candidate_by_fact) & set(production_by_fact)
        identical_facts = sum(
            (normalize(candidate_by_fact[key]["question"]), normalize(candidate_by_fact[key]["answer"]))
            == (normalize(production_by_fact[key]["question"]), normalize(production_by_fact[key]["answer"]))
            for key in shared_facts
        )
        candidate_questions = {normalize(row["question"]): normalize(row["answer"]) for row in self.rows}
        production_questions = {normalize(row["question"]): normalize(row["answer"]) for row in self.production}
        shared_questions = set(candidate_questions) & set(production_questions)
        expected = {
            "candidate_only_fact_ids": len(set(candidate_by_fact) - set(production_by_fact)),
            "production_only_fact_ids": len(set(production_by_fact) - set(candidate_by_fact)),
            "same_fact_id_changed_normalized_qa": len(shared_facts) - identical_facts,
            "same_fact_id_identical_normalized_qa": identical_facts,
            "shared_document_sha256s": len(
                {row["document_sha256"] for row in self.rows}
                & {row["document_sha256"] for row in self.production}
            ),
            "shared_exact_normalized_qa_pairs": len(
                {(normalize(row["question"]), normalize(row["answer"])) for row in self.rows}
                & {(normalize(row["question"]), normalize(row["answer"])) for row in self.production}
            ),
            "shared_fact_ids": len(shared_facts),
            "shared_normalized_questions": len(shared_questions),
            "shared_question_changed_answer": sum(
                candidate_questions[key] != production_questions[key] for key in shared_questions
            ),
            "shared_question_same_answer": sum(
                candidate_questions[key] == production_questions[key] for key in shared_questions
            ),
        }
        self.assertEqual(expected, comparison["versus_active_production"])
        gate = capacity["five_panel_gate"]
        self.assertFalse(gate["eligible"])
        self.assertLess(conflict["count"], gate["required_conflict_units"])
        self.assertEqual(gate["decision"], "fail_closed_insufficient_independent_units")

    def test_05_deterministic_rebuild(self) -> None:
        with tempfile.TemporaryDirectory(prefix="specialist-v283-freeze-") as temp:
            output = Path(temp) / "candidate.jsonl"
            report = Path(temp) / "candidate.report.json"
            datasets = []
            reports = []
            for _ in (1, 2):
                self.builder.build_projection(output, report, self.builder.OUTPUT_PROJECTION_CURATIONS)
                datasets.append(output.read_bytes())
                reports.append(report.read_bytes())
            self.assertEqual(datasets[0], datasets[1])
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(datasets[0], self.artifact.read_bytes())


if __name__ == "__main__":
    unittest.main()
