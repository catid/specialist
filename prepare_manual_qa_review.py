#!/usr/bin/env python3
"""Package generated QA candidates into small source-document review packets.

This script deliberately performs no QA generation or judging.  It only joins
an already leakage-gated candidate pool to the complete raw source document so
a reviewer can make source-grounded decisions with a bounded context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_documents(raw_dir: Path):
    documents = {}
    for path in sorted(raw_dir.glob("*.json")):
        item = json.loads(path.read_text())
        url, text = item.get("url", ""), item.get("text", "")
        if not url or not text:
            continue
        if url in documents:
            raise ValueError(f"duplicate raw source URL: {url}")
        documents[url] = (path, item)
    return documents


def prepare(candidates_path: Path, raw_dir: Path, output_dir: Path,
            urls: list[str], batch: str):
    candidates_by_url = defaultdict(list)
    for line_number, item in read_jsonl(candidates_path):
        url = item.get("url", "")
        if url in urls:
            candidates_by_url[url].append({
                "fact_id": item["fact_id"],
                "question": item["question"],
                "answer": item["answer"],
                "candidate_line": line_number,
            })

    documents = load_documents(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "manual-qa-packets-v1",
        "batch": batch,
        "candidate_input": str(candidates_path.resolve()),
        "candidate_sha256": file_sha256(candidates_path),
        "packets": [],
    }
    for index, url in enumerate(urls, 1):
        if url not in documents:
            raise ValueError(f"raw source not found for URL: {url}")
        if not candidates_by_url[url]:
            raise ValueError(f"no candidates found for URL: {url}")
        source_path, document = documents[url]
        text = document["text"]
        packet_name = f"packet_{index:03d}.json"
        packet = {
            "schema": "manual-qa-packet-v1",
            "batch": batch,
            "source_file": str(source_path.resolve()),
            "source": document.get("source", ""),
            "url": url,
            "title": document.get("title", ""),
            "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_text": text,
            "candidates": candidates_by_url[url],
        }
        (output_dir / packet_name).write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
        manifest["packets"].append({
            "file": packet_name,
            "url": url,
            "candidate_count": len(packet["candidates"]),
            "document_characters": len(text),
        })
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--url", dest="urls", action="append", required=True)
    args = parser.parse_args()
    manifest = prepare(args.candidates, args.raw_dir, args.output_dir,
                       args.urls, args.batch)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
