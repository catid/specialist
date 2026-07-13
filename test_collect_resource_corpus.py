import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import collect_resource_corpus as corpus


class FakeResponse:
    def __init__(
        self,
        body="",
        *,
        status=200,
        headers=None,
        url=None,
        encoding="utf-8",
    ):
        self.content = body if isinstance(body, bytes) else body.encode(encoding)
        self.text = self.content.decode(encoding, errors="replace")
        self.status_code = status
        self.headers = headers or {}
        self.url = url
        self.encoding = encoding


class FakeSession:
    def __init__(self, responses=None):
        self.responses = dict(responses or {})
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url not in self.responses:
            raise AssertionError(f"unexpected network request: {url}")
        response = self.responses[url]
        if isinstance(response, list):
            if not response:
                raise AssertionError(f"no fake responses left for: {url}")
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        if response.url is None:
            response.url = url
        return response


class FakeYouTubeAdapter:
    name = "fixture-youtube"

    def __init__(self, items=None, error=None):
        self.items = items or []
        self.error = error
        self.calls = []

    def collect(self, resource):
        self.calls.append(resource["id"])
        if self.error:
            raise self.error
        return self.items


def make_resource(
    resource_id,
    canonical_url,
    mode,
    *,
    max_pages=None,
    reason=None,
):
    collection = {"mode": mode}
    if max_pages is not None:
        collection["max_pages"] = max_pages
    if reason is not None:
        collection["reason"] = reason
    return {
        "id": resource_id,
        "name": resource_id.replace("_", " ").title(),
        "category": "learning",
        "purpose": "fixture resource",
        "supplied_url": canonical_url,
        "canonical_url": canonical_url,
        "recommendation_question": "Where is the fixture resource?",
        "collection": collection,
    }


def robots(body, *, status=200, headers=None):
    return FakeResponse(
        body,
        status=status,
        headers=headers or {"Content-Type": "text/plain"},
    )


def html_page(text="fixture page body " * 20, *, headers=None, url=None):
    body = f"<html><head><title>Fixture title</title></head><body>{text}</body></html>"
    return FakeResponse(
        body,
        headers=headers or {"Content-Type": "text/html; charset=utf-8"},
        url=url,
    )


def sitemap(xml, *, headers=None):
    return FakeResponse(
        xml,
        headers=headers or {"Content-Type": "application/xml"},
    )


class CollectorTests(unittest.TestCase):
    def run_collection(
        self,
        directory,
        resources,
        session,
        *,
        youtube_adapter=None,
        resource_ids=None,
    ):
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        output_dir = directory / "raw"
        coverage_dir = directory / "coverage"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "rope-resource-manifest-v1",
                    "resources": resources,
                    "category_questions": [],
                }
            )
        )
        summary = corpus.collect_manifest(
            manifest_path,
            output_dir,
            coverage_dir,
            resource_ids=resource_ids,
            session=session,
            youtube_adapter=youtube_adapter,
            auto_youtube_adapter=False,
            delay=0,
            min_text_chars=20,
            extractor=lambda _html, _url: (
                "Fixture title",
                "A narrowly relevant fixture document with enough factual text.",
                "fixture-extractor",
            ),
        )
        return summary, output_dir, coverage_dir

    def load_coverage(self, coverage_dir, resource_id):
        return json.loads(
            (Path(coverage_dir) / f"{resource_id}.coverage.json").read_text()
        )

    def documents(self, output_dir):
        return [json.loads(path.read_text()) for path in sorted(Path(output_dir).glob("*.json"))]

    def test_reference_only_never_touches_network_and_is_reported(self):
        resource = make_resource(
            "theduchy",
            "https://www.theduchy.com/",
            "reference_only",
            reason="Content-Signal declares ai-train=no",
        )
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            summary, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session
            )
            coverage = self.load_coverage(coverage_dir, "theduchy")
            self.assertEqual(session.calls, [])
            self.assertEqual(summary["counts"], {
                "fetched": 0,
                "skipped": 0,
                "blocked": 1,
                "failed": 0,
            })
            self.assertEqual(coverage["status"], "blocked")
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"],
                "manifest_reference_only",
            )
            self.assertEqual(self.documents(output_dir), [])

    def test_crash_restraint_reference_only_terms_block_precedes_network(self):
        resource = make_resource(
            "crash_restraint",
            "https://crash-restraint.com/",
            "reference_only",
            reason="site terms prohibit automated and manual scraping",
        )
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(directory, [resource], session)
            coverage = self.load_coverage(coverage_dir, "crash_restraint")
            self.assertEqual(session.calls, [])
            self.assertEqual(coverage["counts"]["blocked"], 1)
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"],
                "manifest_reference_only",
            )
            self.assertEqual(self.documents(output_dir), [])

    def test_robots_content_signal_blocks_before_page_request(self):
        url = "https://blocked.example/tutorial"
        resource = make_resource("blocked_signal", url, "exact_public_page")
        session = FakeSession(
            {
                "https://blocked.example/robots.txt": robots(
                    "User-agent: *\nDisallow:\n"
                    "Content-Signal: search=yes, ai-train=no, use=reference\n"
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(directory, [resource], session)
            coverage = self.load_coverage(coverage_dir, "blocked_signal")
            self.assertEqual([call[0] for call in session.calls], [
                "https://blocked.example/robots.txt"
            ])
            self.assertTrue(coverage["robots"]["ai_train_no"])
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"],
                "robots_content_signal_ai_train_no",
            )
            self.assertEqual(self.documents(output_dir), [])

    def test_robots_disallow_blocks_exact_page(self):
        url = "https://rules.example/private/tutorial"
        resource = make_resource("robots_block", url, "exact_public_page")
        session = FakeSession(
            {
                "https://rules.example/robots.txt": robots(
                    "User-agent: *\nDisallow: /private/\n"
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, _, coverage_dir = self.run_collection(directory, [resource], session)
            coverage = self.load_coverage(coverage_dir, "robots_block")
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"], "robots_disallow"
            )

    def test_exact_page_writes_provenance_rich_document_and_coverage(self):
        url = "https://public.example/tutorial"
        resource = make_resource("public_page", url, "exact_public_page")
        session = FakeSession(
            {
                "https://public.example/robots.txt": robots("User-agent: *\nDisallow:\n"),
                url: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            summary, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session
            )
            documents = self.documents(output_dir)
            coverage = self.load_coverage(coverage_dir, "public_page")
            self.assertEqual(summary["counts"]["fetched"], 1)
            self.assertEqual(coverage["counts"], {
                "fetched": 1,
                "skipped": 0,
                "blocked": 0,
                "failed": 0,
            })
            self.assertEqual(len(documents), 1)
            document = documents[0]
            self.assertEqual(document["schema"], "resource-corpus-document-v1")
            self.assertEqual(document["source"], "public_page")
            self.assertEqual(document["resource_id"], "public_page")
            self.assertEqual(document["url"], url)
            self.assertEqual(
                document["document_sha256"],
                hashlib.sha256(document["text"].encode()).hexdigest(),
            )
            lineage = document["source_lineage"]
            self.assertEqual(lineage["requested_url"], url)
            self.assertEqual(lineage["robots_url"], "https://public.example/robots.txt")
            self.assertEqual(lineage["extractor"], "fixture-extractor")

    def test_page_content_signal_header_blocks_saving(self):
        url = "https://header.example/tutorial"
        resource = make_resource("header_signal", url, "exact_public_page")
        session = FakeSession(
            {
                "https://header.example/robots.txt": robots("User-agent: *\nDisallow:\n"),
                url: html_page(
                    headers={
                        "Content-Type": "text/html",
                        "Content-Signal": "search=yes, ai-train=no, use=reference",
                    }
                ),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(directory, [resource], session)
            coverage = self.load_coverage(coverage_dir, "header_signal")
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"],
                "page_content_signal_ai_train_no",
            )
            self.assertEqual(self.documents(output_dir), [])

    def test_rope365_uses_only_page_sitemap_and_relevant_paths(self):
        root = "https://rope365.com/"
        index = "https://rope365.com/sitemap_index.xml"
        pages = "https://rope365.com/page-sitemap.xml"
        posts = "https://rope365.com/post-sitemap.xml"
        safety = "https://rope365.com/safety/"
        shop = "https://rope365.com/shop/"
        random = "https://rope365.com/company-history/"
        resource = make_resource("rope365", root, "relevant_public_pages", max_pages=20)
        resource["collection"]["exclude_url_patterns"] = [
            r"/shop/", r"/company-history/"
        ]
        session = FakeSession(
            {
                "https://rope365.com/robots.txt": robots(
                    f"User-agent: *\nDisallow:\nSitemap: {index}\n"
                ),
                index: sitemap(
                    "<sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
                    f"<sitemap><loc>{pages}</loc></sitemap>"
                    f"<sitemap><loc>{posts}</loc></sitemap>"
                    "</sitemapindex>"
                ),
                pages: sitemap(
                    "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
                    f"<url><loc>{root}</loc></url>"
                    f"<url><loc>{safety}</loc></url>"
                    f"<url><loc>{shop}</loc></url>"
                    f"<url><loc>{random}</loc></url>"
                    "</urlset>"
                ),
                root: html_page(),
                safety: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(directory, [resource], session)
            coverage = self.load_coverage(coverage_dir, "rope365")
            called_urls = [call[0] for call in session.calls]
            self.assertIn(pages, called_urls)
            self.assertNotIn(posts, called_urls)
            self.assertNotIn(shop, called_urls)
            self.assertNotIn(random, called_urls)
            self.assertEqual({doc["url"] for doc in self.documents(output_dir)}, {root, safety})
            self.assertEqual(coverage["counts"]["fetched"], 2)
            skipped_reasons = {
                item["reason"] for item in coverage["results"]["skipped"]
            }
            self.assertIn("sitemap_out_of_scope", skipped_reasons)
            self.assertIn("manifest_exclude_pattern", skipped_reasons)

    def test_sitemap_parser_ignores_nested_image_locations(self):
        xml = """<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
          xmlns:image='http://www.google.com/schemas/sitemap-image/1.1'>
          <url><loc>https://example.test/lesson/</loc>
            <image:image><image:loc>https://example.test/lesson.jpg</image:loc></image:image>
          </url>
        </urlset>"""
        kind, locations = corpus.parse_sitemap_locations(xml)
        self.assertEqual(kind, "urlset")
        self.assertEqual(locations, ["https://example.test/lesson/"])

    def test_manifest_plural_sitemaps_and_include_exclude_patterns_are_applied(self):
        root = "https://rope365.com/"
        declared_sitemap = "https://rope365.com/page-sitemap.xml"
        broad_sitemap = "https://rope365.com/sitemap_index.xml"
        safety = "https://rope365.com/safety/"
        learning = "https://rope365.com/learning/foundations/"
        excluded = "https://rope365.com/learning/private/"
        unrelated = "https://rope365.com/company-history/"
        resource = make_resource(
            "rope365", root, "relevant_public_pages", max_pages=3
        )
        resource["collection"].update(
            {
                "sitemaps": [declared_sitemap],
                "include_url_patterns": [r"/(?:safety|learning)/"],
                "exclude_url_patterns": [r"/learning/private/"],
            }
        )
        session = FakeSession(
            {
                "https://rope365.com/robots.txt": robots(
                    f"User-agent: *\nDisallow:\nSitemap: {broad_sitemap}\n"
                ),
                declared_sitemap: sitemap(
                    "<urlset>"
                    f"<url><loc>{safety}</loc></url>"
                    f"<url><loc>{learning}</loc></url>"
                    f"<url><loc>{excluded}</loc></url>"
                    f"<url><loc>{unrelated}</loc></url>"
                    "</urlset>"
                ),
                root: html_page(),
                safety: html_page(),
                learning: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session
            )
            coverage = self.load_coverage(coverage_dir, "rope365")
            called_urls = [call[0] for call in session.calls]
            self.assertNotIn(broad_sitemap, called_urls)
            self.assertNotIn(excluded, called_urls)
            self.assertNotIn(unrelated, called_urls)
            self.assertEqual(
                {document["url"] for document in self.documents(output_dir)},
                {root, safety, learning},
            )
            skipped_reasons = {
                item["reason"] for item in coverage["results"]["skipped"]
            }
            self.assertIn("manifest_exclude_pattern", skipped_reasons)
            self.assertIn("manifest_include_pattern_miss", skipped_reasons)

    def test_manifest_sitemap_filters_work_without_site_hardcoding(self):
        root = "https://manifest.example/"
        sitemap_url = "https://manifest.example/pages.xml"
        lesson = "https://manifest.example/tutorials/single-column"
        excluded = "https://manifest.example/tutorials/private/member-lesson"
        unrelated = "https://manifest.example/store/candles"
        resource = make_resource(
            "manifest_site", root, "relevant_public_pages", max_pages=2
        )
        resource["collection"].update(
            {
                "sitemaps": [sitemap_url],
                "include_url_patterns": [r"/tutorials/"],
                "exclude_url_patterns": [r"/tutorials/private/"],
            }
        )
        session = FakeSession(
            {
                "https://manifest.example/robots.txt": robots(
                    "User-agent: *\nDisallow:\n"
                ),
                sitemap_url: sitemap(
                    "<urlset>"
                    f"<url><loc>{lesson}</loc></url>"
                    f"<url><loc>{excluded}</loc></url>"
                    f"<url><loc>{unrelated}</loc></url>"
                    "</urlset>"
                ),
                root: html_page(),
                lesson: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_collection(directory, [resource], session)
            self.assertEqual(
                {document["url"] for document in self.documents(output_dir)},
                {root, lesson},
            )
            called_urls = [call[0] for call in session.calls]
            self.assertNotIn(excluded, called_urls)
            self.assertNotIn(unrelated, called_urls)

    def test_xpole_exact_mode_collects_both_manifest_urls_with_max_pages_two(self):
        product = "https://xpoleus.com/shop-all/aerial/a-frame/xpole-a-frame/"
        support = "https://xpoleus.com/support/a-frame/"
        resource = make_resource("xpole_a_frame", product, "exact_public_page")
        resource["collection"].update(
            {"urls": [product, support], "max_pages": 2}
        )
        session = FakeSession(
            {
                "https://xpoleus.com/robots.txt": robots(
                    "User-agent: *\nDisallow:\n"
                ),
                product: html_page(),
                support: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session
            )
            coverage = self.load_coverage(coverage_dir, "xpole_a_frame")
            self.assertEqual(coverage["candidate_urls"], [product, support])
            self.assertEqual(coverage["counts"]["fetched"], 2)
            self.assertEqual(
                {document["url"] for document in self.documents(output_dir)},
                {product, support},
            )

    def test_explicit_public_membership_landing_page_is_not_mistaken_for_login(self):
        root = "https://catalog.example/"
        membership = "https://catalog.example/pages/membership"
        resource = make_resource(
            "public_catalog", root, "public_catalog_and_descriptions", max_pages=2
        )
        resource["collection"]["urls"] = [root, membership]
        session = FakeSession(
            {
                "https://catalog.example/robots.txt": robots(
                    "User-agent: *\nDisallow:\n"
                ),
                root: html_page(),
                membership: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session
            )
            coverage = self.load_coverage(coverage_dir, "public_catalog")
            self.assertEqual(coverage["counts"]["fetched"], 2)
            self.assertEqual(len(self.documents(output_dir)), 2)

    def test_explicit_url_list_replaces_canonical_as_fetch_plan(self):
        canonical = "https://supplier.example/"
        first = "https://supplier.example/shop/upline-rope"
        second = "https://supplier.example/shop/hemp-rope"
        resource = make_resource(
            "explicit_supplier", canonical, "reference_plus_narrow_specs"
        )
        resource["collection"].update(
            {"urls": [first, second], "max_pages": 2}
        )
        session = FakeSession(
            {
                "https://supplier.example/robots.txt": robots(
                    "User-agent: *\nDisallow:\n"
                    "Sitemap: https://supplier.example/general-commerce-sitemap.xml\n"
                ),
                first: html_page(),
                second: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session
            )
            coverage = self.load_coverage(coverage_dir, "explicit_supplier")
            self.assertEqual(
                [call[0] for call in session.calls],
                ["https://supplier.example/robots.txt", first, second],
            )
            self.assertNotIn(canonical, [call[0] for call in session.calls])
            self.assertEqual(coverage["candidate_urls"], [first, second])
            self.assertEqual(len(self.documents(output_dir)), 2)

    def test_commerce_sitemap_fetches_only_resource_specific_product(self):
        root = "https://www.subspacedesigns.shop/"
        index = "https://www.subspacedesigns.shop/sitemap_index.xml"
        products = "https://www.subspacedesigns.shop/product-sitemap.xml"
        posts = "https://www.subspacedesigns.shop/post-sitemap.xml"
        plate = "https://www.subspacedesigns.shop/product/suspension-rigging-plate"
        candle = "https://www.subspacedesigns.shop/product/scented-candle"
        resource = make_resource(
            "subspace_designs", root, "relevant_product_pages", max_pages=10
        )
        session = FakeSession(
            {
                "https://www.subspacedesigns.shop/robots.txt": robots(
                    f"User-agent: *\nDisallow:\nSitemap: {index}\n"
                ),
                index: sitemap(
                    "<sitemapindex>"
                    f"<sitemap><loc>{products}</loc></sitemap>"
                    f"<sitemap><loc>{posts}</loc></sitemap>"
                    "</sitemapindex>"
                ),
                products: sitemap(
                    "<urlset>"
                    f"<url><loc>{plate}</loc></url>"
                    f"<url><loc>{candle}</loc></url>"
                    "</urlset>"
                ),
                root: html_page(),
                plate: html_page(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_collection(directory, [resource], session)
            called_urls = [call[0] for call in session.calls]
            self.assertNotIn(posts, called_urls)
            self.assertNotIn(candle, called_urls)
            self.assertEqual(
                {document["url"] for document in self.documents(output_dir)},
                {root, plate},
            )

    def test_youtube_without_adapter_reports_unavailable_without_network(self):
        url = "https://www.youtube.com/watch?v=fixture123"
        resource = make_resource("youtube_fixture", url, "youtube_video")
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(directory, [resource], session)
            coverage = self.load_coverage(coverage_dir, "youtube_fixture")
            self.assertEqual(session.calls, [])
            self.assertEqual(coverage["status"], "blocked")
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"],
                "youtube_adapter_unavailable",
            )
            self.assertEqual(self.documents(output_dir), [])

    def test_youtube_adapter_saves_public_metadata_and_creator_caption(self):
        url = "https://www.youtube.com/watch?v=fixture123"
        resource = make_resource("youtube_fixture", url, "youtube_video")
        adapter = FakeYouTubeAdapter(
            [
                {
                    "url": url,
                    "title": "Single-column tie",
                    "description": "A public lesson description.",
                    "creator": "Fixture educator",
                    "transcript": "Creator-provided caption text.",
                    "caption_kind": "creator-provided:en",
                }
            ]
        )
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, coverage_dir = self.run_collection(
                directory, [resource], session, youtube_adapter=adapter
            )
            coverage = self.load_coverage(coverage_dir, "youtube_fixture")
            document = self.documents(output_dir)[0]
            self.assertEqual(session.calls, [])
            self.assertEqual(adapter.calls, ["youtube_fixture"])
            self.assertEqual(coverage["counts"]["fetched"], 1)
            self.assertIn("Creator-provided caption text.", document["text"])
            self.assertEqual(document["source_lineage"]["authenticated"], False)
            self.assertEqual(
                document["source_lineage"]["caption_kind"], "creator-provided:en"
            )

    def test_youtube_access_block_is_not_retried_or_reported_as_failure(self):
        url = "https://www.youtube.com/playlist?list=fixture"
        resource = make_resource("youtube_playlist", url, "youtube_playlist")
        adapter = FakeYouTubeAdapter(
            error=corpus.YouTubeAccessBlocked("members-only playlist")
        )
        with tempfile.TemporaryDirectory() as directory:
            _, _, coverage_dir = self.run_collection(
                directory, [resource], FakeSession(), youtube_adapter=adapter
            )
            coverage = self.load_coverage(coverage_dir, "youtube_playlist")
            self.assertEqual(coverage["counts"]["blocked"], 1)
            self.assertEqual(coverage["counts"]["failed"], 0)
            self.assertEqual(
                coverage["results"]["blocked"][0]["reason"],
                "youtube_access_blocked",
            )
            self.assertEqual(adapter.calls, ["youtube_playlist"])

    def test_unknown_selected_resource_is_rejected_without_network(self):
        resource = make_resource("known", "https://known.example/", "exact_public_page")
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown resource ids"):
                self.run_collection(
                    directory,
                    [resource],
                    session,
                    resource_ids=["missing"],
                )
        self.assertEqual(session.calls, [])

    def test_invalid_manifest_url_regex_is_rejected_without_network(self):
        resource = make_resource(
            "invalid_pattern",
            "https://invalid-pattern.example/",
            "relevant_public_pages",
        )
        resource["collection"]["include_url_patterns"] = ["["]
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "invalid collection.include_url_patterns"):
                self.run_collection(directory, [resource], session)
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
