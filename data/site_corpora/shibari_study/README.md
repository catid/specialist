# Shibari Study policy-exclusion artifact

This directory is intentionally not an instructional corpus. Shibari Study's live Terms and Conditions prohibit storing or copying Service content, compiling it into a database, and automated or manual collection. Public-route crawl permission in `robots.txt` therefore does not make the material eligible for direct training.

`content_records.jsonl` is empty. The XML inventory exists only to prove that every sitemap URL received an exclusion disposition; it must never be sampled as training text or converted into URL/title questions. No tutorial, lesson, transcript, subtitle, video, image, blog, glossary, FAQ, or substantive page text is stored here.

The ordinary build is offline and deterministic:

```sh
python3 data/site_corpora/shibari_study/build.py
python3 -m unittest discover -s data/site_corpora/shibari_study/tests -v
```

`capture_metadata.py` is a narrow, non-build utility restricted to the published robots and XML sitemap paths. It rejects canonical pages, including the Terms route. If explicit authorization or policy changes later permit a corpus, perform a fresh review and assign each canonical document to a split before producing Markdown chunks or derived QA; all derivatives of one page must remain in that split.
