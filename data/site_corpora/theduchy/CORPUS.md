# The Duchy corpus access notice

**Status:** excluded from direct training. **Content records:** 0. **Direct-training ready:** false. **Non-QA:** true.

The metadata snapshot completed at `2026-07-16T06:38:55.515194Z`. The retrieved [robots policy](https://www.theduchy.com/robots.txt) declares `Content-Signal: search=yes,ai-train=no,use=reference` and separately disallows GPTBot site-wide. That policy conflicts with producing a direct-training corpus, so no canonical page, tutorial, post, media asset, gated route, title, or body text was requested, copied, summarized, paraphrased, or inferred.

The [sitemap index](https://www.theduchy.com/sitemap_index.xml) and its listed XML sitemaps were read only to create a disposition inventory. Every discovered canonical URL is marked `robots_and_content_signal_ai_training_disallowed` in `inventory.jsonl`; the inventory is compliance metadata, not training content. The exact robots response and XML snapshots, retrieval times, and hashes are retained in `source_snapshot/`.

Media and access gates were not assessed because doing so would require canonical-page retrieval. No UI text, image-only procedure, video instruction, purchase material, member material, or category mapping is represented.
