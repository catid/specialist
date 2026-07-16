# Shibari Study corpus access notice

**Status:** excluded from direct training. **Content records:** 0. **Direct-training ready:** false. **Non-QA:** true.

The access audit found that the live [Terms and Conditions](https://shibaristudy.com/pages/terms-conditions) grant personal, noncommercial access and prohibit storing or copying Service content, compiling it into a database, and automated or manual content collection. Although the [robots policy](https://shibaristudy.com/robots.txt) permits crawling selected public routes, crawl permission does not override those use restrictions. The resulting disposition is `terms_of_service_prohibits_content_collection_and_storage`.

Only robots and XML sitemap metadata were retained. No tutorial, blog, FAQ, glossary, description, transcript, subtitle, image, video, lesson, member material, procedure, or factual paraphrase was captured for training. `content_records.jsonl` is intentionally empty, and the sitemap inventory is compliance metadata only.

No taxonomy mapping or instruction inference was attempted. A future corpus requires explicit authorization or a materially changed policy, followed by a new review and a source-document split assigned before any Markdown chunk or derived QA is created.
