# Rope-topia manual dataset review

Reviewed 2026-07-14. The requested `http://rope-topia.com` redirects to
`https://rope-topia.com/`. Its public `robots.txt` contains no `Disallow` rule
and points to a sitemap with 218 URLs.

## Outcome

The current host is a limited recovered-site demo. `/blog/` and every relevant
article/tutorial URL below return HTTP 200 but contain the explicit notice
“The DEMO version only includes 4 pages” instead of an article body. The public
WordPress API and feed endpoints return 404. No gate was bypassed, no archived
or third-party copy was silently substituted, and no fact was inferred from a
title.

The four accessible content pages and their equivalent `.html` aliases are:

| Canonical page | Accessible alias | HTTP | Review use |
| --- | --- | ---: | --- |
| https://rope-topia.com/ | https://rope-topia.com/index.html | 200 | Homepage and title-link index |
| https://rope-topia.com/about/ | https://rope-topia.com/about.html | 200 | Biography; no instructional fact admitted |
| https://rope-topia.com/links/ | https://rope-topia.com/links.html | 200 | Links page; no educational content admitted |
| https://rope-topia.com/portfolio/ | https://rope-topia.com/portfolio.html | 200 | Verified one article title |

`https://rope-topia.com/robots.txt` and
`https://rope-topia.com/sitemap.xml` are also accessible with HTTP 200. The
blog index, the 15 indexed educational resource URLs, and the eight exact
excluded post URLs comprise 24 relevant gated endpoints. The checked
`/wp-json/`, `/wp-json/wp/v2/posts?per_page=100`, `/feed/`, `/blog/feed/`, and
the suffix-free `.../kinbaku-today-rope-is-not-about-rope/` endpoints return
404.

The resulting dataset contains 15 manually written resource-index Q&A records
and zero article-content records. Each answer preserves a canonical URL; each
record says `article_content_available: false` and
`content_use: resource_metadata_only_due_demo_gate`.

## Educational resources indexed

| Listed title | Kind | Canonical URL |
| --- | --- | --- |
| Wicked Fast Bowline (WykD’s fast bowline) | Tutorial | https://rope-topia.com/portfolio-items/wicked-fast-bowline/ |
| Wet treating rope | Tutorial | https://rope-topia.com/portfolio-items/wet-treating-rope/ |
| Joining Rope | Tutorial | https://rope-topia.com/portfolio-items/joining-rope/ |
| Strugglers Knot | Tutorial | https://rope-topia.com/portfolio-items/strugglers-knot/ |
| Yin Yoga for Bondage | Blog post | https://rope-topia.com/2012/09/yin-yoga-for-bondage/ |
| Ichinawa, Ippon me no nawa and One rope | Blog post | https://rope-topia.com/2012/10/ichinawa-ippon-me-no-nawa-and-one-rope/ |
| Safety cutters | Safety page | https://rope-topia.com/safety-cutters/ |
| Luck, self awareness, responsibility & rope bondage injuries | Blog post | https://rope-topia.com/2013/11/luck-self-awareness-responsibility-rope-bondage-injuries/ |
| Identifying predatory behaviour | Community-safety page | https://rope-topia.com/newcomers-information/identifying-predatory-behaviour/ |
| So you’re new to the kink scene | Newcomer page | https://rope-topia.com/newcomers-information/so-youre-new-to-the-kink-scene/ |
| Newcomers information | Newcomer hub | https://rope-topia.com/newcomers-information/ |
| Nerve and Circulation Problems in Shibari | Safety page | https://rope-topia.com/nerve-and-circulation-problems/ |
| Rope Bottom Guide | Bottom-education page | https://rope-topia.com/rope-bottom-guide/ |
| Getting out in the kink community | Newcomer page | https://rope-topia.com/newcomers-information/out-into-the-kink-community/ |
| Kinbaku Today (Rope is not about Rope) | Article | https://rope-topia.com/portfolio-items/kinbaku-today-rope-is-not-about-rope/portfolioCats-102-70-123-72-57/ |

The sitemap supplies the last item with a WordPress portfolio-filter suffix:
`portfolioCats-102-70-123-72-57/`. That exact URL is the canonical answer and
returns HTTP 200 with the demo notice. The superficially cleaner suffix-free
path was checked separately and returns HTTP 404, so it is not represented as
canonical.

## Exact excluded post URLs

These sitemap-listed posts were individually checked, returned the same demo
notice, and were excluded rather than guessed from their titles:

- https://rope-topia.com/2018/05/a-week-of-bondage-at-beachbind-2018/ — event/travel recap.
- https://rope-topia.com/2017/08/wykd-tk-4-week-course/ — time-bound course promotion.
- https://rope-topia.com/2017/08/london-shibari-intensive-at-anatomie-studio-in-2017/ — past workshop promotion.
- https://rope-topia.com/2018/04/tokyo-and-kyoto-2018/ — travel post.
- https://rope-topia.com/2017/09/swamp-shibari-show/ — event/show post.
- https://rope-topia.com/2017/08/a-love-letter/ — title does not establish educational relevance.
- https://rope-topia.com/2013/12/taboo-magazine-cover-shibari-bondage/ — media/portfolio post.
- https://rope-topia.com/2018/01/tattooed-art/ — art/promotion post.

Archive, author, category, tag, service, gallery, shoot, and portfolio-index URLs
were navigation, promotional, image-centric, duplicative, or outside the
educational scope. They remain discoverable in the site's 218-URL sitemap and
were not converted into training records.

## Facts and exclusions

The only facts taught by this tranche are the 15 listed-title-to-canonical-URL
mappings. There are no instructions about tying, anatomy, safety, medical
issues, rope treatment, or community conduct because their source bodies were
not accessible.

The homepage’s definition of *shibari* was readable but excluded: the same
literal-translation/definition fact already appears in `data/eval_qa.jsonl`, so
including it would create semantic evaluation leakage. Personal philosophy,
biography, promotional claims, dates, events, and the external Rope Bottom
Guide site were also excluded.

## Validation

- 15/15 reviewed resources represented by canonical and supplied URL.
- 15 unique questions and fact IDs.
- 505 evaluation facts checked; 0 leakage collisions.
- 2,213 current non-Rope-topia curated-baseline rows checked; 0 duplicate
  questions or pairs.
- 0 unsupported safety/medical claims.
- 0 article-body reproductions; evidence is limited to short titles and one
  sitemap URL entry per record.
- 9 dedicated tests pass, including deterministic output/report rebuilds, gate
  fail-closed, provenance checks, exact URL status, evaluation leakage, and duplicate
  rejection.

Rebuild and test with:

```bash
python3 build_rope_topia_manual.py
python3 -m unittest -v test_build_rope_topia_manual.py
```

Artifacts:

- `sources/rope_topia_manual_v1.json`
- `data/rope_topia_manual_v1.jsonl`
- `data/rope_topia_manual_v1.report.json`
- `build_rope_topia_manual.py`
- `test_build_rope_topia_manual.py`
