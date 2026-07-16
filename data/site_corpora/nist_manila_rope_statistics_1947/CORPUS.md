# Historical statistical-methods digest: three-strand manila rope procurement tests

## Use boundary: historical population analysis, not modern rope guidance

This is a manual digest of a **1947 National Bureau of Standards statistical analysis of government-purchase test records for three-strand manila rope**. It preserves provenance, specimen preparation, measurement endpoints, statistical assumptions, anomalies, and the authors’ unusually explicit sampling caveats. It does not retain equations, tables, plots, numerical strength values, tolerance values, procurement limits, or standards rules. [NBS-1947: journal pp. 551–559]

The observations describe heterogeneous rope submitted during 1938–1941, not a random experiment and not a current product population. The source does not validate modern natural-fiber bondage rope, knots, wet or aged behavior, rope care, body contact, uplines, anchors, or human suspension. It supplies no safe load, rating, safety factor, inspection interval, or retirement rule. [NBS-1947: journal pp. 551–555]

The breaking endpoint was a slow, monotonic laboratory pull on conditioned, eye-spliced specimens. It was not cyclic, impact, shock, knot, body-contact, or human-suspension testing. Only the splices were briefly immersed as a test-control measure; that does not make the results a wet-rope study. [NBS-1947: journal p. 552]

## Article identity and corrected DOI

The source is Sanford B. Newman and J. H. Curtiss, “A Statistical Analysis of Some Mechanical Properties of Manila Rope,” Research Paper **RP1847**, *Journal of Research of the National Bureau of Standards*, volume 39, December 1947, journal pages 551–559. The final page records Washington, May 26, 1947. [NBS-1947: journal pp. 551, 559]

The verified DOI is [`10.6028/jres.039.039`](https://doi.org/10.6028/jres.039.039), which resolves to the official RP1847 PDF and whose registry metadata matches the title, Newman and Curtiss, volume, issue, month, and opening page. The superficially similar DOI ending in `.037` resolves to an unrelated uranium paper and is not an identifier for this work. [DOI-AUDIT: verified and rejected resolver bindings]

## Rights status

This is a National Bureau of Standards federal technical-series article authored by Bureau personnel. NIST’s Research Library says federal NIST publications are generally public domain and not subject to copyright in the United States, with source credit requested. NIST’s technical-series policy likewise says employee-authored works are not protected in the United States and grants broad reprint and derivative permission to the extent foreign rights may exist. [NIST-RIGHTS: Library FAQ; Technical Series policy]

The adaptation credits the authors, title, series, year, DOI, and agency and identifies its changes. Source page images, tables, graphs, equations, cited third-party works, and component-level material are excluded despite the federal-text rights basis. [NIST-RIGHTS: Technical Series policy; NBS-1947: journal pp. 551–559]

## What the dataset actually represents

The abstract describes tests on more than 800 samples; the statistical section gives the exact analysis count as **863 rope observations**. The material was submitted for acceptance testing by a government agency from 1938 through 1941 and had been supplied by rope works and contractors in the preceding years, before wartime loss of manila-fiber sources. [NBS-1947: journal pp. 551–552]

This was an administrative stream of purchased material, not a probability sample designed by the authors. Popular nominal sizes appeared more often, some size groups were sparse or absent, and supply came from a composite of works, contractors, production periods, and prior treatment histories. The authors describe the implied national-output universe as far from homogeneous. [NBS-1947: journal pp. 551–552]

The source analyzes **three-strand** manila rope. Some four-strand data existed, but there was not enough for statistical treatment, so the paper expressly leaves that construction out. No inference from the fitted three-strand relationships to four-strand rope is supported. [NBS-1947: journal p. 551]

The paper attributes property variation to differences in fiber quality, method of fabrication, and previous treatment. Those factors were not separately randomized or modeled as recorded predictors; they are explanations for heterogeneity that remains folded into the observed dispersion. [NBS-1947: journal p. 551]

## What was measured

The study considers circumference, weight per unit length, and breaking strength. It examines circumference against nominal diameter, weight against nominal diameter, strength against nominal diameter, and strength against weight. These are different conditional relationships: a mean strength at fixed weight cannot generally be obtained by algebraically combining mean relationships fitted at fixed nominal diameter. [NBS-1947: journal pp. 552, 554]

### Circumference and weight

Circumference and weight were measured on an unspliced specimen after conditioning for a specified time in a controlled temperature and relative-humidity atmosphere. The specimen was placed in a horizontal machine under a prescribed nominal-size-dependent preload. Those settings are part of the historical protocol and are intentionally not reproduced as a field procedure. [NBS-1947: journal pp. 551–552]

At the specimen midpoint, a single fiber was passed snugly around the loaded rope, cut where it overlapped, and measured to obtain circumference. A specified gauge length was marked while the preload remained applied; after unloading, that section was cut out and weighed to calculate weight per unit length. Thus both properties depend on conditioning, preload, and the paper’s measurement conventions. [NBS-1947: journal pp. 551–552]

### Static breaking endpoint

Breaking tests used a separate specimen with an eye splice at each end. The specimen received the same atmospheric conditioning, after which the splices—not the full rope—were immersed briefly to reduce the chance of splice failure. Pins through the eyes transferred load in a horizontal testing machine. [NBS-1947: journal p. 552]

The protocol used one of two machine types according to the anticipated strength range, advanced the moving head at a prescribed slow rate, and increased tension until maximum load was reached and one or more strands failed. The recorded “breaking strength” is therefore the maximum static tensile load under that apparatus and endpoint definition. The paper does not present a cross-machine equivalence study. [NBS-1947: journal p. 552]

## Why 863 observations were not automatically representative

The authors explain what would have been possible with random sampling from a clearly defined homogeneous population: a large sample could closely estimate distributions and support a more elaborate empirical description. They immediately state that those conditions did not hold. They had no direct control over sampling, and the intended universe was composite and heterogeneous. [NBS-1947: journal p. 552]

They proceeded because the observations showed enough internal consistency for a deliberately simple mathematical description. That choice is conditional: the model was not supposed to provide an elaborate causal account. A large row count increased computational precision around the fitted sample pattern, but it did not remove selection bias, population ambiguity, or unrecorded production differences. [NBS-1947: journal pp. 552–555]

This is the central statistical lesson of the paper. **Sampling validity and model validity are separate from sample size.** A narrow formal error bar under an assumed model can coexist with weak knowledge about whether the fitted curve describes the target technological population. [NBS-1947: journal pp. 554–555]

## Logarithmic regression and its assumptions

The raw dispersion increased with the property mean, so ordinary unweighted regression on the original scale would not have constant variance. The authors transformed all variables to base-ten logarithms and fit simple straight-line relationships in log space by unweighted least squares. They report that preliminary exploration suggested the transformation approximately stabilized variance for circumference, weight, and strength. [NBS-1947: journal pp. 553–554]

The fitted line assumes that mean log response is linear in log predictor and that log-scale standard deviation is constant across predictor values. Converting back to an arithmetic mean requires a correction because exponentiating the mean logarithm produces a geometric mean, not the arithmetic mean. Their correction further assumes a normal distribution on the log scale. [NBS-1947: journal p. 553]

The authors favored a small-parameter power relationship that was easy to combine with the logarithmic transformation and had a physical scaling interpretation. A more flexible comparison gave a similar weight-versus-diameter fit, but that limited check did not prove the simple form exact for every relationship or subgroup. [NBS-1947: journal p. 554]

Calculations were performed with punched-card machinery: logarithms and their squares were placed on data cards, and products and cumulative sums were produced with the Bureau’s equipment. This historical implementation detail explains the computational workflow but adds no validation to the assumptions. [NBS-1947: journal p. 553]

## Means, individual dispersion, and the paper’s tolerance limits

The fitted central curves estimate a mean response for a fixed, predetermined predictor value. The paper separately estimates dispersion of individual observations about those means. Its outer “tolerance” curves concern modeled individual measurements, not uncertainty intervals for the sample means and not operational tolerances for rope use. [NBS-1947: journal pp. 554–556]

Those ranges depend on the fitted curve being the true log-scale mean, the estimated spread being the true log-scale standard deviation, and individual log responses being normally distributed. Rounding conventions and model assumptions are part of their construction. They are historical statistical summaries, not ratings, safe-load intervals, procurement rules, or modern acceptance limits. [NBS-1947: journal p. 555]

The formal standard errors of curve coefficients were small, partly because many observations entered the fit. Yet the authors say their usual random-sampling interpretation was unwarranted. They warn that unknown sampling and possible misspecification make the real accuracy of the curves as a description of the era’s national rope situation largely a matter of faith. [NBS-1947: journal pp. 554–555]

## The anomalies are evidence, not cleanup noise

The paper explicitly records three irregularities. First, several nominal-size groups not recognized in the cited federal specification had circumference, weight, and strength distributions resembling particular larger nominal groups more closely than their labels implied. The authors inferred apparent substitution or misclassification in purchasing. This is a historical data-quality finding, not a procurement rule for today. [NBS-1947: journal p. 558]

Second, observations at fixed nominal diameter tended to cluster into tighter subgroups than the overall model spread. The paper attributes this at least partly to nonrandom sampling. Batch, supplier, contract, or production structure could therefore be hidden inside a nominal-size category rather than represented as independent draws. [NBS-1947: journal p. 558]

Third, one very small nominal-size subgroup had circumference and weight observations poorly represented by the analytic fit. The authors judged global extrapolation based on the much larger dataset potentially more reliable than that anomalous subgroup, while still publishing the subgroup summary. That judgment does not validate extrapolation beyond this historical universe. [NBS-1947: journal p. 558]

Misclassification and clustering increase the pooled residual spread relative to what a cleaner within-group dataset might show. Treating every row as an exchangeable independent observation can therefore make a single dispersion estimate combine manufacturing variation, group mixture, labeling error, and measurement error. [NBS-1947: journal p. 558]

## Limits on historical and modern transfer

The data predate modern manufacturing controls and arose from government acceptance submissions during a disrupted supply era. Conditioning standardized one laboratory exposure before measurement; it did not erase previous treatment or establish aging, moisture cycling, damage progression, or outdoor durability. [NBS-1947: journal pp. 551–552]

The paper compares one strength curve with an earlier study, but those source data and plotted numerical results are excluded here. A difference between two historical procurement samples cannot by itself establish a time trend, material improvement, or current manila-rope property. [NBS-1947: journal pp. 558–559]

The durable lessons are methodological: define the population, preserve construction and procurement provenance, record conditioning and apparatus, distinguish a static endpoint from other load modes, inspect group counts, look for labeling and clustering anomalies, stabilize variance only with checked assumptions, and keep individual dispersion separate from uncertainty about a mean curve. [NBS-1947: journal pp. 551–558]

Do not derive a strength value, load limit, rating, tolerance, purchase rule, knot claim, care practice, inspection interval, or human-load decision from this digest. It deliberately contains no operational equation or numerical property output. [NBS-1947: corpus scope based on complete journal pp. 551–559 review]

## Public-domain adaptation and attribution

Adapted from Sanford B. Newman and J. H. Curtiss, “A Statistical Analysis of Some Mechanical Properties of Manila Rope,” Research Paper RP1847, *Journal of Research of the National Bureau of Standards* 39 (December 1947), 551–559, DOI [10.6028/jres.039.039](https://doi.org/10.6028/jres.039.039). Republished courtesy of the National Institute of Standards and Technology, U.S. Department of Commerce. Not copyrightable in the United States. This digest changes wording, selection, organization, and emphasis and omits equations, tables, figures, numerical property results, procurement rules, third-party material, and operational claims. No endorsement by the authors, NBS, NIST, the Department of Commerce, suppliers, or cited organizations is implied.

## Citation key

- **NBS-1947** — official nine-page RP1847 PDF; citations use printed journal pages 551–559.
- **DOI-AUDIT** — DOI resolver and registry metadata review distinguishing the verified `.039` identifier from unrelated `.037`.
- **NIST-RIGHTS** — official NIST Research Library FAQ and Technical Series copyright/licensing policy.

No source PDF, scan, OCR text, equation, table, graph, or page image is retained in this corpus.
