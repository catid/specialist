# Statistical-method digest: knot efficiency as a ratio distribution

## Use boundary: this is a population model, not a rope-safety calculator

This is a manual digest of a **2022 mathematical treatment of slow-static rope breaking-strength data**. It is not validation for dynamic, cyclic, wet, aged, damaged, or natural-fiber rope; bondage; body contact; uplines; or human suspension. It does not choose a knot, assign a rating, define a safety factor, or calculate a safe load. [SYM-2022: Abstract; §1 Introduction; §4 Conclusions]

The paper studies how uncertainty propagates when knot efficiency is defined as one random breaking strength divided by another. Its probability curves describe a specified statistical model and source population. They do not turn a laboratory breaking test into an operational limit, and they do not erase uncertainty in the parameters estimated from finite samples. [SYM-2022: §1 Introduction; §2.1–§2.3; §3 Discussion]

Equations, example calculators, numerical efficiencies, portable ranges, figures, tables, knot rankings, safety claims, vendor statements, and standards or third-party text are intentionally excluded. The retained value is the reasoning: what was measured, what was calculated, what assumptions connect them, and which summaries are meaningful for a ratio distribution. [SYM-2022: §1 Introduction; §2 Results; §3 Discussion]

## Article identity and license

The source is Ján Šimon and Branislav Ftorek, “Basic Statistical Properties of the Knot Efficiency,” *Symmetry*, volume 14, article 1926, published September 15, 2022. It records receipt on August 14, 2022 and acceptance on September 10, 2022. The DOI is `10.3390/sym14091926`. [SYM-2022: front matter]

The article front matter states © 2022 by the authors and identifies the work as open access under Creative Commons Attribution 4.0 International. This digest attributes the authors, title, journal, date, DOI, and license, identifies itself as an adaptation, and links both source and license. [SYM-2022: front matter license; CC-BY-4.0: Attribution requirements]

## Measured strengths versus calculated efficiency

The paper labels the breaking strength of a rope specimen containing the chosen knot as **X** and the breaking strength of a straight specimen from the same rope population as **Y**. Both are directly observed destructive-test outcomes. Knot efficiency, **η = X/Y**, is calculated from them; it is not directly measured by the test machine. [SYM-2022: §1 Introduction, Definition 1]

The distinction matters because rope is heterogeneous. Different pieces from a nominally common rope population do not break at one fixed force. X and Y therefore have their own distributions, central values, and dispersions. The efficiency inherits uncertainty from both distributions and must itself be treated as a random variable. [SYM-2022: §1 Introduction]

In the model, X and Y are independent draws because testing destroys each specimen, different pieces are used for knotted and straight tests, and pieces are shuffled before assignment. “Independent” is a design and modeling assumption about draws from the two source populations; it does not mean the rope populations are perfectly characterized or that finite-sample estimates are certain. [SYM-2022: §2.1 General PDF]

## Provenance of the normal model

The 2022 paper reports that earlier work examined roughly 200 straight-rope breaking tests and roughly 80 knotted-rope breaking tests using quantile plots and two named goodness-of-fit tests: Kolmogorov–Smirnov and Shapiro–Wilk. The article says those checks did not show a statistically significant departure from normality. Those measurements are provenance from cited earlier work, not a new 2022 experimental campaign. [SYM-2022: §1 Introduction, citing prior work [2]]

Failure to reject a normal model is not proof that the source populations are normal. A goodness-of-fit test asks whether the available sample supplies enough evidence against the null model at a selected threshold. Its answer depends on sample size, test power, dependence, outliers, and how parameters were estimated. Other distributions may also be compatible with the same observations. The normal model is therefore a supported working approximation, not an established physical law. [SYM-2022: §1 Introduction; §2.2 Normal Breaking Strength]

The authors also motivate normality from maximum-entropy and central-limit reasoning. Those arguments explain why trying a normal model is plausible; they do not replace empirical validation for a new rope, condition, knot, or loading regime. [SYM-2022: §2.2 Normal Breaking Strength]

## Two common calculations that mislead

### One knotted break divided by one straight break

A single knotted specimen and a single straight specimen produce one ratio. That ratio is one draw from a potentially broad distribution, not an estimate of a sharp material constant. Which two heterogeneous rope pieces happen to be selected can dominate the result, and the destructive test supplies no repeat measurement of the same piece. [SYM-2022: §1 Introduction, misleading concept 1]

Calling this a “paired” result does not cure the problem: the two outcomes come from different pieces, not two conditions measured on one unchanged specimen. Without replication, the experiment cannot estimate either source distribution or the spread of their ratio. [SYM-2022: §1 Introduction, misleading concept 1; §2.1 General PDF]

### A ratio of two small-sample means

Using several specimens is better than using one, but simply dividing the average knotted strength by the average straight strength does not generally yield the expected value of X/Y. Even under independence, the expectation of a quotient is not the quotient of expectations. Discarding the two input variances removes information that controls the width and shape of the efficiency distribution. [SYM-2022: §1 Introduction, misleading concept 2]

Small samples add another layer: their means and variances are uncertain estimates of population parameters. The paper derives the population ratio distribution after specifying input distributions and parameters; it does not make finite-sample estimation error disappear. A responsible analysis must distinguish variability among rope specimens from uncertainty about the fitted model parameters. [SYM-2022: §1 Introduction; §2.2 Normal Breaking Strength; §3 Discussion]

## From two input distributions to one ratio distribution

The general derivation starts with continuous probability density functions for X and Y and an independent joint distribution. Every possible pair of strength outcomes maps to a ratio. Accumulating probability over all pairs that produce ratios in a small interval yields the probability density function of η. The same construction works beyond normal inputs when the required density and independence assumptions hold. [SYM-2022: §2.1 General PDF]

A **probability density function (PDF)** describes how probability is distributed across possible efficiency values. Its height at one point is not itself a probability. Probability comes from area over an interval. A narrow high peak means outcomes are concentrated; skew, multiple peaks, or long tails mean a single central number hides structure. [SYM-2022: §2.1 General PDF; §2.3 Properties]

A **cumulative distribution function (CDF)** gives the probability of drawing an efficiency at or below a threshold. Subtracting CDF values gives the probability between two thresholds. For the exact ratio-of-normal model, the PDF has an analytic form, but the CDF and many summaries require numerical evaluation rather than a short elementary expression. [SYM-2022: §2.3 Properties, item 3]

Under the normal-input special case, the ratio distribution is controlled by the relative locations and spreads of the two breaking-strength populations. The paper compresses four input descriptors into three dimensionless shape-and-scale parameters. This parameter reduction is mathematical convenience, not evidence that rope material, knot geometry, condition, and loading history cease to matter. [SYM-2022: §2.2 Normal Breaking Strength; §2.3 Properties, items 1 and 4–5]

## Model support outside the physical region

A normal distribution assigns a tiny mathematical probability to negative breaking strength because its support extends over the entire real line. A ratio also becomes extreme when the denominator is arbitrarily close to zero. The authors restrict their practical parameter sector so those nonphysical input tails are negligible, but they remain part of the exact mathematical model. [SYM-2022: §2.2 Normal Breaking Strength, Assumption 1; §2.3 Properties]

Those tails have an important consequence: the full ratio-of-normal distribution does not have a finite global mean or variance. The paper therefore defines mean and variance after conditioning on the physically meaningful efficiency interval from zero through unity. These are **truncated** or conditional summaries, not the ordinary global moments of the unbounded mathematical ratio distribution. [SYM-2022: §2.3 Properties, item 8]

Truncation solves a reporting problem only after the analyst accepts the physical interval and model. It should be stated explicitly because conditioning changes the population being summarized. A truncated mean or variance must not be confused with proof that observations outside the interval are physically valid or with a safe operating band. [SYM-2022: §2.3 Properties, items 6–8; §3.1–§3.2]

## Central tendency: mode, median, and truncated mean

The **mode** is a density peak: the most locally probable value under the fitted curve. The **median** divides cumulative probability in half. The **truncated mean** averages the ratio after conditioning on the physical interval. In an asymmetric distribution these locations need not coincide, and choosing among them depends on the reporting question. [SYM-2022: §2.3 Properties, items 8–10; §3.1 Discussion]

The ratio of the two input means is not generally the exact ratio distribution’s mean. In the paper’s “solid” approximation, that ratio is the median; the authors describe the exact-model median as nearly equal in the practical region. This limited median result does not rehabilitate the ratio of means as a universal expected efficiency. [SYM-2022: §2.3 Properties, item 9; §2.4 Solid Approximation, Theorem 5]

The exact mathematical PDF can be asymmetric and, for some parameter choices, bimodal. In the parameter region the authors regard as practically plausible, a secondary far-tail mode may carry negligible probability. Modality should therefore be checked from the fitted distribution rather than asserted from knot name alone. [SYM-2022: §2.3 Properties, items 2 and 10]

## Dispersion and the paper’s “tolerance intervals”

Dispersion describes how widely outcomes vary. The two source variances influence the ratio curve, while truncated variance describes spread only within the conditioned physical interval. Reporting a center without spread invites false precision, especially for heterogeneous or worn source material. [SYM-2022: §1 Introduction; §2.3 Properties, item 8; §3.1 Discussion]

The paper defines equal-tail central ranges analogous to familiar one-, two-, or three-standard-deviation bands of a normal curve and calls them tolerance intervals. Their endpoints are quantiles of the modeled efficiency distribution. Because the ratio curve can be skewed, the lower and upper distances from the center need not match. [SYM-2022: §2.3 Properties, item 11]

In this paper’s usage, those ranges describe a chosen share of the modeled population. They are not confidence intervals for estimated parameters, do not include a separate confidence level about population coverage, and are not engineering tolerances, safety factors, ratings, or safe-load intervals. Finite-sample uncertainty in the fitted inputs must be handled separately. [SYM-2022: §2.3 Properties, item 11; §3 Discussion]

## Exact model, solid approximation, and normal approximation

The exact ratio-of-normal PDF is algebraically complex, and its CDF lacks a short elementary form. The paper’s **solid approximation** restricts attention to a stated practical parameter sector, drops terms bounded as negligible there, renormalizes the simplified density, and obtains a closed-form CDF. That makes medians, quantiles, and central ranges easier to evaluate. [SYM-2022: §2.4 Solid Approximation]

The approximation does not make every summary elementary: the physically truncated mean and variance still require numerical integration. The paper derives upper bounds for PDF, CDF, truncated-mean, and truncated-variance errors. Those checks are part of the approximation; using the simplified form without verifying its parameter assumptions and error bounds would discard its justification. [SYM-2022: §2.4 Solid Approximation; §2.5 Error Management]

A symmetric normal approximation is simpler again. The discussion shows why it is case-dependent: one inherited worn-rope example is broad and asymmetric enough that the normal approximation is misleading, while one inherited new-rope example is narrow and symmetric enough for it to track the physical region closely. Neither example supplies a universal efficiency value or a rule based only on rope age. [SYM-2022: §2.5 Error Management; §3.1–§3.2 Discussion]

The two examples reuse parameter estimates associated with earlier static-rope work. They illustrate model behavior; they are not a controlled new-versus-worn aging experiment, and the 2022 paper reports no new dynamic, cyclic, wet, damaged-rope, natural-fiber, body-contact, or human-suspension validation. [SYM-2022: §3 Discussion; Data Availability Statement; reference [2]]

## What can be carried forward

The durable lesson is procedural: define the source rope population, randomize specimen assignment, measure enough straight and knotted specimens to characterize both distributions, retain both dispersions, distinguish population variability from parameter-estimation uncertainty, and propagate the two distributions through a ratio model whose assumptions are checked. [SYM-2022: §1 Introduction; §2.1–§2.3]

The PDF communicates shape, the CDF communicates threshold and interval probabilities, central measures answer different questions, and equal-tail population ranges communicate spread. None of those outputs is automatically a safety limit. The model must remain attached to its rope population, knot configuration, loading regime, specimen history, and estimation method. [SYM-2022: §2.3 Properties; §3 Discussion; §4 Conclusions]

Do not use this digest to calculate retained strength, choose a knot, set a working load, derive a safety factor, validate a body-contact system, or transfer results to another rope class. It contains no operational equations or numerical efficiency outputs. [SYM-2022: corpus scope based on §1–§4]

## Attribution and adaptation notice

Adapted from Ján Šimon and Branislav Ftorek, “Basic Statistical Properties of the Knot Efficiency,” *Symmetry* 14 (2022), article 1926, DOI [10.3390/sym14091926](https://doi.org/10.3390/sym14091926). The article states that it is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This digest changes wording, selection, ordering, and emphasis and omits equations, figures, tables, numerical examples, vendor material, third-party content, and operational claims. No endorsement by the authors, institutions, publisher, funders, vendors, or Creative Commons is implied.

## Citation key

- **SYM-2022** — Šimon and Ftorek, 2022, official MDPI PDF and XML, reviewed in full.
- **CC-BY-4.0** — Creative Commons Attribution 4.0 International deed, accessed July 16, 2026.

No source HTML, XML, PDF, extracted full text, equation set, figure, or table is retained in this corpus.
