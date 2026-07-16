# Historical methods digest: internal abrasion and tensile creep-fatigue in synthetic double-braided marine rope

## Use boundary: a 1990 mechanism model, not a rope-life calculator

This is a manual digest of a **1990 MIT Sea Grant technical report about idealized synthetic-fiber double-braided rope under marine cyclic-tension conditions**. It preserves the report’s physical hypotheses, model structure, evidence, and limitations. It does not preserve equations, parameter tables, plotted values, load ratios, cycle counts, predicted service lives, or design recipes. [NOAA-1990: PDF pp. 2, 4–5, 21–22]

The source does not validate natural-fiber rope, knots, rope care or retirement criteria, bondage, body contact, uplines, anchors, or human suspension. It supplies no contemporary rating, safe-load rule, inspection rule, or operational recommendation for those domains. Its nylon and polyester results are bound to the constructions, material inputs, wet-test comparisons, marine context, and idealizations described by the authors. [NOAA-1990: PDF pp. 6–22; complete 41-page review]

The report uses the language of prediction and service life, but its outputs depend on fitted fiber and yarn behavior plus many structural assumptions. A predicted stress-versus-life curve is a model result, not a portable property of “rope,” and the source itself warns against extending severe-test behavior linearly into a different loading region. [NOAA-1990: PDF pp. 6, 14–22]

Figures, tables, formulas, named commercial materials, third-party recommendations, and numerical results are excluded pending separate component and source review. The retained material is conceptual: how the authors connected rope geometry, internal contact, filament wear, time-dependent tensile damage, and observed failure location. [NOAA-1990: PDF pp. 7–41]

## Identity, funding, and public-domain record

The NOAA Institutional Repository indexes the work as Moon Hwo Seo, J. F. (John Forbes) Mandell, and Stanley Backer, *Modeling Of Synthetic Fiber Ropes Deterioration Caused By Internal Abrasion And Tensile Fatigue*, published in 1990 as MIT Sea Grant College Program report MITSG 90-18. The scanned title page spells “Modelling” with two l’s and orders the authors Moon Hwo Seo, Stanley Backer, and John F. Mandell. [NOAA-IR: record metadata; NOAA-1990: PDF p. 2]

The title page and repository bind the report to grant `NA86AA-D-SG089` and project `RT-11`. The repository identifies Massachusetts Institute of Technology, Sea Grant College Program, and the National Sea Grant College Program as corporate authors and labels the main document **Public Domain**. Its published SHA-256 checksum exactly matches the reviewed PDF. [NOAA-IR: identity, funding, rights, and checksum fields; NOAA-1990: PDF p. 2]

## The historical problem the model addresses

The report begins from a mismatch between available fatigue testing and marine use. Long-duration low-tension tests were difficult to run, so much of the literature emphasized more severe cyclic loading. The authors argue that simply extending those severe-test trends into lower-load mooring or towing conditions lacks justification. Their alternative was a mechanism-based model intended to represent different dominant damage processes in different regimes. [NOAA-1990: PDF p. 6]

The report is partly a synthesis. It draws fiber and yarn behavior, structural mechanics, deployment observations, and rope fatigue results from cited earlier studies, then compares model predictions against selected available rope data. It is not a single newly designed experiment covering every input, construction, and validation case. [NOAA-1990: PDF pp. 6–24]

## One rope, two competing deterioration clocks

The central model combines two damage processes. **Tensile creep-fatigue** acts through sustained and repeated axial stress in the constituent fibers, and its experimental input is expressed in terms of accumulated tensile exposure time. **Internal abrasion** acts where rope components press and slide against one another, and its experimental input is expressed in friction cycles. The two clocks therefore respond differently to cycling frequency. [NOAA-1990: PDF pp. 8–11]

In the authors’ predicted stress–life curves, the high-applied-tension region is dominated by tensile creep-fatigue, while a lower-tension region can be dominated by internal wear. The location and extent of those regions depend on fiber wear behavior, rope structure, and cycling frequency. “High” and “low” are model-relative descriptions, not safe or unsafe operating categories. [NOAA-1990: PDF pp. 15, 21–22]

The model treats these mechanisms as competing failure routes rather than as one smooth empirical exponent. This supplies a physical explanation for a change in curve shape: changing stress and frequency can change which process reaches its modeled failure condition first. It does not establish a universal transition point for all synthetic ropes. [NOAA-1990: PDF pp. 7, 9–11, 21–22]

## How rope geometry creates internal wear

The inherited structural model represents a strand path in a double braid as sinusoidal undulation superposed on a circular helix. Rope extension changes strand geometry; strand stress follows from its constitutive behavior; axial components of strand tension are summed to estimate rope load. The model also estimates lateral pressure and relative movement where strands contact. [NOAA-1990: PDF pp. 7–8, 25]

At a strand crossing, curvature redirects tension and creates lateral contact pressure. Cyclic axial deformation can then produce relative motion between opposing components. The model assigns surface-filament wear to these pressurized contact zones while tensile fatigue proceeds through the strand body. Internal abrasion is therefore generated by the rope’s own moving components, not by rubbing the rope against external hardware. [NOAA-1990: PDF pp. 8–9, 26]

The report also notes that transverse vibration and local bending in marine service could add component motion even without obvious external rubbing. Later, the calculation assumes relative movement on every loading cycle. That is a modeling choice motivated by reported deployment pathology; the necessary and sufficient conditions for motion were explicitly described as uncertain. [NOAA-1990: PDF pp. 8–9, 11–12]

The wear representation removes filaments layer by layer at the strand surface of highest curvature. Lost cross-sectional area reduces the remaining tension-bearing section. The scheme assumes either fixed contact width or a prescribed rearrangement that preserves bundle proportions; it does not simulate arbitrary filament migration, damage morphology, or evolving contact geometry. [NOAA-1990: PDF p. 9]

## Evidence for internal fiber-to-fiber abrasion

As historical evidence, the report summarizes earlier microscopy of cyclically tested and marine-deployed lines. It says those studies observed extensive internal filament abrasion and associated reductions at fiber, yarn, strand, or rope scale, including in deployments without identified external abrasion or photochemical degradation. These observations come from cited prior work and were not independently reproduced in this 1990 report. [NOAA-1990: PDF p. 7]

The report also describes internal wear in double-braided polyester and in plaited nylon and polyester specimens tested elsewhere. Some severe cases were judged creep-dominated despite visible wear. The important distinction is that observing abrasion does not by itself prove abrasion controlled final failure; failure-mode attribution depends on damage localization, residual-strength behavior, and the model comparison. [NOAA-1990: PDF pp. 7, 16–17]

## Tensile creep-fatigue input

The tensile component comes from separate fiber and yarn studies on nylon 6.6 and polyethylene terephthalate, abbreviated PET. Within the tested high-stress ranges summarized by the report, fatigue life followed a relationship with accumulated creep time. The authors interpret this as support for a time-based cumulative-damage rule that is independent of cycle frequency within that input model. [NOAA-1990: PDF pp. 9–10]

“Frequency independent” does not mean frequency disappears from the rope problem. If one mechanism accumulates with elapsed tensile exposure and another with friction cycles, changing the cycle period changes their competition for a given number of cycles. The report accordingly compares predictions and measurements at matched cycling frequency. [NOAA-1990: PDF pp. 10, 15, 21–22]

The authors characterize creep-dominated rope failure as comparatively sudden: small wear-related residual-strength changes may precede failure without serving as a reliable progress measure. In one reused rope example, component observations did not align cleanly, illustrating that residual strength alone may not identify the governing mechanism. [NOAA-1990: PDF p. 17]

## Yarn-on-yarn wear input

The abrasion input comes from friction and wear tests on monofilaments and rope yarns. In the report’s interpretation, yarn wear life is governed by the number of friction cycles and varies with normal contact pressure. A three-region empirical wear curve is used to represent low, intermediate, and high pressure behavior. [NOAA-1990: PDF pp. 10–11]

The low-pressure region was a major evidence gap: the authors had extensive data at intermediate and high pressure but little at very low pressure. Where measurements were unavailable, they imposed a smooth extrapolation satisfying chosen boundary and slope conditions. The high-pressure region also showed substantial scatter and was represented by another fitted form. [NOAA-1990: PDF pp. 10–11]

This matters because the model predicts low-pressure yarn wear to control low-applied-tension mid-span behavior. The region of greatest deployment interest was therefore also the region where the basic wear input was least measured. The later sensitivity plots vary hypothetical wear parameters rather than reporting new yarn observations. [NOAA-1990: PDF pp. 21, 38–41]

## Structural and failure assumptions

The structural calculation brackets inter-strand friction with ideal zero-friction and infinite-friction cases. In the former, strand strain is uniform along the strand; in the latter, local strain follows the constrained geometry. Real ropes need not occupy either ideal endpoint, and friction also controls how local damage redistributes load along helical strands. [NOAA-1990: PDF pp. 8, 12–13]

For deterioration, each rope layer is idealized as identical strands containing uniform rectangular fiber bundles. A later calculation uses a square array of identical filaments. Internal wear is confined to the highest-curvature surface, and worn layers are assumed not to alter local strand curvature. [NOAA-1990: PDF pp. 9, 12]

Tensile fatigue is assumed uniform through the bundle and independent of wear except for wear-induced area loss. A strand fails when the fatigue criterion for its filament population is reached or when its section is consumed by wear. At rope scale, residual strength is assigned to the weakest strand location where internal wear is most severe. [NOAA-1990: PDF pp. 9, 12]

These assumptions make the computation tractable but suppress heterogeneity, stochastic filament strength, evolving curvature, partial interactions between fatigue and wear, nonuniform contact, and complex load sharing. They define the model’s population; they are not general observations about all double braids. [NOAA-1990: PDF pp. 8–13]

## Internal wear versus external wear

The report distinguishes internal component-on-component wear from external abrasion against pulleys, capstans, bollards, deck surfaces, test grips, or termination hardware. It assumes internal and external wear are independent and may be superposed. That independence is another model assumption, not an experimentally universal law. [NOAA-1990: PDF p. 12]

Its external-wear extension contrasts a short localized wear zone with a longer, effectively distributed zone. Which idealization applies depends on wear-zone length, strand helix, inter-strand friction, and load transfer. The report’s laboratory eye-splice analysis is deliberately qualitative because contact width and surface-specific wear behavior were not well known. [NOAA-1990: PDF pp. 13, 18–21]

This external-wear discussion is useful mainly as a validation warning. Much available rope fatigue data ended at eyes or tangency points rather than at mid-span, so those observations cannot be treated automatically as validation of an internal-abrasion model. Failure location and termination damage are part of the experimental outcome. [NOAA-1990: PDF pp. 13, 17–20]

## Hysteretic heating as a test confound

The report states that energy dissipated through hysteresis can generate high internal rope temperatures when cyclic tests are not run wet. It uses that statement, together with termination failures, to prefer limited wet mid-span or end-of-splice data for comparison. The report does not present a thermal model, temperature measurements, or a heating threshold in this study. [NOAA-1990: PDF pp. 13–14]

Hysteretic heating is therefore a supported qualitative confound, not a third calibrated deterioration law in this corpus. It also prevents unqualified comparison of dry and wet cyclic tests: a result shaped by self-heating may not isolate the same creep and abrasion mechanisms as the selected wet comparisons. [NOAA-1990: PDF pp. 13–15]

## What the validation comparisons do and do not show

The authors populated the model with measured fiber and yarn properties, then compared predicted curves with previously reported wet-rope results. They also acknowledge that many references did not describe rope structure in enough detail and therefore assume the compared ropes are geometrically similar to their own test set. [NOAA-1990: PDF pp. 14–15]

For the available wet nylon comparisons, the tested region was described as creep-dominated and close to the creep prediction; wear dominance at lower applied tension was a model prediction below that evidence. For wet PET, only a small set of mid-span failures was available, and the authors reported agreement in both modeled regimes while expressly leaving the very-low-tension region uncertain. [NOAA-1990: PDF p. 15]

The residual-strength comparison is mixed rather than uniformly confirmatory. Rope-scale and strand-scale results differed, core and sheath did not always follow the expected direction, and the authors argued that nonuniform wear or frictional load sharing could explain part of the discrepancy. Their “reasonable” assessment is an author judgment tied to the particular reused data and plotting scale. [NOAA-1990: PDF pp. 15–17]

The external-wear extension reproduced a qualitative trend but missed the observations materially. The report identifies assumed contact width, contact pressure, and transferability of yarn-on-yarn wear behavior to hardware contact as possible causes. It had little purpose-designed external-wear data and lacked contact-width information for the comparison set. [NOAA-1990: PDF pp. 18–21]

The report presents no modern uncertainty analysis for fitted inputs, specimen variability, model discrepancy, or service-life prediction. Its comparisons combine studies with differing constructions and test conditions, while many outcomes are termination failures. Agreement in selected plots should therefore be read as historical plausibility evidence, not broad validation. [NOAA-1990: PDF pp. 13–24]

## Durable lessons and non-transfer

The durable conceptual lesson is to separate damage drivers before fitting one life curve. Ask whether exposure time, friction-cycle count, contact pressure, construction geometry, frequency, wetness, self-heating, failure location, and termination abrasion have been distinguished. A change in dominant mechanism can make simple extrapolation across load regimes misleading. [NOAA-1990: PDF pp. 6–22]

The report also demonstrates why model assumptions must travel with a prediction. Its conclusions are for idealized synthetic double-braided nylon and PET systems assembled from particular historical inputs; they do not establish natural-fiber behavior, knot behavior, inspection criteria, retirement timing, or human-load safety. [NOAA-1990: PDF pp. 8–22; complete 41-page review]

Do not derive a load, lifetime, safety factor, rating, inspection interval, or retirement decision from this digest. It contains no operational formula or numerical output and deliberately omits the source’s parameter values and plots. [NOAA-1990: corpus scope based on PDF pp. 6–41]

## Public-domain adaptation and source notice

Manually adapted from Moon Hwo Seo, Stanley Backer, and John F. Mandell, *Modelling of Synthetic Fiber Ropes Deterioration Caused by Internal Abrasion and Tensile Fatigue*, MIT Sea Grant College Program report MITSG 90-18 (1990), available through the [NOAA Institutional Repository record](https://repository.library.noaa.gov/view/noaa/9887). The repository labels the main document **Public Domain**. This digest changes wording, selection, organization, and emphasis and omits equations, tables, figures, numerical examples, commercial identifiers, third-party recommendations, and operational claims. No endorsement by the authors, institutions, funders, NOAA, or cited organizations is implied.

## Citation key

- **NOAA-1990** — reviewed 41-page scanned report; citations use PDF page numbers.
- **NOAA-IR** — official repository record for identity, funding, rights, download binding, and checksum.

No source PDF, page image, OCR text, formula set, figure, or table is retained in this corpus.
