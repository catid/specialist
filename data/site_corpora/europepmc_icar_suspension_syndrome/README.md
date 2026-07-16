# Europe PMC ICAR suspension-syndrome corpus

This directory contains one non-Q&A, direct-training evidence corpus built
from the CC BY 4.0 JATS article identified by PMCID `PMC10710713`. The only
network capture route is:

`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10710713/fullTextXML`

The Springer publisher endpoint and the United States PMC endpoint are outside
the capture boundary. `capture_source.py` accepts only the exact EMBL-EBI URL
and refuses redirects. The accepted XML and provenance live under
`source_snapshot/`; ordinary builds are offline:

```sh
python3 data/site_corpora/europepmc_icar_suspension_syndrome/build.py
python3 -m unittest data/site_corpora/europepmc_icar_suspension_syndrome/test_europepmc_icar_suspension_syndrome.py
```

The Markdown preserves the scoping-review methods, evidence quality, mechanism
uncertainty, internal search-count discrepancy, proposed classification,
graded ICAR recommendations, limitations, and correction of unsupported
blood-pooling and seated-recovery claims. It labels the evidence as harness,
occupational rope-access, climbing, and mountain-rescue evidence—not a
validated bondage-suspension protocol. The queued PMID typo is confined to
capture provenance and the report; direct-training text and manifest retain
only JATS PMID `38071341`.

Before promotion, assign the PMCID and full-document hash to one split and keep
all Markdown chunks and any later Q&A in that split. Protected validation, OOD,
shadow, and sealed-holdout documents remain excluded from training.
