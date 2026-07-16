# Europe PMC rope-neuropathy study corpus

This directory contains one non-Q&A, direct-training Markdown evidence corpus
built from the CC BY 3.0 JATS article identified by PMCID `PMC10294117`. The
only network capture route is the authorized EMBL-EBI Europe PMC endpoint:

`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML`

The Cureus publisher copy and the United States PMC copy are outside the
capture boundary. `capture_source.py` hard-rejects every URL except the exact
EMBL-EBI route and refuses redirects. Its retained XML and provenance are
pinned under `source_snapshot/`. The ordinary corpus build is offline:

```sh
python3 data/site_corpora/europepmc_rope_neuropathy_study/build.py
python3 -m unittest data/site_corpora/europepmc_rope_neuropathy_study/test_europepmc_rope_neuropathy_study.py
```

The Markdown attributes all observations to Khodulev et al. and treats the
article as a retrospective, injury-enriched survey/case series plus a detailed
recurrent case report. It preserves denominators, diagnostic coverage,
recovery ranges, limitations, ethics, and license. The repository-only
identifier discrepancy is confined to capture provenance and the audit report,
not the direct-training Markdown or manifest. The corpus does not provide
tying instructions, population prevalence, causal rules, or safe-duration claims.

Before any training promotion, assign the PMCID and full-document hash to one
document split. All Markdown chunks and any later derived Q&A must remain in
that split. Protected validation, OOD, shadow, and sealed-holdout documents
remain excluded from training.
