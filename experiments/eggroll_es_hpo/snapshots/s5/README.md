# Frozen S5 dataset and evaluation snapshot

This directory freezes the post-audit S5 training and evaluation boundary.
The training JSONL contains 784 unique questions and fact IDs (SHA-256
`62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507`).
Its Arrow file has SHA-256
`c4935458da6e887064ed181e9ec8ee490752cca2b6a1d33ecb7aa58c201c851f`.

The active curation ledgers contain 2,548 decisions: 2,500 drops and 48
source-evidenced edits. The final quality-only Kinbaku promotion reviewed 316
rows from 125 source documents and removed 124 low-quality facts, including
five prior edits replaced by explicit drop decisions. A separate set of 107
rows flagged only for overlap with the obsolete S4 development benchmark was
not removed on that basis.

The domain evaluation artifact is unchanged from its complete manual audit:
41 validation questions from 22 documents and 18 sealed holdout questions
from 11 different documents. The 24-item OOD QA and 16-document OOD prose
artifacts are frozen alongside it. Normalized source identities have zero
training/domain/OOD or validation/holdout collisions. The sealed holdout must
not be used for HPO, horizon selection, seed selection, or early stopping.

Key hashes:

| Artifact | SHA-256 |
| --- | --- |
| Dataset manifest | `7d56567a2116ad11814d8bc9b62e6b9341593dc3e3f0854511c151449b76b056` |
| General curation ledger | `ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb` |
| Curated build report | `3ca1cbf68dcc281176b25ca14a894edd6db7a3c705849d97f4efcdc6240b974a` |
| Domain eval JSONL | `ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b` |
| Validation Arrow | `19181b832e38ef6f97e3ba734362cd1af921f067e8edd249113c5129439443db` |
| Sealed holdout Arrow | `df23a704d0f621bffd8b55fb4a0a296e06a79feaf79cfe1bd357d55bb4f07cf1` |
| OOD QA Arrow | `b201123c6a358d306b7f874e400861068900bb764b1fda80eb663b82ca53dced` |
| OOD prose JSONL | `3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57` |

The Arrow files are retained for byte-exact experiment replay. Rebuilding from
the frozen JSONL artifacts should reproduce the same row content, but Arrow
serialization hashes can depend on the installed `datasets`/PyArrow versions.
