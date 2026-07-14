# S6 V14a full-frame document-estimator preregistration

## Bound aggregate evidence

V13b completed its exact alpha-zero train-only diagnostic with no update. Its
attempt and report pass committed-source, report-binding, exact-restoration,
population-boundary, panel, perturbation, and self-hash validation. The compact
evidence contains hashes and aggregate stability summaries only: no response
vectors, row content, validation, OOD, or heldout content.

- V13b attempt file/content SHA-256:
  `00513c2e32d839c895a2cac5d4d00717be88b89bccfc9f841265c8cff9be6b6c` /
  `188c972e97d531d9ace9e3ae3ca17dee943e449795c6731552b8039cf04285c4`.
- V13b report file/content SHA-256:
  `d53832ab9d021aa4692cef038058014ca84501a772ee935195bf8dfeba85e753` /
  `dfa8c73fae35d0b915dcb1f7c5ef2bca91415a551a178cc90ab534a0646939da`.
- V13b compact evidence file/content SHA-256:
  `d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54` /
  `06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3`.

V13b's optimization cosine median/worst was
`0.47411088498906484` / `0.3900621868364503`; its train-screen cosine
median/worst was `0.3936314430866483` / `0.314941371734614`. This is useful
signal but not stable enough for a model update.

V12 independently closed the proposed consensus candidate. Both positive
alphas had negative lower confidence bounds on C45, C46, A43, and A44; neither
was eligible, no candidate seal was written, and confirmation/release did not
open. Heldout and benchmark-before-seal remained false.

- V12 preseal report file/content SHA-256:
  `1cc6c07a251fcbee61150a3a688940bf1a765934db2cdb1822f37b5339477c4e` /
  `001eac316bb74eb6b2949acade7167d9a6efedb1035f2767c5aedf254ecad831`.
- V12 negative compact evidence file/content SHA-256:
  `4fec87ad1f41e40ba2ebc97dd46b58f4f7bf345e78d364a0ff2e98b9969a6512` /
  `ce259d30481d8de85089116a215ac710db791e27ffd1592f846c5c9a1e56bb59`.

The V12 consensus coefficient and its alpha grid must not be reused. The next
experiment isolates a fresh direction estimator at alpha zero.

## Why full-frame is the next estimator

V13b generated 280 train prompts per direction/sign across five panels, but
only the 168 optimization-panel documents entered its aggregate. V14a instead
generates one frozen row draw from every one of the 310 documents. This costs
only `10.71428571428572%` more prompts than V13b while eliminating document
selection variance from the primary estimator.

The same 310 scored documents also produce five exact, globally disjoint
56-document subpanel estimates matching V14's document allocations. No extra
generation is required. For each of the two screen allocations, its
56-document coefficient is compared with a separately standardized coefficient
from the disjoint 254-document complement. Thus the screen gate is cross-fit;
the screen never contributes to the coefficient it tests.

The full-frame row draw is fixed to V14 iteration zero. Alternate within-page
row selection is deliberately not mixed into this experiment. If V14a passes,
a separate alpha-zero confirmation repeats the full frame at row-draw iteration
one. The frozen snapshot has 171 multi-row and 139 single-row documents, so
that confirmation directly measures the remaining row-choice variance.

## Frozen sampling identities

V14a binds sampler SHA-256
`6981a746d6e0fc0904603abaf584ab71b9cc8a777a9abc00f4d305a98ebd186a`,
the exact 794-row/Arrow source, frame
`ce50531881f4b7044bf82fc3e8fd52d603ba53041fccc4b934aa307840862d6c`,
and policy
`b4c8d038da0c670f2cc8602b822e249de427b124e80fd2d65d69b6c48d980ebc`.
Hard replay is zero.

The 310-document full-frame content / ordered-row identities are:

- `f91b3388226e2d6cfec60e4f62c2cc5e2b28161b8fd5071d89a5700540a587b2`;
- `2f2de46c2e4c35ba03aedba93a7ce58426aa17e5028263f1035d7199e650c798`.

Matched 56-document subpanel content / ordered-row identities are:

- optimization 0:
  `5e0c16a68a2c64a5b1ee9ce95786bb2a1bcd229d4bc5ed0878c723337256898b` /
  `97444de398399cf4f5875258b9c853629bb9d3afdaec858c664f3dab611ad1d1`;
- optimization 1:
  `8109e6a9c5d1de68e41a260bcb863316a415ef5623dff93e1d73502586991b94` /
  `c7c700cca6202a35381df596b994f018da0dfab0cfb45832ede2f1486a8cbad8`;
- optimization 2:
  `8ef3b4f774dc64720fa6dc05104b86c1bea78e4b6a30c991365e16baf26ec685` /
  `0d67a589670532dcd9ee5b447036e38da222101ce196b381ff5c44b8c23a5403`;
- train screen 0:
  `7c7e01ec15bf4e438341bed25763c5f00e4f7eef58d1550da944a1c5952e1a4b` /
  `9855bd37e99eea106a3c53d72f5e8d2153d3f17c7bb1f973006e0d0e0d75f3c9`;
- train screen 1:
  `47cd836f4a0acf995ff07ba5aa67e1f2bb84ec8f97051c39db399518532e3688` /
  `0b3c066f684186fd9ce7025a114088c973009034fb5ff78f6c3766baffad81f6`.

The complete machine-readable preregistration has file/content SHA-256
`d27052ee26d9ba5dd4383491b3d093d0a2f9469ddb4a073909a2b6590e0cba3e` /
`e610c4bd83449b6b9cb3a0055f8e099ebae32ff6827aa64c6521d74705bda59d`.

## Frozen runtime and numeric gate

The future runtime must use Qwen3.6-35B-A3B, the exact middle-late plan, the
same 32-direction basis
`29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11`,
plus then minus, alpha zero, exact restoration after every sign, and four TP=1
engines on GPU IDs 0,1,2,3. Every direction/sign sees the identical 310-row
order. All update RPCs remain unreachable.

For each response vector, central response is `(plus-minus)/2` and
standardization epsilon is `1e-8`. All rules below are conjunctive:

1. matched-56 optimization pairwise cosine median/worst strictly exceed
   `0.47411088498906484` / `0.3900621868364503`;
2. matched-56 pairwise sign median/worst are at least `0.59375` / `0.5625`;
3. full-frame to matched-56 optimization cosine median/worst are at least
   `0.7608236805612648` / `0.7082628389768383`;
4. corresponding sign median/worst are at least `0.8125` / `0.75`;
5. disjoint complement-to-screen cosine median/worst strictly exceed
   `0.3936314430866483` / `0.314941371734614`;
6. corresponding sign median/worst are at least `0.65625` / `0.53125`;
7. all response spreads are nonzero, the full-frame coefficient has 32 finite
   nonzero coordinates, and every provenance/restoration audit passes.

Passage authorizes only the separate row-draw-iteration-one alpha-zero
confirmation. Failure retains V13. Neither result authorizes an update or an
evaluation surface.

## Holdout and architecture firewall

No GPU launch is authorized yet because the full-frame runtime adapter does
not exist. Before launch, add, review, commit, and hash exactly:

- `train_eggroll_es_specialist_anchor_v14a.py`;
- `run_eggroll_es_hierarchical_train_panels_v14a.py`;
- `test_eggroll_es_hierarchical_train_panels_v14a.py`.

The driver must bind both compact evidence files and the machine-readable
preregistration, reject validation/OOD/heldout/benchmark tokens before parsing,
use fresh exclusive attempt/run paths, and require a committed implementation
bundle. Until then there is deliberately no valid GPU command.

V11f is immutable failed evidence superseded by completed V11g. The V12
candidate is closed. The sealed holdout remains closed.

After estimator stability and alternate-row confirmation, architecture HPO is
a separate alpha-zero same-basis experiment. V7's front-only plan failed with
slot cosine `0.0687`; back-only and combined front+back remain untested under
the improved estimator. The preregistered order is back layers 36--39,
front 0--3 plus back 36--39, then middle-late control. No layer insertion or
nonzero update may occur before that comparison, and it must not be mixed into
V14a estimator isolation.
