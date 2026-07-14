# S6 V11b launch-path failure

The first real V11b invocation at `2026-07-14T17:50Z` failed before engine
creation, GPU allocation, journal creation, perturbation, scoring, or model
update. It did not create the fresh V11b run directory and did not consult the
heldout set.

The frozen launch source was commit
`3cb02bc269dc4d36d64bc6ec7a63e6b8059afa12`. Its V11b implementation hashes
were:

- worker `64a0af9c977d8e09282560e8f8e2979a50034d6d78e081387ee1383bee97baa7`
- trainer `2db34d796f7a39c85187964bdbd333d153212af47b257c9dfd0dbe92965c6254`
- driver `ba3e643b57076834b90ece101210465486e655807f400a11f4939423fcf489f2`
- reporter `9b1fb809a2b4cb20541808c8df841b99496b4486778c92c7d88f275c931007c2`
- tests `77b5600a002a64bb47e5538b02a0791376cda43ec9c4c69dd269cf5e9761b6c8`
- protocol `6af1331aee3f4b648a71385a544ec198ae87954c6363009eb87b1b008d4a6a44`

The real (non-dry) driver reached the inherited anchor-data setup and raised:

```text
AttributeError: module 'train_eggroll_es_specialist_anchor_v11b' has no attribute 'load_anchor_prose'
```

The V11b dry-run and focused tests did not exercise that substituted-module
surface, so this is also a test-coverage failure. V11b remains immutable.
Any retry must use new V11c-only sources, a fresh experiment name, and a
launch-shaped regression that covers every anchor-module symbol used by the
inherited non-dry setup before engine creation.
