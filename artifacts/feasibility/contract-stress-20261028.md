# Opposite-region contract sensitivity control

The same positive recorded evidence was evaluated against a deliberately changed placement requirement. Both saved guided perception replay and fresh Spark checks changed from PASS under the original contract to FAIL under the opposite-region contract.

| Exposed recording / source pair | Original requirement and saved result | Authored changed requirement | Guided replay | Fresh Spark |
| --- | --- | --- | --- | --- |
| Candy episode 000000, 12.5 / 13.5 seconds | Candy in COLLECTION: PASS in both systems | Candy in HANDLING | FAIL | FAIL |
| Coffee episode 000007, 20 / 21 seconds | Glass in DISPENSE: PASS in both systems | Glass in STAGING | FAIL | FAIL |

The four original frames were visually inspected again before freezing the changed contracts. The sole white oval candy is visibly in COLLECTION, outside HANDLING; the sole transparent glass is visibly in DISPENSE, outside STAGING. These are authored requirement changes on already exposed evidence, not newly discovered robot errors or additional independent recordings. The controls test sensitivity to task requirements; they provide no comparative accuracy advantage for guided perception.

## Binding and provenance

The frozen schedule retains the original file digests, original approved contract hashes, per-case source image hashes, changed approvals and the unchanged guided implementation digest. Review provenance is `assistant_under_user_authorization`, not human annotation or an independent trusted review timestamp. Both changed contracts were frozen before either new Spark call.

The original object concepts, region calibration, unique-target scene assumption, source timestamps and image bytes were preserved. Only the declared required region, authored requirements, protocol version and expected label changed. Spark was told to evaluate the declared contract rather than the original demonstration goal. It received no annotation or previous verdict.

For guided evaluation, the unchanged guided adapter replayed the saved Spark locator and actual saved SAM responses against the exact current image files. Locator source order, geometry, mask raster, guided provenance and complete pair equality were checked. The original observation objects and verdicts were **not** reused for the changed task. Raw source-bound SAM measurements were grounded again under the new protocol hash, producing new specification-bound evidence IDs. The shared benchmark validator replayed the changed decisions and checked their totals against actual source-bound sidecars.

The original PASS references are explicitly saved results. Changed guided verdicts are `replayed_saved_live_meta_observations`; changed Spark verdicts are fresh `live_meta_api` results. Hashes establish local consistency, not independent model identity, physical authenticity, or trusted capture/review timing.

## Requests and estimated cost

Exactly **two new Spark requests**, zero retries and zero new SAM requests were used. The incremental estimate was **$0.018421**, below the $0.10 stopping threshold. Estimates use the existing project token/image rates and are not a provider billing guarantee.

| Source pair | Original guided perception | Original Spark reference | New guided replay | New Spark |
| --- | ---: | ---: | ---: | ---: |
| Candy | 3 requests / $0.016245 | 1 request / $0.009097 | 0 requests / $0 | 1 request / see JSON |
| Coffee | 3 requests / $0.011487 | 1 request / $0.009475 | 0 requests / $0 | 1 request / see JSON |

Zero incremental guided requests does not mean perception is free: the original locator and masks had already been paid for. Per-pair costs were reconstructed from saved response usage and checked against the original per-request ledgers; they are not whole-experiment costs attributed to one example.

## Reproduce

```sh
.venv/bin/python -m unittest tests.test_contract_stress -v
.venv/bin/python -m proofloop.contract_stress \
  artifacts/local/contract-stress-20261028-01 \
  --output artifacts/local/contract-stress-replayed.json
```

The replay command needs the original private media and response files. It performs no API calls, reads the actual frozen schedule and saved Spark responses, revalidates both old and new contract bindings, and refuses to overwrite an existing output. A public clone can run the authored unit tests without media or an API key.

Public machine-readable results: [contract-stress-20261028.json](contract-stress-20261028.json). The seven earlier online evaluation reports were not modified.
