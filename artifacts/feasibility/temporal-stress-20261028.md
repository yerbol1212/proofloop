# Temporal-stability stress evaluation — 2026-10-28

On eight deliberately constructed development cases, **guided SAM plus the checker matched 7/8 reviewed labels; Spark alone matched 8/8**. Both correctly abstained on the four inconsistent frame pairs and accepted both stable correct placements. Guidance did not outperform Spark alone, and one additional guided abstention prevented a supported FAIL decision.

The [machine-readable summary](online-temporal-stress-20261028.json) preserves the actual live Meta API results. Observation version: `normalized_boxes_v2`. Evaluation role: `development_reprocessing`.

## Inputs and label meaning

This suite reuses the four already exposed ALOHA coffee episodes **000000–000003**, under the same [MIT-licensed dataset source](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/1c52c4c794b6f93d931489263cf15e4331600bfc/README.md). Credit Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn, and Hugging Face / LeRobot for the dataset conversion. These are normal recorded robot demonstrations, not robots following ProofLoop instructions.

All **16 distinct frame hashes**, selections and annotations were frozen before this stress run. Every selected image was visually reviewed. No image was hand edited, composited or generated; frames were decoded from the original source videos. The eight cases comprise:

| Group | Source episodes | Source timestamps per pair | Reviewed result |
| --- | --- | --- | --- |
| Four inconsistent windows | 000000–000003 | 0.500 / 20.000 seconds | UNKNOWN |
| Two stable staging controls | 000000 / 000001 | 2.500 / 3.500 seconds | FAIL |
| Two stable dispense controls | 000002 / 000003 | 19.500 / 21.500 seconds | PASS |

In each inconsistent window, the first image positively shows the sole glass in staging, while the later image shows it in dispense. The reviewed predicate requires **two consistent supporting observations**. One image supporting each of two locations does not satisfy that rule, so the expected result is UNKNOWN. This does **not** mean the latest physical arrangement is visually unknowable: its latest image supports dispense, but a single supporting sample cannot certify the required stable pair.

The region calibration, original semantic concept `transparent drinking glass`, `min_stable_samples=2`, `max_age_ms=30000`, and 0.8 confidence threshold remained unchanged. Frame selection never used this run's outputs. Some constituent images had already been model-tested during earlier development; newly extracted controls still come from familiar source episodes. This is a temporal stress test, **not an independent source holdback or a new estimate of general accuracy**.

Frozen contract SHA-256:

```text
bbfd2b963d398dd34b1b11072a1791860b9a2bb60427daff30132375c359472a
```

## Actual results

| Condition | Correct verdicts | Correct UNKNOWN cases | Total UNKNOWN outputs | Incorrect passes | Correct placements accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guided SAM + checker | 7/8 | 4/4 | 5 | 0 | 2/2 |
| Spark alone | 8/8 | 4/4 | 4 | 0 | 2/2 |

The guided UNKNOWN reasons matter:

- Episodes **000000 and 000001 inconsistent windows** explicitly reached the insufficient-stable-observations guard after grounding.
- Episodes **000002 and 000003 inconsistent windows** abstained because of low reported detection confidence. Their final UNKNOWN verdicts match the labels, but they do **not** demonstrate that the temporal guard specifically detected those two inconsistencies.
- The **episode 000001 stable staging control** also remained UNKNOWN because of low confidence. Its reviewed label is FAIL; this is an unnecessary abstention on that supported control, not an extra success.

Spark returned UNKNOWN for all four mixed-time pairs and supplied explanations about inconsistent placements. It correctly rejected both staging controls and accepted both dispense controls. Neither condition produced an incorrect pass on this selected suite.

An always-UNKNOWN system would match only **4/8** labels here and accept **0/2** correct placements. Reporting correct acceptance alongside abstention therefore prevents blanket uncertainty from appearing to solve the task.

## Compute and integrity checks

| Condition | Actual requests | Estimated USD | Run elapsed seconds |
| --- | ---: | ---: | ---: |
| Guided SAM + checker | 22 | 0.106472 | 100.926 |
| Spark alone | 8 | 0.078527 | 78.010 |

The run made **30 actual requests under a 32-request cap**, estimated at **$0.184999** total. These are usage-based estimates rather than a billing guarantee. Missing or ambiguous targets can use fewer than the maximum guided requests; no repeated call was selected merely for producing a higher score.

The report integrity validator recomputed the metrics, checked **16 condition/case decisions**, replayed **eight grounded decisions**, and replayed both private inference sidecars. Verdicts, source bindings, stored grounded evidence and reported metrics were consistent. This checks internal evidence/report integrity; it is not independent proof that the models perceived the physical scene correctly.

## Scope of the finding

The suite now tests an uncertainty condition missing from the earlier simple placement comparison. It demonstrates appropriate abstention on four authored inconsistent pairs, with the specific temporal guard evidenced in two guided cases. It does not establish a universal false-acceptance rate, continuous tracking, recovery after a requested physical repair, or superiority over Spark.

The separate [fresh-source comparison](online-guided-v2-holdback-20261028.json) remains unchanged: guided **3/4**, Spark **4/4** on episodes 000006 / 000007. This development suite is not added to that holdback denominator or presented as an independent improvement. More diverse sources and causal evidence-acquisition tests remain necessary.
