# Real online-video evaluation — 28 October 2026

Source: real ALOHA coffee-task robot recordings distributed by Hugging Face / LeRobot, with MIT explicitly declared in the dataset card. See [source provenance and credits](../../docs/ONLINE-VIDEO-SOURCES.md). Original videos and extracted frames stay local. These tests need no new participant recording.

The authored gate checks whether a single glass is wholly inside a calibrated dispensing area. Its early staging position fails that readiness gate; its later position can pass. The early position is a normal intermediate state, not a publisher-labeled robot mistake. A passing placement does not certify dispensing, grasping, beverage quality or safety.

## Initial frozen development pilot

Eight selected cases from four episodes, with labels and calibration reviewed before inference. Sixteen SAM calls and eight Spark baseline calls completed.

| Condition | Matches labels | Unknown | Incorrect passes | Correct acceptance | Est. USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM + checker | 0/8 | 8 | 0 | 0/4 | 0.040000 |
| Spark alone | 7/8 | 1 | 0 | 3/4 | 0.077802 |

SAM returned no objects for the chosen transparent-glass concept. The original run is unchanged. Six additional short-concept probes on development episode 000000 also returned no objects. This motivated explicitly reviewed visual prompting rather than weaker thresholds or relabeled examples.

## New-source guided comparison

The guided implementation was frozen after development probes, before inference on episodes 000004 and 000005. Four newly labeled cases used the unchanged regions and semantic concept. All three conditions ran on the same exact input pairs.

The verbatim requirements text retains the initial pilot's descriptive reference to four episodes. The bound source list contains exactly two episodes for this comparison; labels, regions and evaluated placement semantics are unaffected. The immutable evaluated contract is preserved, and the interface derives its short task summary from the executable checks.

| Condition | Matches labels | Unknown | Incorrect passes | Correct acceptance | Actual requests | Est. USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM + checker | 0/4 | 4 | 0 | 0/2 | 8 | 0.020000 |
| Spark alone | 4/4 | 0 | 0 | 2/2 | 4 | 0.041702 |
| Spark-guided SAM + checker | 2/4 | 2 | 0 | 1/2 | 10 | 0.040694 |

The two comparisons total 46 actual requests and an estimated **$0.220198**, excluding separate compiler and development probes. Estimates use adapter Standard rates and are not provider billing guarantees. Raw responses and request-level usage remain private.

The guided condition improved decision coverage over concept-only SAM on these four cases but did not beat Spark's acceptance judgments. Its additional output is an inspectable mask; that does not prove the mask's semantic identity independently of Spark. This result supports an engineering improvement and identifies remaining perception limits, not a claim of superior model accuracy.

## Reproducibility and limits

- [Original pilot summary](online-coffee-20261028.json), contract `a80ecfc424e5ae7de2204a653192389a430a7e83f920bff3773d2a064d3d62f2`.
- [New-source summary](online-coffee-new-source-20261028.json), contract `e78ef11c8a8086c4d1f49784e2ea186dca020a060e33cd637de875b0506991cf`.
- Source hashes, exact timestamps and license links are public; raw source images, locator responses and decoded masks are private.
- Full-run labels were not evaluated. Cases from the same episode are correlated, the camera/task are nearly identical, and this is not an official ALOHA benchmark.
- No robot followed the planner, no live physical correction was validated, and the results cannot establish production reliability or demand.
- The [benchmark guide](../../docs/BENCHMARK.md) documents validation, bounded execution and allowlisted export. Unknown and missing predictions are counted explicitly; an always-unknown system cannot pass by merely avoiding incorrect acceptance.
