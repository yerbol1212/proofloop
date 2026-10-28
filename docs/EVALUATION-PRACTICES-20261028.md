# Evaluation practices adopted from robotics projects

Researched against primary project documentation on October 28, 2026. These are transferable evaluation practices, not claims that ProofLoop matches a robot controller or competes on the same benchmark. No third-party source code or assets were transplanted.

| Project | What the project actually does | Practice transferred to ProofLoop |
| --- | --- | --- |
| [LeRobot benchmark integration](https://huggingface.co/docs/lerobot/en/adding_benchmarks) | Runs evaluations across episodes, tasks and suites, with an environment-provided success signal and hierarchical aggregation. | Keep the check, case and source views. Compare conditions on the same approved evidence, and inspect individual disagreements instead of inferring behavior from one average. |
| [SimplerEnv](https://github.com/allenai/SimplerEnv) | Evaluates robot policies with visual matching and variant aggregation, including changes to background, lighting, distractors and textures. It provides reproducible evaluation scripts. | Use a predeclared perturbation matrix and retain failure cases. Reprocessing the same source is development evidence, not a new independent scene. A distractor challenge must check target identity separately from mask geometry. |
| [robomimic datasets](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) | Separates demonstrations by task and collection type, including multiple humans of differing quality and both simulation and real data. Its docs warn that dataset/environment versions can affect reproduced results. | Keep dataset roles, source groups, task identity and observation versions explicit. Do not pool new-task pilots, reused coffee scenes and new-source evaluation into a single headline score. |
| [RLBench task conditions](https://github.com/stepjam/RLBench/blob/master/rlbench/backend/task.py) and [condition composition](https://github.com/stepjam/RLBench/blob/master/rlbench/backend/conditions.py) | Tasks declare explicit success/failure conditions. Its condition system can combine predicates and preserve ordering state. | Preserve per-check comparisons: identical aggregate FAIL values can hide a changed subcheck. A final placement image cannot prove that earlier workflow steps occurred in the required order. |
| [Meta SAM prompt documentation](https://dev.meta.ai/docs/sam/segmenting) | Distinguishes concept prompts, which match named concepts, from box/point prompts, which select specified objects. It advises applying spatial filtering to returned coordinates. | A successful box-prompted mask is not independent confirmation that Spark selected the correct semantic target. Keep identity challenges and task acceptance separate from mask confidence. |

The metric definitions below are ProofLoop design decisions. They are not attributed to a robotics project's published results.

## Implemented now

`proofloop.evaluation.evaluation_summary(envelope, report, raw_results=None)` first uses the existing shared `validate_report`. It refuses inconsistent totals, incomplete case/check coverage and replay contradictions. Optional private adapter sidecars receive the same replay validation. The new summary contains no raw media, masks, prompts, provider response bodies or local paths.

For each completed condition, it reports both case-level and check-level metrics:

- **Known-truth coverage:** PASS/FAIL predictions on PASS/FAIL labels, divided by the number of PASS/FAIL labels. UNKNOWN labels are excluded from this denominator.
- **Known-truth selective accuracy:** correct PASS/FAIL predictions divided by decided PASS/FAIL predictions on known truth. This is unavailable when the system abstains on all known cases. Pair it with coverage; abstaining can raise selective accuracy while making the app less useful.
- **Known FAIL acceptance:** FAIL labels predicted PASS, divided by all FAIL labels.
- **UNKNOWN acceptance:** UNKNOWN labels predicted PASS, divided by all UNKNOWN labels. This is unsupported acceptance under the annotation contract, rather than evidence of a known physical failure.
- **Acceptance label error:** accepted PASS predictions whose label is FAIL or UNKNOWN, divided by all PASS predictions. The two components remain separately visible.
- **Correct acceptance:** PASS labels predicted PASS, divided by all PASS labels.
- Overall three-label accuracy, the full confusion matrix and abstention counts remain available. Empty denominators produce `null`, never an invented zero or perfect score.

Pairwise diagnostics list every aggregate and per-check disagreement with the case ID, source hash, expected label, both predictions and which prediction matched the label. They also count cases where only one condition was correct. They do not provide a statistical significance claim.

Source groups collapse exact source-file SHA-256 aliases and show metrics for each group. Different encoded file hashes do **not** establish independent captures: transcodes, related episodes and repeated scenes can still be correlated. The number of selected cases or checks must not be advertised as an independent sample count. The summary deliberately provides no confidence interval or population reliability estimate.

`comparison_id` deterministically hashes the approved contract, evaluation role, declared condition versions and validated benchmark decisions. It is stable when list ordering changes. It identifies a decision comparison, not a unique API run, trusted timestamp, model authorship or image authenticity.

Illustrative cases are excluded from all new scores and pairwise comparisons. Running/error conditions are explicitly excluded instead of receiving fabricated UNKNOWN predictions. Existing published benchmark summaries remain unchanged.

## Reproduce without new model calls

From the repository root:

```sh
.venv/bin/python -m unittest tests.test_evaluation -v
.venv/bin/python -m proofloop.evaluation \
  artifacts/local/online-benchmark-guided-v2-holdback-20261028-01/approved-manifest.json \
  artifacts/local/online-benchmark-guided-v2-holdback-20261028-01/report.json \
  --output artifacts/local/guided-v2-paired-diagnostics.json
```

The second command needs the existing private recorded-run files and refuses to overwrite an output. It performs no network calls and requires no API key. A public clone can run the authored unit tests; those tests are not live model performance evidence.

Seventeen authored diagnostic tests passed under both Python 3.12 and the system Python 3.9. All seven existing completed online-run reports also passed this diagnostic computation with their private inference sidecars. No new inference requests were made for that replay.

## What replay of the existing experiments reveals

| Existing experiment | Exact source-file groups | Guided known-truth coverage | Guided selective accuracy | Spark known-truth coverage / selective accuracy |
| --- | ---: | ---: | ---: | ---: |
| Normalized v2 new-source comparison | 2 | 3/4 | 3/3 | 4/4 and 4/4 |
| Fixed future evidence pairs, development | 3 | 1/3 | 1/1 | 3/3 and 3/3 |
| Temporal uncertainty stress, development | 4 | 3/4 | 3/3 | 4/4 and 4/4 |

The fixed-future experiment also contains three UNKNOWN-labeled cases; the temporal experiment contains four. They remain separate from these known-truth denominators. A perfect selective accuracy here hides reduced coverage: it is not evidence that guided perception beats Spark. None of these curated runs recorded a FAIL-label or UNKNOWN-label PASS acceptance, but these small exposed cohorts do not establish an operational error bound. Rows share previously exposed source families and must not be pooled into independent observations.

## Next evidence that would change the decision

1. Freeze a fresh source-group challenge plan **before** inference: same-class distractors, partial occlusion, region-boundary proximity, appearance change and a calibration-invalidating camera change. Preserve the original sources and declare every transformation. Do not assign exact-count or absence labels unsupported by the current contract.
2. Test a uniquely specified target alongside a similar distractor. Record semantic selection and final task check separately. A wrong object in the desired region is the essential negative control.
3. Publish the paired disagreement table and source grouping even when SAM loses to Spark. If SAM improves mask geometry but not acceptance decisions, claim only the measured geometry result; that requires independent mask annotations.
4. Establish physical following of a displayed request using a genuinely new recording after the request. Existing prerecorded future frames can evaluate a controller's stopping decisions but cannot prove the person or robot followed ProofLoop.

Do not switch thresholds, delete difficult cases, or choose favorable future frames to make an already exposed dataset look stronger. Improve the system on development data, then freeze it before another new-source comparison.
