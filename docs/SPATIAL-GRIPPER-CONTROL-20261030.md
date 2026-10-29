# Six manual spatial gripper controls

This is the final diagnostic in the [competitive assessment](COMPETITIVENESS-20261030.md), following two failed candidates. [Text-prompt gripper inference](IMAGE-GRIPPER-RESULTS-20261030.md) produced no objects, so it could not test foreground proximity. These six controls isolate that proposal failure from the distance calculation.

Keep the six references declared before the text test: cloth frames 65 and 70, cup frames 0, 5, 58 and 63. A separate assistant reviewed all six source images and authored one generous pixel box around the terminal gripper and fingers; the root assistant reviewed those same images and boxes before inference. These are **manual anchors**, not automatic localization, verified semantic identity or ground-truth masks.

Freeze each original JPEG, its target SAM response, the manual box, the exact spatial token, implementation/dependency hashes and the existing three-source-pixel margin. SAM receives only the spatial token, with no mixed noun phrase. Make exactly one request per view, maximum six, with no Spark call, retries, prompt changes or favorable substitution. The public declaration must precede the first paid admission. Failed responses and unavailable comparisons remain explicit.

The diagnostic gate requires a measured near warning on cloth frame 65 and available comparisons without a near warning on the five separated references. The two cloth views may additionally illustrate the unchanged negative-only policy withholding the parent's erroneous PASS, but this is a single exposed, manually initialized diagnostic. No nine-case task-accuracy result is replaced or pooled with it.

Every actual result, mask count, confidence, distance, unavailable comparison, request and cost will be published. No anchor adjustment is allowed after inference. The controls cannot fix 0/2 clear-task-positive retention, establish automatic gripper perception, qualify production or substitute for an operator following ProofLoop's request.

The [machine-readable declaration](../artifacts/feasibility/spatial-gripper-plan-20261030.json) binds private plan `80fecc1e9179d145a17b31bf210912603cf6dbc35bd6e1013e43c75e15735407`. Its file SHA-256 is `66fc03fb5c41be69449d11d9ef38ff079c9ad834c235ca725195797c7863eb32`. The [probe implementation](../tools/probe_spatial_gripper.py) and seven focused tests are included in the repository.

## Completed result

The limited six-view gate passes. The declaration was public in commit `8c95e4efdbe65aef9be90f50e60a02282b510ae7` before any of its six SAM admissions. All calls complete with one gripper proposal per view and available geometry. The scores below belong to **gripper masks**, not the original task targets; those target scores and the failed 0/2 clear-positive task retention remain unchanged.

| Source view | Gripper SAM score | Foreground distance in source pixels | Negative warning |
| --- | ---: | ---: | --- |
| Cloth 65 | 0.930 | 2 | Near proposal |
| Cloth 70 | 0.945 | Beyond 3 | None observed |
| Cup 0 | 0.945 | Beyond 3 | None observed |
| Cup 5 | 0.941 | Beyond 3 | None observed |
| Cup 58 | 0.949 | Beyond 3 | None observed |
| Cup 63 | 0.961 | Beyond 3 | None observed |

The [independent audit](../artifacts/feasibility/spatial-gripper-audit-20261030.json) re-decodes all six raw responses, matches exact spatial payload hashes against the persistent journal, recomputes distances using independent source-pixel sets and reviews all six masks against their original images. The silhouettes plausibly follow the terminal gripper. Fine fingers near the blue cup rim, filled internal gaps and exact boundaries remain uncertain; this is not pixelwise accuracy or independently established semantic identity.

Applying the unchanged negative-only projection to the two actual cloth diagnostics changes that parent's erroneous PASS to UNKNOWN through a measured near warning. This is **one exposed, manually initialized control**, rather than a replacement nine-case comparison or a production correction. The automatic text method remains failed.

The six requests cost an estimated **$0.015**, with no recorded errors or accounting uncertainty. Across the three assessment experiments, the total is **44 requests, estimated $0.116607**, no automatic retries. One private launch failed to import the local probe before setup/admission; correcting only that launcher's import path made no change to frozen code or data and repeated no paid request. [Full result](../artifacts/feasibility/spatial-gripper-results-20261030.json).

No additional model tuning, heldback-source test or live acceptance promotion follows this control. The next capability gates are automatic proposal availability and useful positive support, followed by the [prepared actual operator trial](OPERATOR-TRIAL-20261030.md).
