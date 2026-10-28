# Guided perception and real-video findings

The first external-video pilot exposed a real limitation: SAM returned zero detections for all sixteen images using the concept “transparent drinking glass.” The checker correctly abstained, but useful decision coverage was zero. Shortening the phrase to “cup,” “glass” or “drinking glass” also produced zero detections in six development-only probes. We preserve these failures.

Meta documents [box and point prompting](https://dev.meta.ai/docs/sam/segmenting) as an alternative when a user or another model can locate the target. The optional guided mode asks Spark to propose candidate boxes and asks SAM to refine their boundaries. The current version uses Spark's documented normalized coordinate convention, then converts to SAM source pixels. It uses actual model output; it has no hand-authored box oracle.

`proofloop.guided.guided_segment_pair` receives only exact source images, dimensions and object concepts. The locator receives **no expected regions, task requirements, target placements or ground-truth labels**. It must identify unambiguous candidate objects from each frame. Missing or multiple candidates produce unavailable evidence; they are never converted into fabricated completed SAM responses. Source hashes are checked around requests. SAM confidence thresholds are unchanged.

This does not create independent semantic verification: Spark assigns the class identity, while SAM provides the prompted mask. Common mistakes remain possible. The mode is explicitly selected and reviewed, rather than silently substituted when text detection fails.

## Historical v1 frozen new-source comparison

After development on episode 000000, the implementation and prompts were frozen. Two previously untested source episodes, 000004 and 000005, supplied four pre-labeled placement cases with the same calibration. They are a small source holdback in the same task and camera setup; they do not establish task or environment generalization.

| Condition | Matches reviewed labels | Unknown | Incorrect passes | Correct placements accepted |
| --- | ---: | ---: | ---: | ---: |
| Text SAM + checker | 0/4 | 4 | 0 | 0/2 |
| Spark alone | 4/4 | 0 | 0 | 2/2 |
| Spark-guided SAM + checker | 2/4 | 2 | 0 | 1/2 |

Guidance recovered some geometric support relative to text-only SAM. **It did not outperform Spark alone on this sample.** Zero incorrect passes in four cases is not a general reliability guarantee. More unknowns are not automatically better. All systems received the same requirements and exact frame pairs for evaluation; only the guided locator was isolated from the acceptance requirements. Compute costs differ and are reported separately.

The [machine-readable result](../artifacts/feasibility/online-coffee-new-source-20261028.json) and [experiment narrative](../artifacts/feasibility/online-video-20261028.md) preserve the results and limitations. The **Video lab** tab shows original recorded frames, calibration, expectations, per-condition reasons and the source license. These recordings are normal teleoperated demonstrations, not robots following ProofLoop's recommendations.

## Diagnosing the two v1 unknowns

The failures occurred at different stages. For **episode 000004 dispense**, the locator marked its candidate ambiguous in both images, so SAM was never called. The v1 flag combined uncertainty about identity, visibility and exact boundaries; the trace does not establish which kind caused the flag. For **episode 000005 staging**, the locator returned boxes that visibly omitted the upper glass rim and extended below the glass. The actual SAM confidence values were **0.36328125 and 0.2392578125**, correctly kept below the unchanged 0.8 acceptance threshold. The box defect is an evidenced input problem and a plausible contributor to low confidence, rather than a proven isolated cause.

The [official Spark grounding documentation](https://dev.meta.ai/docs/image-understanding#localize-objects-with-perception-grounding) describes normalized 0–1000 coordinates. V1 requested absolute image pixels instead. This motivated a versioned observation change; it does not justify changing historical results or forcing ambiguous observations into a decision.

## Current method: `normalized_boxes_v2`

The locator now emits normalized **outer box edges**, with `(0,0)` at the image's upper-left and `(1000,1000)` at its lower-right. Integer values must lie within this range and define a positive extent. For source dimensions `W × H`, conversion into the inclusive SAM wire coordinates is:

```text
pixel_x1 = floor(normalized_x1 × W / 1000)
pixel_y1 = floor(normalized_y1 × H / 1000)
pixel_x2 = ceil(normalized_x2 × W / 1000) - 1
pixel_y2 = ceil(normalized_y2 × H / 1000) - 1
```

This covers the declared outer extent, including a box ending at the image edge. Invalid geometry is rejected rather than clamped. The conversion adds no object-specific padding and never uses an expected region or label to move a box. SAM's inclusive wire convention is documented in its [prompt API](https://dev.meta.ai/docs/sam/segmenting#prompt-with-boxes-and-points); the [official builder](https://dev.meta.ai/docs/sam/client-libraries#build-box-and-point-prompts) likewise converts half-open right/bottom edges into inclusive values.

The new schema separates three judgments:

- **`identity_ambiguous`**: uncertainty about class or instance identity. True still blocks refinement. Multiple candidates are not reduced to a preferred instance.
- **`fully_visible`**: whether the target's full visible extent is supported. False, including an occluded or truncated target, still blocks refinement.
- **`boundary_uncertain`**: imprecision in edge placement despite a unique visible target. This is recorded separately and may proceed to SAM, whose purpose is boundary refinement.

The older `ambiguous` flag is never reinterpreted retrospectively. Missing candidates, ambiguous identity, unsupported visibility, inadequate SAM confidence and inconsistent frame evidence continue to produce unknown. A refined mask alone never grants a pass.

The request cap is unchanged: **one Spark locator call plus at most two SAM calls per object**, with no automatic retries. A single-object pair therefore uses at most three calls; unavailable targets can use fewer. The checker still requires two distinct supporting images, the reviewed timing and region contract, and the same **0.8** confidence threshold. No threshold was optimized to the known labels.

Guided task drafts, active approvals and saved capture provenance now include `perception_version`. A draft or active task reviewed under another guided version is rejected until a new draft is reviewed and activated. A method update therefore cannot silently reuse an older observation approval or certify a previous capture.

Spark still assigns semantic identity. SAM refines the prompted mask and reports its own confidence; it does not independently prove that the mask belongs to the named class. Neither model's confidence is presented as a calibrated probability of physical correctness.

### Development replay

V2 matched **4/4** labels when replayed on the already exposed episode 000004 / 000005 cases, with no unknowns. The run used **12 requests**, estimated at **$0.050461** under the recorded rates. This is a **development replay, not held-out evidence**: these images and their prior failures informed the change. The normalized coordinates and separated uncertainty flags changed together, so this run does not isolate either change's individual effect. The original v1 result remains 2/4 in the historical table.

### Fresh-source check

Episodes **000006 and 000007** supplied four additional cases with the original requirements, concepts, timestamps and region rectangles. Their labels and source hashes were reviewed and frozen before inference; the v2 module and schema were frozen separately before testing. The completed comparison is preserved in the [fresh-source result](../artifacts/feasibility/online-guided-v2-holdback-20261028.json).

| Condition | Matches reviewed labels | Unknown | Incorrect passes | Correct placements accepted |
| --- | ---: | ---: | ---: | ---: |
| Text SAM + checker | 0/4 | 4 | 0 | 0/2 |
| Spark alone | 4/4 | 0 | 0 | 2/2 |
| V2 Spark-guided SAM + checker | 3/4 | 1 | 0 | 1/2 |

| Condition | Actual requests | Estimated USD | Run elapsed seconds |
| --- | ---: | ---: | ---: |
| Text SAM + checker | 8 | 0.020000 | 7.784 |
| Spark alone | 4 | 0.043140 | 47.586 |
| V2 Spark-guided SAM + checker | 11 | 0.052156 | 49.320 |

The remaining guided unknown was **episode 000006 dispense**. The first frame's locator reported `fully_visible:false`, so no SAM call was made for that image. The second frame produced an actual SAM confidence of **0.74609375**, below the unchanged 0.8 threshold. Both limitations remain in the trace; no fallback, threshold relaxation or synthetic evidence converted this case to a pass. The authored label concerns supported visible placement, while this observation method requires full target visibility as an additional conservative gate.

V2 supplied geometric support for three fresh cases, but **did not outperform Spark alone**. Its 3/4 fresh result and v1's 2/4 historical result use different source episodes, so they are not a paired measurement of a one-case improvement. The 4/4 development replay remains separately labeled above. The weakness is reduced on these tested inputs, not fully resolved.

There are only **four cases from two source episodes**, including two reviewed failing placements and two reviewed passing placements. Neither this sample nor zero incorrect passes establishes a general reliability rate. It contains no ground-truth unknown class, so it cannot measure appropriate abstention across ambiguous tasks. These near-identical scenes test narrow source transfer, not new-task or environment generalization, full procedure success, or interactive physical recovery. See [source provenance](ONLINE-VIDEO-SOURCES.md).

## Commissioning object prompts

```sh
python -m proofloop.concepts --images /path/view1.jpg /path/view2.jpg \
  --candidate cup --candidate glass --max-requests 4 \
  --input-origin licensed_online --output artifacts/local/my-commissioning-run
```

This bounded diagnostic reports detections and reported confidence, not accuracy. It never selects or approves a winner. Tune only on declared development inputs; freeze a new contract before evaluating additional sources. Changing a prompt mapping in the Task tab requires a new reviewed draft and invalidates old evidence.
