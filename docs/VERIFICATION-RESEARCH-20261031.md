# Full-source verification investigation — 31 October 2026

The next capability question is whether the [native crop proposals](CONTEXT-CROP-RESULTS-20261030.md) can support useful task decisions while withholding the known gripper-edge ambiguity. The prior result is proposal feasibility only: four eligible pairs include both clear cups, one wrong-side fork and one ambiguous cloth. All nine exposed pairs must remain in the next comparison. No test on these examples establishes new-source generalization.

## Research and implementation choices

| Primary source | Relevant observation | Concrete consequence for ProofLoop |
| --- | --- | --- |
| [Meta image understanding](https://dev.meta.ai/docs/image-understanding) | Spark accepts several source images and can describe relationships or propose coordinates. | Use two exact original views to propose terminal tools, then separately review actual target/tool masks against both original scenes. Coordinate proposals remain fallible model outputs. |
| [Meta SAM prompting](https://dev.meta.ai/docs/sam/segmenting) | Spatial prompts can use boxes supplied by another model. Concept and spatial prompts are separate modes; spatial coordinates use the media's dimensions. | Replace failed text gripper proposals with automatically proposed source-coordinate terminal-tool boxes. Send spatial tokens alone to SAM on each original image. Keep targets from the frozen crop experiment rather than rerunning favorable target proposals. |
| [Meta segmentation output](https://dev.meta.ai/docs/sam/reading-segmentation) | Boxes use inclusive endpoints and each record can have confidence. Stable IDs have a bounded response scope. | Preserve exact crop-response bytes and separately translate geometry. A returned ID, high score or empty detection does not independently establish semantic identity, release or physical absence. |
| [REFLECT](https://robot-reflect.github.io/) | The authors connect recorded observations to failure explanation and correction planning. | Keep the scene observation, inferred interpretation and allowed next step inspectable. Do not infer physical contact, hidden grasp force or an executed correction from image-zone agreement. This is an adaptation of the evidence principle, not a reproduction of their robot evaluation. |
| [KnowNo](https://robot-help.github.io/) | The authors calibrate prediction sets on a calibration dataset before asking for help. | Preserve uncertainty and specific missing-evidence outcomes. ProofLoop's current thresholds and small exposed sample are not conformal calibration and inherit no statistical completion guarantee. |

This architecture is an inference from those sources and our measured failures. It is not a recommendation or endorsement from Meta. The local adapter supports eight Spark images, so the final review can include two originals, two native contexts, two derived full-source target masks and two derived full-source terminal-tool masks without changing the provider integration.

## Bounded new method

1. Preserve every original case, reference, timestamp, crop receipt, source byte and confidence threshold. Only the four pairs already eligible under the completed crop test enter new inference; the others remain explicit UNKNOWN outcomes from their completed target gate.
2. Ask Spark to locate visible terminal end-effectors, including all contact-relevant fingers, in the two original images. Bound the output to zero, one or two tools; unresolved counts and ambiguous localization cannot qualify support. Do not provide desired regions, reference labels or earlier predictions in either perception prompt.
3. Send at most one SAM spatial request per original view, batching the declared tool boxes under exact IDs. No manual anchors, noun-phrase retry, best-of selection or invented full-source target response is allowed. Completed empty or weak proposals withhold support; failed required requests remain unscored.
4. Compare the actual tool masks with the explicitly derived target geometry at the unchanged three-source-pixel negative warning margin. A near proposal can withhold support. A far proposal cannot itself establish separation or visibility.
5. Ask Spark separately to assess full-scene target identity, uniqueness, visibility, extent and mask purity; apparent release/contact; visible tool count; and whether returned tool masks cover all contact-relevant terminal parts. Contradiction, uncertainty, missing coverage or a near warning withholds support even if another field is positive. Agreement between calls to the same model can share errors.
6. Feed only qualified, explicitly labeled derived observations to the unchanged whole-region, distinct-source, stability and core task checks. The original image, its crop and its mask count as one physical view. Task outcomes here are recorded-source experimental results, not current authority or physical execution.

The maximum for four eligible pairs is eight Spark requests and eight SAM requests, sixteen total. Every fresh client has allowance one, retries zero, and shares a persistent journal. The exact frozen plan and implementation must be public before paid admission. HTTP 401, 402, 403 or 429 stop further admissions. No automatic resume is allowed.

The development gate requires both clear task positives, the wrong-side FAIL and all six UNKNOWN references, with complete original-case accounting. A failure stops this candidate rather than triggering prompt tuning on the same outcomes. A favorable result permits a separately declared new-source evaluation; it does not qualify production deployment.

## Production findings being corrected separately

Independent review reproduced two ordinary-workflow defects with injected model fixtures and no paid calls. Upload freshness began at worker dispatch, allowing queue dwell to disappear. A normal inspection after changing tasks could also attach the previous task's recovery receipt. Production corrections must preserve admission age through completion, reject stale queued images before inference, and bind receipt attribution to the exact task, protocol and revision. Historical receipt files remain historical evidence. These are software correctness issues, distinct from perception accuracy.
