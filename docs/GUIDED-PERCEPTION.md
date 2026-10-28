# Guided perception and real-video findings

The first external-video pilot exposed a real limitation: SAM returned zero detections for all sixteen images using the concept “transparent drinking glass.” The checker correctly abstained, but useful decision coverage was zero. Shortening the phrase to “cup,” “glass” or “drinking glass” also produced zero detections in six development-only probes. We preserve these failures.

Meta documents [box and point prompting](https://dev.meta.ai/docs/sam/segmenting) as an alternative when a user or another model can locate the target. The optional guided mode therefore asks Spark to propose source-pixel candidate boxes and asks SAM to refine their boundaries. It uses actual model output; it has no hand-authored box oracle.

`proofloop.guided.guided_segment_pair` receives only exact source images, dimensions and object concepts. The locator receives **no expected regions, task requirements, target placements or ground-truth labels**. It must identify unambiguous candidate objects from each frame. Missing or multiple candidates produce unavailable evidence; they are never converted into fabricated completed SAM responses. Source hashes are checked around requests. SAM confidence thresholds are unchanged.

This does not create independent semantic verification: Spark assigns the class identity, while SAM provides the prompted mask. Common mistakes remain possible. The mode is explicitly selected and reviewed, rather than silently substituted when text detection fails.

## Frozen new-source comparison

After development on episode 000000, the implementation and prompts were frozen. Two previously untested source episodes, 000004 and 000005, supplied four pre-labeled placement cases with the same calibration. They are a small source holdback in the same task and camera setup; they do not establish task or environment generalization.

| Condition | Matches reviewed labels | Unknown | Incorrect passes | Correct placements accepted |
| --- | ---: | ---: | ---: | ---: |
| Text SAM + checker | 0/4 | 4 | 0 | 0/2 |
| Spark alone | 4/4 | 0 | 0 | 2/2 |
| Spark-guided SAM + checker | 2/4 | 2 | 0 | 1/2 |

Guidance recovered some geometric support relative to text-only SAM. **It did not outperform Spark alone on this sample.** Zero incorrect passes in four cases is not a general reliability guarantee. More unknowns are not automatically better. All systems received the same requirements and exact frame pairs for evaluation; only the guided locator was isolated from the acceptance requirements. Compute costs differ and are reported separately.

The [machine-readable result](../artifacts/feasibility/online-coffee-new-source-20261028.json) and [experiment narrative](../artifacts/feasibility/online-video-20261028.md) preserve the results and limitations. The **Video lab** tab shows original recorded frames, calibration, expectations, per-condition reasons and the source license. These recordings are normal teleoperated demonstrations, not robots following ProofLoop's recommendations.

## Commissioning object prompts

```sh
python -m proofloop.concepts --images /path/view1.jpg /path/view2.jpg \
  --candidate cup --candidate glass --max-requests 4 \
  --input-origin licensed_online --output artifacts/local/my-commissioning-run
```

This bounded diagnostic reports detections and reported confidence, not accuracy. It never selects or approves a winner. Tune only on declared development inputs; freeze a new contract before evaluating additional sources. Changing a prompt mapping in the Task tab requires a new reviewed draft and invalidates old evidence.
