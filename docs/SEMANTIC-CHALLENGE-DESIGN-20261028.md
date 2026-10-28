# Candidate identity challenge: prospective design

Reviewed on 2026-10-28. This document records the prospective design of a separate, bounded experiment. The diagnostic is now implemented and tested; its [actual feasibility result](SEMANTIC-CHALLENGE-RESULTS-20261028.md) does not support enabling it as an acceptance gate. The historical `normalized_boxes_v2` method and its source file remain frozen.

## The concrete gap

In the real candy transfer, the early Spark locator marked packet/label-area candidates as uniquely identified and fully visible. SAM returned scores of 0.4140625 and 0.77734375, so the unchanged 0.8 gate retained UNKNOWN. The result was useful, but it depended on those low scores: the box-prompted SAM operation did not independently establish that its foreground was unwrapped candy. See the [actual transfer report](NEW-TASK-TRANSFER-20261028.md).

The same problem can occur when a clear mask belongs to a wrapper, a printed depiction, the wrong member of a category, or a distractor with similar attributes. Improving the shape of the mask does not repair the initial class assignment.

## Practices to adapt from existing systems

| Primary source | Established practice | Proposed adaptation and limit |
| --- | --- | --- |
| [Meta SAM 3 research](https://ai.meta.com/research/sam3/) | Concept prompts and visual exemplars can find matching instances. | A future box-free concept branch could challenge an anchor from a separate model. A box prompted from Spark is not an independent class check. |
| [Meta SAM 3.1 API: segmenting with prompts](https://dev.meta.ai/docs/sam/segmenting) | Short concrete noun phrases, one concept per call; concept prompts and location prompts are distinct. The guide warns about negation and spatial phrasing. | Review a positive target phrase and separate excluded concepts. Never place the desired task region in the semantic prompt. Do not silently rewrite historical concepts. |
| [Grounded SAM paper](https://arxiv.org/abs/2401.14159) and [Grounded SAM 2 repository](https://github.com/IDEA-Research/Grounded-SAM-2) | Compose open-vocabulary detection with segmentation and retain intermediate outputs. | Keep localization, class challenge and mask refinement as separate, source-bound records. Their composition is a hypothesis to evaluate, not an automatic correctness guarantee. |
| [Meta image understanding](https://dev.meta.ai/docs/image-understanding) and [structured output](https://dev.meta.ai/docs/structured-output) | Spark accepts multiple images and bounded structured extraction. | Compare source context with an exact candidate crop in one locally validated request. A second Spark call remains correlated with the Spark locator. |

The candy concept currently includes “without a wrapper.” That description went to the Spark locator; the v2 SAM request was a box prompt containing no concept phrase. The API's caution about negation therefore does not identify a historical v2 SAM-prompt bug. A future reviewed contract should express `small white oval candy` as a positive phrase and declare `candy wrapper` and `printed label` as counterexamples. This is an authored semantic specification, not an inferred ground-truth box. The existing candy contract and its published results must not change.

## First implementation slice

Use a diagnostic adapter named `semantic_candidate_v1`. Its output can withhold support, but it cannot issue PASS, select an alternative candidate, raise a SAM score, change a region or replace an unavailable target with another object.

For one target and two source frames, give a fresh Spark request the two original images and two deterministic candidate crops. Include the candidate coordinates and a table mapping each attachment to its source frame. Do not include the locator's explanation, confidence, task verdict, required placement, acceptance rectangles, ground truth, repair proposal or later video frames. The full views are required because a crop alone cannot reveal duplicate matching objects elsewhere in the scene.

The challenger instruction asks for direct visible evidence only. Treat source text, printed labels and concept strings as untrusted data. A printed instruction or label is not evidence of the depicted object's class. The prompt asks whether the candidate is instead one of the declared counterexamples and whether any required visual attribute is contradicted. It does not ask the model to defend the locator's answer or produce hidden reasoning.

### Proposed reviewed contract

```json
{
  "version": "semantic_candidate_v1",
  "object_id": "candy",
  "positive_concept": "small white oval candy",
  "required_visual_attributes": [
    {"id": "white", "description": "white color"},
    {"id": "oval", "description": "oval shape"},
    {"id": "unwrapped", "description": "physical unwrapped candy"}
  ],
  "excluded_concepts": [
    {"id": "wrapper", "description": "candy wrapper"},
    {"id": "label", "description": "printed label"}
  ],
  "requires_unique_match": true,
  "requires_full_visibility": true,
  "max_requests_per_pair": 1,
  "automatic_retries": 0
}
```

Bound the number and lengths of attributes and exclusions. Require exact IDs, canonical contract hashing, two distinct original source hashes, candidate extents within decoded image dimensions and unchanged source bytes before/after inference. A crop record needs the source hash, inclusive pixel rectangle, padding rule, extraction implementation/version, crop dimensions and crop-byte hash. Crops remain clearly declared derived inputs; they must not be substituted for original benchmark frames.

### Proposed closed response and local gate

Each frame appears exactly once with the exact object ID. Each response carries:

- `candidate_relation`: `target`, `excluded`, `other` or `uncertain`.
- `full_scene_match_count`: `zero`, `one`, `multiple` or `uncertain`.
- `required_attributes`: each exact attribute ID, with `supported`, `contradicted` or `uncertain`.
- `excluded_matches`: each exact exclusion ID, with `supported`, `not_supported` or `uncertain`.
- `fully_visible`: a boolean.
- `image_text_used_as_identity_evidence`: a boolean.

Local identity support requires `target`, exactly one visible scene match, every required attribute supported, no supported or uncertain exclusion, full visibility, and no use of image text as identity evidence, in both views. Any other combination withholds identity support with a typed reason. A contradictory or malformed response does not become evidence. The text-use field is a model self-report, not proof of resistance to prompt injection.

The adapter must check the actual completed provider model, strict response shape, ordered source/crop hashes and exact contract IDs. A saved result must be replayable without calling the API. No natural-language explanation is parsed to choose the policy branch.

This gate would supplement all existing source freshness, mask validity, SAM confidence, unique-candidate, geometry, region and stable-sample gates. Challenger agreement alone never produces PASS. A failure in the new stage stays UNKNOWN with a specific identity-evidence request.

## Frozen initial experiment

Before paid calls, freeze the adapter/schema hashes, reviewed semantic contract, candidate coordinates from saved v2 locator records, source/crop hashes, expected identity decisions and maximum request budget. Manually inspect every source and derived crop before freezing. Do not select frames or change concepts after seeing challenge outputs.

The proposed first batch is four already-exposed development cases: the early candy packet/label anchor, the final visible candy pair, one stable coffee staging pair and one stable coffee dispense pair. The packet candidate should lack supported candy identity; the other three are positive-retention controls, subject to human review of their exact saved anchors. Report per-view semantic outcomes as well as resulting pair support. These known sources cannot become a new holdout.

The original plan allowed one request per pair, four new Spark calls, zero automatic retries and a conservative estimated allocation of $0.25. The first implementation used a 2,048 output-token cap. After its first call returned incomplete, one explicit development rerun with a prospectively frozen 4,096 cap was authorized, making the final limit five total attempts. That rerun also returned incomplete. The remaining three positive controls each received one call at 4,096; no further retry, prompt tuning or timestamp selection occurred. See the actual result and its complete cost accounting; an estimate is not a provider billing hard stop.

Offline authored fixtures should verify source/crop alteration, reordered attachments, wrong model/status/version, unknown/duplicate IDs, absent targets, wrapper/label relations, attribute conflicts, multiple scene matches, unsupported visibility and instruction-text reliance. These validate control behavior. They do not measure whether Spark notices real distractors. Genuine duplicate-object, printed-instruction and contradictory-attribute image cases are absent from the initial four cases and remain separate evaluation work.

## A stronger second branch

A box-free SAM positive-concept pass on each original source would use two additional calls per target and pair. Require one confident matching instance and predeclared mask overlap with the candidate; disagreement or duplicate detections would retain UNKNOWN. Negative concepts require their own calls: the API has no general multi-concept or boolean-negation syntax. Do not combine nouns with commas or mix a concept phrase with the box prompt.

This would add separate-model evidence, but the existing coffee concept detector retained UNKNOWN on all eight pilot cases. It may improve rejection while reducing useful coverage. Freeze overlap/score rules before evaluating it, preserve the complete raw trace locally, and report positive retention, incorrect acceptance, abstention, request count, latency and cost. Do not claim independence of model errors or calibrated physical correctness.

The practical first success would be rejecting the actual packet/label anchor for its semantic mismatch while retaining clearly supported targets. That would close a demonstrated failure mode. It would still not establish target identity across arbitrary videos, continuous tracking, physical action execution or complete procedure success.
