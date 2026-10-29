# Native context-crop proposal feasibility

This is a bounded diagnostic of the failed clear-cup controls, not a new acceptance method. The [previous concept-mask trial](CONCEPT-MASK-RESULTS-20261030.md) returned nearly identical cup masks between paired views, with scores on opposite sides of the unchanged 0.8 threshold. All fifteen nonempty retained concept-SAM records re-decode successfully and have raster dimensions matching their inclusive boxes. No observed parser or coordinate bug explains the score gap.

The current [hosted SAM documentation](https://dev.meta.ai/docs/sam/reading-segmentation) defines `c` as confidence for one frame's record. Its [prompting documentation](https://dev.meta.ai/docs/sam/segmenting) permits variation across frames and repeated runs. Neither documents a calibration guarantee or a hosted score formula. The [open-source image processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py) resizes inputs in its local implementation; this does not establish the hosted SAM 3.1 implementation. Scale/context sensitivity is a hypothesis to test, not an established cause or justification for lowering a threshold.

## Fixed transformation and scope

Retain all original eighteen source views and nine now-exposed cases. Select one crop for each valid single original concept proposal, including low-score proposals and those already vetoed by cached visibility judgments. Three empty original proposals select no crop; all fifteen available crops receive exactly one new SAM call with the unchanged positive noun phrase.

Set square side to `max(256, 4 × max(inclusive box width, inclusive box height))`. Center using integer floor and clamp within the source. If that square cannot fit the source's smaller dimension, use the full original source rectangle. Decode the original JPEG to RGB before cropping and write lossless PNG, with no resizing or generated pixels. This also changes transport encoding; improvements cannot be attributed solely to spatial context without a separate control.

Keep each actual crop response unchanged and separate from its derived full-source geometry. Translation adds only the integer crop origin to box coordinates; the official decoded raster and reported confidence remain unchanged. Reject missing/multiple crop detections, unavailable or below-0.8 scores, artificial crop-edge contact and every original cached identity, uniqueness or visibility veto. Crop and original count as one physical view.

The probe reports **proposal eligibility only**. Two eligible source projections do not establish the named target, full visibility, correct placement or gripper separation. No full-scene semantic calls, task verdicts, replacement accuracy score, production promotion or physical execution claim are provided.

## Public declaration and gate

The [exact machine-readable declaration](../artifacts/feasibility/context-crop-plan-20261030.json) binds private plan `d92eb2c322ed7f92cdb4f430da767f8ef38bb9799f70003b7135b82e67361207`, all original/prepared/parent/source/code/dependency hashes and all nine references. Its file SHA-256 is `806a76abd70fcc7fc3532286106a1059163a75af11648c332f7b1f778b1ac9fa`. It must be public before the first paid admission.

The maximum is **fifteen new SAM requests**, zero Spark requests and zero retries. Fresh one-request clients share a persistent fifteen-request journal. HTTP 401/402/403/429 stop admission; failures remain unscored. No prompt change, best-of selection, hidden fallback or favorable substitution is allowed.

Continue this branch only if both clear-PASS reference pairs obtain two eligible projections, with all eighteen views/nine pairs and complete accounting preserved. If that gate fails, stop this crop branch. If it succeeds, full-source semantic/gripper/region checks and separately frozen new-source validation remain necessary. The original two task positives are not retroactively accepted by this probe.
