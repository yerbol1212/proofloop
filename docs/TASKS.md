# Reviewed task contracts

The **Task** tab creates executable placement checks through a real Muse Spark request. It supports two to six named targets and two calibrated areas, LEFT and RIGHT. Each target class must have exactly one visually distinct intended instance; this is a declared controlled-scene assumption, not detector-proven completeness.

Describe the requirements, review the SAM prompts and perception mode, confirm the intended unique targets, then choose **Draft checks with Spark**. The active task and its evidence remain unchanged while a proposal is drafted. Read every proposed object/region check before approving it. Changing the draft inputs blocks activation until a new matching draft is generated.

**Activate reviewed task** binds an exact draft ID and hash. It changes the executable specification, clears prior images and verdicts, and requires two new views. SAM prompt mappings and perception mode are included in the reviewed task hash; changing either cannot reuse an earlier task's certification.

Supported compilation is deliberately limited to positive location predicates. The compiler rejects count, absence, quality, attachment and temporal requirements instead of silently translating them into placement checks. The core has additional predicate types, but that does not imply the current camera pipeline can observe them.

## Perception modes

- **SAM concept prompts:** one SAM call per target per image. Short nouns are recommended; detection can still fail.
- **Spark-guided SAM:** one task-independent Spark call on the two images proposes candidate boxes; actual SAM calls refine unambiguous boxes. Missing, ambiguous or low-confidence support stays unknown. Spark assigns object identity; a prompted SAM mask does not independently establish that identity. Guided uploads currently require upright metadata-free JPEGs up to 1280 pixels. See the [guided evaluation](GUIDED-PERCEPTION.md).

The runtime caps concept inspections at `2 × target count` requests and guided inspections at `1 + 2 × target count`, with no automatic retries. The inspection request identifies what further evidence is needed, including an expired pair, unsupported region, ambiguous target, low confidence or inconsistent views. It does not infer that a non-detected object is absent or hidden.

Guided drafts and active tasks also bind `perception_version` into the exact review hash. A changed guided method requires a newly reviewed draft before activation or inspection; a returned observation with another version cannot verify the task. The current `normalized_boxes_v2` uses normalized Spark boxes converted to SAM source pixels and retains separate identity, full-visibility and boundary-precision flags.

## Local API

All mutations use the same Origin and `X-ProofLoop-Token` boundary as inspection:

- `POST /api/task/draft`: `title`, `requirements`, `objects`, `confirm_unique_targets: true`; optional `concepts` mapping and `perception_mode` (`concept` or `guided`). One bounded Spark call.
- `POST /api/task/activate`: exact `draft_id`, `draft_hash`, `confirm_requirements: true`, `confirm_unique_targets: true`. No model call.
- `GET /api/state`: active task, unapproved draft, safe draft errors and inspection guidance.

Assistant-driven integration tests identify `review_origin: assistant_under_user_authorization` and save the authorization reference privately. Such review is not described as a person having reviewed the exact contract. Ordinary local confirmation is operator-declared, not authenticated identity proof.

The [live task mutation record](../artifacts/feasibility/task-editor-live-20261028.json) records a real Spark draft swapping fork/spoon regions, preservation of the old active state before approval, and immediate evidence invalidation after assistant-authorized activation. It establishes the task-editing integration, not physical completion of the new task.
