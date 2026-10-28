# Placement preparation workflow

ProofLoop now connects a reviewed placement requirement to current visual evidence and a separate declaration of intent to continue. The coffee preset makes the scope concrete: **the declared unique transparent drinking glass must occupy DISPENSE, rather than STAGING**. Its starting rectangles come from the fixed camera used in the licensed coffee development videos. An operator must review both rectangles against the actual view.

This is an authored visible preparation checkpoint. It does not verify dispensing, beverage quality, hidden contents, physical safety, an executed repair, or a complete coffee procedure. Neither Meta endorsement nor an accuracy advantage over Spark alone is claimed.

## Four explicit gates

1. **Review the exact contract.** Activate the draft containing the object inventory, two named region IDs, exact placement checks, perception method and task hash. The fixed coffee preset requires review even though preparing it makes no compilation request. A custom draft uses one bounded Spark request. Local operator review and assistant review under user authorization are labeled separately.
2. **Verify the current preparation.** All required placement checks must PASS in the latest pair of new images. Historical references, UNKNOWN, FAIL, an incomplete pair, expired evidence, an inspection error or a running inspection block continuation.
3. **Replay saved evidence.** “Check preparation evidence” rehashes the saved source files and model results and replays the grounded checks. The proof binds the exact task hash, protocol hash and inspection revision. A successful earlier check cannot grant permission after files or the inspection change.
4. **Record a separate declaration.** Once preparation passes, the operator must explicitly declare the decision to continue. The continuation endpoint replays the saved evidence again and checks freshness after replay before saving the declaration. The record says `operator_declared_continuation_not_observed`; it does not report an observed action.

Uploaded photo freshness and spacing remain operator declarations. Camera timestamps are client-reported. Byte hashes and replay check consistency with the saved evidence; they do not authenticate capture time, sign a record or independently prove the scene's uniqueness.

## Local API and records

These authenticated local control routes make no additional model requests:

- `POST /api/workflow/check` accepts the current `task_hash` and integer `revision`.
- `POST /api/workflow/continue` accepts the same binding and `confirm_operator_continuation: true`.

The normal browser flow records `review_origin: local_operator`. An assistant-origin declaration additionally requires `review_origin: assistant_under_user_authorization` and a direct user authorization reference. That reference is retained privately and is excluded from public snapshots. The interface identifies assistant declarations explicitly.

Declarations are written as new private `artifacts/local/interactive-*/workflow-*.json` files with mode `0600`. Each includes the task/revision binding, saved placement evidence, declaration origin and a content hash. New inspections invalidate the active continuation gate. Task changes require new evidence. An older declaration remains a historical record of intent and cannot approve a changed task or revision.

## Online video evidence

Licensed robot demonstrations can test the authored coffee placement contract and missing-evidence policy. Their timestamps and later movements belong to prerecorded demonstrations. They cannot establish that a current operator followed ProofLoop's repair or continuation request. Recorded benchmark results therefore cannot unlock this current-scene preparation gate.

The [source register](ONLINE-VIDEO-SOURCES.md) documents the recordings and evaluation limits. The [linked recovery record](RECOVERY.md) documents failure, proposed repair and subsequent observations. Recovery and preparation remain separate: a fresh passing placement can support the checkpoint without proving how that placement was achieved.

## Verification

Offline semantic and engine tests cover unreviewed or changed contracts, missing/UNKNOWN/FAIL checks, historical or expired evidence, expiration during replay, altered media after a successful check, stale revisions, injected replay flags, duplicate declarations and private authorization filtering. Authored test fixtures validate control behavior; they are not model-quality measurements or physical demonstrations.
