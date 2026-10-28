# Linked recovery receipts

A recovery receipt connects a recorded result, a requested repair or inspection, and later image evidence under the same task. It makes the saved reasoning independently replayable without another model call. It does **not** establish that a person or robot executed the proposed motion.

## Create and inspect a record

When a supported FAIL has a physical move proposal, **Start repair** records the earlier checks, source references and proposed actions before invalidating the current evidence. A request made from UNKNOWN creates an **inspection** record, with no physical repair actions. Expired evidence from a fresh capture also requires inspection rather than authorizing an old move proposal. A historical recording remains explicitly historical.

Submit two new qualifying views after the request. The later result updates the linked record as follows:

| Outcome | Meaning |
| --- | --- |
| `awaiting_fresh_evidence` | The request is recorded; no later qualifying result is available. |
| `fresh_goal_verified_after_requested_repair` | New image evidence passes the requested goal after a supported repair request. Execution of the move remains unproved. |
| `fresh_goal_verified_after_inspection` | New image evidence passes after an inspection request. No physical repair is asserted. |
| `still_failed` | New image evidence still supports a failed required placement. |
| `needs_evidence` | The later result is UNKNOWN and cannot verify recovery. |
| `error` | Inspection, receipt storage or evidence replay could not support a recovery claim. |
| `cancelled_task_change` | A pending recovery was closed because a different task was activated. |

Open **Evidence log → Linked recovery record** to see the earlier checks, proposed repair or requested inspection, and later checks. **Check local evidence** replays the saved record. A successful check of a request alone says **Request record checked**; it does not imply a passing correction. A completed replay can say **Saved evidence replay passed** even when the recorded outcome is still failed or UNKNOWN: `valid` describes consistency of the record, not success of the physical task.

The same local replay is exposed by read-only `GET /api/recovery`. It returns a sanitized `recovery` summary and a `verification` result with `valid`, `scope`, `receipt_hash`, `outcome`, `reason` and `physical_action_execution_verified: false`. The endpoint makes no model calls and returns no private paths, encoded media, raw provider outputs or IDs, capture tokens, credentials or authorization text.

## What the verifier checks

Each receipt binds the executable protocol and task hash, object descriptions, perception mode and guided method version. The before and after evidence each bind their own reviewed capture rectangles, source timestamps and image hashes. Different camera calibrations are preserved separately; they cannot silently replace the earlier calibration.

Offline replay checks:

- Image bytes, referenced result files, parent records and canonical contract/receipt hashes.
- Two distinct source hashes, declared timestamp spacing, linked revisions and no reuse of the earlier views as a correction.
- Completed source-bound SAM records, source-image dimensions, decoded mask raster metadata and digests, and conservative grounding against the saved rectangles.
- For guided mode, the saved source-bound locator, method version, target identity/full-visibility gates and corresponding actual SAM results. Spark still assigns semantic identity; SAM boundary refinement does not independently prove it.
- Exact check coverage and unique IDs, recomputed per-check and aggregate results, and consistency of the claimed outcome.
- For a repair request, the bounded declared-action planner's action IDs and displayed steps recomputed from the earlier evidence. A `move-` prefix alone is insufficient.

The verifier evaluates historical evidence at the evaluation time saved in the record. It does not turn an old image into current evidence.

## Private storage and immutable parents

Records live under the ignored directory `artifacts/local/interactive-<session>/`, using filenames such as `recovery-<receipt-id>-0000.json`. Files are created once with private permissions rather than overwritten. A later result refers to its immutable request parent by both file-byte hash and canonical receipt hash. Parent verification also checks continuity of the task, earlier evidence, proposed plan and revisions, with a budget of 32 parent links. Missing, altered or excessive chains fail verification.

Keep all referenced local images, SAM results, guided locator records and parent receipts. A sanitized browser summary alone is not a complete offline packet. Referenced files must remain inside the project’s private `inputs/local` or `artifacts/local` directories. They are intentionally absent from a fresh public clone.

A successful closed receipt remains a historical record when a later unrelated inspection runs. Its success does not keep the live inspection fresh: current capture evidence expires after 30 seconds. Changing the task cancels a **pending** recovery; it does not rewrite a previously closed result as evidence for the new task.

## Run offline verification

From the repository root, list locally saved receipts:

```sh
rg --files --no-ignore artifacts/local -g 'recovery-*.json'
```

Replace the uppercase placeholders below with an actual listed filename. Select the later sequence for the outcome you want to inspect; retain its parent files.

```sh
.venv/bin/python -m proofloop.recovery \
  'artifacts/local/interactive-SESSION/recovery-RECEIPT-0001.json'
```

The command prints only a safe verification result and exits with `0` for a consistent record or `1` for failed verification. A valid requested record is not a verified goal. No API key or network call is needed for this replay.

If the private project tree was copied to another location, preserve the relative layout and provide its root:

```sh
.venv/bin/python -m proofloop.recovery \
  '/absolute/path/to/copied-project/artifacts/local/interactive-SESSION/recovery-RECEIPT-0001.json' \
  --root '/absolute/path/to/copied-project'
```

## Meaning and limits of a valid receipt

`valid: true` means the referenced saved artifacts and historical results replay consistently under the recorded checks. Hashes detect changes relative to that record. They are **not** signatures, trusted timestamps, proof of Meta API authenticity, proof that a source was owned or licensed, or independent confirmation of the reviewer’s identity.

Uploaded-photo freshness and spacing remain operator declarations. Camera capture times are client-reported. Scene uniqueness and calibration are declared assumptions. Editing an old photo can defeat a simple known-byte reuse guard. Two pictures and a correct resulting placement cannot prove the proposed motion occurred or caused the change.

If replay fails, ProofLoop removes the verified **recovery** claim and shows the evidence record as unverified. That does not automatically change a separately supported captured placement verdict. For example, altering an earlier receipt parent can invalidate the linked recovery while the current image pair still supports PASS. The two claims are displayed separately.

## Validation and remaining work

At this milestone, **362 Python 3.12 regression tests pass**. Recovery tests cover wrong repairs, UNKNOWN results, task cancellation, reused views, changed media/masks/contracts/plans, duplicate checks, bad revisions and altered parents. They also check guided locator replay and safe exports.

Browser checks using an explicitly authored offline fixture exercised a passing linked record and rejection of altered media, with no app console errors and no horizontal overflow at 390 pixels. These checks validate the software flow; they are not new live-model measurements or physical execution trials.

The initial owned phone experiment used real Meta models on a prerecorded observation loop. A person following a displayed proposal and supplying fresh corrective captures is **still pending**. Other review priorities—including more capable evidence acquisition and authored workflow gates—also remain pending; recovery receipts do not add action recognition or sequential procedure verification.

The current receipt path was also checked against the genuine historical misplaced-fork reference. Starting a repair preserved and replayed its original Meta-backed source evidence and bounded plan. [Safe request check](../artifacts/feasibility/recovery-request-20261028.json) records `scope: requested` and awaiting new evidence. This made no new model calls and did not create or claim a fresh physical correction.
