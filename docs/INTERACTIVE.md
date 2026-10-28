# Local interactive inspection

The workbench connects reviewed task contracts, SAM grounding, placement checks, typed missing-evidence requests, a bounded repair planner, linked recovery receipts and a preparation checkpoint. A historical owned recording can initialize a misplaced fork; it is labeled **Recorded reference**. A current placement can pass only from a newly submitted supported pair of images. A PASS does not establish how that placement was achieved. There is no model fallback.

## Start

Use Node 24 and Python 3.10 or newer for the official SAM parser. From the repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[sam]'
cd web
npm ci
npm run build
cd ..
.venv/bin/python -m proofloop.server --port 8790
```

Open <http://127.0.0.1:8790>. Keep `META_API_KEY` in the ignored `.env.local`; the browser never receives it. If the local owned-phone report and original files are available, the server checks their hashes and replays its wrong-region result. A fresh public clone starts with no evidence. The interface needs the Python server; Vite's development server by itself has no model backend.

## Define and review a task

Open **Task**, name one to six visually distinct target classes, give both regions distinct lowercase IDs such as `dispense` and `staging`, and describe the required placements using those IDs. Review the SAM object prompts or Spark locator descriptions, confirm the unique-target scene assumption, and choose **Draft checks with Spark**. This creates a locally validated draft without changing the current task or verdict.

Alternatively, choose **Use coffee video preset** for one transparent drinking glass in DISPENSE/STAGING. Confirm the declared unique-target assumption, then choose **Review authored coffee checks**. This fixed authored draft makes no model request. Editing its requirements, inventory, regions, concept or perception mode routes it through a new custom Spark draft. The starting rectangles describe the licensed fixed coffee camera; review them against your actual view. New region IDs receive neutral starting rectangles that also require calibration.

Review the exact executable checks, region IDs, perception mode and guided method version, then choose **Activate reviewed task** with the required confirmations. Activation invalidates previous evidence and capture tokens. Changing the requested placement, target descriptions, region IDs or perception mode requires a new reviewed contract. Renaming a region does not certify the old rectangle; activation invalidates the prior result and both rectangles must be reviewed against the new pair. A guided implementation version change also requires renewed review; an older draft or active contract cannot silently use the new method. The current editor compiles observable placement; it does not infer hidden contents, attachment, exact counts or step order.

The default **SAM concept prompts** mode uses at most two SAM calls per target for a fresh pair. Optional **Spark-guided SAM** adds one task-independent Spark localization call, then at most two box-prompted SAM calls per target. Spark receives concepts without desired placements, acceptance regions or ground-truth boxes. **Spark assigns semantic identity; SAM refines the boundary and does not independently prove identity.**

Current guided v2 uses a normalized 0–1000 coordinate contract before conversion to image pixels. It distinguishes uncertain identity, incomplete visibility and approximate boundary precision. Identity ambiguity or incomplete visibility skips that target; boundary imprecision alone may proceed to SAM refinement. Missing or ambiguous anchors remain unknown, and the approved SAM score threshold is unchanged. Guided uploads require upright JPEGs with orientation metadata removed and a longest side no greater than 1280 pixels.

## Follow a specific UNKNOWN request

When the result is UNKNOWN, the **Next inspection** panel names the target, affected view and missing support. Typed fields from the structured locator, bound SAM results and checker observations distinguish identity ambiguity, incomplete visibility, insufficient confidence, unsupported geometry/region boundaries, inconsistent views, insufficient distinct samples and expired evidence. Generated explanation text is not parsed to choose a policy branch.

Follow those instructions, then choose **Capture requested views**. Keep all required targets visible and the arrangement unchanged between the two entirely new views, at least one second apart. UNKNOWN requests observation; it does not authorize moving an unseen target. A missing detection proves neither absence nor the cause of a physical mistake. A supported wrong placement can instead produce a bounded repair proposal; its predicted move never becomes measured evidence.

The panel displays **0 / 2**, **1 / 2** or **2 / 2 follow-up pairs submitted**. The server consumes a submitted follow-up before model dispatch, including an inference error. Starting another inspection does not reset an unresolved chain. Two follow-ups that still leave insufficient evidence stop acquisition; further inspection/model calls are blocked until the task or capture method is deliberately reviewed. Earlier image hashes cannot satisfy a fresh request. Unsupported predicates require another validated observation method rather than more sampling, and confidence thresholds stay fixed.

A new reviewed task/session is a new scope, not retroactive success for the old result. See [evidence acquisition](ACQUISITION.md) for policy and source constraints.

## Explore existing videos

Open **Video lab** to inspect the reviewed online comparisons. Choose a run and case to see the source pair, calibrated zones, authored expectation and each method's verdict/explanation. The lab is read-only and opening it makes no model calls. It requires the private source media and a matching reviewed public summary; those files are intentionally absent from a fresh public clone.

The corpus contains eight original coffee recordings and 16 unique authored placement pairs. Historical results remain visible: the initial text-SAM pilot matched **0/8** labels (all UNKNOWN), versus Spark **7/8**; guided v1 matched **2/4** on episodes 000004/000005, versus text SAM **0/4** and Spark **4/4**.

Choose **Normalized grounding — development replay** for v2's **4/4** result on those already tested sources. Choose **Normalized grounding — new sources** for episodes 000006/000007: guided **3/4**, one UNKNOWN, versus text SAM **0/4** and Spark **4/4**. V2 covers these four replay and four fresh pairs only. The remaining partly occluded case lacks a first-view mask and its second-view score of **0.746** misses the unchanged **0.8** threshold. It remains UNKNOWN.

Choose **Specific evidence requests — fixed future pairs** to see the new typed request/replay chain. Six pairs were selected before inference; later pairs are strictly after the previous pair and do not reuse its images. Guided matched **4/6**, Spark **6/6**. Guided episode 000000 progresses UNKNOWN → UNKNOWN → PASS in two permitted follow-ups. Episode 000001 remains UNKNOWN when the source ends. Spark resolves both on the first follow-up; the unnecessary second pair is explicitly excluded from policy cost. The terminal third timeline remains UNKNOWN for both. These reuse old coffee episodes and add no independent recordings. [Follow-up report](../artifacts/feasibility/fixed-followup-20261028.md).

Choose **New task transfer — candy collection** for one different manipulation recording. Both methods matched **2/2**, UNKNOWN → PASS. The early locator's packet/label-area anchors were rejected by SAM scores below the unchanged threshold; this did not independently solve semantic identity. There is no FAIL control. [Transfer report](NEW-TASK-TRANSFER-20261028.md).

A public clone can use the [reproduction CLI](REPRODUCING.md) to download a small pinned source and extract fixed frames without an API key, model calls or automatic contract approval. Preparing a draft does not install the original private lab runs or reproduce their live inference outputs.

These are scoped placement checks on normal demonstrations, not natural-error detection, action recognition or validated physical recovery. V2 is a partial improvement and has not outperformed Spark. See the [guided-fix report](../artifacts/feasibility/guided-fix-20261028.md) and [historical online report](../artifacts/feasibility/online-video-20261028.md).

## Phone workflow

1. Click **Start repair**. This clears the current verdict immediately. Move the fork wholly into the handwritten LEFT area; leave the spoon wholly in RIGHT.
2. Keep the paper flat and the camera fixed. Include all of both utensils and both labeled areas. Remove your hand and avoid glare. Take two new still photos at least one second apart after starting the repair.
3. Transfer both photos to this computer and choose **Upload two photos**. Use upright JPEG or PNG files, each at most 5 MiB, with matching dimensions and orientation. HEIC is not supported in this first version; export upright JPEG/PNG before uploading. A laptop camera is an alternative: **Use camera** requires explicit permission and captures a pair after you position it.
4. Mark **LEFT** and **RIGHT** on the first image. Confirm the regions match both photos and the scene has exactly one fork and one spoon. For uploaded photos, confirm you took the pair after starting this repair, at least one second apart.
5. Click **Verify correction**. For the default two-object concept mode, the backend makes at most four SAM requests with no automatic retries; guided mode adds one Spark localization request. It checks actual returned masks against the reviewed regions. A wrong location fails; ambiguity or missing evidence stays unknown.

The earlier portrait recording has LEFT above the divider and RIGHT below it. Your new camera orientation may differ: the handwritten labels determine the regions. Avoid the divider and image edges so each utensil is clearly inside its area.

After any movement, click **New inspection** before capturing again. Evidence from fresh captures ages out after 30 seconds, including earlier FAIL results and their move proposals. It describes the inspected images, not continuous monitoring.

## Check preparation before a declared continuation

Open **Workflow**. The **Placement preparation checkpoint** has four explicit steps: review the exact contract, support all current placement checks, replay the saved evidence, then record a separate declaration of intent to continue. A historical reference, UNKNOWN, FAIL, missing pair, expired evidence, running inspection or error blocks continuation.

Once a fresh inspection supports every required placement, choose **Check preparation evidence**. This rehashes saved sources/results and replays the checks without new model calls. The task hash and inspection revision bind the replay. Check the declaration confirmation and choose **Record continuation declaration** only if you intend to proceed. The server replays the evidence and rechecks freshness again before writing the record.

The record explicitly says continuation was **declared, not observed**. It cannot verify dispensing, hidden contents, quality, physical safety, execution of a proposed move or a complete coffee procedure. New inspections and task changes invalidate the active gate; an old declaration stays historical. Recorded Video lab results cannot unlock this current-scene checkpoint. See [workflow scope and API](WORKFLOW.md).

## Check a linked recovery

**Start repair** saves the supported earlier failure and proposed move before clearing current evidence. When the earlier result is UNKNOWN, the request is recorded as an inspection with no asserted physical repair. Later qualifying views can record a fresh goal PASS, a continued failure, UNKNOWN or an error under the same task. Activating a different task cancels a pending recovery.

Open **Evidence log → Linked recovery record**, then choose **Check local evidence**. ProofLoop rehashes saved sources, masks, task/version bindings and immutable request parents, and recomputes checks and the bounded repair plan. Guided receipts also replay the saved source-bound locator. This makes no model calls. A valid request record alone does not certify a correction; a valid completed replay can consistently describe a failed or UNKNOWN result.

The read-only `GET /api/recovery` endpoint returns only a safe summary and verification result. Raw records and their private source references stay local. Request and later-result JSON files are write-once, with hashed parent links checked to a 32-link budget. See [recovery receipts](RECOVERY.md) for exact offline CLI commands and packet requirements.

A closed successful receipt remains historical when current evidence expires or a later inspection runs. Replay success does not refresh the live result. Hashes are not signatures, trusted timestamps or proof of API authenticity, capture freshness or physical motion. If the receipt fails verification, its recovery claim is removed; a separately supported captured placement verdict is kept distinct.

## Evidence and limits

The evidence log exposes the source, revision, reasons and invalidation trace. Upload timing is an operator declaration; camera timing is client-reported. Tokens and known image hashes reject accidental reuse but cannot prove capture time or defeat a dishonest operator who modifies an old photo. The unique-target assumption is explicitly confirmed, not inferred from an absence of detections.

Private uploads, raw responses and run records remain in ignored `artifacts/local/interactive-*` directories. The server binds only to IPv4 loopback and validates Host, Origin and a per-server mutation token. It serves compiled assets and explicitly registered media, never the repository directory. It has no remote hosting or phone LAN access in this version.

The full local Python 3.12 suite passed **450/450 tests**, including loopback HTTP checks, and the strict production frontend build passed. Offline injected-client tests exercise pass/fail/unknown, linked recovery and tamper rejection, typed requests, finite acquisition budgets, portable contracts and preparation/continuation gating. They are labeled test fixtures and are never substituted in the running app. Browser QA of an authored offline fixture passed linked-record replay, rejected altered media, produced no app console errors and showed no horizontal overflow at 390 pixels.

The initial phone observation experiment used real Meta models. A new person following the displayed proposal and supplying fresh corrective captures is still pending; these software checks do not establish physical interactive closure. The new typed acquisition and authored preparation gates validate software control behavior; semantic identity, broader procedures, a walkthrough by another person and physical operation remain review priorities.
