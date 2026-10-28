# Local interactive inspection

The workbench connects editable reviewed task contracts, SAM grounding, placement checks, targeted inspection requests and a bounded repair planner. A historical owned recording can initialize a misplaced fork; it is labeled **Recorded reference**. A correction is verified only from a newly submitted pair of images. There is no model fallback.

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

Open **Task**, name two to six visually distinct target classes, and describe their required placement in LEFT and RIGHT. Review the SAM object prompts or Spark locator descriptions, confirm the unique-target scene assumption, and choose **Draft checks with Spark**. This creates a locally validated draft without changing the current task or verdict.

Review the exact executable checks, perception mode and guided method version, then choose **Activate reviewed task** with the required confirmations. Activation invalidates previous evidence and capture tokens. Changing the requested placement, target descriptions or perception mode requires a new reviewed contract. A guided implementation version change also requires renewed review; an older draft or active contract cannot silently use the new method. The current editor compiles observable placement; it does not infer hidden contents, attachment, exact counts or step order.

The default **SAM concept prompts** mode uses at most two SAM calls per target for a fresh pair. Optional **Spark-guided SAM** adds one task-independent Spark localization call, then at most two box-prompted SAM calls per target. Spark receives concepts without desired placements, acceptance regions or ground-truth boxes. **Spark assigns semantic identity; SAM refines the boundary and does not independently prove identity.**

Current guided v2 uses a normalized 0–1000 coordinate contract before conversion to image pixels. It distinguishes uncertain identity, incomplete visibility and approximate boundary precision. Identity ambiguity or incomplete visibility skips that target; boundary imprecision alone may proceed to SAM refinement. Missing or ambiguous anchors remain unknown, and the approved SAM score threshold is unchanged. Guided uploads require upright JPEGs with orientation metadata removed and a longest side no greater than 1280 pixels.

When the result is UNKNOWN, follow **Next inspection** before attempting a physical repair. It asks for evidence appropriate to the grounding reason—for example, clearer region boundaries, an unambiguous target or a stable fresh pair. A missing detection does not prove absence or identify an unseen physical mistake. A supported wrong placement can produce a repair proposal; the predicted move never becomes measured evidence.

## Explore existing videos

Open **Video lab** to inspect the reviewed online comparisons. Choose a run and case to see the source pair, calibrated zones, authored expectation and each method's verdict/explanation. The lab is read-only and opening it makes no model calls. It requires the private source media and a matching reviewed public summary; those files are intentionally absent from a fresh public clone.

The corpus contains eight original coffee recordings and 16 unique authored placement pairs. Historical results remain visible: the initial text-SAM pilot matched **0/8** labels (all UNKNOWN), versus Spark **7/8**; guided v1 matched **2/4** on episodes 000004/000005, versus text SAM **0/4** and Spark **4/4**.

Choose **Normalized grounding — development replay** for v2's **4/4** result on those already tested sources. Choose **Normalized grounding — new sources** for episodes 000006/000007: guided **3/4**, one UNKNOWN, versus text SAM **0/4** and Spark **4/4**. V2 covers these four replay and four fresh pairs only. The remaining partly occluded case lacks a first-view mask and its second-view score of **0.746** misses the unchanged **0.8** threshold. It remains UNKNOWN.

These are scoped placement checks on normal demonstrations, not natural-error detection, action recognition or validated physical recovery. V2 is a partial improvement and has not outperformed Spark. See the [guided-fix report](../artifacts/feasibility/guided-fix-20261028.md) and [historical online report](../artifacts/feasibility/online-video-20261028.md).

## Phone workflow

1. Click **Start repair**. This clears the current verdict immediately. Move the fork wholly into the handwritten LEFT area; leave the spoon wholly in RIGHT.
2. Keep the paper flat and the camera fixed. Include all of both utensils and both labeled areas. Remove your hand and avoid glare. Take two new still photos at least one second apart after starting the repair.
3. Transfer both photos to this computer and choose **Upload two photos**. Use upright JPEG or PNG files, each at most 5 MiB, with matching dimensions and orientation. HEIC is not supported in this first version; export upright JPEG/PNG before uploading. A laptop camera is an alternative: **Use camera** requires explicit permission and captures a pair after you position it.
4. Mark **LEFT** and **RIGHT** on the first image. Confirm the regions match both photos and the scene has exactly one fork and one spoon. For uploaded photos, confirm you took the pair after starting this repair, at least one second apart.
5. Click **Verify correction**. For the default two-object concept mode, the backend makes at most four SAM requests with no automatic retries; guided mode adds one Spark localization request. It checks actual returned masks against the reviewed regions. A wrong location fails; ambiguity or missing evidence stays unknown.

The earlier portrait recording has LEFT above the divider and RIGHT below it. Your new camera orientation may differ: the handwritten labels determine the regions. Avoid the divider and image edges so each utensil is clearly inside its area.

After any movement, click **New inspection** before capturing again. A passing snapshot ages out after 30 seconds. It describes the inspected images, not continuous monitoring.

## Evidence and limits

The evidence log exposes the source, revision, reasons and invalidation trace. Upload timing is an operator declaration; camera timing is client-reported. Tokens and known image hashes reject accidental reuse but cannot prove capture time or defeat a dishonest operator who modifies an old photo. The unique-target assumption is explicitly confirmed, not inferred from an absence of detections.

Private uploads, raw responses and run records remain in ignored `artifacts/local/interactive-*` directories. The server binds only to IPv4 loopback and validates Host, Origin and a per-server mutation token. It serves compiled assets and explicitly registered media, never the repository directory. It has no remote hosting or phone LAN access in this version.

The current suite has 334 passing Python 3.12 tests. Earlier Python 3.9/3.12 CI passed at b6a5ebe. Offline injected-client tests exercise pass/fail/unknown and failure paths. They are labeled test fixtures and are never substituted in the running app. The initial phone observation experiment used real Meta models; a new person-following-the-proposal correction needs fresh user captures to establish physical interactive closure.
