# ProofLoop

Executable checks and evidence-linked correction for bounded physical workflows.

ProofLoop turns a confirmed procedure into inspectable checks, evaluates observations as **pass**, **fail** or **unknown**, and searches a small declared action model for a correction. Corrections require fresh observations before the result changes. Muse Spark drafts checks and SAM 3.1 grounds image observations.

## Current status

This is the event-built feasibility implementation for the Meta Global AI Developer Hackathon. The workbench turns an explicit placement requirement into a reviewed task, requests evidence it cannot yet support, and keeps later observations separate from proposed actions.

- **Portable task contracts:** one to six visually distinct target classes, two named regions, exact draft review and invalidation when the active contract changes. The authored coffee preset uses one glass in DISPENSE/STAGING; custom drafts use Muse Spark.
- **Bounded UNKNOWN follow-up:** typed object/view gaps request full visibility, distinguishing features, supported boundaries, confidence or a stable pair. The server permits at most two follow-up submissions in an unresolved chain, rejects reused images and stops when the budget or recorded timeline is exhausted.
- **Placement preparation checkpoint:** reviewed requirements, current PASS and replayed evidence gate a separately recorded operator continuation declaration. That declaration reports intent; it does not verify dispensing or a complete procedure.
- **Inspectable evidence:** the read-only Video lab preserves source attribution, actual model comparisons and fixed future-evidence trajectories. Linked recovery receipts replay task, plan, parent, media and mask consistency without asserting executed motion.
- **Grounded explanations and failure review:** each explanation binds the reviewed requirement, supported observation and permitted next step without inventing a physical cause. Video lab shows coverage, accuracy among decided cases, unsupported acceptance and selectable model disagreements with explicit denominators.
- **Durable inspection service:** asynchronous SQLite jobs bind the exact reviewed task, protocol, session and revision; duplicate requests reuse the job, expired workers cannot retry inference, and verified historical results remain separate from current evidence. A persistent inspection-request allowance survives clients and restarts. See [runtime boundaries](docs/DURABLE-INSPECTIONS.md).
- **Read-only MCP bridge:** four versioned tools expose capabilities, one known job, verified events and request accounting to local agent clients. It never submits images or authorizes physical action. See [MCP setup and scope](docs/MCP.md) and the [Meta infrastructure research](docs/INFRASTRUCTURE-REVIEW-20261028.md).
- **Public replay inspector:** Video lab includes a key-free walkthrough of four tracking recordings and 32 reviewed checkpoints, even with empty private storage. Compare the three recomputed policies, source-bound confirmations, false warning and unscored outcome. [Run the public walkthrough](docs/PUBLIC-REPLAY.md). Cached inputs contain no original video pixels or new inference.
- **Integrated service infrastructure:** shared compilation/inspection/tracking admission, lost-response recovery, bounded operational traces/readiness, a separate durable video diagnostic API and Python client, portable zero-call policy replay and installed-wheel CI. See [implementation, commands and limits](docs/INFRASTRUCTURE-IMPLEMENTATION-20261029.md). Tracking diagnostics remain separate from placement acceptance.

Muse Spark drafts custom checks and supplies task-independent object anchors in guided mode; actual SAM 3.1 calls refine those anchors. Spark assigns semantic identity. A prompted mask does not independently prove that identity, and unsupported results remain UNKNOWN. The runtime uses Meta models; Codex assists development.

Recent real Meta comparisons remain separate cohorts:

| Evaluation | Guided Spark → SAM | Spark alone | Scope |
| --- | ---: | ---: | --- |
| [Fresh coffee source pairs](artifacts/feasibility/guided-fix-20261028.md) | 3/4 | 4/4 | Two new episodes after the v2 fix; one guided UNKNOWN |
| [Temporal stress](artifacts/feasibility/temporal-stress-20261028.md) | 7/8 | 8/8 | Four previously exposed recordings |
| [Fixed future-evidence pairs](artifacts/feasibility/fixed-followup-20261028.md) | 4/6 | 6/6 | Three reused recordings; typed requests and chronological stopping |
| [Candy task transfer](docs/NEW-TASK-TRANSFER-20261028.md) | 2/2 | 2/2 | One different task/scene; UNKNOWN and PASS only |

In the fixed follow-up replay, guided resolved one of two trajectories with later evidence; Spark resolved both. A third terminal trajectory remained UNKNOWN for both. The candy result adds one recording from another manipulation task, not proof of broad generalization or independently verified identity. The [historical text-SAM 0/8 and guided-v1 2/4 results](artifacts/feasibility/online-video-20261028.md) remain unchanged. Adding SAM has not shown a label-matching advantage over Spark alone.

These are scoped image-zone checks on normal prerecorded demonstrations. A later PASS does not prove that a robot followed ProofLoop's request. The [owned phone recording](artifacts/feasibility/phone-session-20261028.md) produced PASS → FAIL → UNKNOWN → PASS, but was recorded before its planner proposal. Fresh physical operation following the displayed request, broader procedure verification and practical user demand remain unvalidated.

**1,278/1,278 Python 3.12 tests passed; the Python 3.9 suite passed with two expected optional-parser skips. The strict production frontend build passed.** [Publication CI](https://github.com/yerbol1212/proofloop/actions/workflows/tests.yml) runs both Python versions, frontend checks and clean-wheel installation outside the checkout. These software checks do not establish perception accuracy.

See the [review and remaining work](docs/AI-SYSTEMS-REVIEW-20261028.md), [feasibility evidence](docs/FEASIBILITY.md), [three-minute demo script](docs/DEMO-SCRIPT.md), [source credits](docs/ONLINE-VIDEO-SOURCES.md) and [event contributions](CONTRIBUTIONS.md). Software tests exercise control behavior separately from model-quality measurements.

The [comparison and implementation plan](docs/COMPETITIVE-PLAN-20261028.md) identifies practices adapted from REFLECT, Tulip, Retrocausal, SayCan, KnowNo and Rerun. This is an inspectable prototype verification layer; it has not been compared experimentally with those systems or validated in production.

Two [opposite-region contract controls](artifacts/feasibility/contract-stress-20261028.md) now reuse the exact candy and coffee images: both systems change PASS to FAIL under the changed requirement. Guided re-grounding uses saved perception with zero new calls; the two fresh Spark checks cost an estimated $0.018421. Original inference costs remain recorded. These authored task changes are development controls, not new physical failures or independent sources.

The [historical candidate challenge](docs/SEMANTIC-CHALLENGE-RESULTS-20261028.md) completed only two of four cases and remains preserved. The [low-reasoning v2 check](docs/SEMANTIC-V2-RESULTS-20261028.md) now completed **4/4** on the same inputs, matching three authored identity annotations and withholding the misleading candy anchor. On six manually anchored real-video cases from two additional recordings, v2 completed **6/6**, retained both targets, and incorrectly supported one of four distractors. A separate crop-only development retest withheld that battery candidate while retaining both targets; the original **5/6** result remains unchanged. These diagnostics are **not enabled for task acceptance**. Open **Video lab → Development experiments** to inspect each cohort, source view, candidate crop and failure without model calls.

The [new video-tracking experiment](docs/TRACKING-RESULTS-20261029.md) tracked **100 sampled frames from two new licensed recordings** and compared 16 prospectively labeled checkpoints. Manually initialized SAM retained 7/7 clear targets but supported 3/9 occluded/unclear states; the semantic policy retained only 1/7 clear targets and still supported 1/9 unclear states. A separate cached mask-based correction removed 57 false rectangle-overlap warnings without changing those checkpoint results or making new API calls. Initialization and compute differ from the still baselines. This is experimental candidate tracking, outside live placement acceptance. A separate bounded-reasoning repair completed the original incomplete baseline pair; its original failure and denominator remain preserved.

The separate [full-frame visibility comparison](docs/VISIBILITY-RESULTS-20261029.md) improved direct clear-target retention from **3/7 to 5/7** on exposed frames, with unclear views called clear falling **2/9 to 0/9**. On two new recordings, clear retention improved **5/6 to 6/6**, while unclear support increased **0/10 to 1/10** at a borderline gripper contact. The unchanged two-view policy withheld that view; downstream new-recording retention stayed **2/6** in both versions. The component gain is not an overall tracking solution. Its 50 new calls cost an estimated **$0.415522**, with original results, labels and thresholds preserved.

The subsequent [contact/retention comparison](docs/GRIPPER-VETO-RESULTS-20261029.md) adds current-view confirmation and negative-only gripper warnings. Combined retention improves **2/6 → 3/6** on exposed recordings and **3/7 → 4/7** on two additional recordings, with **0/10 and 0/9 unclear checkpoints supported**. Both misleading contact candidates become UNKNOWN. The warning also withholds one clear view; temporal confirmation alone retains 4/6, while the failed strict gripper gate retains 0/6. The additional cohort includes one unscored network outcome in its seven-positive denominator. The warning comparison is a post-inspection cached development retest with zero new calls, outside live placement acceptance. Earlier trials and failures remain published.

## Open the interactive workbench

Build the frontend with `cd web && npm ci && npm run build`, then from the repository run `.venv/bin/python -m proofloop.server --port 8790` and open <http://127.0.0.1:8790>. Python 3.10+ with `.[sam]`, Node 24, and a locally configured Meta key are needed for live inspection. The [interactive guide](docs/INTERACTIVE.md) includes setup and exact phone capture instructions. Use **Task** to draft and review requirements, **Inspection** to follow specific evidence requests, **Workflow** to check the placement preparation gate, **Evidence log** for provenance, and **Video lab** to explore recorded comparisons. Opening the lab makes no model calls. The interface labels historical evidence and requires two new views after evidence is invalidated. A public clone has no private recordings or raw lab runs. The [reproduction guide](docs/REPRODUCING.md) prepares a small pinned online-video example with zero API calls and no automatic approval; paid Meta inference is a separate command.

The [session guide](docs/SESSIONS.md) connects timestamped original frames to placement checks, evidence invalidation, a hypothetical repair plan and fresh verification. The first recorded phone run passed this bounded observation loop. The current [recording preset](examples/cutlery/README.md) uses one fork in LEFT and one spoon in RIGHT, with the fork moved and corrected during the clip. Earlier [generated-image tests](artifacts/feasibility/live-meta-20261026.md) and synthetic oracle diagnostics retain their separate provenance.

## Run the offline checks

Python 3.9 or newer is sufficient for the dependency-free checker and HTTP adapter:

```bash
python3 -m unittest discover -s tests -v
python3 -m proofloop.feasibility --output artifacts/feasibility
```

The diagnostic report records the exact cases and evidence origin. It is not an industrial accuracy benchmark.

## Run real model access checks

Copy `.env.example` to `.env.local` and save `META_API_KEY` locally. The local file is ignored by Git. Meta account billing requirements must be satisfied. No key belongs in source files, screenshots, chat or public artifacts.

For an editable local password form, run `python3 tools/key_setup.py`, open the printed loopback URL, save the key, then stop the server with Ctrl+C. The key stays in `.env.local`.

```bash
python3 -m proofloop.smoke --models-only
python3 -m proofloop.smoke --spark-only
python3 -m proofloop.smoke --image /absolute/path/to/owned-image.png --concept fork
```

The smoke runner calls `https://api.meta.ai/v1` with `muse-spark-1.3` and `sam-3.1`. SAM mask decoding uses Meta's official parser and requires Python 3.10 or newer. In a Python 3.10+ virtual environment, install `pip install '.[sam]'` before the image check. The project development environment uses Python 3.12 and parser version 0.1.0. Spark-only access checks and the deterministic core remain compatible with Python 3.9.

Each request has bounded output/size/time and a local attempt budget; estimated cost is not a provider billing stop. Unreviewed responses and source media remain under ignored local directories. Missing access produces an explicit error, never a substituted model result.

## Draft a protocol with Spark

```bash
python3 -m proofloop.compiler \
  --requirements 'Exactly one fork in the left region and exactly one spoon in the right region.' \
  --objects fork spoon --regions left right
```

The compiler calls Spark and validates the resulting protocol against the allowed inventory and checker vocabulary. Exact count is the default predicate mode; visibility and location are explicit alternatives. It rejects unsupported or ambiguous requirements and does not infer mandatory step order from a successful video. A draft is always unapproved and saved locally for explicit review; it is not immediately used to judge a run. The workbench requires separate activation of the exact reviewed contract and records reviewer provenance. Temporal checks and actions currently come from explicit authored specifications.

For positive placement checks, explicitly select location mode and supply the unique target objects confirmed by the operator:

```bash
python -m proofloop.compiler \
  --requirements 'The unique fork must be located in left and the unique spoon in right.' \
  --objects fork spoon --regions left right \
  --predicate-kind location --unique-objects fork spoon
```

This contract says the scene contains one visually distinct target instance of each supplied class. The model cannot create that assumption. A stable positive detection in the wrong region can fail the placement check; zero or multiple detections stay unknown. The [initial tool-scene live location draft](artifacts/feasibility/live-location-20261026.json) matched its synthetic test requirements and remains unapproved. The current [fork/spoon preset](examples/cutlery/README.md) is authored; the separate [live SAM cutlery probe](artifacts/feasibility/cutlery-concepts-20261027.md) tests concepts on one generated view. Visibility mode is also available with `--predicate-kind visible`. Exact counts are never silently converted into placement.

Two initial count compilation drafts matched the requested requirements, including an edit that swapped required regions. They remain unapproved. Mocked compiler tests additionally exercise contracts and rejection behavior.

## Generated scene inputs

The [synthetic fixture set](inputs/synthetic/README.md) includes correct, wrong-region and occluded conditions, plus a second correct variant. Exact prompts and source hashes are included. These generated images exercised real API transport, official mask decoding and grounding; they are not camera observations or a completed physical repair.

```bash
python -m proofloop.live_scenes --output artifacts/local/my-new-scene-run
```

This makes at most eight SAM calls with no retries. Reports and raw outputs stay local. Use a new output directory, or explicitly `--resume` to reuse matching completed responses. The harness labels reuse and checks source hashes.

`proofloop.grounding` connects parsed SAM image masks to approved checker predicates. It preserves provenance, requires distinct supporting image hashes, and returns unknown for missing/ambiguous perception. Positive detections supply visibility or count lower bounds, never automatic proof of absence or an exact count.

The original live scene harness uses a separate authored visibility protocol. Location predicates connect explicit unique-target placement requirements to positive SAM evidence. Exact-count checks still need a justified completeness contract before this perception path can certify them.

## Scope

Start with a fixed camera, a visible tray and distinct macro objects. The current checker tests declared observable properties. It cannot certify hidden contents, attachment, torque, electrical continuity or physical safety. A minimum repair is minimum cost within the supplied finite action model.

The source of each fact, specification version, observation epoch and freshness matter. Occlusion and possible content changes invalidate earlier evidence. A new attempt cannot retroactively fix the record of an earlier one.

## Development

Only project code, reviewed fixtures and public-safe results are published. Owned camera media, local credentials, account information and private research remain local. Commits identify event-created work and external components. Build milestones are pushed after relevant checks pass.
