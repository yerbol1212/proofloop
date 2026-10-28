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

**545/545 Python 3.12 tests passed, and the strict production frontend build passed.** Publication CI is checked separately after pushing the milestone. These software checks do not establish perception accuracy.

See the [review and remaining work](docs/AI-SYSTEMS-REVIEW-20261028.md), [feasibility evidence](docs/FEASIBILITY.md), [three-minute demo script](docs/DEMO-SCRIPT.md), [source credits](docs/ONLINE-VIDEO-SOURCES.md) and [event contributions](CONTRIBUTIONS.md). Software tests exercise control behavior separately from model-quality measurements.

The [comparison and implementation plan](docs/COMPETITIVE-PLAN-20261028.md) identifies practices adapted from REFLECT, Tulip, Retrocausal, SayCan, KnowNo and Rerun. This is an inspectable prototype verification layer; it has not been compared experimentally with those systems or validated in production.

Two [opposite-region contract controls](artifacts/feasibility/contract-stress-20261028.md) now reuse the exact candy and coffee images: both systems change PASS to FAIL under the changed requirement. Guided re-grounding uses saved perception with zero new calls; the two fresh Spark checks cost an estimated $0.018421. Original inference costs remain recorded. These authored task changes are development controls, not new physical failures or independent sources.

The separate [candidate identity challenge](docs/SEMANTIC-CHALLENGE-RESULTS-20261028.md) is **not enabled for task acceptance**. Of four selected development cases, two returned classifications: one supported a clear glass and one withheld support for visible candy. The misleading packet/label case never produced a classification. Five total attempts included three incomplete requests, estimated at $0.076588. The experiment exposes a completion and positive-retention problem; it does not demonstrate successful semantic distractor rejection. Open **Video lab → Development experiments** to inspect both controls without model calls.

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
