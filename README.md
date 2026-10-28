# ProofLoop

Executable checks and evidence-linked correction for bounded physical workflows.

ProofLoop turns a confirmed procedure into inspectable checks, evaluates observations as **pass**, **fail** or **unknown**, and searches a small declared action model for a correction. Corrections require fresh observations before the result changes. Muse Spark drafts checks and SAM 3.1 grounds image observations.

## Current status

This repository is the feasibility implementation for the Meta Global AI Developer Hackathon. The local workbench now supports **editable reviewed task contracts**, targeted requests for better evidence, fresh correction checks and a read-only **Video lab**. Spark drafts placement checks for an explicit inventory; activating the reviewed contract invalidates previous evidence. Unknown observations prompt inspection before a physical repair is proposed. Optional Spark-guided SAM now uses normalized coordinate grounding and separates identity, visibility and boundary uncertainty. Its method version is bound to task review; Spark-assigned identity remains explicit.

The online corpus contains **eight original LeRobot coffee recordings and 16 unique authored placement pairs**. The original pilot remains **text SAM 0/8** label matches (all UNKNOWN), versus **Spark 7/8**. Historical guided v1 matched **2/4** on episodes 000004/000005, versus text SAM **0/4** and Spark **4/4**; those results are unchanged.

Normalized guided v2 matched **4/4 on a development replay** of those already tested sources. On **four fresh pairs from episodes 000006/000007**, v2 matched **3/4**, with one UNKNOWN; text SAM matched **0/4**, Spark **4/4**. V2 was tested on four replay pairs and four fresh pairs, not the whole corpus. The remaining case lacks a supported first-view mask and has a second-view SAM score of 0.746 below the unchanged 0.8 threshold. This is partial progress, not a complete fix or better accuracy than Spark. See the [guided-fix report](artifacts/feasibility/guided-fix-20261028.md) and [historical results](artifacts/feasibility/online-video-20261028.md).

These are scoped image-zone checks on normal demonstrations. They do not establish natural-error recognition, action recognition or physical closed-loop recovery. [Source credits](docs/ONLINE-VIDEO-SOURCES.md) and the [benchmark guide](docs/BENCHMARK.md) document provenance and correlated-case limits.

The earlier owned phone recording produced **correct PASS → misplaced fork FAIL → covered UNKNOWN → fresh corrected PASS** using real Spark/SAM calls. The [phone report](artifacts/feasibility/phone-session-20261028.md) retains its selected-frame limits: the person did not follow a planner proposal generated afterward. Physical interactive closure remains unvalidated.

**334 Python 3.12 tests pass**. Earlier GitHub CI passed Python 3.9 and 3.12 at [b6a5ebe](https://github.com/yerbol1212/proofloop/commit/b6a5ebe); software tests do not establish perception accuracy.

See [feasibility status](docs/FEASIBILITY.md), [capture checklist](docs/CAPTURE.md) and [contributions](CONTRIBUTIONS.md).

## Open the interactive workbench

Build the frontend with `cd web && npm ci && npm run build`, then from the repository run `.venv/bin/python -m proofloop.server --port 8790` and open <http://127.0.0.1:8790>. Python 3.10+ with `.[sam]`, Node 24, and a locally configured Meta key are needed for live inspection. The [interactive guide](docs/INTERACTIVE.md) includes setup and exact phone capture instructions. Use **Task** to draft and review requirements, **Inspection** to check fresh views, **Evidence log** for provenance, and **Video lab** to explore the recorded comparisons. Opening the lab makes no model calls. The interface labels historical evidence and requires two new views after evidence is invalidated. A public clone has no private recordings or raw lab runs.

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
