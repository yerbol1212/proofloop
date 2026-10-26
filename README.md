# ProofLoop

Executable checks and evidence-linked correction for bounded physical workflows.

ProofLoop turns a confirmed procedure into inspectable checks, evaluates observations as **pass**, **fail** or **unknown**, and searches a small declared action model for a correction. Corrections require fresh observations before the result changes. Muse Spark and SAM 3.1 are the intended reasoning and perception models.

## Current status

This repository is the initial feasibility implementation for the Meta Global AI Developer Hackathon. Offline diagnostic cases are explicitly synthetic oracle observations. Live access tests use real Meta APIs when a valid key and billing access are available. Neither synthetic fixtures nor mocked API tests establish physical perception quality.

See [feasibility status](docs/FEASIBILITY.md), [capture checklist](docs/CAPTURE.md) and [contributions](CONTRIBUTIONS.md).

## Run the offline checks

Python 3.9 or newer is sufficient for the dependency-free checker and HTTP adapter:

```bash
python3 -m unittest discover -s tests -v
python3 -m proofloop.feasibility --output artifacts/feasibility
```

The diagnostic report records the exact cases and evidence origin. It is not an industrial accuracy benchmark.

## Run real model access checks

Copy `.env.example` to `.env.local` and save `META_API_KEY` locally. The local file is ignored by Git. Meta account billing requirements must be satisfied. No key belongs in source files, screenshots, chat or public artifacts.

```bash
python3 -m proofloop.smoke --models-only
python3 -m proofloop.smoke --spark-only
python3 -m proofloop.smoke --image /absolute/path/to/owned-image.png --concept screwdriver
```

The smoke runner calls `https://api.meta.ai/v1` with `muse-spark-1.3` and `sam-3.1`. SAM mask decoding uses Meta's official parser and requires Python 3.10 or newer. In a Python 3.10+ virtual environment, install `pip install '.[sam]'` before the image check. The project development environment uses Python 3.12 and parser version 0.1.0. Spark-only access checks and the deterministic core remain compatible with Python 3.9.

Each request has bounded output/size/time and a local attempt budget; estimated cost is not a provider billing stop. Unreviewed responses and source media remain under ignored local directories. Missing access produces an explicit error, never a substituted model result.

## Draft a protocol with Spark

```bash
python3 -m proofloop.compiler \
  --requirements 'Exactly one screwdriver in the left region and one pair of pliers in the right region.' \
  --objects screwdriver pliers --regions left right
```

The compiler calls Spark and validates the resulting protocol against the allowed inventory and checker vocabulary. This initial lowering supports exact-count predicates. It rejects unsupported or ambiguous requirements and does not infer mandatory step order from a successful video. A draft is always unapproved and saved locally for human review; it is not immediately used to judge a run. Temporal checks and actions currently come from explicit authored specifications.

No live compilation result is claimed until the account-access gate is complete. Mocked compiler tests only test contracts and rejection behavior.

## Generated scene inputs

The [synthetic fixture set](inputs/synthetic/README.md) includes correct, wrong-region and occluded conditions, plus a second correct variant. Exact prompts and source hashes are included. These generated images can exercise real API transport and parsing once account access is ready; they are not camera observations or a completed physical repair.

`proofloop.grounding` connects parsed SAM image masks to approved checker predicates. It preserves provenance, requires distinct supporting image hashes, and returns unknown for missing/ambiguous perception. Positive detections supply visibility or count lower bounds, never automatic proof of absence or an exact count.

## Scope

Start with a fixed camera, a visible tray and distinct macro objects. The current checker tests declared observable properties. It cannot certify hidden contents, attachment, torque, electrical continuity or physical safety. A minimum repair is minimum cost within the supplied finite action model.

The source of each fact, specification version, observation epoch and freshness matter. Occlusion and possible content changes invalidate earlier evidence. A new attempt cannot retroactively fix the record of an earlier one.

## Development

Only project code, reviewed fixtures and public-safe results are published. Owned camera media, local credentials, account information and private research remain local. Commits identify event-created work and external components. Build milestones are pushed after relevant checks pass.
