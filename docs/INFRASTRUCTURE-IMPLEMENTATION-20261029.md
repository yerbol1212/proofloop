# Implemented verification-service infrastructure

All six infrastructure priorities now have executable implementations. This milestone adds no Meta inference and does not change the published perception scores or live placement acceptance. Frozen perception modules and historical artifacts remain unchanged.

## Shared paid-call admission

The production server constructs one persistent `AttemptJournal` and `RuntimeControl` before creating the engine. Spark task compilation, placement inspection and real tracking share the same initial 120-attempt allowance across processes/restarts. Compilation has a one-request operation; placement retains its existing maximum thirteen; tracking has a fixed ten-request parent operation and fresh one-request clients for its component adapters. The parent allowance cannot be reset by creating another child client. Every uncertain reservation remains consumed; no paid request is automatically retried or refunded. Research CLIs retain their separate explicit experiment budgets. This local request allowance is not account-wide provider billing control or a hard dollar cap.

`RuntimeCompilerClient` is a distinct production type, preserving the difference between production transport and injected test factories. A completed provider response can still fail protocol/schema verification. Provider `ok` is not accepted evidence or a valid protocol.

## Lost-response recovery

Before an inspection POST, the browser persists and rereads a versioned opaque intent containing its random idempotency key, exact task binding and SHA-256 of the precise request-body bytes. Images, capture tokens and mutation tokens stay out of browser storage. Failure to persist the capability prevents dispatch. The server verifies the optional fingerprint header against the raw body and atomically stores a private append-only intent record with the original job.

`POST /api/v1/inspections/lookup` accepts only that versioned capability. It returns the original integrity-checked historical job or a bounded not-found response, without resending images, admitting work or calling a model. An absent record is only a point-in-time observation: the browser retains its intent and blocks capture across reloads until explicit New inspection or reviewed task activation changes the authoritative binding. This fences a late original request. Existing saved v1 job references remain readable.

## Durable tracking diagnostics

The separate `TrackingService` accepts opaque registered video asset IDs, reviewed frame-zero seeds, a visibility contract and two to eight ordered checkpoints. Assets are MP4/H.264, at most 8 MiB, twelve seconds, 120 frames and the existing decoder's source-size limits. A SQLite reservation enforces a maximum of 64 assets across concurrent service instances; at most 32 jobs can be pending. Callers cannot provide a server filesystem path or download URL.

Real jobs use the existing source-bound SAM video, Spark visibility, current-view confirmation and negative-only gripper adapters. The optional gripper review supplies both IDs 3/4; target ID 1 and optional distractor ID 2 remain distinct. New paid jobs require worker/journal/storage/allowance readiness and local parser/media-tool preflight before admission. Exact existing jobs remain retrievable when a worker, dependency or allowance later becomes unavailable. Claims are atomic, lease expiry is terminal, and expired work is never reassigned for inference. Source bytes and metadata are rechecked before dispatch and publication.

Results have an explicit diagnostic schema with `supported`, `unknown` and per-frame `unscored`, source hashes, warning codes and current-ending confirmation evidence. They never produce placement PASS or authorize an action. An injected callback/provider cannot claim live Meta provenance. Polling validates persisted result hashes and the closed public schema. A separate event stream supports reload/poll recovery.

The cached example projects the exposed `remote_new` policy and ends UNKNOWN at the known contact frame. Its mode is always `cached_demo`, origin is `cached_development_projection`, and request/cost counts are zero. The complete portable comparison remains available separately.

## Portable public policy replay

```sh
python -m proofloop.replay_bundle
# After installation:
proofloop replay
```

The packaged 583,546-byte bundle reproduces six published method reports across four recordings, including **2/6 → 3/6** and **3/7 → 4/7** combined retention, zero support on 10/9 unclear references, the clear false warning and the one unscored positive. It includes cached numerical perception inputs, mask-measurement receipts, reviewed seeds, references and exact policy dependencies. It contains no original video, encoded provider masks, API credentials or arbitrary provider text.

This is **policy-and-score replay**, not full video/mask reconstruction, independent remeasurement of proximity, API-origin attestation or held-out accuracy validation. Byte digests, the exact dependency inventory, closed schemas, source bindings, warning consistency, baseline-subset checks and expected scores are verified. The historical full-media audit remains a separate private operation. Localization sensitivity is replayed at IoU 0.2/0.3/0.5.

## Operations and local developer client

| Endpoint | Implemented behavior |
| --- | --- |
| `GET /api/health` | Process liveness |
| `GET /api/ready` | Local worker, journal, storage, telemetry and allowance checks; 503 when unavailable |
| `GET /api/v1/operations` | Redacted bounded spans, finite-label counters, sampled p50/p95 and readiness |
| `GET /api/v1/operations/otel` | Local OpenTelemetry-shaped JSON export; no network exporter |
| `GET /api/v1/replay` | Recompute the portable cached policy comparison, zero inference |
| `GET /api/v1/tracking/capabilities` | Versioned limits, cached scope and local model prerequisites |
| `POST /api/v1/tracking/assets` | `{media_type:"video/mp4", data:<base64>}` → opaque reviewed-source asset reference |
| `POST /api/v1/tracking/jobs` | Idempotent cached or explicitly reviewed real diagnostic admission |
| `GET /api/v1/tracking/jobs/<job_id>` | Verified historical diagnostic result |
| `GET /api/v1/tracking/jobs/<job_id>/events` | Bounded event page, with optional `after_sequence` |

All HTTP routes retain exact loopback Host/Origin checks; mutations retain the local CSRF capability. These are local single-user boundaries. Readiness makes no provider connectivity, authentication, credit balance or future model-quality assertion.

Telemetry extends existing job/attempt correlation with admission, queue, validation, execution, model, verification and publication timing. Storage retains at most 1,024 spans and exposes at most 100; labels are allowlisted and UNKNOWN reasons are normalized. Prompts, images, filesystem paths, provider error strings and credentials are excluded. Cached work has model `none` and no provider span. Telemetry failure preserves a possibly charged operation's result/accounting and blocks subsequent readiness instead of rewriting history. The local JSON export has no externally tested collector integration.

```python
from proofloop.client import ProofLoopClient

client = ProofLoopClient(port=8790)
job = client.cached_tracking("my-stable-demo-intent")
result = client.wait_tracking(job["job_id"], timeout=30)
assert result["result"]["mode"] == "cached_demo"
```

The SDK accepts a loopback port, explicit video bytes and closed requests. It exposes no arbitrary remote URL, generic command executor or file-reading tool. Polling timeouts retain the original reference and never resubmit inference. The existing MCP bridge remains read-only; no actual Meta-agent or glasses integration is claimed by these software checks.

## Installation and clean-environment verification

```sh
# Build a complete workbench wheel from the checkout:
cd web && npm ci && npm run build
cd ..
python -m pip install build==1.6.1 setuptools==84.2.1 wheel==0.48.0
python -m build --wheel --no-isolation
python scripts/verify_install.py dist/proofloop-0.1.0-py3-none-any.whl
```

The release wheel includes frontend assets and the replay fixture. A Python-only library installation without a prior frontend build remains supported; `doctor` reports missing UI assets. Installed services use a writable data directory independent of site-packages. Source checkout defaults stay inside the project; installed defaults use the user's data directory, with explicit `--data-dir`/`PROOFLOOP_DATA_DIR` overrides.

```sh
proofloop doctor
proofloop replay
proofloop serve --offline --data-dir artifacts/local/installed-workbench --port 8790
# In another terminal, same service:
proofloop track-demo --port 8790
```

`--offline` disables live model dispatch while retaining cached replay. `doctor` makes no model request or credential-authentication claim. Real tracking additionally needs Python ≥3.10, `pip install '.[sam]'`, compatible FFmpeg/FFprobe, an API key and reviewed inputs.

CI now runs both Python versions, the portable replay, fourteen Node recovery checks, the production frontend build and a separate installed-workbench job. The installation job builds a wheel, installs it into a fresh isolated environment outside the checkout, serves its bundled frontend, executes an asynchronous cached tracking job through the SDK and verifies zero reserved provider attempts.

## Validation evidence and limits

- **1,254 Python 3.12 tests pass**; Python 3.9 passes with two expected optional-parser skips.
- **14 Node recovery tests pass**, and the strict production frontend build passes.
- Clean-wheel installation outside the checkout serves the frontend, replays the bundle and completes the asynchronous cached API flow with zero provider attempts.
- Real browser QA uses the actual engine, HTTP handler, SQLite store and worker with authored fake clients. Dropping only the acceptance response recovers the same job after reload, with one dispatch and three fake calls. Tests also cover persistent late-admission fencing, storage failure and legacy references. Desktop 1440×1000 and mobile 390×844 pass render/interaction/overflow checks; the intentionally aborted request is the sole expected console error. Browser plugin was unavailable; Playwright used installed Chrome. Private scripts, screenshots and detailed QA results remain under ignored local infrastructure artifacts.
- The 20 frozen perception implementation hashes remain unchanged. No Meta inference or new model-accuracy experiment occurs in this milestone. The **3/6 and 4/7** perception results, one clear false warning and one unscored network case remain unchanged.

Remote multi-user deployment, actual external-agent interoperability, unseen-video validation and fresh physical operation remain separate work. The service does not turn a repair proposal, operator declaration or historical result into observed physical execution.

## Publication verification

Implementation commit `194b85211d26bffd36697d9270df4569a1903796` passed [all four CI jobs](https://github.com/yerbol1212/proofloop/actions/runs/40849139921): Python 3.9, Python 3.12, frontend and the installed workbench outside the checkout. The [sanitized validation record](../artifacts/feasibility/infrastructure-checks-20261029.json) records the local checks and their limits. The updated workbench was restarted locally; its SDK readiness, existing read-only MCP bridge, cached contact diagnostic and read-only browser rendering passed with zero added provider attempts. The browser smoke found no console or page errors. Private screenshots and detailed browser logs stay in the ignored local infrastructure directory.
