# Durable inspection runtime

Implemented on 2026-10-28. This is a local, single-user service around ProofLoop's existing reviewed placement checks. Durability improves failure handling and correlation; it does not improve model accuracy or establish physical action completion.

## Execution contract

1. Review and activate the task, then start an inspection using the existing workbench. Two new views and the engine-issued one-use capture token are still required.
2. `POST /api/v1/inspections` accepts exactly `{idempotency_key, binding, inspection}`. `binding` contains exactly `session_id`, `task_hash`, `protocol_hash`, and `revision`; `inspection` is the existing two-view capture payload. It returns HTTP 202 with an opaque job ID.
3. Admission validates the reviewed task and capture under the engine lock without consuming the token. A private SQLite transaction stores the canonical payload fingerprint and binding. The same key and exact request returns the original job, including after completion or restart. A changed request under that key conflicts.
4. One worker claims a job once. It checks the binding again, records that execution may dispatch, and calls the authoritative engine. The engine rechecks the binding inside its admission lock, closing the gap between a queue check and token consumption.
5. During production inspection inference, the worker guard checks its lease and binding before every provider request. A separate heartbeat extends the lease while a request is in flight. Each request reserves persistent allowance before transport. Calls use the existing Meta adapter with zero automatic retries.
6. Publication validates the completion snapshot through a closed policy projection, atomically saves its result, and commits a result hash/reference. Polling verifies the journal, file hash and result schema before returning the stored result. Tampered/missing result bytes remove both public result projections and set `integrity.result_valid=false`.

The browser submits through this path. It stores only the opaque job ID in session storage, allowing a reload in the same tab to resume polling. It refreshes the authoritative engine state after completion; it never applies a persisted result as current authority. A new server session can legitimately have a lower revision and replaces the prior session's UI state.

## API

All routes retain the existing exact loopback Host/Origin checks. Mutations also require the token returned by `/api/state`, sent as `X-ProofLoop-Token`. These are local request boundaries, not remote account authentication. Duplicate JSON keys are rejected at any nesting level.

| Route | Meaning |
| --- | --- |
| `GET /api/v1/capabilities` | Versioned supported predicates, evidence requirements and claim limits; job service availability |
| `POST /api/v1/inspections` | Admit or retrieve an exact idempotent inspection |
| `GET /api/v1/jobs/<id>` | Redacted job metadata, integrity status and verified historical completion result |
| `GET /api/v1/jobs/<id>/events?after_sequence=0` | Bounded event journal page, with hash links |
| `POST /api/v1/jobs/<id>/cancel` with `{}` | Cancel a queued job before dispatch; running inference cannot be cancelled through this route |
| `GET /api/v1/accounting` | Coherent snapshot of persistent inspection attempt admission and provider-reported token totals |

The old synchronous `/api/inspect` route is disabled when the durable service is enabled, preventing an unmetered inspection bypass. Test-only servers without a job service keep their existing injectable transport behavior.

## Failure and restart behavior

States are `queued`, `running`, `succeeded`, `failed`, `interrupted`, and `cancelled`. An expired running claim becomes terminal `interrupted`; it is never reassigned. Before an execution marker, the code is `interrupted_no_dispatch`; after the marker it is `outcome_unknown`. That marker means provider dispatch may occur, not proof that an HTTP request was sent. The independent attempt journal records reservations closer to transport.

Leases fence stale completion and subsequent provider calls. They cannot undo an HTTP request already sent. On shutdown the service stops new dispatch; an in-flight request may outlive the short shutdown join. Its reservation remains consumed even if the process cannot settle it. A restarted engine creates a new session; queued jobs bound to an older session fail validation instead of spending calls, and old completed results remain historical.

The event hash chain detects local inconsistency against the retained job projection. It is not a digital signature, independent timestamp, or defense against someone who can replace the entire database. SQLite uses DELETE journaling with short `BEGIN IMMEDIATE` write transactions and full synchronization. This is local-host storage, not a distributed queue or a network-filesystem deployment claim.

Private state and evidence JSON now use same-directory temporary files, file fsync, atomic replacement and directory fsync. Failed serialization or interrupted replacement preserves the last complete file. A filesystem/hardware failure can still make an operation unavailable; the implementation does not claim cross-file transactional recovery of a live engine session.

## Request allowance and accounting

The production server defaults to a persistent **120 inspection-request attempts** allowance across clients, processes and restarts. The bound operation allowance remains the actual engine budget: two requests per target, plus one Spark localization request in guided mode, at most 13. The initial global allowance is configurable using `--request-limit`; reopening the same journal with a different allowance is rejected. A new adapter cannot silently raise a previously registered operation allowance.

Every reservation counts, including unfinished and failed requests. There are no automatic refunds or retries. The journal stores only operation ID, endpoint, supported model, request hash, timestamps, outcome and allowlisted reported token counts. It stores no prompt, source image, credential or provider message body. HTTP/transport `ok` means the response completed; later schema/evidence validation may still reject it.

This is a request-count bound, **not a provider-enforced dollar cap**. Unreported usage and a lost connection can make actual billing unknown. This allowance covers the durable inspection path; separate explicit research commands and the task-drafting call are not routed through this journal. Check provider billing for account-wide spend.

## Validation and remaining boundaries

Offline tests use authored pixels, injected clients and mocked HTTP. They cover competing submissions/claims, exact-key conflicts, lease expiry, interrupted work, stale worker fencing, persistent allowances, reported usage on incomplete responses, failed artifact writes, tampering, binding changes, HTTP origin/CSRF/duplicate-key checks, and actual engine/worker interoperability. They are software reliability evidence, not fresh model accuracy measurements.

Browser QA used an isolated fixture on port 8791 with authored PNGs and injected inference: upload → submit → reload while running → restored polling → completed result. It made no Meta API calls. The production workbench remains on port 8790.

Known boundaries:

- If the admission HTTP response is lost before the browser receives/stores its job ID, the UI cannot recover that job by its idempotency key after a reload. It does not automatically resubmit; authoritative engine state remains available. API clients retaining their full request/key can retrieve the original job by exact resubmission.
- Stored `current_at_snapshot` describes completion time. Polling later does not renew freshness or authorize workflow continuation or physical actions.
- Reviewed unique-target location checks in two regions are the live supported contract. Broader deterministic predicates are listed separately; live count/absence/procedure correctness are not implied.
- Semantic crop experiments remain separate from live acceptance. The measured SAM-vs-Spark accuracy limitations are unchanged.
- There is no remote multi-user authentication, robot executor, distributed scheduler or glasses integration.

Primary-source rationale and the next independently evaluated tracking experiment are in [the infrastructure review](INFRASTRUCTURE-REVIEW-20261028.md).
