# Infrastructure review — 2026-10-28

Reviewed source baseline: `f3df82d12504e09f0e866d944a951c682bb10c84`. Source line references below describe that baseline; subsequent implementation may move them. This is an architectural review and primary-source research, not evidence of Meta endorsement, product demand, robot integration, or improved perception accuracy.

The strongest direction is a small verification service that other agents can call: admit a bounded inspection, preserve exactly what happened, evaluate the reviewed contract, and return either a supported result or the specific evidence still missing. Extra model names and infrastructure layers would not strengthen that contribution by themselves.

## Existing strengths to preserve

- The HTTP server binds to IPv4 loopback, validates Host/Origin and mutation tokens, and exposes explicit operations rather than arbitrary filesystem or command access (`proofloop/server.py:137`, `:203`, `:313`). Do not remove those boundaries to make agent integration easier.
- Evidence binds task hash, protocol hash, source bytes, source timing basis and perception version. Replay recomputes grounding and decisions; hashes are deliberately not presented as signatures or independent capture-time proof (`proofloop/recovery.py:1`, `:50`, `:86`).
- Placement preparation and an operator's continuation declaration are already separate. Current source, freshness, exact check coverage and bound replay are required; the module explicitly grants no machine-action permission (`proofloop/workflow.py:1`, `:36`, `:85`).
- Failed/incomplete model responses remain errors. The provider adapter records reported token usage even for an incomplete response and does not retry ambiguous connection errors (`proofloop/meta.py:329`, `:359`). Preserve those semantics when adding durability.
- The frozen guided perception adapter and separate semantic experiments make model regressions visible. A more convenient service must not silently promote an experimental identity judgment into live acceptance.

## Priority 1 — Make paid work durable before adding more autonomy

**Observed gap.** HTTP mutations execute synchronously on a request thread (`proofloop/server.py:313–327`). The engine's busy flag, active token and current operation live in process memory (`proofloop/interactive.py:187–249`). A restart creates a new session. JSON persistence uses `O_TRUNC`, so an interrupted write can leave an incomplete projection (`:452–460`). Inspection consumes the capture token and records hashes before external inference, then saves provider results and evaluation separately (`:957–1014`, `:1030–1066`). These are conservative rejection semantics, but there is no durable operation identity for a disconnected caller to query.

**Consequence.** A browser/agent cannot distinguish “request rejected before dispatch,” “remote inference may have run,” and “completed locally but response was lost” after a process failure. Repeating the whole inspection risks new charges and creates an unrelated operation; treating an incomplete local snapshot as success would be worse.

**Bounded implementation.** Add a local SQLite operation journal with a unique idempotency key and canonical request fingerprint. Persist admission, exact task/session/revision binding, source hashes, operation kind and attempt budget in one short transaction. A duplicate matching request returns the existing operation; a reused key with changed content is rejected. Publish explicit states such as admitted, running, completed, failed and interrupted. On startup, reconcile unfinished work to interrupted/needs review; never automatically resend an operation whose dispatch outcome is uncertain.

Keep remote calls outside transactions and state locks. Save private artifacts atomically through a same-directory temporary file, flush/fsync, replacement and directory synchronization where supported; only publish a completed operation after all referenced bytes and the replayable result are committed. A durable journal need not restore a prior PASS as current evidence: old evidence still expires and must retain its historical provenance.

SQLite permits multiple readers but one simultaneous writer, and `BEGIN IMMEDIATE` acquires the write transaction up front. This fits short admission transitions, not long inference-held transactions. [SQLite transaction documentation](https://www.sqlite.org/lang_transaction.html)

WAL is optional rather than a scaling claim. It is local-host storage with checkpointing requirements. The current Python environment reports SQLite `3.53.1`; portable installations should inspect their runtime before selecting WAL. SQLite documents a rare concurrent WAL reset issue fixed in `3.51.3` and selected backports. A simple rollback journal is sufficient for this single-user workbench. [SQLite WAL documentation](https://www.sqlite.org/wal.html)

**Acceptance evidence.** Concurrent duplicate submissions produce one operation and one dispatch; key/content conflicts produce no dispatch; failures injected before/after admission, before dispatch, after a fake provider result and during artifact publication never produce a current PASS without its complete replayable artifacts. Restart preserves the operation's terminal or interrupted state. All of these can run with fake clients and zero API spend.

## Priority 2 — Journal each provider attempt and enforce admission across sessions

**Observed gap.** `MetaClient.ledger` is an in-memory list (`proofloop/meta.py:283–296`). The attempt is appended immediately before transport, but persistence happens later in engine evaluation/failure files (`:313–336`; `proofloop/interactive.py:1065`, `:1122`). A fresh client has a fresh request cap. The current cost summary correctly excludes unreported and ambiguous usage and explicitly is not a billing hard stop (`proofloop/meta.py:467–498`). There is no project-wide durable request admission policy.

**Bounded implementation.** Give each paid attempt an operation-linked attempt ID. Persist a redacted request fingerprint, model/version, response schema or prompt hash, source hashes, maximum output tokens and dispatch state before sending. Save response ID/status, reported usage, latency, local result hash and error category after receiving. Store no bearer token or raw base64 payload in the journal. Unknown remote outcome must remain a distinct field rather than zero cost or known failure.

Use an atomic project-level attempt allowance and conservative reservation policy for new admissions, with per-operation request caps still enforced. Expose estimated cost, unreported usage and uncertain attempts separately. Do not advertise exact provider spending control from local token estimates. A later reconciliation import can record actual billed cost without rewriting original observations.

Meta's current published Standard prices distinguish Spark input/output/cached tokens, SAM images and SAM video frames; video has a 50-frame minimum. Team-level rate limits are shared across API keys. This makes request type and rate-card date necessary in the ledger rather than a single generic token counter. [Meta pricing and rate limits](https://dev.meta.ai/docs/pricing-rate-limits)

**Acceptance evidence.** Restart cannot reset the project allowance; concurrent admissions cannot oversubscribe it; refused or incomplete responses preserve reported usage; a transport timeout preserves “outcome unknown”; fake-provider latency and call count are observable without exposing request content.

## Priority 3 — Expose an agent-facing contract while retaining the policy boundary

**Observed gap.** Current HTTP operations serve the workbench and return whole state snapshots (`proofloop/server.py:288–327`). The verification policy exists, but there is no small versioned tool interface another agent can discover and consume. Adapting the browser mutation token into a general agent credential would blur its purpose.

**Bounded implementation.** Start with a local Python tool service and a stdio MCP adapter, or explicit typed HTTP tools under the existing local boundary. Read tools should expose capabilities, current task binding, operation status, current verdict, evidence requests and replay verification. An inspection submission should return an operation ID promptly and use the durable admission path. Each tool returns a versioned narrow DTO with exact task hash/revision and explicit claim limits. Runtime validation must reject extra or malformed parameters even if the model was given a strict schema.

Keep task activation and continuation declarations distinct from read operations. Do not expose a generic `execute_action`, shell command, path-reading or unrestricted URL-download tool. Proposed repairs are suggestions; the preparation gate and an explicit declaration still do not prove or authorize physical execution. The agent cannot manufacture review or consent by placing `confirmed: true` into an argument. The existing engine, rather than the model's explanation or tool result display, remains authoritative.

Meta's tool-calling contract has the application execute developer-defined functions. Its current `strict` option validates tool schema definitions; `max_tool_calls` limits built-in tools, so client-side function loops still need application limits. The current documented `tool_choice` surface accepts `auto`, so a test must accept either a function call or a normal model response. [Meta tool calling](https://dev.meta.ai/docs/tool-calling)

Muse Code currently documents stdio and Streamable HTTP MCP servers. That makes a local MCP adapter an immediately relevant integration surface, though it is not evidence that Meta AI itself accepted the service. [Muse Code MCP configuration](https://dev.meta.ai/docs/muse-code/extending)

The MCP tools specification supports input/output schemas and structured results. Its behavior annotations are descriptive metadata rather than trusted authorization; actual enforcement belongs in the service. Pin a tested protocol version and negotiate capabilities instead of claiming compatibility with every MCP client. [MCP tools specification, 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

**Acceptance evidence.** A real local stdio client initializes, discovers the read tools, retrieves a schema-valid result and polls one fake inspection. Unsupported tool names, extra parameters, stale bindings, replayed idempotency keys with changed content, exhausted budgets and attempted physical-action calls fail before dispatch. This verifies the tool integration without needing paid inference.

## Priority 4 — Make verification a versioned policy and regression target

**Observed gap.** The policy is already stronger than a free-form VLM answer, but its requirements are distributed across upload validation, acquisition, task review, grounding, evidence freshness and workflow replay. A new HTTP/MCP caller could accidentally implement its own incomplete interpretation of “ready.” The frozen semantic challenger remains experimental and has real distractor failures; durability must not launder those into accepted identity evidence.

**Bounded implementation.** Publish a machine-readable capability/policy document that lists supported predicates, required source count, freshness and timestamp basis, uniqueness assumptions, perception version, semantic-challenger status and physical-action prohibition. Derive decisions from the existing executable policy. Bind a policy version/hash into operation results and receipts. Describe the evaluated hypothesis precisely: visible region placement under a reviewed two-region contract, rather than whole-procedure correctness.

Add a deterministic fault matrix around the authoritative boundary: missing target, distractor ambiguity, stale image, reused source, task revision change, incomplete provider response, artifact tampering, duplicate submission and interrupted work. Record expected outcome and dispatch count. Keep software fault-injection cases separate from real-video model performance and from fresh physical recovery.

**Acceptance evidence.** Browser and agent APIs report the same policy hash and decision for the same stored operation; historical evidence stays historical; a semantic candidate assessment cannot unlock continuation; a changed task cannot reuse the original task's authorization or verdict; failure of persistence or replay yields UNKNOWN or blocked status.

## Priority 5 — Add temporal tracking only as a separately evaluated adapter

Two still views support a placement checkpoint. They do not establish that the object remained continuously visible, that the same physical instance persisted, or that a repair move occurred. Continuous tracking is a promising next research branch after durable execution and an interoperable API, rather than a reason to enlarge the live acceptance claim now.

SAM's documented video API accepts concept or box/point prompts, preserves supplied object IDs, supports directional propagation and can emit frames out of chronological order. Documented limits include 16 tracked objects and 15,000 frames. A future adapter should bind original video hash, frame index/timebase, rotation, prompt coordinates and track availability, and must not infer absence from an omitted/occluded object. A track ID still does not independently verify semantic identity. [Meta SAM segmentation and tracking](https://dev.meta.ai/docs/sam/segmenting)

First experiment: freeze two licensed clips and manually label target-visible, occluded, distractor-overlap and reappearance intervals; compare the existing two-view policy with the separate track adapter on those intervals. Preserve UNKNOWN through occlusion and report identity switches and interval coverage separately. Replay the exact saved stream with frames sorted by source markers, not arrival order. This is a new benchmark, not an upgrade justified by the current candy/coffee results.

## Current Meta alignment and honest future scope

An interoperable verification service fits Meta's public developer direction because developer-defined tools are available now, and Muse Code documents MCP integrations. This is an engineering inference from the cited APIs, not a statement about internal roadmaps or what judges will prefer.

Meta AI Connectors currently offers selected developers a preview for APIs/MCP and account-linked tools across Meta AI surfaces. Access and public distribution are not guaranteed; a local ProofLoop tool service must not be described as a live glasses integration. [Meta AI Connectors](https://dev.meta.ai/products/connectors)

The official V-JEPA project describes research on visual prediction and planning; its repository includes V-JEPA 2.1 with temporally consistent dense features. These are potential future perception adapters, not a hosted Responses endpoint or a demonstrated ProofLoop capability. Downloading and integrating a large checkpoint without a new hypothesis, reproducible evaluation and compatible hardware would add risk before adding evidence. [Meta V-JEPA project](https://ai.meta.com/research/vjepa/), [official V-JEPA repository](https://github.com/facebookresearch/vjepa2)

The immediate milestone should therefore be durable admission and attempt accounting, one tested agent-facing interface, a versioned policy boundary and deterministic failure injection. The claim to demonstrate is that an agent gets a bounded, replayable, task-bound answer even when requests fail or repeat. Perception accuracy, physical recovery and broader user usefulness remain separate validation work.
