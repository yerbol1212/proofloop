# Queue admission and recovery references

Investigation date: 31 October 2026, Asia/Almaty. Two production defects were reproduced through the real inspection engine and asynchronous job service with authored pictures, fake model clients and injectable clocks. These checks make no Meta API calls and measure neither perception accuracy nor physical execution.

## Queued uploads acquired a second freshness window

The original reproduction admitted an uploaded pair at server time `100000`, waited `31001` milliseconds before starting the worker, then completed the inspection. The job recorded its original creation time but spent four fixture model calls and returned PASS with current age zero: the engine started its receipt clock at worker dispatch.

A server-owned admission record now binds the original wall and monotonic anchors to the exact request hash, job binding, decoded source hashes and local journal scope. Queue waiting and model inference consume the same nondecreasing elapsed window. The default worker and production server callback validate the original record before dispatch and pass it through completion. Synchronous library inspection calls also inherit known pending source age when mixed with the same engine's queued work. The standard server already disables its legacy synchronous HTTP route when the job service is configured; this additional check hardens the supported library path. An expired queued pair fails validation before model calls. A twenty-second queue followed by fifteen seconds of fixture inference produces UNKNOWN at thirty-five seconds; a ten-second queue followed by fifteen seconds retains its supported twenty-five-second result.

Polling a running job can also observe a later wall-clock high-water than the engine saw at dispatch. The queue and active evidence share that floor, and terminal cleanup preserves it in the current revision. A later wall-clock rollback cannot revive that evidence.

Changing an idempotency key, calibration or view order cannot make an already admitted source younger. Pending pairs sharing either exact decoded source inherit the earliest relevant pending receipt clock. When an accepted admission ends through completion, failure, cancellation or task change, its bytes enter the existing source-reuse guard. **Cancellation requires different fresh views for another inspection, even if the cancelled pair was still within its window.** This conservative behavior retains known image age without keeping an unbounded history of live clocks. Crops, edited pixels and newly encoded copies are not authenticated new captures; upload freshness still depends on the operator's declaration.

The private live-clock registry is bounded to thirty-two entries and pruned on terminal or cancelled jobs. Services using the same engine share its records. A new runtime without an original anchor rejects pending work before model calls; it does not reconstruct freshness from a historical job's creation time. The immutable request payload, SQLite schema and historical event chains remain unchanged. Idempotent historical retrieval remains read-only and never grants a new freshness window.

## An unrelated recovery could label a new inspection

A second reproduction completed a repair under task A, activated task B and submitted a normal inspection. Previously, publication attached A's receipt identifier to B's successful job, despite different task hashes and revisions. A later revision of the same task could also inherit an earlier completed recovery reference.

Task activation now cancels an open old request where applicable, clears the current recovery references and preserves the saved historical files. New job publication attaches a recovery identifier only after freshly replaying its saved evidence and checking the exact task hash, protocol hash, completed after-revision, receipt hash and outcome. Missing or inconsistent evidence omits the reference; it does not rewrite a separately supported visual verdict. A genuinely matching new correction job still attaches its completed receipt.

Previously published historical job events and receipts are not rewritten. Their hashes describe their saved local history; they are neither signed provenance nor current workflow authority.

## Decision time remains separate from delivery

The [earlier freshness correction](FRESHNESS-INVESTIGATION-20261030.md) continues to apply. A receipt records its decision point on the declared source clock advanced by the nondecreasing server elapsed floor. This is not a trusted wall-clock timestamp or an independently measured capture time. If subsequent persistence or delivery consumes the window, the genuinely fresh evaluation can remain valid history while current state and workflow expire. A thirty-second ceiling is an upper bound; individual checks can expire earlier.

## Focused validation

Twenty new queue/recovery integration regressions cover expired pre-dispatch rejection, combined queue/inference age, a fresh positive control, idempotent repeats, monotonic aging, queued and running high-water rollback, missing runtime anchors, same-engine service handoff, bounded cancellation cleanup, request rekeying, reused views, changed calibration, reversed view order, task changes, unrelated older receipts, missing receipt files, a genuinely matching completed repair reference, and expired/fresh direct-route controls.

The five-module queue, job, freshness and recovery suite passes **87 tests on Python 3.12 and Python 3.9**. Sixty loopback server/API/adversarial-runtime tests also pass. The final full suite runs **1,570 tests successfully on Python 3.12**, with Python 3.9 passing and five expected optional-dependency skips. Independent review retested the original defects and rekey/cancellation controls with no remaining P1/P2 in the reviewed changes. CodeRabbit completed a separate thirteen-file review with zero issues. These are software correctness checks using injected fixtures. No paid Meta calls, benchmark predictions or production perception thresholds are changed by this investigation.
