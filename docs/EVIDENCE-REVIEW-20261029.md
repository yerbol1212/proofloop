# Second independent evidence review

29 October 2026, Asia/Almaty. This is a skeptical repository review, not a Meta endorsement or a prediction of hackathon placement. It follows the [first harsh review](HARSH-REVIEW-20261029.md). The review and accounting regression checks below made zero provider requests.

ProofLoop's engineering is substantially better demonstrated than its practical advantage. The existing placement comparisons favor direct Spark, and the latest tracking warning loses a clear target while withholding a misleading contact candidate. More tests or infrastructure cannot turn those measurements into perception superiority. The most valuable next result is a small prospective comparison that can fail, with all of its denominators and source reuse visible.

## Confirmed P2 defect: lost Spark responses could appear fully accounted

The durable tracking adapter accumulated a known-cost subtotal from each visibility assessment. It marked possible unreported charges when a SAM video had no cost estimate or the frozen Spark usage summary counted missing token fields. The summary intentionally excludes unsuccessful Spark attempts with no retained usage. A lost response therefore produced one `network_error` request, zero reported token usage and a zero Spark subtotal, while tracking could publish `unreported_charge_possible: false`.

This was reproduced twice without network access: first with the actual Meta transport and an authored opener that raises `URLError`, then through the tracking adapter composition with three authored lost Spark responses. Before the fix, the composed job succeeded as a diagnostic, retained five attempted requests and three unscored checkpoints, yet denied possible unreported charges. The unscored observations were correct; the billing uncertainty flag was incorrect.

The mutable tracking service now examines every child Spark ledger as well as reported subtotals. Started, lost and invalid-JSON responses remain financially uncertain. Missing, malformed or negative token counts also remain uncertain. Two reported nonnegative integer counts, including explicit zeros, are distinguishable from absent usage. Reported costs remain a known subtotal, not a billing cap. No transport, journal, perception policy, threshold or historical result was changed.

Composition regressions cover lost responses, malformed JSON, incomplete responses without usage, completed responses without usage, malformed token counts and explicit reported zeros. All 32 tracking-service tests pass on Python 3.12 and Python 3.9. Each injected model child still has one request and zero retries; no additional model attempt is dispatched as a response to uncertainty.

## Prospective experiment brief

Use two previously untested licensed task/source families and one original recording per family. Select six two-view cases before inference: one clear correct placement, one clear wrong-region placement and one genuinely unsupported pair per recording. The unsupported pair can be an inconsistent transition or a view where the target cannot be established. Do not rename a normal intermediate demonstration as a publisher-observed robot error.

Freeze the exact video and image bytes, decoded timestamps, attributed dataset revision and license evidence, target concept, calibrated regions, unique-target assertion, case rationale, case labels and implementation hashes. Pairs must use distinct source images at least one second apart. Keep the existing confidence threshold of 0.8, two-view requirement, guided-v2 method and direct Spark baseline. No region, label, prompt, checkpoint or acceptance threshold may change after the first response. An ambiguous label must be resolved or excluded before inference, with the exclusion recorded.

For six cases with one target, the existing placement harness needs at most 24 requests: up to 18 for guided Spark plus SAM, and six for the independent direct Spark baseline. Set an explicit persistent attempt cap and zero retries. An advisory allocation of one dollar is a planning estimate, not a provider-enforced spending limit. Preserve incomplete responses, reported usage and possible unreported charges. A failed condition must not erase the other method's planned denominator.

Report each family's three cases separately, then the six-case total. Include exact label matches, incorrect PASS decisions over four negative cases, retained correct placements over two positive cases, abstentions, incomplete/missing predictions, request attempts, known estimated cost, financial uncertainty and elapsed time. Six pairs from two videos are two source recordings, not six independent videos. New task families within the same robot/camera collection still do not establish transfer to arbitrary cameras, people or industrial work.

The executable minimum reuses `benchmark.approve_manifest`, `verify_media`, `request_requirements`, `run_benchmark`, `validate_report` and `public_summary`, with `capture._extract_frame` for actual decoded frame timestamps. `run_benchmark` currently stops at the first condition error; run methods in independently bounded directories or add a separate prospective runner that retains a complete case/method status table. Keep the frozen benchmark comparison semantics intact.

The older tracking experiment is unsuitable for silently relabeling this experiment: its attribution validator fixes the role to an exposed dataset family; its freeze function fixes exactly two clips, eight checkpoints and 50-fps native input sampled to 5 fps. The later contact experiment additionally requires episode 2 and episode 3 from its same two prior families. `reproduce.py` prepares only already exposed coffee profiles. A new prospective source-preparation wrapper must state the new source role explicitly rather than rewriting these historical restrictions.

## Capability and product gates that remain open

- The defining current physical loop still requires an operator to receive ProofLoop's request, make a real correction and supply new observations afterward. Existing prerecorded outcomes and software fixtures do not establish this.
- A useful demonstration needs a wrong repair and an UNKNOWN branch as well as a correct repair, preserving the original request and receipts. A passing visible state does not prove the requested action occurred.
- No unfamiliar operator has yet demonstrated independent setup or explained whether the requested evidence makes their next action clearer. A short observed usability session is more valuable than another generic dashboard.
- The strongest supportable user story is an evidence gate for a narrowly reviewed visible-placement task. Manufacturing quality, full procedure completion, safety, hidden state and autonomous robot recovery need additional observations and validation.

These are research and product limitations. They cannot be closed by weakening an acceptance gate, increasing the software-test count or presenting cached policy replay as new model inference.
