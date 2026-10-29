# Concept proposals and returned-mask verification

This is a targeted response to two measured failures: displaced Spark anchors on clear cups and high-confidence SAM masks that merge the fork with a spoon. The [competitiveness assessment](COMPETITIVENESS-20261030.md) ranks useful perception ahead of additional infrastructure.

## Fixed intervention

SAM receives the original full image and the exact existing positive target concept. This removes the old spatial anchor from segmentation. A proposal is usable only when there is exactly one officially decoded object with reported confidence at least 0.8. The original locator's identity, full-visibility and single-candidate vetoes still apply.

For an eligible pair, Spark receives six bound images in fixed order: two original scenes, two deterministic context crops and two derived full-source binary masks. The masks show the actual returned foreground, rather than a suggested locator rectangle. They are visual aids, not extra physical observations. Spark must support identity, full physical visibility, complete target coverage, target-only foreground and one full-scene match in both views. Uncertain or contradicted fields withhold support. No task requirement, desired region or scoring label enters the challenge prompt.

Supported SAM records then enter the original grounding/checker. The confidence threshold, two distinct source views, original timestamps and whole-region requirements remain unchanged. Spark does not invent a replacement confidence or directly decide task completion. Missing and multiple detections remain unsupported; paid request failures remain unscored.

The [hosted SAM API](https://dev.meta.ai/docs/sam/segmenting) supports separate concept or spatial prompts. This method uses its concise concept form, not a mixture of text and box tokens. The task-independent mask challenge is an additional model judgment; it cannot independently prove physical identity or occlusion.

## Development evaluation

Retain all nine now-exposed cases from the [published pilot](NEW-SOURCE-RESULTS-20261029.md): two PASS, one FAIL and six UNKNOWN. Keep all original source bytes, concepts, labels, calibrations and thresholds. The runner attempts all eighteen concept-SAM views for diagnostic coverage before at most nine eligible-pair Spark challenges. Each client has one request, no retries and persistent admission. The main trial's maximum is 27 new requests.

A separate control reuses the two real high-confidence mixed fork/spoon masks from the rejected padding experiment. It adds at most one Spark challenge and no SAM request. Its provenance is explicitly retained spatial-SAM evidence; it is not mislabeled as new concept inference or included in the nine-case accuracy denominator. This tests whether the mask-aware veto can withhold a known bad foreground, rather than merely succeed on clean proposals.

The combined allowance is **at most 28 new requests**. The exact public plan fixes code/dependency hashes, original evidence and the control before paid inference. Errors and unused slots remain visible; no automatic retries, fallback model or favorable case substitution are allowed.

The [machine-readable declaration](../artifacts/feasibility/concept-mask-plan-20261030.json) binds private main-plan hash `437737c8e97c5de61ae93d86c19f37ff97e84d5662a51fc918625c51def73dbb` and mixed-mask preparation hash `22e2da86420aa37e634100ced443335c67276341aac629559f57500aa521e0ea`. Its JSON file SHA-256 is `2b4e42b91488bbfb67d9a71f7836e3b627697e35b8dacb8e2f89c1e539fbb765`.

Before inference, an independent adversarial review reproduced and closed two accounting-integrity areas: failed semantic attempts and aggregate experiment accounting. Its five concrete tampering checks now reject altered records. All 59 focused tests pass, including actual FFmpeg and official-parser integration. The full 1,415-test suite passes on Python 3.12; Python 3.9 passes with three expected optional-dependency skips. These establish software checks, not perception accuracy.

The predeclared development gate is:

- Retain both clear positive controls and the single supported wrong-placement FAIL.
- Withhold all six UNKNOWN references without converting request failures into successful abstention.
- Withhold the separately identified mixed-mask control through a completed semantic assessment.
- Retain complete source/response/accounting consistency and publish all costs and errors.

A favorable result on these exposed cases does not justify a general accuracy claim or production promotion. Two additional recordings have been selected by fixed metadata rules and prepared without model calls; their proposed six cases contain two PASS and four UNKNOWN references, with no supported FAIL. They share the existing two source families and related cameras. Any later use requires root review, a separate public preregistration and an unchanged method before inference. An actual current operator-followed correction and user benefit remain separate validation gates.
