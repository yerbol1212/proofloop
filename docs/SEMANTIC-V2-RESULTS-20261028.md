# Semantic challenger v2: completion and real-video tests

The explicit low-reasoning request completed all four original cases, with no automatic retries. The earlier experiment completed two of four cases across five attempts. This fixes the observed completion problem on this development sample; it does not establish a general completion guarantee or semantic accuracy advantage.

## Paired completion check

The original two candy and two coffee prepared pairs, source/crop bytes, semantic descriptions, attributes, exclusions, prompt, schema and decision rule were retained. The new adapter explicitly requests `reasoning: {effort: "low"}`, with a 4,096 output-token cap and exactly one attempt per pair. Its schema name is versioned separately. The four selected cases and implementation/input hashes were frozen before any new call. Earlier results remain unchanged.

Meta's [reasoning guide](https://dev.meta.ai/docs/reasoning) documents that reasoning tokens share the output budget. The earlier negative requests used almost all of their 2,048/4,096 token caps for reasoning. The [Responses documentation](https://dev.meta.ai/docs/protocols/responses) describes the nested reasoning-effort setting. The new setting is opt-in; existing Spark calls retain their prior default payload.

| Original candidate pair | Reviewed identity expectation | Low-reasoning result |
| --- | --- | --- |
| Early candy packet/label anchor | Unsupported | UNKNOWN; excluded candidate and conflicting attributes |
| Visible white candy | Supported | Supported |
| Coffee 000004 glass on tray | Supported | Supported |
| Coffee 000005 separate glass | Supported | UNKNOWN; transparency and candidate identity uncertain |

All **4/4 requests completed**, **3/4** decisions matched the authored annotations, **2/3** positive candidates retained support, and the **1/1** misleading negative candidate was withheld. This is semantic support, not placement PASS or physical recovery. The clear glass that had been supported in v1 now abstained, so completion improvement must not be presented as universal semantic improvement.

The four calls cost an estimated **$0.035822**, with reported tokens from every attempt included. The old five-attempt estimate was $0.076588; the attempt counts and inference configurations differ, and the cases are repeated development inputs. This is not a statistical or causal performance study. Spark also assigned these historical anchors, so its subsequent semantic challenge is correlated rather than independent identity proof.

The [public paired result](../artifacts/feasibility/semantic-v2-completion-20261028.json) is regenerated from the local freeze, exact preparations, retained responses and actual usage ledgers. The original guided artifacts separately establish the historical anchors' origin. All new errors retain available raw provider responses locally and remain unscored; they are never converted into UNKNOWN predictions.

## Software verification

The adapter has an opt-in, validated effort field; omitted settings preserve the previous payload. Source/crop changes and malformed responses fail closed. The new trial runner prevents overwriting or rerunning an attempted trial, freezes reviewed labels/configuration/code, uses a fresh one-request client per case and reports completed, unscored and unattempted cases separately. Public output omits local paths, provider IDs and raw responses.

The challenger remains separate from live placement acceptance. The next section records the independently selected real-video candidate stress cohort once its bounded inference is complete.

## Real distractors from two additional recordings

The next cohort used two pinned, openly licensed ALOHA recordings: [screwdriver, MIT](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/blob/51448b4e7638c1773fd07c1e7febce8ebb104635/README.md) and [battery/remote body, Apache-2.0](https://huggingface.co/datasets/lerobot/aloha_static_battery/blob/dd8bda435f5068a0298bd109005828768ecbfa12/README.md). Six candidate pairs reuse their same 0/1-second frames. The two source recordings, four source frames, twelve review crops, semantic descriptions, boxes and labels were visually reviewed and frozen before these requests. Five clips were screened; three were excluded for weak visibility or truncation before inference. Details and exact attribution are in [the source record](SEMANTIC-DISTRACTOR-SOURCES-20261028.md).

These are **manually authored candidate boxes**, not automatic Spark detections. They test whether the challenger stays with a selected candidate when similar or unrelated real objects appear nearby. Full views and deterministic context crops were retained; no compositing, object insertion, generated distractors or source upsampling was used. A local verifier checks pinned video/card/info hashes and re-extracted all four source frames byte-for-byte before the run. Cases are correlated, not six independent recordings.

| Reviewed candidate | Expected support | Global-context v2 result |
| --- | --- | --- |
| Actual white-handle screwdriver | Supported | Supported |
| White tape strip, screwdriver scene | Unsupported | UNKNOWN |
| Blue cup, screwdriver scene | Unsupported | UNKNOWN |
| Actual white remote body | Supported | Supported |
| White tape strip, remote scene | Unsupported | UNKNOWN |
| Loose battery beside remote | Unsupported | **Supported: incorrect acceptance** |

All **6/6** requests completed, **5/6** matched the authored labels, **2/2** positive controls retained support, and **1/4** unsupported candidates was accepted. Estimated cost: **$0.047778**. The [original result](../artifacts/feasibility/semantic-v2-distractors-20261028.json) remains unchanged.

The failure is concrete: the selected battery box covers only 22×26 source pixels, while the actual remote lies elsewhere in the full frame. The model reported the remote's body and compartment attributes for the battery candidate. This supports a candidate-drift diagnosis from recorded outputs, not a claim about the model's hidden reasoning.

## Separate crop-only confirmation

After observing that error, a new development retest was prospectively frozen for **all three globally supported candidates**: both actual targets and the false-support battery. It used a generic prompt, with no battery/remote example, expected label or global verdict in the request. Only the two deterministic candidate crops and translated inner-box geometry were attached. The local schema deliberately omits full-scene match count: crops cannot establish global uniqueness.

The combined rule requires both previously recorded global support and completed local support. It cannot upgrade a global UNKNOWN, and an incomplete local request remains unscored. The [new result](../artifacts/feasibility/semantic-crop-confirmation-20261028.json) completed **3/3** requests:

| Candidate | Original global support | After crop-only confirmation |
| --- | --- | --- |
| Screwdriver target | Supported | Supported |
| Remote-body target | Supported | Supported |
| Loose battery | Supported | UNKNOWN |

The local battery assessment was uncertain about identity, attributes, exclusions and full visibility. It **withheld unsupported support**; it did not positively identify the battery's type. Both true targets retained support. Three additional calls cost an estimated **$0.015086**. This is an exposed development retest after seeing the error, not a fresh holdout, general accuracy guarantee or independent identity proof. The original 5/6 result is not rewritten into a stronger headline score.

Across the three separate runs, **13 new requests** completed at an estimated **$0.098686**. The runs have different roles and must not be pooled into one accuracy claim. The frozen guided-v2 placement pipeline, its published failures and all older semantic results remain unchanged. These diagnostics stay outside live placement acceptance pending independent new-source and automatic-anchor evaluation.

## Workbench inspection and reproduction

Open **Video lab → Development experiments**. Completion, authored distractors and crop-only confirmation have separate result sections. Candidate selection opens the exact original source frames, outlined boxes and derived crops; missing detail is explicitly requested when support is withheld. This read-only inspection makes no model calls. Public summaries are compared exactly with private source/response replay before display. Media tokens serve only hash-bound image bytes from that validated snapshot; refreshing results rebuilds the image allowlist.

Raw videos, frames, crops, model responses, source review and trial freezes remain local. Public source/result records contain credit, safe hashes, bounded classifications and usage; they contain no API key or private paths. The dependency-free authored tests run without requiring FFmpeg; real crop preparation/replay requires the declared FFmpeg version. A CI failure caught an unintended extractor lookup in one authored unit fixture, which was isolated from the external executable before publication of the completed milestone.

## Final validation

The full Python 3.12 suite passed **626/626** tests. System Python 3.9 ran the same 626 tests with one expected optional-parser skip. TypeScript checking and the production Vite build passed. The failed CI fixture lookup was corrected without changing frozen inference code or experimental results.

The rendered flow was: localhost → Video lab → Development experiments → select remote-battery candidate → inspect its exact full views/crops → compare the separate crop-only confirmation and missing-evidence request. Browser plugin was unavailable; the built-in CUA Playwright surface was used under the runtime's browser-control instructions. Opening/replaying the inspector performs no inference.

| Browser check | Result |
| --- | --- |
| Page identity and nonblank content | ProofLoop at http://127.0.0.1:8790/; expected panels rendered |
| Framework overlay / console | No overlay; no relevant warnings or errors |
| Interaction | Selecting the battery changed the case, preserved the false-support warning and loaded the matching source pair/crops |
| Source images | All four selected images loaded; source views 640×480 and exact derived crops 38×42 pixels |
| Responsive layout | Desktop 1536×1000 and phone 390×844; page width did not overflow; the comparison table scrolls within its own region |
| Correction explanation | Both targets remain supported; battery support withheld; clearer full-outline/distinguishing-feature evidence requested |

Screenshots are saved outside the repository in the Meta Hackathon `qa/proofloop-ui` folder. This validation covers the local inspector and software contracts, not fresh physical recovery, automatic localization, independent identity verification or broader new-source reliability.
