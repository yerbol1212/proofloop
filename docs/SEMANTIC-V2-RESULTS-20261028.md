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
