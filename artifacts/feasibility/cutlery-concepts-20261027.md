# Fork/spoon concept probe — 27 October 2026, Almaty

Real SAM 3.1 calls returned one fork detection on the left and one spoon detection on the right of a single generated image. Both masks decoded with the official parser.

| Concept | Detections | Side of divider | Reported confidence | Latency |
| --- | --- | --- | --- | --- |
| fork | 1 | left | 0.96098 | 3.332 seconds |
| spoon | 1 | right | 0.96491 | 2.294 seconds |

Two image calls have an estimated Standard list-price cost of **$0.005**, not an invoice. The [JSON result](cutlery-concepts-20261027.json) contains source binding, decoded foreground counts and usage. Raw responses remain local.

The [source fixture and exact prompt](../../inputs/synthetic-cutlery/manifest.json) are explicitly generated with the built-in image-generation tool. A single selected view supports continuing to the phone check. It cannot establish real cutlery detection quality, temporal stability, an exact count, absence or a physical correction. Reported confidence is not a calibrated correctness probability.

Current phone setup: one fork LEFT and one spoon RIGHT; move only the fork for the deviation and correction. The [recording guide](../../docs/CAPTURE.md) and [unapproved cutlery starter](../../examples/cutlery/README.md) preserve the same timing and approval requirements.
