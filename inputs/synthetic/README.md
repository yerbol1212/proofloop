# Generated scene fixtures

These four images were created with OpenAI's built-in image generation tool on 26 October 2026, at the user's request, for initial ProofLoop tests. They are **synthetic inputs**, not phone recordings, physical measurements, independent trials or evidence of a completed repair.

| Image | Intended visible condition |
| --- | --- |
| `correct.png` | Red screwdriver left; blue pliers right; both visible |
| `wrong-region.png` | Screwdriver and pliers both right |
| `occluded.png` | Left compartment covered; its contents must remain unknown |
| `correct-variant.png` | Another generated correct condition with small position changes |

The base was generated from text. The other three are edits of that base, so they share substantial visual content. The assistant visually reviewed all outputs. [manifest.json](manifest.json) stores the exact prompts, image hashes, dimensions, generation method, expected visible conditions and manually specified regions. Those annotations are separate from model predictions.

The first model test should establish access, mask decoding and positive grounding. Zero SAM detections must not be interpreted as proof that a covered object is absent. Generated scenes do not replace the later owned-camera validation gate.
