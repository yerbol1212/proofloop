# Full-frame visibility comparison

This is a prospective development experiment for a separate visibility assessor. The historical tracking comparison and mask retest remain unchanged. No new model request is made until the implementation, scoring references and public plan are frozen and pushed.

## Problem and intervention

The original crop guard rejected four clearly visible remote-control states despite accepting their identity and attributes. It also supported a gripped screwdriver fragment when its shrinking tracker box and context crop omitted the hidden handle end. The original Boolean contains no explanation of those internal decisions, so their exact model causes are unknown.

The new `visibility_guard` receives two registered images of the same current frame: the full source JPEG and its unchanged deterministic candidate crop. It distinguishes semantic identity from current-view physical visibility. A nearby robot or tape strip is not automatically an occluder; recognizable fragments are not automatically clear. Clear evidence requires an observable projected boundary, visible extremities, at least two named terminal landmarks, resolved full-frame extent and no external hiding, frame truncation or assessment-blocking blur. Natural unseen backsides and interiors are not required. Uncertain or contradictory evidence withholds support.

No reference image, later annotation, later box, score, expected class or prior model answer enters a new assessor request. The actual SAM candidate box only identifies a proposal; it does not define the object's complete extent. Identity and visibility remain fallible model evidence. Meta supports multiple images in one Spark request. [Image-understanding documentation](https://dev.meta.ai/docs/image-understanding)

## Fixed comparison

The two cohorts each have sixteen fixed checkpoints and remain separate:

- **Exposed development:** the same episode-1 screwdriver and remote frames used in the published tracking experiment. Original SAM responses and old guard answers are reused byte-for-byte; only the new assessor is called.
- **New recordings:** episode 2 from the same exposed dataset families, newly downloaded and visually annotated before any model calls. SAM tracks each clip once. Both the unchanged crop guard and new full-frame guard assess the same actual tracker candidates.

Source preparation selects every tenth native 50-fps frame into 5-fps lossless H.264, without resizing/cropping. All 100 new selected decoded frames match the native decoded YUV420p frames. The diagnostic JPEGs are extracted deterministically as q2/yuvj444p with metadata stripped. Only frame-zero target and nearby tape boxes initialize tracking; later authored boxes are scoring-only.

Fresh screwdriver checkpoints are frames 0, 3, 5, 8, 13, 17, 30 and 37. Fresh remote checkpoints are 0, 3, 5, 10, 20, 30, 40 and 59. The first three views in each recording are labeled clear. Later views are labeled insufficient visibility, including the remote's final frame whose left edge remains uncertain under the nearby gripper. A visually promising end frame is not automatically labeled clear re-exposure.

Seven diagnostic methods are reported: SAM presence; old/new visibility flags alone; old/new identity-plus-visibility candidate support; and old/new semantic policies with the already evaluated mask-based distractor correction. The policy's confidence threshold 0.8, two-view requirement and 500-ms spacing remain unchanged. Visibility-only confusion counts are separate from identity and localization scores. Localization uses IoU ≥0.3, with 0.2/0.5 sensitivity. Failed classifications are unscored; a successful UNKNOWN is a different outcome. Report completion, positive retention, unclear support and coverage together.

This tests the combined full-frame context and explicit-rubric intervention. It cannot isolate the value of either change alone. The two new recordings share prior tasks, viewpoints and dataset families; they are not an independent cross-domain benchmark. Full disappearance, identity switches, physical recovery and practical customer demand remain unproved.

## Budget and replay

At most **50 new requests** are admitted persistently: sixteen new-guard calls on exposed frames, two fresh video calls, sixteen unchanged-guard calls and sixteen new-guard calls on fresh frames. Missing candidates consume no guard call. Spark uses explicit low reasoning with a 4,096-token cap. Every call is once-only, with zero automatic retries. An estimated $3 allocation is advisory, not a provider-enforced dollar cap. Original reused inference costs are not charged as new work. Video usage uses Meta's reported billed frames and its frame pricing. [Pricing](https://dev.meta.ai/docs/pricing-rate-limits)

The runner keeps raw responses and private media locally, binds all source/derived/crop bytes and implementation hashes, replays classifications from retained raw output and checks the persistent attempt journal. Public artifacts contain allowlisted classifications, hashes, metrics and source credits. Local hashes detect changes; they are not signatures proving capture time or API origin. The new assessor stays outside live placement acceptance while this evaluation runs.

## New source credits

Both videos were created by the external ALOHA authors and distributed through LeRobot. ProofLoop did not create or train these recordings or Meta's models. Original media, local license copies and raw provider output are not redistributed in this repository.

| Dataset / recording | Publisher-declared license | Pinned revision | Original SHA-256 |
| --- | --- | --- | --- |
| [Screwdriver episode 2](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/resolve/51448b4e7638c1773fd07c1e7febce8ebb104635/videos/chunk-000/observation.images.cam_high/episode_000002.mp4) | [MIT](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/blob/51448b4e7638c1773fd07c1e7febce8ebb104635/README.md) | `51448b4e7638c1773fd07c1e7febce8ebb104635` | `a6f47aec329a9b9068cc4ddd720d56ccbe692de82c3777e2cc56dd563629b697` |
| [Battery/remote episode 2](https://huggingface.co/datasets/lerobot/aloha_static_battery/resolve/dd8bda435f5068a0298bd109005828768ecbfa12/videos/chunk-000/observation.images.cam_high/episode_000002.mp4) | [Apache-2.0](https://huggingface.co/datasets/lerobot/aloha_static_battery/blob/dd8bda435f5068a0298bd109005828768ecbfa12/README.md) | `dd8bda435f5068a0298bd109005828768ecbfa12` | `dd6b75cf6ccaec1972ec788d056abf236b01def11a77b30b6b987830c2165ab4` |

Earlier episode-1 source details remain in [tracking source credits](TRACKING-SOURCES-20261029.md). The local preparation manifest is `artifacts/local/visibility-review-20261029/prepared-sources.json`; the source-only freeze is `artifacts/local/visibility-experiment-20261029-01/fresh-sources/freeze.json`. Neither is itself an inference run.
