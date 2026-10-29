# Boundary contact and current-view confirmation protocol

This experiment addresses two separate failures: an unclear white-on-white gripper contact was classified clear, and a valid current clear view was withheld because the most recent prior confirmation was too close in time. Prior visibility/tracking results and their code remain unchanged.

## Two interventions

`contact_guard` v3 receives the full current frame, its registered native context crop, and a deterministic 3× nearest-neighbor PNG enlargement of that crop. The enlargement repeats existing decoded pixels; it adds no detail or independent evidence. Besides the existing identity/visibility fields, the model assesses the target's actual projected outline in four image-relative regions: left, right, top and bottom. Relations distinguish visible background separation, a traceable target boundary, external overlap, unresolved contact and other uncertainty. All four regions must be directly resolved for clear support. A white neighbor merging into the target boundary yields UNKNOWN without inventing a physical cause. An actual externally hidden part remains occluded. The proposal rectangle is not assumed to contain the whole target.

`confirmation_policy` v2 finds a qualifying sequence ending at the current observed frame. For clear views at 0, 600 and 1,000 ms, the last view can use the 0-ms view rather than being rejected solely because the latest 600-ms view is only 400 ms away. The requirement remains two distinct, positively assessed source views at least 500 ms apart. All previous confidence, mask, frame-edge, distractor, identity and visibility hard gates remain. Every counted view additionally requires its own positive source-bound semantic assessment. Gaps and identifier changes clear all history. Unassessed current views remain UNKNOWN; old confirmation never substitutes for current visible evidence. Repeated JPEGs cannot count as new views within a continuity segment. Selection uses a bounded earliest-ending dynamic program, not future frames or reference labels.

## Fixed comparison

The two factors are crossed: old/new guard observations each feed the original greedy policy and the new current-view policy. Guard visibility and identity-plus-visibility support are also scored separately. This prevents an effective temporal veto from hiding a guard classification mistake, or a better guard from concealing a temporal regression.

- **Exposed episode 2:** the same sixteen references, exact SAM responses and v2 guard responses from the published visibility run. Only sixteen new v3 calls are made. This includes the previously observed `remote_new` frame-59 error; it is a development regression case, not fresh validation.
- **New episode 3:** two new source recordings in the same exposed dataset families. Each clip is tracked once, and both guards assess the same actual SAM candidates at eight preselected checkpoints. Reference labels and later boxes are scoring-only. Only authored frame-zero target/tape boxes initialize tracking.

Fresh screwdriver checkpoints are 0, 3, 5, 8, 15, 22, 32 and 37; the first three are clear. Fresh remote checkpoints are 0, 3, 5, 8, 15, 25, 40 and 59; the first four are clear. The remote's final white-on-white boundary remains UNKNOWN under the frozen visual review. Seven clear and nine insufficient-visibility references are authored before any new model call. No borderline cases are removed or relabeled after inference.

Native 50-fps videos are sampled every tenth frame into lossless 5-fps H.264 without resizing/cropping. All 100 new decoded frame MD5 values match the selected originals, and all 100 q2/yuvj444p JPEGs byte-replay from the derived clips. Thresholds remain confidence 0.8, two views, 500-ms spacing and localization IoU ≥0.3, with 0.2/0.5 sensitivity. Completion, coverage, positive retention and unsupported unclear support are reported together for each cohort. No first view is retrospectively supported by a future view.

## Execution and boundaries

The implementation, inputs, code hashes, labels, configuration and public plan are frozen and pushed before new inference. Admission is at most **50 new requests**, with zero automatic retries: sixteen v3 calls on exposed frames, two fresh SAM video calls, sixteen fresh v2 calls and sixteen fresh v3 calls. Missing candidates consume no guard call. Each Spark call uses explicit low reasoning and a 4,096-token cap. The estimated $3 allocation is advisory, not a provider dollar limit. Historical reused costs and new costs stay separate.

Offline replay validates raw-output classifications, exact source/crop/enlargement bytes and registered image order, source derivations, actual candidate boxes, baseline-score equality, request accounting and input immutability. Public exports contain source credits, classifications, hashes and metrics; raw media/provider bodies and arbitrary model observations stay private. Local hashes are consistency receipts, not signatures proving API origin.

The recordings share prior domains/cameras/tasks, and the near-contact references are assistant-authored visual judgments. The experiment does not establish cross-domain reliability, full-disappearance identity recovery, continuous visibility, physical execution or independent measurements of occlusion. Both new modules remain in experimental tracking, outside live placement acceptance.

## New online source credits

The external ALOHA authors created these recordings, distributed through LeRobot. Publisher-declared licenses were verified from freshly downloaded pinned dataset cards. The code does not redistribute the original media.

| Source | Declared license | Original SHA-256 |
| --- | --- | --- |
| [Screwdriver episode 3](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/resolve/51448b4e7638c1773fd07c1e7febce8ebb104635/videos/chunk-000/observation.images.cam_high/episode_000003.mp4) | [MIT](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/blob/51448b4e7638c1773fd07c1e7febce8ebb104635/README.md) | `6b69b8cab6a8c0702f12fb7a270eda723aecf2f9671bca3f517c4e5d48562c90` |
| [Battery/remote episode 3](https://huggingface.co/datasets/lerobot/aloha_static_battery/resolve/dd8bda435f5068a0298bd109005828768ecbfa12/videos/chunk-000/observation.images.cam_high/episode_000003.mp4) | [Apache-2.0](https://huggingface.co/datasets/lerobot/aloha_static_battery/blob/dd8bda435f5068a0298bd109005828768ecbfa12/README.md) | `2579796191bd0d20dec5d77463062357ef74b39b34227f2935b387b773acf9f3` |

The prospective public plan retains derived hashes, exact selection correspondence, scoring references and pinned revisions. Earlier episode-2 sources and the original remaining error are preserved in [the visibility report](VISIBILITY-RESULTS-20261029.md).
