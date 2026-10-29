# Preregistered new-source-family placement pilot

This declaration is published before the study's Meta requests. It fixes nine paired cases from three recordings in two original dataset families, using 18 extracted original views. Current guided v2 and the existing same-pair direct Spark prompt receive the same evidence. Both use the existing low-reasoning transport profile; no confidence gate or perception policy is changed. The persistent study allowance is at most 36 attempts, with no automatic paid retries.

The [machine-readable preregistration](../artifacts/feasibility/new-source-preregistration-20261029.json) fixes all source/frame hashes, concepts, protocols, region calibration, annotations, code hashes, per-part request limits and both clock mappings. Private media, raw responses and direct-user authorization stay ignored locally. The annotation review is by two agents under the user's instruction, not an independent human panel.

| Source | Target gate | Fixed labels |
| --- | --- | --- |
| BridgeData V2 scripted trajectory 0 | Blue-handled fork wholly RIGHT | FAIL, UNKNOWN, UNKNOWN |
| Berkeley UR5 episode 0 | Green cloth wholly LEFT | UNKNOWN, UNKNOWN, UNKNOWN |
| Berkeley UR5 episode 4 | Brown receiving cup wholly LEFT | PASS, PASS, UNKNOWN |

The first six fixed cases contained no strict positive reference. They are all retained; the third source adds a pre-inference clear-target control. All nine cases, including incomplete and unattempted outcomes, remain in their denominators. Results will be reported per recording and source family. This is a curated pilot, not an official benchmark or a representative generalization claim.

## Source rights and clocks

BridgeData V2's [original publisher](https://rail-berkeley.github.io/bridgedata/) explicitly licenses all data CC BY 4.0. The pinned [VyoJ conversion card](https://huggingface.co/datasets/VyoJ/BridgeData-V2-Scripted-Videos/blob/d6c6064ca8aca851cae1b4f05e72732177490420/README.md) identifies the original scripted subset and preserves original rights. Credit Homer Walke and coauthors, CoRL 2023, RAIL-Berkeley and the VyoJ conversion. The clip is encoded at 1 fps for 50 seconds, while its published original capture timestamps span 11.768 seconds. The study maps the selected decoded frame ordinals to the original per-frame capture timestamps, retaining encoded PTS separately. It does not treat slowed playback as physical stability.

The pinned [LeRobot Berkeley UR5 dataset card](https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5/blob/6306aa91c00b9b0de28831b7fceaa42b8096bb4f/README.md) declares CC BY 4.0 for the dataset. The original homepage's license was not established; the basis is explicitly the redistributor's dataset declaration. Credit Lawrence Yunliang Chen, Simeon Adebola, Ken Goldberg and Hugging Face LeRobot. These clips retain native 5 fps decoded PTS. No original video pixels are republished here.

These frames belong to normal prerecorded demonstrations. A task-relative FAIL does not mean the original robot made a procedural error. Neither a later PASS nor the brown-cup control proves nesting, task completion or an operator following our repair. New-to-project footage may have appeared in model training.

## Planned evaluation

Report all class denominators, completed predictions, missing/incomplete attempts, confusion, unsupported acceptance, correct acceptance, UNKNOWN rate, reported token usage, estimated cost, uncertain charges and wall time. A failed model response is unscored, not fabricated UNKNOWN. Auth, billing and rate-limit HTTP failures stop further admissions; ordinary case failures do not erase the other method. Every private sidecar and source binding is replay-verified for publication. The source-selection and annotation process is authored and exposed; thresholds will not be changed after seeing results.
