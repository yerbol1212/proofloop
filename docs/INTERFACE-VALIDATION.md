# Interactive interface validation — 28 October 2026

## Task editor and recorded Video Lab milestone

- Python 3.12: 321 offline tests pass. The backend milestone `b6a5ebe` passed GitHub CI on Python 3.9 and 3.12. The final interface production build also passes strict TypeScript checks.
- A real Spark draft correctly changed the fork requirement to RIGHT and spoon to LEFT. Drafting preserved the active contract and its evidence. Exact-draft activation was separately exercised through HTTP under the user's development authorization, recorded as assistant review, and invalidated old evidence. Browser checks confirmed activation stays disabled before review, and edited inputs require a new draft. No assistant action was labeled human approval.
- Task mode and visual descriptions are part of the reviewed contract. The interface identifies text prompts to SAM separately from descriptions sent to the guided Spark locator. Supported compilation remains observable placement, with temporal and hidden-state requirements explicitly excluded.
- The read-only Video Lab displayed both actual experiments: eight pilot cases from four coffee recordings, then four cases from two new recordings. It preserved the original SAM failures, baseline results, UNKNOWN outcomes, original frame pairs, frozen labels, license attribution and links to publisher videos. Opening or switching lab views made no model calls.
- The guided comparison displayed 2/4 label matches, concept-only SAM 0/4 and Spark alone 4/4. These results establish neither superiority nor general accuracy. The footage is prerecorded; it does not show an operator or robot following our proposed correction.
- An isolated server used the existing authored mixed-evidence fixture: known wrong fork plus unknown spoon. The browser displayed **More evidence is needed**, a targeted spoon inspection, and **Capture requested views**, without physical move instructions. Clicking it cleared the old evidence and required new captures. This establishes UI behavior only, not perception accuracy.
- Desktop 1536 × 1024 and mobile 390 × 844 were checked in the built-in browser. The task editor and Video Lab both kept document/body width at 390 px. The comparison table scrolls inside its own focusable region. Source attribution, experiment switching and case switching worked.
- Private final screenshots: `qa/proofloop-ui/video-lab-final-desktop.jpg`, `video-lab-final-mobile.jpg`, `task-editor-final-mobile.jpg`, and `mixed-evidence-inspection.jpg` in the parent workspace. Original phone photos, downloaded videos, raw responses and QA images remain outside the public repository.

## Original camera interface milestone

## Checks and provenance

- Python 3.12: 228 offline tests pass. Python 3.9: 228 run, one optional official-parser test skipped.
- Frontend: strict TypeScript checks and Vite production build pass. Dependencies are pinned in `web/package-lock.json`.
- Browser: built-in Codex in-app browser, desktop at 1536 × 1024 and mobile at 390 × 844. No Browser plugin was installed; the built-in browser supplied DOM controls, file choosers, console inspection and screenshots. No external Playwright process was used.
- The normal local server loaded the genuine historical misplaced-fork result after verifying the original media, frozen contract and saved SAM responses. Starting repair immediately displayed unknown and fresh-capture controls.
- An isolated local test server injected explicitly authored masks. Through the browser, two-photo upload → numeric calibration → scene/timing confirmations → wrong-placement result → Start repair → two different inputs → passing result worked. The evidence log identified the injected client and declared timing. These inputs and outputs establish no perception or physical success.
- The isolated API fixture also produced occlusion/unknown and a failed request; browser inspection confirmed unknown/unverified screens. Stopping that server produced the disconnected state and disabled mutations. No provider body or credential appeared.
- The capture token expired during a prolonged browser file-picker operation. Submission was rejected with specific new-inspection guidance; no model request ran.
- Verification stayed disabled before both images and confirmations. A new revision cleared prior uploads and confirmations. Browser console inspection showed no application errors during the completed flows.
- At mobile width, document and body widths were 390 px: no horizontal overflow. Real hardware camera capture and a new operator-following-the-repair physical correction were not exercised.

## Design comparison ledger

The image-generated desktop and mobile concepts preceded frontend implementation. Their private reference photos and all QA screenshots remain local. Both concepts and the latest screenshots were inspected using the image viewer.

| Point | Concept → implementation |
| --- | --- |
| Header and hierarchy | Plain ProofLoop wordmark, centered navigation, New inspection, large problem heading and single supporting sentence match the intended hierarchy. The new Task and Video Lab tabs extend the original two-tab navigation. |
| Evidence composition | Two portrait views, region labels and detection boxes remain side by side on desktop. The app preserves the entire original photos with neutral letterboxing; the generated concept had tightened their framing. Evidence is never a generated mockup. |
| Inspector and action | Per-target fail/pass rows lead into the bounded repair and indigo Start repair button. Compact instruction copy keeps the action and four-step rail visible at the concept viewport. |
| Typography and color | Self-hosted Inter, cool white canvas, navy type, thin borders, indigo controls and text-plus-color status indicators follow the design tokens. |
| Responsive capture flow | Mobile stacks the heading, evidence inputs, calibration, confirmations and action in the concept order. Functional upload/camera buttons and a numeric calibration alternative are added. |

## Copy differences

The repair specifies “from RIGHT to LEFT” because the planner knows the wrong region. “Keep both items visible and hands clear” condenses the primary instruction; capture spacing is explicit at submission. Capture mode uses “Placement to capture,” and an unknown result uses “Required placement,” so a target description is not presented as a measured repair proposal. Placeholder text says “Choose a photo,” because the local phone workflow transfers files to this computer. The app adds truthful historical-source, orientation, expiry, upload-timing and error guidance absent from the static concept.

## Remaining validation

The runtime uses the real Meta client with no retries. With two to six targets, concept mode permits at most 4–12 SAM calls; guided mode adds one Spark request for a 5–13 request cap. The original camera-interface milestone spent no additional model credits; the later live task draft and recorded comparisons have their own public-safe reports. A new person-following-the-proposal test still needs two fresh supported photos after Start repair. Guided mode requires upright JPEGs no larger than 1280 pixels. Phone LAN access, HEIC conversion and real camera hardware behavior remain outside this version's verified scope. The earlier real Spark/SAM phone experiment retains its separate recorded-session evidence.
