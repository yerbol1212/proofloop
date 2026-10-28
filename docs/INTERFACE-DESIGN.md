# Interactive workbench design

The local interface follows two image-generated concepts made before frontend implementation: a desktop mistaken-placement view and a mobile capture view. The concepts include private owned photos and remain in ignored local artifacts; they are design references, never evidence or application backgrounds.

## Visual system

Inter, self-hosted; navy text `#172334`, muted text `#657387`, cool background `#F6F8FA`, white panels, borders `#DDE3EA`, indigo action `#3157DD`, red failure `#B93A32`, green success `#18744E`. Desktop header and content use 32–48 px gutters, 24 px panel padding, 16–24 px gaps, 10–12 px corners. The desktop work area is approximately two-thirds evidence and one-third inspector. Mobile stacks the same workflow without horizontal scrolling.

## Components and behavior

- Header: ProofLoop, Inspection / Evidence log, New inspection.
- Heading: the current problem or next action, with one sentence of context.
- Evidence panel: two entire source images, view labels, normalized region overlays and actual returned detection boxes. Recorded references are always labeled historical.
- Inspector: per-object status and required/observed region, followed by a suggested repair and primary action.
- Capture panel: explicit camera activation or two file uploads, two previews, LEFT/RIGHT rectangle calibration, scene and freshness confirmations, then Verify correction.
- Trace: actual evidence invalidation, observation and verification events; future steps are never shown complete.

Start repair invalidates the old result immediately. A proposal is hypothetical. Verification requires two distinct new inputs and four live SAM requests, then the existing deterministic checker. Errors and ambiguous observations stay unknown. The server keeps credentials and private media outside the public bundle.

## Essential copy

“Fork in the wrong area.” / “Move the fork to LEFT, then show two new views.”

“Current evidence” / “Recorded reference” / “Placement checks” / “Suggested repair” / “Start repair”

“Capture the correction.” / “Fork in LEFT. Spoon in RIGHT.” / “Old evidence cleared. Two new views required.”

“Upload two photos” / “Use camera” / “Set LEFT” / “Set RIGHT” / “Verify correction”

“The marked areas match my paper, with one fork and one spoon.”

“I took both photos after starting this repair, at least one second apart.”

Uploads use operator-declared timing; camera times are client-reported. Neither is proof against a dishonest operator. Known byte-identical prior inputs are rejected. The app reports an observation at a time, never continuous certification of the physical scene.
