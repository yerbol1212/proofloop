# First physical capture

Use owned objects and keep private material, faces, documents and identifying backgrounds out of view. Put files in `inputs/local/`; these are ignored by Git. PNG/JPEG images are enough for the first access and observation gate. Short phone MP4 clips follow after still-image grounding works.

## Scene

Fix the phone above a plain tray or tabletop with left and right regions. Choose two or three large distinct objects, ideally a screwdriver and pliers plus a visible divider. Avoid tiny screws and visually identical parts. Leave the target objects unobstructed for a few seconds at checkpoints.

## Four independent captures

1. **Correct:** the required objects are fully visible in their declared regions.
2. **Visible deviation:** move one object into the wrong region. This gives positive evidence of a contradiction without relying on a detector's zero matches to prove absence.
3. **Unknown:** cover the critical region or close the tray. The system must request another view, not certify hidden contents.
4. **Correction:** restore the correct state and take a fresh image or recording. This must be new evidence after the action, not the original correct frame reused.

Keep the original files. Record your declared object locations and which areas are visible separately before model evaluation. Any manual anchors, region selections or repeated captures must be counted in setup effort. Images test a checkpoint; they do not establish temporal tracking or a complete workflow from video.

## Useful optional clips

Record one valid alternate ordering and one case where a previously checked item is moved while obscured by a hand. The former must be accepted if the confirmed protocol permits it; the latter must invalidate the old check until a fresh view is available.
