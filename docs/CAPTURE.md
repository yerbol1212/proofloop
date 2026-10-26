# Record the first real ProofLoop check

Record **one continuous 47–50 second phone video**. It will test positive tool location, wrong placement, abstention when the tray is hidden, and fresh evidence after you correct it. It does not yet validate exact counts, automatic action recognition, or industrial reliability.

## Set up before pressing record

1. Use **one screwdriver and one pair of pliers**. Choose ordinary, clearly different tools large enough to see. No tiny screws or additional tools.
2. Put a plain tray or a sheet of paper on a well-lit table. Mark two generous regions with a vertical divider. Put **LEFT** and **RIGHT** labels outside the regions, with left and right defined as seen in the video. Keep the objects completely inside their regions, away from the divider and edges.
3. Fix the phone above the table in **landscape, 1080p, 30 fps**. A tripod or stable stand is enough. The whole tray and both outside labels must remain in frame. Keep this position and zoom for the entire clip. Use even lighting and avoid strong shadows and glare.
4. Start with the **screwdriver in LEFT and pliers in RIGHT**, separated and fully visible. Have an **opaque board or a plain piece of cardboard** ready that can cover the whole tray and both tools.
5. Keep faces, documents, screens, addresses and other private background material out of view. Use your own recording and objects.

## The exact sequence

Use the phone's recording timer or a stopwatch outside the frame. The stable holds matter more than perfect timing. Keep recording if you are a second or two late; the annotation offsets can be corrected afterward.

| Time | What you do | What must be visible during the hold |
| --- | --- | --- |
| **0–8 seconds** | Start in the correct state. Get your hands out by second 2. Hold still. | Screwdriver LEFT, pliers RIGHT; both fully visible. |
| **8–13 seconds** | Move **only the screwdriver** into RIGHT. Put it beside the pliers without overlapping. Remove your hands. | The actual movement stays in the recording. |
| **13–21 seconds** | Hold the wrong state without touching anything. | Both tools RIGHT; both fully visible. |
| **21–26 seconds** | Cover the **whole tray and both tools** with the opaque board. Remove your hands. | The cover hides both tools completely. |
| **26–34 seconds** | Leave the board still and your hands out. | Neither tool is visible through or around the board. |
| **34–39 seconds** | Remove the board, then move the screwdriver back to LEFT. Leave the pliers RIGHT. Remove your hands. | The uncovering and correction stay in the recording. |
| **39–47 seconds** | Hold the corrected state still, then stop recording around second 48–50. | Screwdriver LEFT, pliers RIGHT; both fully visible in fresh frames. |

**Keep the original continuous video.** Do not trim it, splice sections, add overlays, interpolate frames, or replace the final correct state with the first one. The two correct holds must be different moments after a visible physical correction.

## Where to put it

Copy the original MOV or MP4 to this project folder:

```text
inputs/local/phone/01-correction-loop.MOV
```

Use `.mp4` if that is your phone's original format; capitalization of the extension does not matter. Copy the original file directly from the phone. These originals, extracted frames and capture annotations are ignored by Git and stay local. Tell Codex when the file is there. If the timing differs, say which part changed; we will inspect and adjust the plan before any model calls.

## What we will check next

We will inspect the original and approve the LEFT/RIGHT rectangles on a displayed extracted frame. Those rectangles must match the actual camera image; the generated-scene rectangles are not reused automatically. The default plan samples **3 and 5 seconds**, **16 and 18 seconds**, **29 and 31 seconds**, and **42 and 44 seconds**. Each pair comes from different original timestamps separated by two seconds, with hands out and the scene stable.

The timeline and your labels are human annotations. They do not tell the model what to answer. SAM will receive the extracted images and tool concepts. A zero detection does not prove absence or reveal a covered tool. A changed or covered scene invalidates older observations; the final correct state needs new post-correction evidence. Our first phone pass will use eight images and two concepts, at most 16 new SAM segmentation requests. No live phone-camera result is claimed until that run succeeds.

If both extracted images are byte-identical, we will retain them honestly and the verifier will not count their hashes as two distinct visual sources. We may need another longer hold or a fresh capture. Do not edit a frame to manufacture distinct evidence.

## Optional second clip: a valid alternate order

After the first clip works, record a separate 20–25 second clip in the same setup. Begin with both tools beside the tray, put the **pliers RIGHT first**, then the **screwdriver LEFT**, and finish with at least six seconds of both tools fully visible and your hands out. Save the original as `inputs/local/phone/02-alternate-order.MOV` or `.mp4`. This can check that a protocol with no required order accepts another valid sequence. The current still-image extraction alone cannot establish action tracking or infer a causal ordering rule.

## Local preflight and extraction commands

The local tool uses `ffprobe` and `ffmpeg` from PATH or beside the Python executable (the configured project runtime has them in `.venv/bin/`). Use a current FFmpeg release; the local integration check used version 9.0. The tool never installs software, reads API credentials or calls a model. `preflight` records original dimensions, phone display rotation, duration and SHA-256. Extraction applies display rotation, limits the longest JPEG edge to 1280 pixels, strips copied metadata, checks the 5 MiB input ceiling and preserves **actual decoded media timestamps**, rather than assuming a requested seek offset was the frame returned. Timestamps are relative to the first decoded video frame, not wall-clock capture time.

Run from the repository root:

```bash
.venv/bin/python -m proofloop.capture preflight inputs/local/phone/01-correction-loop.MOV
.venv/bin/python -m proofloop.capture plan --output inputs/local/phone/capture-plan.json
```

Watch the original, edit the plan's stage start/end times and sample offsets to match stable, hands-free moments, then explicitly confirm that review:

```bash
.venv/bin/python -m proofloop.capture extract inputs/local/phone/01-correction-loop.MOV \
  --plan inputs/local/phone/capture-plan.json \
  --output inputs/local/phone/run-01 \
  --confirm-offsets
```

Output is `inputs/local/phone/run-01/manifest.json` plus eight local JPEGs. Use a fresh output directory for another extraction; the tool never overwrites an evidence run. Stage IDs are `initial_correct`, `wrong_region`, `occluded`, and `corrected`. Each frame has its relative filename, hash, requested offset, actual source timestamp in seconds and milliseconds, dimensions and size. The source file path is relative to the manifest directory, so a complete local capture folder can be moved together. Failed extractions leave `manifest.incomplete.json`, which is not usable as a completed capture.

`--confirm-offsets` confirms human annotation review. It does not prove physical ownership, model accuracy, complete visibility or a successful repair. The manifest preserves these distinctions and remains pending visual review.
