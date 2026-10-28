# Local interactive inspection

The workbench connects the browser to the existing SAM grounding, placement checks and bounded repair planner. A historical owned recording can initialize a misplaced fork; it is labeled **Recorded reference**. A correction is verified only from a newly submitted pair of images. There is no model fallback.

## Start

Use Node 24 and Python 3.10 or newer for the official SAM parser. From the repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[sam]'
cd web
npm ci
npm run build
cd ..
.venv/bin/python -m proofloop.server --port 8790
```

Open <http://127.0.0.1:8790>. Keep `META_API_KEY` in the ignored `.env.local`; the browser never receives it. If the local owned-phone report and original files are available, the server checks their hashes and replays its wrong-region result. A fresh public clone starts with no evidence. The interface needs the Python server; Vite's development server by itself has no model backend.

## Phone workflow

1. Click **Start repair**. This clears the current verdict immediately. Move the fork wholly into the handwritten LEFT area; leave the spoon wholly in RIGHT.
2. Keep the paper flat and the camera fixed. Include all of both utensils and both labeled areas. Remove your hand and avoid glare. Take two new still photos at least one second apart after starting the repair.
3. Transfer both photos to this computer and choose **Upload two photos**. Use upright JPEG or PNG files, each at most 5 MiB, with matching dimensions and orientation. HEIC is not supported in this first version; export upright JPEG/PNG before uploading. A laptop camera is an alternative: **Use camera** requires explicit permission and captures a pair after you position it.
4. Mark **LEFT** and **RIGHT** on the first image. Confirm the regions match both photos and the scene has exactly one fork and one spoon. For uploaded photos, confirm you took the pair after starting this repair, at least one second apart.
5. Click **Verify correction**. The backend makes at most four SAM requests, with no automatic retries, and checks the returned masks against the approved regions. A wrong location fails; ambiguity or missing evidence stays unknown.

The earlier portrait recording has LEFT above the divider and RIGHT below it. Your new camera orientation may differ: the handwritten labels determine the regions. Avoid the divider and image edges so each utensil is clearly inside its area.

After any movement, click **New inspection** before capturing again. A passing snapshot ages out after 30 seconds. It describes the inspected images, not continuous monitoring.

## Evidence and limits

The evidence log exposes the source, revision, reasons and invalidation trace. Upload timing is an operator declaration; camera timing is client-reported. Tokens and known image hashes reject accidental reuse but cannot prove capture time or defeat a dishonest operator who modifies an old photo. The unique-target assumption is explicitly confirmed, not inferred from an absence of detections.

Private uploads, raw responses and run records remain in ignored `artifacts/local/interactive-*` directories. The server binds only to IPv4 loopback and validates Host, Origin and a per-server mutation token. It serves compiled assets and explicitly registered media, never the repository directory. It has no remote hosting or phone LAN access in this version.

Offline injected-client tests exercise pass/fail/unknown and failure paths. They are labeled test fixtures and are never substituted in the running app. The initial phone observation experiment used real Meta models; a new person-following-the-proposal correction needs fresh user captures to establish physical interactive closure.
