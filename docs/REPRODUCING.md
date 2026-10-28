# Reproduce a small online-video demo

A public clone can prepare a bounded, licensed recording example without an API key. Preparation downloads exact pinned source bytes, extracts a fixed set of original frames, retains attribution and license evidence, and writes an **unapproved** contract. It never makes a model request. The original private model responses and recordings are not in Git.

## Check the dependency-free software first

```bash
git clone https://github.com/yerbol1212/proofloop.git
cd proofloop
python3 -m unittest discover -s tests -v
python3 -m proofloop.reproduce
```

The final command prints the source revision, expected SHA-256, byte budget and four fixed frame timestamps. It does not access the network or inspect local credentials. Python 3.9 or newer is sufficient for these commands.

## Prepare four frames from one real recording

Install FFmpeg on your system and ensure `ffmpeg` is on `PATH` or beside the Python executable. Then run:

```bash
python3 -m proofloop.reproduce prepare --download \
  --output artifacts/local/reproduction-coffee
```

This acquires only **9,970,951 bytes**: the pinned 9,964,446-byte episode 000000 and its 6,505-byte dataset card. Both require their exact frozen sizes and SHA-256 digests. Downloads stop at that bound and cannot replace an existing file; wrong or incomplete data is rejected. No fallback mirror or newer dataset revision is substituted.

The four requested source timestamps are **0.500, 1.500, 20.000 and 21.000 seconds**. Actual decoded timestamps are retained, rather than treating seek requests as measured capture time. The frames remain 640 × 480, without generated content, compositing or changed object placement. JPEG bytes can vary by FFmpeg version, so the newly decoded hashes are bound into this clone's own contract instead of being presented as bit-identical historical inference inputs.

Files remain in ignored folders:

```text
inputs/local/reproduction/coffee-placement/
  dataset-card.md
  MIT-aloha-LICENSE.txt
  ATTRIBUTION.txt
  episode_000000.mp4
  frames/
artifacts/local/reproduction-coffee/
  contract-draft.json
  extraction.json
  preparation.json
```

The draft labels say the glass is first in staging (**FAIL** for the authored dispense gate), then inside the dispense region (**PASS**). This is a normal demonstration's intermediate placement; it does not show the robot making an error or following ProofLoop's recommendation.

View all four frames. Review the region rectangles, concept, unique-target assumption, two-sample stability rule and labels in `contract-draft.json`. Only after reviewing the exact prepared contract, use the hash printed by preparation:

```bash
python3 -m proofloop.reproduce approve \
  --output artifacts/local/reproduction-coffee \
  --reviewed-hash PASTE_THE_EXACT_REVIEWED_CONTRACT_HASH
```

This records the invoking operator's review assertion and verifies the media again; it is not independent proof that review occurred. A changed draft or media fails validation. Existing preparation or approval artifacts are not overwritten. For another preparation, use a fresh checkout or remove only the generated reproduction profile and its output folder after keeping any evidence you need.

## Optional real Meta comparison

Use Python 3.10 or newer, install `pip install '.[sam]'`, and save `META_API_KEY` in the ignored `.env.local`. The project uses Muse Spark and SAM through Meta's API. Approval and preparation make zero API calls; this **separate** command permits paid inference:

```bash
python3 -m proofloop.benchmark run \
  artifacts/local/reproduction-coffee/manifest-approved.json \
  --output artifacts/local/reproduction-coffee-run-01 \
  --condition guided --condition spark --max-requests 8 \
  --evaluation-role development_reprocessing
```

The cap is eight requests for two cases, with automatic retries disabled. The actual count may be lower when localization is unsupported. Cost estimates are not provider billing limits. A fresh call can produce different model results; preparation alone does not reproduce historical accuracy or authenticate an API response. The model method, labels, thresholds and source bytes remain explicit.

## Fixed future-evidence profile

For the follow-up experiment's exact public schedule, select:

```bash
python3 -m proofloop.reproduce plan --profile coffee-acquisition
python3 -m proofloop.reproduce prepare --profile coffee-acquisition --download \
  --output artifacts/local/reproduction-acquisition
```

This is **29,889,192 bytes**, three previously exposed source episodes and twelve fixed frames. The [follow-up report](../artifacts/feasibility/fixed-followup-20261028.md) explains the two-attempt limit, strictly later images, early stopping, unused counterfactual inference and source exhaustion. Preparation does not execute that policy or run inference. This profile contains three UNKNOWN and three PASS labels and no FAIL control; it cannot establish a wrong-placement false-acceptance rate.

## Source rights and verification

The [pinned dataset card](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/1c52c4c794b6f93d931489263cf15e4331600bfc/README.md) explicitly declares the **dataset** MIT. Its card digest is `da783ef780c38b7a85c85db2a03dad304adfcd1de48b69b0618cf1eeab4024c8`. ALOHA authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine and Chelsea Finn, RSS 2023; credit Hugging Face / LeRobot for conversion. The [original ALOHA project](https://tonyzhaozh.github.io/aloha/) and [MIT notice](https://github.com/tonyzhaozh/aloha/blob/main/LICENSE) are retained as attribution, rather than treating a code license alone as the basis for video rights.

After publication, the walkthrough was executed from an anonymous HTTPS clone at commit `5e143149626fd6677a9305c8d70ab39c39ee08ee`, with no source overlays, API key or approval. Python 3.9.6 ran all 450 tests successfully, with one expected optional SAM-parser skip. Preparation downloaded exactly 9,970,951 bytes, decoded four distinct frames, verified source/frame hashes and wrote an unapproved draft with zero Meta calls. The tracked checkout remained clean. Nine offline safeguards also cover bounded downloads, exact hashes, missing download authorization, unchanged cached bytes, private-path confinement and exact-hash approval. Private verification logs remain ignored; published model results retain their separate provenance. This agent-run check does not establish another person's ease of use.
