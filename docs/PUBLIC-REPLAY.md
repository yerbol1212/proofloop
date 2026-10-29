# Public tracking replay walkthrough

This read-only walkthrough works from a public checkout or complete installed workbench with empty data storage and no Meta key. It recomputes tracking policy decisions and scores from published cached measurements. It includes no original video pixels, new model inference, independent contact remeasurement or physical execution.

## Start an isolated offline workbench

Build the frontend from the checkout first:

```sh
cd web
npm ci
npm run build
cd ..
.venv/bin/python -m proofloop.server --offline --port 8791 --data-dir artifacts/local/public-walkthrough
```

For a complete installed wheel, use `proofloop serve --offline --port 8791 --data-dir artifacts/local/public-walkthrough`. Open <http://127.0.0.1:8791/> and select **Video lab → Public tracking replay**. The public inspector works even when there are no private recorded-video runs. The key is not needed; offline mode rejects new live inspection, tracking and Spark compilation requests before admission. Existing admitted jobs and authored task presets remain separate.

`proofloop doctor` now checks the offline workbench profile by default, including the actual replay, frontend assets and writable data storage. Use `proofloop doctor --profile core` only when intentionally checking a Python-only installation without the UI. Neither command checks provider authentication or credits.

## Explore three failures, then a supported proof

1. **Contact candidate withheld** selects `remote_new`, frame 59. Compare the three recomputed method decisions and the near-gripper proposal. Current-view confirmation is already UNKNOWN; the warning did not uniquely prevent acceptance here. A near proposal is uncertainty, not verified contact.
2. **Clear target unnecessarily withheld** selects `screwdriver_new`, frame 5. The scoring annotation is clear. Current-view confirmation supports it; the combined warning method returns UNKNOWN. One gripper proposal has reported confidence about 0.544 and a cached distance of three source pixels. That is a concrete false warning, preserved in the cohort totals.
3. **Incomplete assessment stays unscored** selects `remote_3`, frame 5. The original network outcome had no completed assessment. The positive remains in the denominator of seven, and the inspector distinguishes the effective policy state from the unscored outcome.
4. Select `remote_new`, frame 5, then expand the current-view confirmation proof. The source-bound proof ends at the selected checkpoint, uses distinct image receipts and satisfies the original 500-ms spacing. Hashes check consistency with the packaged record; they do not authenticate capture time or API origin.

Both cohort tables remain visible. On episode 2, current-view confirmation retains **4/6** clear references, while adding warnings retains **3/6**. On episode 3 both retain **4/7**, including one unscored positive in that denominator. Both current-view variants support **0/10** and **0/9** unclear references here. Retention additionally requires localization IoU at least 0.3. These small exposed development cohorts do not establish future reliability or superiority over Spark alone.

## What the walkthrough proves

The endpoint verifies the packaged bundle and frozen policy dependencies, recomputes all method decisions, checks the published score reports, and exposes a closed projection of 32 authored checkpoints across four recordings. Reference annotations remain scoring-only fields; policy inputs do not contain those labels. Selecting checkpoints, reloading the replay and opening proof details use only local GET requests and leave the active inspection task unchanged.

The [source register](ONLINE-VIDEO-SOURCES.md) and the inspector retain pinned dataset revisions, license links and source hashes. Original imagery, raw model responses and credentials remain private. The [harsh review](HARSH-REVIEW-20261029.md) records unresolved physical-loop, independent-validation and user-demand gaps. The [three-minute demo](DEMO-SCRIPT.md) follows this public path without presenting cached analysis as live behavior.
