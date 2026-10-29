# Second correctness review

29 October 2026. An independent adversarial agent reviewed the recovery, workflow, SDK and public replay paths at commit `59ea985`. This is a repository and software-behavior review, not a Meta endorsement or evidence of model accuracy.

The review found no new confirmed P1 or P2 defect in those paths. It did find one P3 presentation error: a point-in-time submission lookup returning `found=false` was described as proof that the original request had never been admitted. An original POST can still be in flight. The existing code correctly retained its opaque recovery intent and blocked another capture until an explicit new inspection replaced the binding.

`web/src/App.tsx` now says that no original job was found yet, that submission may still be in flight, and that the operator can reload to check again or choose New inspection before capturing new evidence. The change preserves intent retention, binding invalidation and the single-submission behavior. No policy, model prompt, confidence threshold or frozen perception implementation changed.

## Checks

The targeted Python 3.12 command passed **66 tests**:

```sh
.venv/bin/python -m unittest tests.test_recovery tests.test_workflow tests.test_replay_bundle tests.test_infrastructure_http tests.test_job_recovery
```

The first sandboxed invocation could not bind a loopback socket. The authorized retry passed; the initial permission failure was not an application failure. These checks used authored fixtures and made no Meta requests.

The strict TypeScript and production frontend build passed. Browser plugin was unavailable, so the existing Playwright harness used installed Chrome against its own isolated local server, with authored fake inference. Desktop 1440×1000 and mobile 390×844 checks confirmed the revised message is visible, fresh capture remains blocked across reloads, and New inspection changes the revision before enabling capture. The harness also recovered one deliberately lost 202 response to the original job: one submission, one dispatch and three authored fake calls. Denied browser storage sent no additional inspection POST.

Page identity, meaningful content, absence of a framework overlay, screenshot evidence, interaction checks and mobile horizontal overflow checks passed. There were no page errors or unexpected console errors; the deliberately aborted 202 caused the expected transport-console error. The isolated server was stopped after verification. The active workbench on port 8790 was not mutated.

Private QA scripts, JSON report and screenshots are saved under `artifacts/local/prospective-20261029/`, including `submission-recovery-browser-qa.json`, `submission-lookup-pending-desktop.png` and `submission-lookup-pending-mobile.png`.

## Limits

These checks establish the tested software behavior. They do not establish perception superiority, camera freshness, a completed physical correction, unseen-source performance, independent usability or practical demand. No real model inference or paid Meta usage was part of this review or wording fix. Prospective model evaluation is a separate experiment with its own evidence and limitations.
