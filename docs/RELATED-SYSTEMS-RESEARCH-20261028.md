# Related systems and practices to adopt

Research date: 28 October 2026. This is a comparison with published systems, not a claim that ProofLoop reproduces their results. The sources below are papers, author project pages and official repositories. No external source code, model weights or media were copied for this comparison.

ProofLoop already combines reviewed task contracts, three-valued checks, bounded evidence requests, finite repair planning, linked receipts and a preparation gate. Its measured guided Spark→SAM pipeline has not outperformed Spark alone on the present label-matching checks. The useful next contribution is to make the reviewer see **what the evidence supports, where it diverges from the reviewed requirement, and which explanation or next action remains unproven**. A failure explanation and a later passing state are different claims.

## Six closest precedents

| System | What the primary source establishes | Relationship to ProofLoop | Transferable practice |
| --- | --- | --- | --- |
| **Inner Monologue** — CoRL 2022 | Combines success detection, passive scene descriptions and active questions as feedback to language-based robot planning; evaluated on simulated and real manipulation. [Author project page](https://innermonologue.github.io/), [paper](https://proceedings.mlr.press/v205/huang23c/huang23c.pdf). | Feedback-driven replanning and requests for information are established ideas. ProofLoop does not yet execute robotic skills or demonstrate physical feedback-driven recovery. | Keep success checks, scene observations and information requests as separate event types. Associate each new observation with the exact question or intended action, rather than treating every uploaded image as a generic next turn. |
| **REFLECT / RoboFail** — CoRL 2023 | Builds hierarchical sensory, event and subgoal summaries; analyzes failed subgoals before broader planning failures; supplies explanations to a correction planner. Its public examples include wrong objects, dropped items and missed procedure steps. [Author project page](https://robot-reflect.github.io/), [official repository](https://github.com/real-stanford/reflect). | The closest precedent for explainable failure→repair. ProofLoop currently has stronger explicit distinctions between an evidence-consistency receipt, operator declarations and observed action, but it supports a far narrower task vocabulary. This is an implementation distinction, not a demonstrated performance advantage. | Show an expected-versus-observed discrepancy card backed by check IDs and source references. Separate the supported symptom from an unknown physical cause. Do not generate a confident causal story from two final-state images. |
| **KnowNo / Robots That Ask For Help** — CoRL 2023 | Uses a calibration dataset and conformal prediction to construct action prediction sets; ambiguity triggers human help. Its guarantees depend on its setup and assumptions, including the help protocol. [Author project page](https://robot-help.github.io/), [paper](https://arxiv.org/abs/2307.01928), [official code](https://github.com/google-research/google-research/tree/master/language_model_uncertainty). | ProofLoop's typed UNKNOWN and finite follow-up budget address a related interaction problem. A fixed SAM score threshold is not a calibrated probability or conformal guarantee. | Preserve a finite help policy and measure both answered checks and requests consumed. If calibration is later attempted, reserve independent recordings and report calibration assumptions; do not relabel current heuristic thresholds as guaranteed confidence. |
| **SayCan** — 2022 | Grounds language-based skill selection in available pretrained skills and their value functions, distinguishing task relevance from contextual feasibility. [Author project page](https://say-can.github.io/), [official code](https://github.com/google-research/google-research/tree/master/saycan). | ProofLoop's repair is minimum cost only within its declared action model. A semantically sensible instruction does not establish that a person or robot can safely execute it in the current scene. | Keep proposed moves distinct from executable capabilities. A future hardware adapter should expose its own capability/precondition check and execution result; a UI click must not supply either fact. |
| **AHA / FailGen** — ICLR 2025 | Learns failure reasoning using procedurally perturbed simulation demonstrations, then evaluates explanations and downstream recovery in multiple settings. [Author project page](https://robofail-vlm.github.io/), [paper](https://arxiv.org/abs/2410.00371), [official repository](https://github.com/NVlabs/AHA). | A generic VLM already competes on failure explanation. ProofLoop has no comparable training set or independently established semantic identity detector. | Use an explicit failure taxonomy and paired controls, rather than adding more successful clips. Separate real recording tests, edited/synthetic perturbations and task-contract counterfactuals in all reporting. |
| **Robo2Process / RoboProcessBench** — October 2026 paper revision | Diagnoses process understanding with separate static and dynamic question families, source-native trace labels, local process units, and episode/recording isolation. Temporal ordering and local progress are evaluated separately from end-state recognition. [Versioned paper](https://arxiv.org/html/2606.13040v4), [version history](https://arxiv.org/abs/2606.13040). | Current ProofLoop placement checks test a narrow static property. Agreement on a final image does not measure contact, action order, local progress, or whether the requested action caused the result. | Split state-check performance from process-check performance. Only introduce a process claim when the chosen recording supplies its necessary observations and labels. Preserve source isolation; multiple windows from one episode are not independent sources. |

The table contains selective summaries. Numerical claims from the papers are deliberately omitted because their tasks, sensors, models and evaluation units differ from ProofLoop's small development experiments.

## Ranked implementation recommendations

The priorities below are engineering judgments based on the comparison and the current repository, rather than prescriptions or claims from any one paper.

### 1. Evidence-backed discrepancy explanations

Adapt REFLECT's separation of observed execution from explanation, using the data ProofLoop actually has. For every nonpassing check, construct a deterministic diagnostic record containing:

- Reviewed requirement: check ID, target, predicate and expected region.
- Observed support: supported region or the typed evidence gap, plus relevant source/view identifiers.
- Decision: PASS, FAIL or UNKNOWN and the direct checker reason.
- Scope: an observed mismatch, an evidence insufficiency, or a freshness/contract issue.
- Next authorized action: the existing bounded repair or acquisition request.
- Causal limit: the physical reason for the mismatch is unobserved unless a separately supported process check establishes it.

For example: “The reviewed task requires the glass in DISPENSE. Both supported observations place it in STAGING. Placement fails. A move to DISPENSE is proposed; why the glass is there and whether a move occurred are unverified.” For an uncertain mask, the explanation must instead name missing support and request the relevant view. This avoids free-form failure stories that could overstate identity, absence or causality. No additional inference is needed to render existing structured evidence.

Acceptance checks: mixed FAIL/UNKNOWN does not become an authorized move; changing the active task invalidates old explanations; stale/historical observations are labeled; diagnostic sources bind to the same revision and task; no hidden raw API responses, local media paths or keys enter a public summary. A diagnostic is useful when a reviewer can recover its statement from the recorded check, not when its language sounds confident.

### 2. Paired failure controls with traceable labels

Adapt FailGen's emphasis on failures and RoboProcessBench's diagnostic task separation. Start with real licensed footage already acquired and maintain a frozen list of pair IDs, source hashes, task hashes, labels, failure category and exposure status before inference. The current candy check has only UNKNOWN and PASS labels; add wrong-placement controls before using it to discuss false acceptance.

The fastest first test is **a reviewed task-contract counterfactual**: keep a supported recorded observation fixed and evaluate two explicit expected regions. A supported target in DISPENSE should satisfy a DISPENSE requirement and fail a STAGING requirement. This tests contract sensitivity and checker behavior; it is neither an independently observed physical failure nor a new source. Guided locators should remain task-independent, and any reused calls must be reported as reused. A Spark judgment conditioned on a changed contract is a new condition and must not silently reuse an old verdict.

Follow with genuinely different episodes containing wrong objects, multiple visually similar targets, occlusion and camera changes. If real failure clips cannot be licensed or unambiguously labeled, keep those cases as separate authored tests rather than presenting them as real-world accuracy. Report per-category false acceptance, correct rejection, abstention, resolution attempts and actual extra inference cost alongside overall label matching. Do not pool new controls with previously published experiments or count edited frames as independent recordings.

Acceptance checks: preserved baseline prompt/model version; no retuning labels after model responses; exact sample denominator; uncertainty about identity remains UNKNOWN; no completed-procedure claim from a zone check. The test should expose mistakes, including mistakes by ProofLoop, rather than manufacture an improved headline score.

### 3. One small genuine process check

Add an authored, observable prerequisite→outcome check only after the first two improvements are complete. An online sequence can establish chronological source observations when timestamps and intermediate views are present. It cannot establish that the robot responded to ProofLoop. Missing the transition must remain “outcome observed; execution unverified.” A public preparation gate is not yet full procedure verification.

### 4. Calibrated help and hardware adapters later

Calibrating abstention requires more independent labeled recordings than the current pilots supply. A hardware adapter likewise needs an actual capability/execution interface. Both are reasonable future directions; implementing their labels without evidence would weaken the demonstration. Preserve the current bounded help policy and operator-declared continuation while building the evaluation base.

## Candid differentiation

“AI sees a failed task, explains it and suggests a repair” overlaps substantially with existing work. “AI asks a clarifying question” does too. The project should not present either as a previously unconsidered research problem.

ProofLoop's candidate contribution is a small, usable review workbench for **reviewed visual requirements, conservative evidence gaps, explicit repair scope, and replayable decision lineage** using Meta models. The source-bound receipt validates consistency; it is not a signature, a trusted capture timestamp, proof of API origin or proof of physical execution. The strongest next demo shows an honest failure, the exact evidence request, a blocked gate, fresh supporting observations, and a reviewer replaying the record. Demonstrated superiority to the systems above, practical customer demand, independent identity validation and physical closed-loop recovery remain open.

Adopt these design practices through new event-created implementation. Any future reuse of external code, weights, datasets or media requires checking its specific license and recording attribution separately; an open paper or repository link alone does not grant rights to every associated asset.
