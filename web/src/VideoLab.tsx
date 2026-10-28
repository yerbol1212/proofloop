import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import type { AcquisitionPlan, Status } from './types'
import { Icon } from './icons'
import { AcquisitionPanel } from './AcquisitionPanel'
import { EvaluationPanel } from './EvaluationPanel'
import type { EvaluationDTO } from './EvaluationPanel'

interface FollowupStep {
  case_id: string
  status: Status
  policy_status: AcquisitionPlan['status']
  requests: AcquisitionPlan['requests']
  source_timestamps_ms: number[]
  actual_request_count: number
}
interface FollowupTrajectory {
  id: string
  final_status: Status
  followup_attempts_used: number
  stop_reason: string
  steps: FollowupStep[]
}
interface FollowupReplay {
  maximum_followup_attempts: number
  conditions: Record<
    string,
    {
      policy_consumed_estimated_usd: number
      policy_consumed_request_count: number
      unused_counterfactual_case_ids: string[]
      trajectories: FollowupTrajectory[]
    }
  >
}

interface Metric {
  correct: number
  eligible_cases: number
  incorrect_passes: number
  correct_acceptances: number
  unknown_results: number
  ground_truth_counts: Record<Status, number>
}
interface LabCase {
  id: string
  expected: Status | null
  sam: Status | null
  spark: Status | null
  guided?: Status | null
  source_publisher: string
  source_url: string
  source_media_url: string
  license: { name: string; url: string; attribution: string }
  reasons: Record<string, string>
  annotation_reason: string
  images: {
    url: string
    width: number
    height: number
    source_timestamp_ms: number
  }[]
  regions: Record<string, [number, number, number, number]>
}
interface LabRun {
  id: string
  title: string
  contract_hash: string
  requirements: string
  task_summary?: string
  case_count: number
  source_count: number
  evaluation_role?: 'development_reprocessing' | 'new_source_holdback'
  evaluation_note?: string
  report_integrity?: {
    verdict_consistency_checked: boolean
    metrics_recomputed: boolean
    grounded_decisions_replayed: number
    case_decisions_checked: number
  }
  followup_policy_replay?: FollowupReplay
  evaluation?: EvaluationDTO
  metrics: Record<string, Metric>
  conditions: Record<
    string,
    {
      estimated_usd: number
      wall_seconds: number
      observation_version?: string
    }
  >
  limits: string[]
  cases: LabCase[]
}
const names: Record<string, string> = {
  sam: 'SAM + checker',
  spark: 'Spark alone',
  guided: 'Spark → SAM + checker',
}
function Result({ value }: { value: Status | null }) {
  return (
    <span className={`lab-result ${value || ''}`}>
      {value ? value[0].toUpperCase() + value.slice(1) : 'Unavailable'}
    </span>
  )
}

function RecordedFollowups({
  replay,
  selectedCase,
  onSelect,
}: {
  replay: FollowupReplay
  selectedCase: string
  onSelect: (caseId: string) => void
}) {
  const guided = replay.conditions.guided
  const selected = guided?.trajectories
    .flatMap((trajectory) =>
      trajectory.steps.map((step, index) => ({ step, index, trajectory })),
    )
    .find(({ step }) => step.case_id === selectedCase)
  return (
    <section
      className="panel followup-replay"
      aria-label="Bounded recorded follow-ups"
    >
      <h2>Ask for evidence, then stop</h2>
      <p>
        These future pairs were fixed before inference. Every follow-up is later
        than the preceding pair and uses new image bytes. This is recorded
        timeline analysis; no camera or robot followed these requests.
      </p>
      <details className="followup-details">
        <summary>Compare the frozen follow-up trajectories</summary>
        <div className="followup-conditions">
          {Object.entries(replay.conditions).map(([name, condition]) => (
            <div key={name}>
              <h3>{names[name] || name}</h3>
              {condition.trajectories.map((trajectory) => (
                <article className="followup-trajectory" key={trajectory.id}>
                  <strong>{trajectory.id.replaceAll('-', ' ')}</strong>
                  <div className="followup-steps">
                    {trajectory.steps.map((step, index) => (
                      <button
                        className={`button secondary ${step.case_id === selectedCase ? 'selected' : ''}`}
                        key={step.case_id}
                        onClick={() => onSelect(step.case_id)}
                        aria-pressed={step.case_id === selectedCase}
                      >
                        {index === 0 ? 'Initial pair' : `Follow-up ${index}`} ·{' '}
                        {step.source_timestamps_ms
                          .map((time) => `${(time / 1000).toFixed(1)}s`)
                          .join(' / ')}
                        <Result value={step.status} />
                      </button>
                    ))}
                  </div>
                  <p>
                    {trajectory.followup_attempts_used} /{' '}
                    {replay.maximum_followup_attempts} follow-ups ·{' '}
                    {trajectory.stop_reason === 'not_needed'
                      ? trajectory.final_status === 'pass'
                        ? 'goal verified; no further pair requested'
                        : 'known failure; no further pair requested'
                      : trajectory.stop_reason.replaceAll('_', ' ')}
                  </p>
                </article>
              ))}
              <p className="lab-limit">
                Policy consumed {condition.policy_consumed_request_count}{' '}
                requests · estimated $
                {condition.policy_consumed_estimated_usd.toFixed(4)}.
                {condition.unused_counterfactual_case_ids.length
                  ? ` ${condition.unused_counterfactual_case_ids.length} separately scored future pair was unused by this policy.`
                  : ''}
              </p>
            </div>
          ))}
        </div>
      </details>
      {selected && selected.step.status === 'unknown' ? (
        <>
          <h3>Guided pipeline: missing evidence at the selected pair</h3>
          <AcquisitionPanel
            plan={{
              schema_version: 1,
              status: selected.step.policy_status,
              inspection_only: true,
              physical_action_allowed: false,
              attempts_used: selected.index,
              max_attempts: replay.maximum_followup_attempts,
              attempts_remaining: Math.max(
                0,
                replay.maximum_followup_attempts - selected.index,
              ),
              unknown_check_ids: selected.step.requests.map(
                (request) => request.check_id,
              ),
              observed_fail_check_ids: [],
              requests: selected.step.requests,
              stop_reason:
                selected.step.policy_status === 'timeline_exhausted'
                  ? 'No further qualifying pair remains in the approved source schedule. This result stays unverified.'
                  : selected.step.policy_status === 'budget_exhausted'
                    ? 'The two-follow-up limit is reached. This result stays unverified.'
                    : null,
            }}
          />
        </>
      ) : null}
    </section>
  )
}

export function VideoLab() {
  const [runs, setRuns] = useState<LabRun[]>([])
  const [selectedRun, setSelectedRun] = useState('')
  const [selectedCase, setSelectedCase] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void fetch('/api/lab', { signal: controller.signal, cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok)
          throw new Error('The recorded video results could not be loaded.')
        const data = (await response.json()) as {
          available: boolean
          runs: LabRun[]
        }
        if (!controller.signal.aborted) setRuns(data.runs || [])
      })
      .catch((caught) => {
        if (!controller.signal.aborted)
          setError(
            caught instanceof Error
              ? caught.message
              : 'The video lab is unavailable.',
          )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [revision])
  const run =
    runs.find((item) => item.id === selectedRun) ||
    runs.find((item) => item.evaluation_role === 'new_source_holdback') ||
    runs[0]
  const current =
    run?.cases.find((item) => item.id === selectedCase) || run?.cases[0]
  if (loading)
    return (
      <div className="loading-panel" role="status">
        <span className="spinner" />
        Loading reviewed video results…
      </div>
    )
  return (
    <div className="video-lab">
      <div className="lab-toolbar">
        <p>
          <Icon name="image" />
          Prerecorded research footage · Results from actual Meta calls
        </p>
        <button
          className="button secondary"
          onClick={() => setRevision((value) => value + 1)}
        >
          <Icon name="refresh" />
          Refresh results
        </button>
      </div>
      {error ? (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      ) : null}
      {!run ? (
        <section className="panel empty-task">
          <h2>No reviewed runs are available.</h2>
          <p>
            Run a licensed recorded-video comparison and review its public
            summary to make it available here.
          </p>
        </section>
      ) : (
        <>
          <section className="panel lab-overview">
            <div className="lab-run-heading">
              <div>
                <h2>Evidence before claims</h2>
                <p>
                  {run.case_count} selected cases from {run.source_count}{' '}
                  recorded episodes.
                </p>
              </div>
              {runs.length > 1 ? (
                <label>
                  Experiment
                  <select
                    value={run.id}
                    onChange={(event) => {
                      setSelectedRun(event.target.value)
                      setSelectedCase('')
                    }}
                  >
                    {runs.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.title} · {item.contract_hash.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </div>
            <p className="lab-requirements">
              {run.task_summary || run.requirements}
            </p>
            {run.report_integrity?.metrics_recomputed &&
            run.report_integrity.verdict_consistency_checked ? (
              <p className="lab-audit-note">
                <Icon name="check" size={16} />
                Report totals checked against recorded decisions.{' '}
                {run.report_integrity.grounded_decisions_replayed} grounded
                decisions replayed.
              </p>
            ) : null}
            {run.evaluation_role ? (
              <p className="lab-limit">
                {run.evaluation_note ||
                  (run.evaluation_role === 'development_reprocessing'
                    ? 'Development replay: these examples were already exposed during debugging. This is not new-source evidence.'
                    : 'New-source check: these recordings were reserved until the revised method was frozen. The camera and task setup are still shared.')}
              </p>
            ) : null}
            <div
              className="lab-table-scroll"
              tabIndex={0}
              role="region"
              aria-label="Model comparison results; scroll horizontally to compare all columns"
            >
              <table className="lab-metrics">
                <caption>Observed results on this recorded sample</caption>
                <thead>
                  <tr>
                    <th scope="col">Condition</th>
                    <th scope="col">Matches labels</th>
                    <th scope="col">Incorrect passes</th>
                    <th scope="col">Accepts correct</th>
                    <th scope="col">Unknown</th>
                    <th scope="col">Est. API cost</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(run.metrics).map(([name, metric]) => (
                    <tr key={name}>
                      <th scope="row">
                        {names[name] || name}
                        {name === 'guided' ? (
                          <small className="lab-method-version">
                            {run.conditions[name]?.observation_version ===
                            'normalized_boxes_v2'
                              ? 'Normalized grounding v2'
                              : 'Original guided method'}
                          </small>
                        ) : null}
                      </th>
                      <td>
                        {metric.correct} / {metric.eligible_cases}
                      </td>
                      <td>{metric.incorrect_passes}</td>
                      <td>
                        {metric.correct_acceptances} /{' '}
                        {metric.ground_truth_counts.pass}
                      </td>
                      <td>{metric.unknown_results}</td>
                      <td>
                        ${(run.conditions[name]?.estimated_usd || 0).toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="lab-limit">
              These are selected placement checkpoints in recorded
              demonstrations. They do not establish general accuracy or a robot
              following ProofLoop’s repair.
            </p>
            {run.followup_policy_replay ? (
              <p className="lab-limit">
                The table includes all scored pairs. Policy totals below exclude
                unused future pairs.
              </p>
            ) : null}
            {run.evaluation ? (
              <EvaluationPanel
                evaluation={run.evaluation}
                selectedCase={current?.id || ''}
                onSelect={setSelectedCase}
              />
            ) : null}
          </section>
          {run.followup_policy_replay && current ? (
            <RecordedFollowups
              replay={run.followup_policy_replay}
              selectedCase={current.id}
              onSelect={setSelectedCase}
            />
          ) : null}
          {current ? (
            <div className="lab-case-workbench">
              <aside className="panel lab-case-list">
                <h2>Recorded cases</h2>
                <div className="case-buttons">
                  {run.cases.map((item) => (
                    <button
                      key={item.id}
                      className={current.id === item.id ? 'selected' : ''}
                      aria-pressed={current.id === item.id}
                      onClick={() => setSelectedCase(item.id)}
                    >
                      <span>{item.id.replaceAll('-', ' ')}</span>
                      <small>
                        Reviewed label: {item.expected || 'unscored'}
                      </small>
                    </button>
                  ))}
                </div>
              </aside>
              <section className="panel lab-case-detail">
                <div className="panel-heading">
                  <h2>{current.id.replaceAll('-', ' ')}</h2>
                  <span className="source-label">Recorded reference</span>
                </div>
                <div className="lab-photo-grid">
                  {current.images.map((image, index) => (
                    <figure key={image.url}>
                      <div
                        className="lab-image"
                        style={{
                          aspectRatio: `${image.width}/${image.height}`,
                        }}
                      >
                        <img
                          src={image.url}
                          width={image.width}
                          height={image.height}
                          alt={`Recorded robot view ${index + 1} for ${current.id}`}
                        />
                        {Object.entries(current.regions).map(
                          ([name, rect], n) => (
                            <div
                              key={name}
                              className={`lab-region region-${n}`}
                              style={
                                {
                                  left: `${rect[0] * 100}%`,
                                  top: `${rect[1] * 100}%`,
                                  width: `${(rect[2] - rect[0]) * 100}%`,
                                  height: `${(rect[3] - rect[1]) * 100}%`,
                                } as CSSProperties
                              }
                            >
                              <span>{name}</span>
                            </div>
                          ),
                        )}
                      </div>
                      <figcaption>
                        View {index + 1} ·{' '}
                        {(image.source_timestamp_ms / 1000).toFixed(1)} s
                      </figcaption>
                    </figure>
                  ))}
                </div>
                <p className="case-label">
                  <strong>Reviewed expectation:</strong>{' '}
                  <Result value={current.expected} /> ·{' '}
                  {current.annotation_reason}
                </p>
                <div className="lab-explanations">
                  {Object.keys(run.metrics).map((condition) => (
                    <div key={condition}>
                      <h3>
                        {names[condition] || condition}
                        <Result
                          value={
                            current[condition as 'sam' | 'spark' | 'guided'] ||
                            null
                          }
                        />
                      </h3>
                      <p>{current.reasons[condition]}</p>
                    </div>
                  ))}
                </div>
                <details className="lab-attribution">
                  <summary>Source, license and experiment limits</summary>
                  <p>
                    <strong>Approved requirements, verbatim:</strong>{' '}
                    {run.requirements}
                  </p>
                  <p>{current.license.attribution}</p>
                  <p>
                    <a
                      href={current.source_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Publisher dataset
                    </a>{' '}
                    ·{' '}
                    <a
                      href={current.license.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {current.license.name} license statement
                    </a>{' '}
                    ·{' '}
                    <a
                      href={current.source_media_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Original video
                    </a>
                  </p>
                  <ul>
                    {run.limits.map((limit) => (
                      <li key={limit}>{limit}</li>
                    ))}
                  </ul>
                </details>
              </section>
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}
