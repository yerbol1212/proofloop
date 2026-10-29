import { useEffect, useState } from 'react'
import './ReplayInspector.css'

const methods = ['v2_greedy', 'v2_current', 'v2_veto_current'] as const
type Method = (typeof methods)[number]
type Decision = 'supported' | 'unknown'
interface MethodResult {
  status: Decision | 'unscored'
  policy_status: Decision
  gap_codes: string[]
  confirmation_views: number
  confirmation_ends_at_current: boolean | null
  confirmation_evidence: {
    frame_index: number
    timestamp_ms: number
    frame_sha256: string
  }[]
  reported_confidence: number | null
  semantic_decision: string
}
interface Checkpoint {
  frame_index: number
  timestamp_ms: number
  frame_sha256: string
  reference: { expected: Decision; condition: string }
  scoring_status: 'scored' | 'unscored'
  warning: {
    code: string
    veto: boolean
    comparisons: {
      gripper_track_id: number
      reported_confidence: number | null
      distance_px: number | null
      distance_is_lower_bound: boolean
      available: boolean
      near: boolean
    }[]
  }
  methods: Record<Method, MethodResult>
}
interface ReplayData {
  schema_version: 'tracking_replay_inspector_v1'
  mode: 'cached_demo'
  scope: 'cached_policy_and_score_replay_only'
  provider_requests: 0
  video_pixels_included: false
  changes_live_placement_acceptance: false
  physical_action_authorized: false
  bundle_sha256: string
  config?: {
    min_confidence: number
    min_stable_views: number
    min_spacing_ms: number
    warning_margin: number
    iou_threshold: number
  }
  method_order: Method[]
  method_labels: Record<Method, string>
  cohorts: {
    id: string
    title: string
    methods: Record<Method, {
      positives_retained: number
      positive_count: number
      positive_unscored: number
      unsupported_support_on_unknown: number
      unknown_count: number
      labels_total: number
      labels_unscored: number
    }>
  }[]
  clips: {
    clip_id: string
    cohort: string
    source_sha256: string
    source_attribution: {
      dataset: string
      episode: number
      license: string
      license_url: string
      url: string
      revision: string
    }
    checkpoints: Checkpoint[]
  }[]
  highlights: {
    id: string
    title: string
    clip_id: string
    frame_index: number
    note: string
  }[]
  limitations: string[]
}

const words = (value: string) => value.replaceAll('_', ' ')
const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const natural = (value: unknown): value is number =>
  finite(value) && Number.isInteger(value) && value >= 0
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string')
const hash = (value: unknown): value is string =>
  typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)
const decision = (value: unknown): value is Decision =>
  value === 'supported' || value === 'unknown'
const nullableNumber = (value: unknown) => value === null || finite(value)
const safeLink = (value: string) => /^https?:\/\//.test(value) ? value : undefined

// Reject an incompatible projection instead of rendering partial or invented scores.
function parseReplay(value: unknown): ReplayData {
  const invalid = () => { throw new Error('The public replay report is incomplete or incompatible. Retry after updating the workbench.') }
  if (!record(value) || !record(value.inspector)) return invalid()
  const data = value.inspector
  if (
    data.schema_version !== 'tracking_replay_inspector_v1' ||
    data.mode !== 'cached_demo' || data.scope !== 'cached_policy_and_score_replay_only' ||
    data.provider_requests !== 0 || data.video_pixels_included !== false ||
    data.changes_live_placement_acceptance !== false || data.physical_action_authorized !== false ||
    !hash(data.bundle_sha256) || !record(data.method_labels) ||
    !Array.isArray(data.method_order) || data.method_order.join(',') !== methods.join(',') ||
    !methods.every((method) => typeof (data.method_labels as Record<string, unknown>)[method] === 'string') ||
    !Array.isArray(data.cohorts) || data.cohorts.length !== 2 ||
    !Array.isArray(data.clips) || data.clips.length !== 4 ||
    !Array.isArray(data.highlights) || !strings(data.limitations)
  ) return invalid()
  for (const cohort of data.cohorts) {
    if (!record(cohort) || typeof cohort.id !== 'string' || typeof cohort.title !== 'string' || !record(cohort.methods)) return invalid()
    for (const method of methods) {
      const score = cohort.methods[method]
      if (!record(score) || !['positives_retained', 'positive_count', 'positive_unscored', 'unsupported_support_on_unknown', 'unknown_count', 'labels_total', 'labels_unscored'].every((field) => natural(score[field]))) return invalid()
    }
  }
  for (const clip of data.clips) {
    if (!record(clip) || typeof clip.clip_id !== 'string' || typeof clip.cohort !== 'string' || !hash(clip.source_sha256) || !record(clip.source_attribution) || !natural(clip.source_attribution.episode) || !['dataset', 'license', 'license_url', 'url', 'revision'].every((field) => typeof (clip.source_attribution as Record<string, unknown>)[field] === 'string') || !Array.isArray(clip.checkpoints) || clip.checkpoints.length !== 8) return invalid()
    for (const point of clip.checkpoints) {
      if (!record(point) || !natural(point.frame_index) || !natural(point.timestamp_ms) || !hash(point.frame_sha256) || !record(point.reference) || !decision(point.reference.expected) || typeof point.reference.condition !== 'string' || (point.scoring_status !== 'scored' && point.scoring_status !== 'unscored') || !record(point.warning) || typeof point.warning.code !== 'string' || typeof point.warning.veto !== 'boolean' || !Array.isArray(point.warning.comparisons) || !record(point.methods)) return invalid()
      for (const comparison of point.warning.comparisons) {
        if (!record(comparison) || !natural(comparison.gripper_track_id) || !nullableNumber(comparison.reported_confidence) || !nullableNumber(comparison.distance_px) || typeof comparison.distance_is_lower_bound !== 'boolean' || typeof comparison.available !== 'boolean' || typeof comparison.near !== 'boolean') return invalid()
      }
      for (const method of methods) {
        const result = point.methods[method]
        if (!record(result) || (!decision(result.status) && result.status !== 'unscored') || !decision(result.policy_status) || !strings(result.gap_codes) || !natural(result.confirmation_views) || (result.confirmation_ends_at_current !== null && typeof result.confirmation_ends_at_current !== 'boolean') || !nullableNumber(result.reported_confidence) || typeof result.semantic_decision !== 'string' || !Array.isArray(result.confirmation_evidence)) return invalid()
        for (const proof of result.confirmation_evidence) {
          if (!record(proof) || !natural(proof.frame_index) || !natural(proof.timestamp_ms) || !hash(proof.frame_sha256)) return invalid()
        }
      }
    }
  }
  for (const highlight of data.highlights) {
    if (!record(highlight) || !['id', 'title', 'clip_id', 'note'].every((field) => typeof highlight[field] === 'string') || !natural(highlight.frame_index) || !data.clips.some((clip) => clip.clip_id === highlight.clip_id && clip.checkpoints.some((point: Checkpoint) => point.frame_index === highlight.frame_index))) return invalid()
  }
  if (data.config !== undefined && (!record(data.config) || !['min_confidence', 'min_stable_views', 'min_spacing_ms', 'warning_margin', 'iou_threshold'].every((field) => finite((data.config as Record<string, unknown>)[field])))) return invalid()
  return data as unknown as ReplayData
}

function MethodCard({ label, result }: { label: string; result: MethodResult }) {
  return (
    <article className="replay-method">
      <h4>{label}</h4>
      <strong className={`replay-decision ${result.status}`}>{result.status.toUpperCase()}</strong>
      {result.status === 'unscored' ? <p>Policy decision: {result.policy_status.toUpperCase()}. No completed assessment; this positive remains in the retention denominator.</p> : null}
      <p>{result.confirmation_views} confirmation views · {result.confirmation_ends_at_current === true ? 'ends at this checkpoint' : result.confirmation_ends_at_current === false ? 'does not end at this checkpoint' : 'current endpoint not exposed'}</p>
      <p><strong>Gaps:</strong> {result.gap_codes.length ? result.gap_codes.map(words).join('; ') : 'None reported'}</p>
      <p>Effective policy semantic decision: {words(result.semantic_decision)}. Reported confidence: {result.reported_confidence === null ? 'unavailable' : result.reported_confidence.toFixed(3)}.</p>
      <details>
        <summary>Confirmation proof ({result.confirmation_evidence.length} references)</summary>
        {result.confirmation_evidence.length ? (
          <ol className="replay-proof">
            {result.confirmation_evidence.map((proof) => <li key={`${proof.frame_index}-${proof.frame_sha256}`}>Frame {proof.frame_index} · {(proof.timestamp_ms / 1000).toFixed(3)} s<code>{proof.frame_sha256}</code></li>)}
          </ol>
        ) : <p>No confirmation evidence returned for this method.</p>}
      </details>
    </article>
  )
}

export function ReplayInspector() {
  const [data, setData] = useState<ReplayData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [revision, setRevision] = useState(0)
  const [selectedClip, setSelectedClip] = useState('remote_new')
  const [selectedFrame, setSelectedFrame] = useState(59)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void fetch('/api/v1/replay', { signal: controller.signal, cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) throw new Error('The public replay could not be loaded. No model request was made.')
        const report = parseReplay(await response.json())
        if (!controller.signal.aborted) setData(report)
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : 'The public replay is unavailable.')
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [revision])
  const clip = data?.clips.find((item) => item.clip_id === selectedClip) || data?.clips[0]
  const point = clip?.checkpoints.find((item) => item.frame_index === selectedFrame) || clip?.checkpoints[0]
  function select(clipId: string, frameIndex?: number) {
    setSelectedClip(clipId)
    setSelectedFrame(frameIndex ?? data?.clips.find((item) => item.clip_id === clipId)?.checkpoints[0]?.frame_index ?? 0)
  }
  return (
    <section className="panel replay-inspector" aria-labelledby="replay-title">
      <div className="replay-heading">
        <div><h2 id="replay-title">Public tracking replay</h2><p>Cached measurements · 0 API calls · works without a Meta key</p></div>
        <button className="button secondary" disabled={loading} onClick={() => setRevision((value) => value + 1)}>{loading ? 'Loading replay…' : 'Reload public replay'}</button>
      </div>
      <p className="replay-scope">Recompute tracking decisions and scores from published measurements. No video pixels are included and no new inference runs. This read-only diagnostic cannot change live placement acceptance or authorize a physical action.</p>
      {loading ? <p role="status">Loading the portable policy and score replay…</p> : error ? <div className="notice notice-error" role="alert">{error}</div> : data && clip && point ? (
        <>
          <div className="replay-cohorts">
            {data.cohorts.map((cohort) => (
              <section className="replay-cohort" key={cohort.id} aria-label={`${cohort.title} scores`}>
                <h3>{cohort.title}</h3>
                <div className="lab-table-scroll" tabIndex={0} role="region" aria-label={`${cohort.title} score comparison; scroll horizontally`}>
                  <table className="lab-metrics"><caption>Retain clear targets without supporting unclear ones</caption><thead><tr><th scope="col">Method</th><th scope="col">Clear retained</th><th scope="col">Unclear supported</th><th scope="col">Unscored</th></tr></thead><tbody>
                    {methods.map((method) => { const score = cohort.methods[method]; return <tr key={method}><th scope="row">{data.method_labels[method]}</th><td>{score.positives_retained} / {score.positive_count}</td><td>{score.unsupported_support_on_unknown} / {score.unknown_count}</td><td>{score.labels_unscored} ({score.positive_unscored} positive)</td></tr> })}
                  </tbody></table>
                </div>
              </section>
            ))}
          </div>
          <p className="replay-scope">These are exposed development clips. Reference labels are used for scoring only; they are not inputs to the tracking policy. A zero in the unclear-supported column is a result on this small sample, not a reliability guarantee.</p>
          <p className="replay-scope">Clear retained requires supported localization{data.config ? ` at IoU ≥ ${data.config.iou_threshold}` : ' at the frozen IoU threshold'}. Unscored positives remain in the retention denominator. {data.config ? `Policy settings: confidence ≥ ${data.config.min_confidence}, ${data.config.min_stable_views} views spaced ≥ ${data.config.min_spacing_ms} ms; gripper warning margin ${data.config.warning_margin} px.` : ''}</p>
          <div className="replay-highlights" aria-label="Important replay cases">
            {data.highlights.map((highlight) => <button className="button secondary" key={highlight.id} aria-pressed={clip.clip_id === highlight.clip_id && point.frame_index === highlight.frame_index} onClick={() => select(highlight.clip_id, highlight.frame_index)}><strong>{highlight.title}</strong><span>{highlight.note}</span></button>)}
          </div>
          <div className="replay-clips" role="group" aria-label="Select a replay clip">
            {data.clips.map((item) => <button className="button secondary" key={item.clip_id} aria-pressed={clip.clip_id === item.clip_id} onClick={() => select(item.clip_id)}>{words(item.clip_id)}<span>{words(item.cohort)}</span></button>)}
          </div>
          <div className="replay-checkpoints" role="group" aria-label={`Reviewed checkpoints for ${clip.clip_id}`}>
            {clip.checkpoints.map((item) => <button key={item.frame_index} className="button secondary" aria-pressed={point.frame_index === item.frame_index} onClick={() => setSelectedFrame(item.frame_index)}>Frame {item.frame_index}<span>{(item.timestamp_ms / 1000).toFixed(2)} s · {item.scoring_status === 'unscored' ? 'unscored' : item.reference.expected}</span></button>)}
          </div>
          <section className="replay-checkpoint" aria-labelledby="replay-checkpoint-title">
            <h3 id="replay-checkpoint-title">{words(clip.clip_id)} · frame {point.frame_index} · {(point.timestamp_ms / 1000).toFixed(3)} s</h3>
            <p><strong>Scoring reference:</strong> {point.reference.expected.toUpperCase()} · {words(point.reference.condition)}. {point.scoring_status === 'unscored' ? 'This positive has no completed assessment and remains in the retention denominator.' : 'This checkpoint is included in the score.'}</p>
            <div className={`replay-warning ${point.warning.veto ? 'active' : ''}`}>
              <strong>Gripper warning: {words(point.warning.code)}</strong>
              <p>{point.warning.veto ? 'The warning vetoes support in the combined method. It is not evidence that contact really occurred.' : 'No warning veto applies at this checkpoint. This does not establish absence of contact.'}</p>
              {point.warning.veto && point.methods.v2_current.policy_status === 'unknown' ? <p>The current-view method already returns UNKNOWN here. The warning does not uniquely prevent acceptance at this checkpoint.</p> : null}
              {point.warning.comparisons.length ? <ul>{point.warning.comparisons.map((comparison) => <li key={comparison.gripper_track_id}>Track {comparison.gripper_track_id}: {comparison.available ? comparison.distance_px === null ? 'distance unavailable' : `${comparison.distance_is_lower_bound ? 'at least ' : ''}${comparison.distance_px.toFixed(2)} px cached mask distance` : 'comparison unavailable'} · {comparison.near ? 'near target' : 'no near-target flag'} · reported confidence {comparison.reported_confidence === null ? 'unavailable' : comparison.reported_confidence.toFixed(3)}</li>)}</ul> : null}
            </div>
            <div className="replay-methods">{methods.map((method) => <MethodCard key={method} label={data.method_labels[method]} result={point.methods[method]} />)}</div>
            <details className="replay-source"><summary>Source binding, attribution and replay limits</summary>
              <dl><dt>Dataset / episode</dt><dd>{clip.source_attribution.dataset} / {clip.source_attribution.episode}</dd><dt>Source revision</dt><dd><code>{clip.source_attribution.revision}</code></dd><dt>Source SHA-256</dt><dd><code>{clip.source_sha256}</code></dd><dt>Checkpoint image SHA-256</dt><dd><code>{point.frame_sha256}</code></dd><dt>Replay bundle SHA-256</dt><dd><code>{data.bundle_sha256}</code></dd></dl>
              <p><a href={safeLink(clip.source_attribution.url)} target="_blank" rel="noreferrer">Source dataset</a> · <a href={safeLink(clip.source_attribution.license_url)} target="_blank" rel="noreferrer">{clip.source_attribution.license}</a></p>
              <ul>{data.limitations.map((limit) => <li key={limit}>{limit}</li>)}</ul>
            </details>
          </section>
        </>
      ) : <p role="status">No reviewed public checkpoints are available.</p>}
    </section>
  )
}
