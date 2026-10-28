import { useEffect, useState } from 'react'
import type { Status } from './types'
import { Icon } from './icons'
import { SemanticTrialPanel } from './SemanticTrialPanel'
import type { SemanticTrialSection } from './SemanticTrialPanel'
import { CropConfirmationPanel } from './CropConfirmationPanel'
import type { CropConfirmationSection } from './CropConfirmationPanel'

interface Cost {
  request_count: number
  estimated_usd: number
}
interface Control {
  case_id: string
  source: { publisher: string; page_url: string; license: { name: string } }
  original_contract: {
    required_region: string
    guided: { status: Status }
    spark: { status: Status }
  }
  changed_contract: {
    required_region: string
    guided: { status: Status }
    spark: { status: Status }
  }
  costs: {
    original_guided_perception: Cost
    original_spark_reference: Cost
    incremental_guided: Cost
    incremental_spark: Cost
  }
}
interface ContractSummary {
  case_count: number
  new_independent_source_count: number
  incremental_spark_request_count: number
  incremental_guided_request_count: number
  incremental_estimated_usd: number
  controls: Control[]
  limits: string[]
}
interface SemanticCase {
  case_id: string
  source_family: string
  source_id: string
  expected_identity: 'supported' | 'unknown'
  challenge: {
    decision: 'supported' | 'unknown'
    reasons: string[]
    usage: { request_attempts: number; estimated_usd: number }
  }
}
interface SemanticSummary {
  cases: SemanticCase[]
  unavailable_cases?: {
    case_id: string
    expected_identity: 'supported' | 'unknown'
    status: string
    attempt_indices: number[]
    failure_evidence?: string
    raw_response_retained?: boolean
  }[]
  metrics: {
    requested_cases: number
    completed_cases: number
    source_recordings: number
    expected_identity_matches: number
    negative_candidate_pairs: number
    scored_negative_pairs: number
    negative_pairs_withheld: number
    positive_candidate_pairs: number
    scored_positive_pairs: number
    positive_pairs_supported: number
  }
  cost_summary: {
    actual_requests: number
    completed_scored_requests: number
    unscored_incomplete_requests: number
    estimated_usd: number
  }
  limitations: string[]
}
interface Section<T> {
  available: boolean
  status: 'unavailable' | 'invalid' | 'validated'
  title: string
  note: string
  summary?: T
}
export interface ResearchLabDTO {
  schema_version: number
  read_only: true
  paid_requests_on_open: 0
  note: string
  semantic: Section<SemanticSummary>
  contract_stress: Section<ContractSummary>
  completion_v2?: SemanticTrialSection
  distractors_v2?: SemanticTrialSection
  crop_confirmation?: CropConfirmationSection
}

function Verdict({ value }: { value: Status }) {
  return <span className={`lab-result ${value}`}>{value}</span>
}

export function ResearchLab() {
  const [data, setData] = useState<ResearchLabDTO | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void fetch('/api/research-lab', {
      cache: 'no-store',
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok)
          throw new Error('The research evidence is unavailable.')
        const value = (await response.json()) as ResearchLabDTO
        if (!controller.signal.aborted) setData(value)
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setData(null)
          setError('The saved research evidence could not be replayed.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [revision])
  const contracts = data?.contract_stress.summary
  const semantics = data?.semantic.summary
  return (
    <section className="panel" aria-label="Research controls">
      <h2>Research controls</h2>
      <p className="task-intro">
        Development tests challenge the candidate and the requirement
        separately. They use previously exposed recordings and do not add
        physical recovery evidence.
      </p>
      <button
        className="button secondary"
        disabled={loading}
        onClick={() => setRevision((value) => value + 1)}
      >
        <Icon name="refresh" />
        {loading ? 'Replaying saved evidence…' : 'Replay saved evidence'}
      </button>
      <p className="capture-footnote">
        This read-only replay makes no model requests and changes no active
        task.
      </p>
      {error ? <p role="status">{error}</p> : null}
      {data ? (
        <>
          {data.completion_v2 ? (
            <SemanticTrialPanel section={data.completion_v2} />
          ) : null}
          {data.distractors_v2 ? (
            <SemanticTrialPanel section={data.distractors_v2} />
          ) : null}
          {data.crop_confirmation ? (
            <CropConfirmationPanel section={data.crop_confirmation} />
          ) : null}
          <section
            className="acquisition-plan"
            aria-label="Candidate identity challenge"
          >
            <h3>{data.semantic.title}</h3>
            <p>{data.semantic.note}</p>
            {semantics ? (
              <>
                <p>
                  Supported means this additional same-model check supports the
                  candidate. It grants no placement PASS or independent proof of
                  identity.
                </p>
                <p>
                  {semantics.metrics.completed_cases} /{' '}
                  {semantics.metrics.requested_cases} reviewed pairs produced a
                  scored result. Label agreement:{' '}
                  {semantics.metrics.expected_identity_matches} /{' '}
                  {semantics.metrics.completed_cases} scored pairs. Scored
                  negative controls: {semantics.metrics.scored_negative_pairs} /{' '}
                  {semantics.metrics.negative_candidate_pairs}.
                </p>
                <p>
                  Positive controls:{' '}
                  {semantics.metrics.positive_pairs_supported} supported,{' '}
                  {semantics.metrics.scored_positive_pairs -
                    semantics.metrics.positive_pairs_supported}{' '}
                  UNKNOWN,{' '}
                  {semantics.metrics.positive_candidate_pairs -
                    semantics.metrics.scored_positive_pairs}{' '}
                  unscored, out of {semantics.metrics.positive_candidate_pairs}{' '}
                  reviewed pairs.
                </p>
                {semantics.cases.map((item) => (
                  <article className="evidence-gap" key={item.case_id}>
                    <h3>{item.case_id.replaceAll('-', ' ')}</h3>
                    <p>
                      Reviewed expectation: {item.expected_identity}. Recorded
                      challenge: <strong>{item.challenge.decision}</strong>.
                    </p>
                    <p>
                      {item.challenge.reasons.length
                        ? item.challenge.reasons
                            .map((reason) => reason.replaceAll('_', ' '))
                            .join('; ')
                        : 'The reviewed candidate criteria were supported in both views.'}
                    </p>
                    <p className="contract-id">
                      {item.challenge.usage.request_attempts} additional request
                      · estimated $
                      {item.challenge.usage.estimated_usd.toFixed(4)}
                    </p>
                  </article>
                ))}
                {semantics.unavailable_cases?.map((item) => (
                  <article className="evidence-gap" key={item.case_id}>
                    <h3>{item.case_id.replaceAll('-', ' ')}</h3>
                    <p>
                      Unscored:{' '}
                      {item.status === 'incomplete'
                        ? 'the run recorded an incomplete response.'
                        : item.status === 'api_error'
                          ? 'the model request failed.'
                          : 'the response did not pass validation.'}{' '}
                      No supported or UNKNOWN classification is available.
                    </p>
                    <p>{item.attempt_indices.length} recorded attempts.</p>
                    {item.raw_response_retained === false ? (
                      <p className="lab-limit">
                        Only the observed error classification and usage ledger
                        were retained for these attempts. Their raw provider
                        response bodies are unavailable; the recorded completion
                        status cannot be independently replayed from those
                        bodies.
                      </p>
                    ) : null}
                  </article>
                ))}
                <p className="lab-limit">
                  {semantics.cost_summary.actual_requests} total requests,
                  including{' '}
                  {semantics.cost_summary.unscored_incomplete_requests}{' '}
                  incomplete, unscored requests · estimated $
                  {semantics.cost_summary.estimated_usd.toFixed(4)}. Same-model
                  agreement remains correlated.
                </p>
                <details className="followup-details">
                  <summary>Semantic experiment limits</summary>
                  <ul className="gap-instructions">
                    {semantics.limitations.map((limit) => (
                      <li key={limit}>{limit}</li>
                    ))}
                  </ul>
                </details>
              </>
            ) : null}
          </section>
          <section
            className="acquisition-plan"
            aria-label="Changed requirement controls"
          >
            <h3>{data.contract_stress.title}</h3>
            <p>{data.contract_stress.note}</p>
            {contracts ? (
              <>
                <p>
                  Each pair uses the same images with a separately reviewed
                  opposite-region requirement. These authored controls are not
                  natural execution failures or new-source accuracy tests.
                </p>
                {contracts.controls.map((control) => (
                  <article className="evidence-gap" key={control.case_id}>
                    <h3>{control.case_id.replaceAll('-', ' ')}</h3>
                    <div className="followup-conditions">
                      <div>
                        <strong>Original requirement</strong>
                        <p>
                          {control.original_contract.required_region.toUpperCase()}
                        </p>
                        <p>
                          Guided{' '}
                          <Verdict
                            value={control.original_contract.guided.status}
                          />
                        </p>
                        <p>
                          Spark{' '}
                          <Verdict
                            value={control.original_contract.spark.status}
                          />
                        </p>
                      </div>
                      <div>
                        <strong>Changed requirement</strong>
                        <p>
                          {control.changed_contract.required_region.toUpperCase()}
                        </p>
                        <p>
                          Guided{' '}
                          <Verdict
                            value={control.changed_contract.guided.status}
                          />
                        </p>
                        <p>
                          Spark{' '}
                          <Verdict
                            value={control.changed_contract.spark.status}
                          />
                        </p>
                      </div>
                    </div>
                    <p>
                      Original guided perception:{' '}
                      {control.costs.original_guided_perception.request_count}{' '}
                      requests, estimated $
                      {control.costs.original_guided_perception.estimated_usd.toFixed(
                        4,
                      )}
                      . Changed requirement:{' '}
                      {control.costs.incremental_guided.request_count} new
                      guided requests and{' '}
                      {control.costs.incremental_spark.request_count} new Spark
                      request, estimated $
                      {control.costs.incremental_spark.estimated_usd.toFixed(4)}
                      .
                    </p>
                    <p className="contract-id">
                      <a
                        href={control.source.page_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {control.source.publisher} ·{' '}
                        {control.source.license.name}
                      </a>
                    </p>
                  </article>
                ))}
                <p className="lab-limit">
                  {contracts.case_count} controls ·{' '}
                  {contracts.new_independent_source_count} new independent
                  sources · {contracts.incremental_spark_request_count} new
                  Spark requests · estimated $
                  {contracts.incremental_estimated_usd.toFixed(4)}. Saved guided
                  perception is replayed, not free original perception.
                </p>
                <details className="followup-details">
                  <summary>Changed requirement experiment limits</summary>
                  <ul className="gap-instructions">
                    {contracts.limits.map((limit) => (
                      <li key={limit}>{limit}</li>
                    ))}
                  </ul>
                </details>
              </>
            ) : null}
          </section>
        </>
      ) : null}
    </section>
  )
}
