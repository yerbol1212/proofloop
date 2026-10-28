import { useState } from 'react'

type Identity = 'supported' | 'unknown'
interface TrialImage {
  source_url: string
  crop_url: string
  width: number
  height: number
  candidate_box?: { x1: number; y1: number; x2: number; y2: number }
}
interface TrialCase {
  case_id: string
  expected_identity: Identity
  source_family: string
  source_id: string
  source_timestamps_ms: number[]
  anchor_origin: 'saved_spark_locator' | 'authored_candidate'
  source_attribution: { page_url: string; license: string; title?: string }
  attempt_status: 'completed' | 'unscored' | 'not_attempted'
  matches_expected: boolean | null
  challenge: {
    status: 'completed' | 'unscored'
    challenge: {
      decision: Identity
      reasons: string[]
      contract: { positive_concept: string }
      frames: {
        frame_index: number
        assessment: {
          decision: Identity
          reasons: string[]
          candidate_relation: string
        }
      }[]
    } | null
    failure: { code: string | null; raw_response_retained: boolean } | null
  } | null
  images?: TrialImage[]
}
export interface SemanticTrialSummary {
  freeze_hash: string
  phase: string
  method_config: {
    reasoning_effort: string
    max_output_tokens: number
    automatic_retries: number
  }
  cases: TrialCase[]
  metrics: {
    case_count: number
    attempted: number
    completed: number
    unscored: number
    not_attempted: number
    scored: number
    correct: number
    known_positive: { count: number; completed: number; supported: number }
    known_negative: {
      count: number
      completed: number
      unsupported_accepted: number
    }
  }
  cost_summary: { request_attempts: number; estimated_usd: number }
  limitations: string[]
}
export interface SemanticTrialSection {
  available: boolean
  status: 'unavailable' | 'invalid' | 'validated'
  title: string
  note: string
  summary?: SemanticTrialSummary
}
const fraction = (value: number, total: number) =>
  total ? `${value} / ${total}` : 'No scored controls'
const words = (value: string) => value.replaceAll('_', ' ').replaceAll('-', ' ')
const outcome = (item: TrialCase) =>
  item.attempt_status === 'completed'
    ? item.challenge?.challenge?.decision.toUpperCase() || 'Unscored'
    : item.attempt_status === 'unscored'
      ? 'Unscored request'
      : 'Not attempted'

export function SemanticTrialPanel({
  section,
}: {
  section: SemanticTrialSection
}) {
  const [selectedCase, setSelectedCase] = useState('')
  const summary =
    section.available && section.status === 'validated'
      ? section.summary
      : undefined
  const selected =
    summary?.cases.find((item) => item.case_id === selectedCase) ||
    summary?.cases[0]
  const challenge = selected?.challenge?.challenge
  return (
    <section
      className="acquisition-plan semantic-trial"
      aria-label={section.title}
    >
      <h3>{section.title}</h3>
      <p>{section.note}</p>
      {summary ? (
        <>
          <div className="semantic-trial-metrics">
            <p>
              <strong>Request completion</strong>
              <span>
                {fraction(summary.metrics.completed, summary.metrics.attempted)}{' '}
                attempted
              </span>
            </p>
            <p>
              <strong>Reviewed identity agreement</strong>
              <span>
                {fraction(summary.metrics.correct, summary.metrics.scored)}{' '}
                scored
              </span>
            </p>
            <p>
              <strong>Positive candidates retained</strong>
              <span>
                {fraction(
                  summary.metrics.known_positive.supported,
                  summary.metrics.known_positive.completed,
                )}{' '}
                completed positive controls
              </span>
            </p>
            <p>
              <strong>Unsupported candidates accepted</strong>
              <span>
                {fraction(
                  summary.metrics.known_negative.unsupported_accepted,
                  summary.metrics.known_negative.completed,
                )}{' '}
                completed negative controls
              </span>
            </p>
          </div>
          <p className="lab-limit">
            {summary.metrics.case_count} reviewed pairs:{' '}
            {summary.metrics.unscored} unscored requests and{' '}
            {summary.metrics.not_attempted} not attempted. Reviewed controls
            include {summary.metrics.known_positive.count} positive and{' '}
            {summary.metrics.known_negative.count} negative pairs. A failed
            request contributes no supported or UNKNOWN prediction.
          </p>
          <details className="followup-details">
            <summary>Inspect the reviewed candidates and source views</summary>
            <ul className="disagreement-list semantic-case-list">
              {summary.cases.map((item) => (
                <li key={item.case_id}>
                  <button
                    className="button secondary"
                    aria-pressed={selected?.case_id === item.case_id}
                    onClick={() => setSelectedCase(item.case_id)}
                  >
                    <strong>{words(item.case_id)}</strong>
                    <span>
                      Reviewed {item.expected_identity.toUpperCase()} ·{' '}
                      {outcome(item)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {selected ? (
              <article
                className="evidence-gap semantic-case-detail"
                key={selected.case_id}
              >
                <h4>{words(selected.case_id)}</h4>
                <p>
                  Reviewed expectation:{' '}
                  <strong>{selected.expected_identity.toUpperCase()}</strong>.{' '}
                  Recorded result: <strong>{outcome(selected)}</strong>.
                </p>
                {selected.matches_expected === false ? (
                  <p className="notice">
                    This result disagrees with the reviewed expectation. The
                    challenger remains experimental and cannot grant task
                    acceptance.
                  </p>
                ) : null}
                <p>
                  {selected.anchor_origin === 'authored_candidate'
                    ? 'The candidate box was manually reviewed before inference. This tests the challenger, not automatic localization.'
                    : 'The candidate box comes from the saved Spark locator. Locator and challenger use the same model; their errors can be correlated.'}
                </p>
                {challenge ? (
                  <>
                    <p>Target: {challenge.contract.positive_concept}.</p>
                    <p>
                      {challenge.reasons.length
                        ? challenge.reasons.map(words).join('; ')
                        : 'The model reported support for the reviewed semantic criteria in both source views.'}
                    </p>
                  </>
                ) : (
                  <p>
                    {selected.attempt_status === 'unscored'
                      ? `The request was not scored (${words(selected.challenge?.failure?.code || 'request failed')}).`
                      : 'No model request has been recorded for this case.'}
                  </p>
                )}
                {selected.images?.length ? (
                  <div className="semantic-trial-views">
                    {selected.images.map((image, index) => {
                      const box = image.candidate_box
                      const assessment = challenge?.frames.find(
                        (frame) => frame.frame_index === index,
                      )?.assessment
                      return (
                        <figure key={`${selected.case_id}-${index}`}>
                          <figcaption>
                            View {index + 1} · declared source time{' '}
                            {(
                              (selected.source_timestamps_ms[index] || 0) / 1000
                            ).toFixed(2)}{' '}
                            s
                            {assessment
                              ? ` · model says candidate ${words(assessment.candidate_relation)}`
                              : ''}
                          </figcaption>
                          <div
                            className="semantic-source-view"
                            style={{ position: 'relative' }}
                          >
                            <img
                              src={image.source_url}
                              width={image.width}
                              height={image.height}
                              loading="lazy"
                              alt={`Recorded source view ${index + 1}; the outline marks the reviewed candidate.`}
                            />
                            {box ? (
                              <svg
                                aria-hidden="true"
                                viewBox={`0 0 ${image.width} ${image.height}`}
                                style={{
                                  position: 'absolute',
                                  inset: 0,
                                  width: '100%',
                                  height: '100%',
                                  pointerEvents: 'none',
                                }}
                              >
                                <rect
                                  x={box.x1}
                                  y={box.y1}
                                  width={box.x2 - box.x1 + 1}
                                  height={box.y2 - box.y1 + 1}
                                  fill="none"
                                  stroke="#ffcf61"
                                  strokeWidth="3"
                                  vectorEffect="non-scaling-stroke"
                                />
                              </svg>
                            ) : null}
                          </div>
                          <p className="capture-footnote">
                            Derived crop of the same source; no new capture.
                          </p>
                          <img
                            className="semantic-candidate-crop"
                            src={image.crop_url}
                            loading="lazy"
                            alt={`Derived candidate crop from source view ${index + 1}.`}
                          />
                        </figure>
                      )
                    })}
                  </div>
                ) : (
                  <p className="capture-footnote">
                    Source views are not attached to this saved summary.
                  </p>
                )}
                <p className="contract-id">
                  <a
                    href={selected.source_attribution.page_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {selected.source_attribution.title || selected.source_id} ·{' '}
                    {selected.source_attribution.license}
                  </a>
                </p>
              </article>
            ) : null}
          </details>
          <p className="lab-limit">
            UNKNOWN withholds candidate support; it does not prove physical
            absence. Semantic support grants no placement PASS or physical
            execution proof.
          </p>
          <p className="contract-id">
            {summary.cost_summary.request_attempts} requests · estimated $
            {summary.cost_summary.estimated_usd.toFixed(4)} ·{' '}
            {summary.method_config.reasoning_effort} reasoning ·{' '}
            {summary.method_config.max_output_tokens} output-token cap ·{' '}
            {summary.method_config.automatic_retries} automatic retries
          </p>
          <details className="followup-details">
            <summary>Experiment limits</summary>
            <ul className="gap-instructions">
              {summary.limitations.map((limit) => (
                <li key={limit}>{limit}</li>
              ))}
            </ul>
          </details>
        </>
      ) : null}
    </section>
  )
}
