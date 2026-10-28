import type { Status } from './types'
import { Icon } from './icons'

export interface DiagnosticDTO {
  schema_version: number
  protocol_hash: string
  evidence_origin: string
  current: boolean
  overall_decision: Status
  cause_not_observed: true
  physical_action_allowed: false
  repair_proposal_eligible: boolean
  checks: {
    check_id: string
    object_id: string | null
    expected_region: string | null
    observed_region: string | null
    decision: Status
    diagnostic_code: string
    gap_codes: string[]
    observed_fact: string
    expected_fact: string
    support_limit: string
    next_action: string
    evidence_ids: string[]
    evidence_scope: 'current' | 'historical' | 'not_current'
    physical_cause_observed: false
  }[]
  claim_limit: string
}

export function DiagnosticPanel({
  diagnostic,
}: {
  diagnostic: DiagnosticDTO | null
}) {
  if (!diagnostic) return null
  return (
    <section className="panel" aria-label="What the evidence supports">
      <h2>What the evidence supports</h2>
      <p className="task-intro">
        Compare the reviewed requirement with its recorded observations. The
        physical reason for a mismatch remains unobserved.
      </p>
      {!diagnostic.current ? (
        <p className="notice">
          <Icon name="info" />
          <span>
            These observations are historical, expired, or invalidated. Start a
            current inspection before considering a repair.
          </span>
        </p>
      ) : null}
      <div className="acquisition-plan">
        {diagnostic.checks.map((check) => (
          <article className="evidence-gap" key={check.check_id}>
            <h3>
              {check.object_id || 'Ordered task requirement'}{' '}
              <span className={`lab-result ${check.decision}`}>
                {check.decision}
              </span>
            </h3>
            <p>
              <strong>Required.</strong> {check.expected_fact}
            </p>
            <p>
              <strong>Supported.</strong> {check.observed_fact}
            </p>
            <p>
              <strong>Next.</strong> {check.next_action}
            </p>
            <p className="contract-id">
              Check {check.check_id} ·{' '}
              {check.evidence_scope.replaceAll('_', ' ')}
              {check.evidence_ids.length
                ? ` · ${check.evidence_ids.length} supporting evidence reference${check.evidence_ids.length === 1 ? '' : 's'}`
                : ''}
            </p>
          </article>
        ))}
      </div>
      <p className="capture-footnote">{diagnostic.claim_limit}</p>
    </section>
  )
}
