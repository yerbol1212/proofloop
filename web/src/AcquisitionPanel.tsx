import type { AcquisitionPlan } from './types'
import { Icon } from './icons'

const labels: Record<string, string> = {
  not_detected: 'No supported detection',
  multiple_candidates: 'More than one candidate',
  identity_ambiguous: 'Identity unconfirmed',
  partial_visibility: 'Full visibility unconfirmed',
  low_confidence: 'Score below threshold',
  boundary_unconfirmed: 'Region boundary unconfirmed',
  invalid_detection: 'Unsupported geometry',
  mask_unavailable: 'Mask unavailable',
  temporal_inconsistency: 'Views disagree',
  insufficient_distinct_views: 'Stable pair missing',
  stale_evidence: 'Evidence expired',
  missing_current_evidence: 'Current observation missing',
  calibration_required: 'Calibration required',
}

export function AcquisitionPanel({ plan }: { plan: AcquisitionPlan }) {
  const stopped = plan.status !== 'ready'
  return (
    <section
      className="acquisition-plan"
      aria-label="Missing evidence requests"
    >
      <p className="acquisition-budget">
        <Icon name="camera" size={18} />
        {plan.attempts_used} / {plan.max_attempts} follow-up pairs submitted
        {!stopped ? ` · ${plan.attempts_remaining} remaining` : ' · stopped'}
      </p>
      {plan.requests.map((request) => (
        <article className="evidence-gap" key={request.check_id}>
          <h3>{request.object_label}</h3>
          <p className="gap-view">
            {request.view_indices.length
              ? request.view_indices
                  .map((index) => `View ${index + 1}`)
                  .join(' and ')
              : 'A new pair of views'}
            {' · '}
            {request.region_label.toUpperCase()}
          </p>
          <ul className="gap-tags">
            {request.gap_codes.map((code) => (
              <li key={code}>{labels[code] || code.replaceAll('_', ' ')}</li>
            ))}
          </ul>
          <p>{request.reason}</p>
          <ul className="gap-instructions">
            {request.instructions.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </article>
      ))}
      {plan.stop_reason ? (
        <p className="acquisition-stop" role="status">
          {plan.stop_reason}
        </p>
      ) : null}
      <p className="acquisition-note">
        Every follow-up needs two entirely new views. Keep all required targets
        visible. An UNKNOWN result does not authorize a physical repair.
      </p>
    </section>
  )
}
