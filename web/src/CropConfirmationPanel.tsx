interface Confirmation {
  status: 'completed' | 'unscored'
  global_decision: 'supported'
  final_support: 'supported' | 'unknown' | null
  local_assessment: { reasons: string[] } | null
}
export interface CropConfirmationSection {
  available: boolean
  status: string
  title: string
  note: string
  summary?: {
    metrics: { eligible: number; completed: number; unscored: number }
    cases: { case_id: string; confirmation: Confirmation | null }[]
    cost_summary: { request_attempts: number; estimated_usd: number }
    limitations: string[]
  }
}

export function CropConfirmationPanel({
  section,
}: {
  section: CropConfirmationSection
}) {
  const summary =
    section.available && section.status === 'validated'
      ? section.summary
      : undefined
  return (
    <section className="acquisition-plan" aria-label={section.title}>
      <h3>{section.title}</h3>
      <p>{section.note}</p>
      {summary ? (
        <>
          <p>
            {summary.metrics.completed} / {summary.metrics.eligible} crop-only
            requests completed; {summary.metrics.unscored} unscored. Only the
            two candidate crops were attached. This additional check can
            withhold global support, and cannot upgrade an unsupported
            candidate.
          </p>
          <div
            className="lab-table-scroll"
            role="region"
            tabIndex={0}
            aria-label="Global and crop-only candidate support"
          >
            <table className="lab-metrics">
              <caption>
                Development retest after the observed false support
              </caption>
              <thead>
                <tr>
                  <th scope="col">Candidate</th>
                  <th scope="col">Initial global check</th>
                  <th scope="col">After crop-only confirmation</th>
                </tr>
              </thead>
              <tbody>
                {summary.cases.map((item) => (
                  <tr key={item.case_id}>
                    <th scope="row">{item.case_id.replaceAll('-', ' ')}</th>
                    <td>
                      {item.confirmation?.global_decision || 'Not attempted'}
                    </td>
                    <td>{item.confirmation?.final_support || 'Unscored'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {summary.cases
            .filter((item) => item.confirmation?.final_support === 'unknown')
            .map((item) => (
              <article className="evidence-gap" key={item.case_id}>
                <h4>{item.case_id.replaceAll('-', ' ')}: support withheld</h4>
                <p>
                  {item.confirmation?.local_assessment?.reasons
                    .map((reason) => reason.replaceAll('_', ' '))
                    .join('; ')}
                  .
                </p>
                <p>
                  <strong>Missing evidence.</strong> Obtain two clearer, closer
                  views of the same candidate, with its full outline and
                  distinguishing features visible. Enlarging this crop does not
                  add source detail.
                </p>
              </article>
            ))}
          <p className="lab-limit">
            {summary.cost_summary.request_attempts} additional requests ·
            estimated ${summary.cost_summary.estimated_usd.toFixed(4)}. This
            retest followed the observed error on the same recordings. It is not
            a fresh-source accuracy test or independent identity proof. The
            original 5/6 result is preserved above.
          </p>
          <details className="followup-details">
            <summary>Crop confirmation limits</summary>
            <ul>
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
