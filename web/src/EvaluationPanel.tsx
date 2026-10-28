import type { Status } from './types'

interface OutcomeMetrics {
  known_truth_count: number
  known_truth_decided: number
  known_truth_decided_correct: number
  known_fail_count: number
  known_fail_accepted: number
  unknown_truth_count: number
  unknown_truth_accepted: number
}
interface Disagreement {
  case_id: string
  check_id?: string
  expected: Status
  left_status: Status
  right_status: Status
}
export interface EvaluationDTO {
  comparison_id: string
  benchmark_case_count: number
  source_file_group_count: number
  conditions: Record<
    string,
    { case_metrics: OutcomeMetrics; check_metrics: OutcomeMetrics }
  >
  pairwise: {
    left: string
    right: string
    cases: { compared: number; disagree: number; disagreements: Disagreement[] }
    checks: {
      compared: number
      disagree: number
      disagreements: Disagreement[]
    }
  }[]
}
const names: Record<string, string> = {
  guided: 'Spark → SAM',
  sam: 'SAM',
  spark: 'Spark alone',
}
const ratio = (value: number, total: number, empty = 'No controls') =>
  total ? `${value} / ${total}` : empty

export function EvaluationPanel({
  evaluation,
  selectedCase,
  onSelect,
}: {
  evaluation: EvaluationDTO
  selectedCase: string
  onSelect: (caseId: string) => void
}) {
  const pairs = evaluation.pairwise.filter(
    (pair) => pair.cases.disagree || pair.checks.disagree,
  )
  return (
    <details className="evaluation-panel">
      <summary>Inspect failure modes and model disagreements</summary>
      <p>
        {evaluation.benchmark_case_count} scored cases from{' '}
        {evaluation.source_file_group_count} distinct encoded source files.
        Windows from a recording are correlated; file hashes do not prove
        independent captures.
      </p>
      <div
        className="lab-table-scroll"
        role="region"
        tabIndex={0}
        aria-label="Failure modes with explicit denominators"
      >
        <table className="lab-metrics">
          <caption>Coverage and errors on the same reviewed cases</caption>
          <thead>
            <tr>
              <th scope="col">Condition</th>
              <th scope="col">Known outcomes decided</th>
              <th scope="col">Correct among decided</th>
              <th scope="col">Known FAIL accepted</th>
              <th scope="col">UNKNOWN label accepted</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(evaluation.conditions).map(
              ([name, { case_metrics: metric }]) => (
                <tr key={name}>
                  <th scope="row">{names[name] || name}</th>
                  <td>
                    {ratio(
                      metric.known_truth_decided,
                      metric.known_truth_count,
                      'No known labels',
                    )}
                  </td>
                  <td>
                    {ratio(
                      metric.known_truth_decided_correct,
                      metric.known_truth_decided,
                      'No decided cases',
                    )}
                  </td>
                  <td>
                    {ratio(metric.known_fail_accepted, metric.known_fail_count)}
                  </td>
                  <td>
                    {ratio(
                      metric.unknown_truth_accepted,
                      metric.unknown_truth_count,
                    )}
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      <p className="lab-limit">
        Being correct on a smaller decided subset can hide excessive abstention.
        Known outcomes exclude UNKNOWN ground truth; its unsupported acceptance
        is reported separately. Zero observed errors in a small selected sample
        do not establish reliability.
      </p>
      <h3>Review disagreements</h3>
      {pairs.length ? (
        pairs.map((pair) => (
          <section
            className="evaluation-pair"
            key={`${pair.left}-${pair.right}`}
          >
            <h4>
              {names[pair.left] || pair.left} /{' '}
              {names[pair.right] || pair.right}
            </h4>
            <p>
              {pair.cases.disagree} / {pair.cases.compared} case decisions
              differ; {pair.checks.disagree} / {pair.checks.compared} individual
              checks differ.
            </p>
            <ul className="disagreement-list">
              {pair.checks.disagreements.map((row) => (
                <li key={`${row.case_id}-${row.check_id}`}>
                  <button
                    className="button secondary"
                    onClick={() => onSelect(row.case_id)}
                    aria-pressed={selectedCase === row.case_id}
                  >
                    <strong>{row.case_id.replaceAll('-', ' ')}</strong>
                    <span>
                      {row.check_id} · reviewed {row.expected} ·{' '}
                      {row.left_status} / {row.right_status}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))
      ) : (
        <p>
          No disagreements in this selected cohort. Agreement does not establish
          correctness.
        </p>
      )}
      <p className="contract-id">
        Decision comparison {evaluation.comparison_id.slice(0, 12)} · recomputed
        from validated records
      </p>
    </details>
  )
}
