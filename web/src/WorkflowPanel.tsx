import { useState } from 'react'
import { Icon } from './icons'
import type { Status } from './types'

export interface WorkflowDTO {
  schema_version: string
  title: string
  task_hash: string | null
  protocol_hash: string | null
  revision: number
  phase: 'blocked' | 'ready' | 'declared'
  gate_checks: { id: string; title: string; status: Status; reason: string }[]
  can_check: boolean
  can_continue: boolean
  blocked_reason: string | null
  declaration: {
    outcome: 'operator_declared_continuation_not_observed'
    task_hash: string
    protocol_hash: string
    revision: number
    review_origin: 'local_operator' | 'assistant_under_user_authorization'
    human_declared: boolean
    physical_action_verified: false
    declared_at_ms?: number
    record_id?: string
    record_hash?: string
  } | null
  claim_limit: string
}

export function WorkflowPanel({
  workflow,
  pending,
  onAction,
}: {
  workflow: WorkflowDTO | null
  pending: boolean
  onAction: (
    path: string,
    payload: Record<string, unknown>,
    destination?: 'workflow',
  ) => Promise<boolean>
}) {
  const [confirmedFor, setConfirmedFor] = useState<string | null>(null)
  if (!workflow) {
    return (
      <section className="panel">
        <h2>Placement preparation checkpoint</h2>
        <p>Preparation gates are unavailable. Refresh the local session.</p>
      </section>
    )
  }
  const binding = `${workflow.task_hash}:${workflow.revision}`
  const confirmed = confirmedFor === binding
  const payload = {
    task_hash: workflow.task_hash,
    revision: workflow.revision,
  }
  return (
    <section className="panel workflow-panel" aria-labelledby="workflow-title">
      <h2 id="workflow-title">{workflow.title}</h2>
      <p className="task-intro">
        Check the reviewed placement evidence, then record a separate decision
        to continue. Each gate belongs to the current task and inspection.
      </p>
      <ol className="task-checks">
        {workflow.gate_checks.map((gate) => (
          <li key={gate.id}>
            <Icon name={gate.status === 'pass' ? 'check' : 'info'} />
            <span>
              <strong>{gate.title}</strong>{' '}
              <span className={`lab-result ${gate.status}`}>{gate.status}</span>
              <small>{gate.reason}</small>
            </span>
          </li>
        ))}
      </ol>
      {workflow.blocked_reason ? (
        <p className="notice" role="status">
          {workflow.blocked_reason}
        </p>
      ) : null}
      <button
        className="button secondary full-width"
        disabled={pending || !workflow.can_check}
        onClick={() =>
          void onAction('/api/workflow/check', payload, 'workflow')
        }
      >
        {pending ? <span className="spinner" /> : <Icon name="refresh" />}
        Check preparation evidence
      </button>
      {workflow.declaration ? (
        <div className="notice" role="status">
          <Icon name="check" />
          <span>
            <strong>
              {workflow.declaration.human_declared
                ? 'Operator continuation declared.'
                : 'Assistant continuation declared under user authorization.'}
            </strong>
            <br />
            This record declares intent. No physical continuation was observed.
          </span>
        </div>
      ) : (
        <>
          <div className="confirmations">
            <label>
              <input
                type="checkbox"
                checked={confirmed}
                disabled={pending || !workflow.can_continue}
                onChange={(event) =>
                  setConfirmedFor(event.target.checked ? binding : null)
                }
              />
              <span>
                I reviewed this preparation checkpoint and declare my decision
                to continue. This does not report an observed action.
              </span>
            </label>
          </div>
          <button
            className="button primary full-width"
            disabled={pending || !workflow.can_continue || !confirmed}
            onClick={() =>
              void onAction(
                '/api/workflow/continue',
                {
                  ...payload,
                  confirm_operator_continuation: true,
                },
                'workflow',
              )
            }
          >
            <Icon name="check" />
            Record continuation declaration
          </button>
        </>
      )}
      <p className="contract-id">
        Task {workflow.task_hash?.slice(0, 12) || 'requires review'} · Revision{' '}
        {workflow.revision}
      </p>
      <p className="capture-footnote">{workflow.claim_limit}</p>
    </section>
  )
}
