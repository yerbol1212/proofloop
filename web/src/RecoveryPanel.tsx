import { useEffect, useRef, useState } from 'react'
import { Icon } from './icons'
import type { RecoveryReceipt, Status } from './types'

interface Verification {
  valid: boolean
  scope: 'requested' | 'completed'
  receipt_hash: string | null
  outcome: RecoveryReceipt['outcome'] | null
  reason: string
  physical_action_execution_verified: false
}
const outcomes: Record<RecoveryReceipt['outcome'], string> = {
  awaiting_fresh_evidence: 'Awaiting new evidence',
  fresh_goal_verified_after_requested_repair:
    'Fresh goal verified after repair request',
  fresh_goal_verified_after_inspection: 'Fresh goal verified after inspection',
  still_failed: 'Required placement still fails',
  needs_evidence: 'More evidence required',
  error: 'Evidence record is unverified',
  cancelled_task_change: 'Closed because the task changed',
}
const titleCase = (text: string) =>
  text.replaceAll('_', ' ').replaceAll('-', ' ')
function CheckList({ checks }: { checks: RecoveryReceipt['before_checks'] }) {
  return (
    <ul className="recovery-checks">
      {checks.map((check) => (
        <li key={check.check_id}>
          <span>{titleCase(check.object_id || check.check_id)}</span>
          <span className={`lab-result ${check.status}`}>{check.status}</span>
        </li>
      ))}
    </ul>
  )
}
function Result({ status }: { status: Status | null }) {
  return status ? (
    <span className={`lab-result ${status}`}>{status}</span>
  ) : (
    <span className="recovery-pending">Pending</span>
  )
}

export function RecoveryPanel({
  receipt,
  activeProtocolHash,
  activeTaskHash,
}: {
  receipt: RecoveryReceipt | null
  activeProtocolHash?: string
  activeTaskHash?: string | null
}) {
  const [verification, setVerification] = useState<Verification | null>(null)
  const [error, setError] = useState<{ hash: string; message: string } | null>(
    null,
  )
  const [pending, setPending] = useState(false)
  const controller = useRef<AbortController | null>(null)
  useEffect(
    () => () => {
      controller.current?.abort()
      controller.current = null
    },
    [],
  )
  async function verify() {
    if (!receipt || pending) return
    const requestedHash = receipt.receipt_hash
    const request = new AbortController()
    controller.current = request
    setPending(true)
    setError(null)
    const timeout = window.setTimeout(() => request.abort(), 20_000)
    try {
      const response = await fetch('/api/recovery', {
        cache: 'no-store',
        signal: request.signal,
      })
      if (!response.ok)
        throw new Error('Local evidence replay could not be completed.')
      const data = (await response.json()) as { verification: Verification }
      if (data.verification.receipt_hash !== requestedHash)
        throw new Error(
          'The recovery record changed. Check the current record again.',
        )
      setVerification(data.verification)
    } catch (caught) {
      if (controller.current === request)
        setError({
          hash: requestedHash,
          message:
            caught instanceof Error && caught.name !== 'AbortError'
              ? caught.message
              : 'Evidence replay was interrupted. Try again.',
        })
    } finally {
      window.clearTimeout(timeout)
      if (controller.current === request) {
        controller.current = null
        setPending(false)
      }
    }
  }
  const currentVerification =
    verification?.receipt_hash === receipt?.receipt_hash ? verification : null
  const currentError =
    error && receipt && error.hash === receipt.receipt_hash
      ? error.message
      : null
  const historical = Boolean(
    receipt &&
      ((activeProtocolHash && activeProtocolHash !== receipt.protocol_hash) ||
        (activeTaskHash &&
          receipt.task_hash &&
          activeTaskHash !== receipt.task_hash)),
  )
  return (
    <section className="recovery-panel" aria-labelledby="recovery-title">
      <div className="recovery-heading">
        <div>
          <h3 id="recovery-title">Linked recovery record</h3>
          <p>Follow the evidence from a request to a new observation.</p>
        </div>
        {receipt ? (
          <button
            className="button secondary"
            disabled={pending}
            onClick={() => void verify()}
          >
            {pending ? <span className="spinner" /> : <Icon name="refresh" />}
            {pending ? 'Replaying evidence…' : 'Check local evidence'}
          </button>
        ) : null}
      </div>
      {!receipt ? (
        <p className="recovery-empty">
          Start a repair or request new views to create a linked record. No
          recovery has been recorded in this session.
        </p>
      ) : (
        <>
          {historical ? (
            <p className="notice">
              Historical record: these observations belong to a different task
              contract.
            </p>
          ) : null}
          <p className="recovery-outcome">
            {
              outcomes[
                currentVerification
                  ? currentVerification.valid
                    ? currentVerification.outcome || receipt.outcome
                    : 'error'
                  : receipt.outcome
              ]
            }
          </p>
          <div className="recovery-stages">
            <div className="recovery-stage">
              <h4>
                1. Recorded before <Result status={receipt.before_status} />
              </h4>
              <p>Revision {receipt.before_revision}</p>
              <CheckList checks={receipt.before_checks} />
            </div>
            <div className="recovery-stage">
              <h4>
                2.{' '}
                {receipt.kind === 'repair'
                  ? 'Proposed repair'
                  : 'Requested inspection'}
              </h4>
              <p>A request describes intent, not an observed action.</p>
              {receipt.plan_steps.length ? (
                <ol>
                  {receipt.plan_steps.map((step, index) => (
                    <li key={`${index}-${step}`}>{step}</li>
                  ))}
                </ol>
              ) : (
                <p>Capture new views under the active task.</p>
              )}
            </div>
            <div className="recovery-stage">
              <h4>
                3. Recorded after <Result status={receipt.after_status} />
              </h4>
              <p>
                {receipt.after_revision === null
                  ? 'Two new qualifying views are still required.'
                  : `Revision ${receipt.after_revision}`}
              </p>
              <CheckList checks={receipt.after_checks} />
            </div>
          </div>
          {receipt.outcome === 'error' && !currentVerification ? (
            <p className="recovery-note">{receipt.verification_note}</p>
          ) : null}
          <p className="recovery-receipt-id">
            Record {receipt.receipt_hash.slice(0, 16)} · Protocol{' '}
            {receipt.protocol_hash.slice(0, 12)}
          </p>
          {currentVerification ? (
            <div
              className={`notice ${currentVerification.valid ? '' : 'notice-error'}`}
              role="status"
            >
              <Icon name={currentVerification.valid ? 'check' : 'info'} />
              <span>
                <strong>
                  {currentVerification.valid
                    ? currentVerification.scope === 'requested'
                      ? 'Request record checked.'
                      : 'Saved evidence replay passed.'
                    : 'Evidence replay failed.'}
                </strong>
                <br />
                {currentVerification.reason}
              </span>
            </div>
          ) : null}
          {currentError ? (
            <div className="notice notice-error" role="alert">
              <Icon name="info" />
              <span>{currentError}</span>
            </div>
          ) : null}
          <p className="recovery-limit">
            Successfully replayed observations support the recorded goal, not
            the execution of a move. Capture timing remains operator-declared.
            Current inspection freshness is shown separately.
          </p>
        </>
      )}
    </section>
  )
}
