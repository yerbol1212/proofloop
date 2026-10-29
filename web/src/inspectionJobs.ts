// Store only a versioned opaque recovery capability. Capture/mutation tokens,
// images and the original request body remain in memory and on the local server.
const storageKey = 'proofloop.inspection-intent.v2'
const legacyStorageKey = 'proofloop.inspection-job.v1'
const jobId = /^[0-9a-f]{32}$/
const hash = /^[0-9a-f]{64}$/
const opaqueId = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/
const intentKey = /^[0-9a-f]{64}$/
const jobStates = new Set(['queued', 'running', 'succeeded', 'failed', 'interrupted', 'cancelled'])

export type InspectionBinding = {
  session_id: string
  revision: number
  task_hash: string
  protocol_hash: string
}
export type InspectionIntent = {
  schema_version: 2
  idempotency_key: string
  binding: InspectionBinding
  submission_fingerprint: string
}
type LegacyInspection = { schema_version: 1; job_id: string }
export type PendingInspection = InspectionIntent | LegacyInspection
type Job = { id: string; state: string; binding?: InspectionBinding; error_code?: string | null }

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
function keys(value: Record<string, unknown>, expected: string[]): boolean {
  return Object.keys(value).length === expected.length && expected.every((key) => Object.hasOwn(value, key))
}
function validBinding(value: unknown): value is InspectionBinding {
  return object(value) && keys(value, ['session_id', 'revision', 'task_hash', 'protocol_hash']) &&
    typeof value.session_id === 'string' && opaqueId.test(value.session_id) &&
    typeof value.task_hash === 'string' && hash.test(value.task_hash) &&
    typeof value.protocol_hash === 'string' && hash.test(value.protocol_hash) &&
    typeof value.revision === 'number' && Number.isSafeInteger(value.revision) && value.revision >= 0
}
function sameBinding(left: unknown, right: InspectionBinding): boolean {
  return validBinding(left) && left.session_id === right.session_id && left.revision === right.revision &&
    left.task_hash === right.task_hash && left.protocol_hash === right.protocol_hash
}
function validIntent(value: unknown): value is InspectionIntent {
  return object(value) && keys(value, ['schema_version', 'idempotency_key', 'binding', 'submission_fingerprint']) &&
    value.schema_version === 2 && typeof value.idempotency_key === 'string' && intentKey.test(value.idempotency_key) &&
    typeof value.submission_fingerprint === 'string' && hash.test(value.submission_fingerprint) && validBinding(value.binding)
}
function sameIntent(left: InspectionIntent, right: InspectionIntent): boolean {
  return left.idempotency_key === right.idempotency_key && left.submission_fingerprint === right.submission_fingerprint &&
    sameBinding(left.binding, right.binding)
}

export function pendingInspection(): PendingInspection | null {
  let encoded: string | null
  let legacy: string | null
  try {
    encoded = sessionStorage.getItem(storageKey)
    legacy = sessionStorage.getItem(legacyStorageKey)
  } catch {
    throw new Error('Browser session storage is unavailable. Enable it before submitting; no new inspection has been sent.')
  }
  if (encoded !== null) {
    let parsed: unknown
    try { parsed = JSON.parse(encoded) } catch { /* Reject below. */ }
    if (!validIntent(parsed))
      throw new Error('The saved inspection recovery reference is invalid. No inspection was resubmitted.')
    return parsed
  }
  if (legacy !== null) {
    if (!jobId.test(legacy)) throw new Error('The saved inspection job reference is invalid.')
    return { schema_version: 1, job_id: legacy }
  }
  return null
}

function clearPending(expected: PendingInspection): void {
  // Do not erase a newer intent written by another in-tab flow.
  try {
    const current = pendingInspection()
    if (expected.schema_version === 2 && current?.schema_version === 2 && sameIntent(current, expected))
      sessionStorage.removeItem(storageKey)
    else if (expected.schema_version === 1 && current?.schema_version === 1 && current.job_id === expected.job_id)
      sessionStorage.removeItem(legacyStorageKey)
  } catch { /* Keeping an old read-only recovery reference is safe. */ }
}

export async function prepareInspectionSubmission(binding: InspectionBinding, inspection: unknown): Promise<{
  body: string
  intent: InspectionIntent
}> {
  if (!validBinding(binding) || !object(inspection)) throw new Error('The inspection binding is invalid.')
  if (pendingInspection())
    throw new Error('Recover the previous inspection status before starting another capture.')
  const bound = { ...binding }
  const bytes = crypto.getRandomValues(new Uint8Array(32))
  const idempotency_key = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
  const body = JSON.stringify({ idempotency_key, binding: bound, inspection })
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(body))
  const submission_fingerprint = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
  const intent: InspectionIntent = { schema_version: 2, idempotency_key, binding: bound, submission_fingerprint }
  const encoded = JSON.stringify(intent)
  try {
    if (pendingInspection()) throw new Error('Another inspection is already pending')
    // The persisted capability must be readable BEFORE the only POST.
    sessionStorage.setItem(storageKey, encoded)
    if (sessionStorage.getItem(storageKey) !== encoded) throw new Error('Recovery reference was not retained')
  } catch {
    throw new Error('The inspection recovery reference could not be saved. No inspection was sent; enable browser session storage and capture again.')
  }
  return { body, intent }
}

export async function recoverInspection(intent: InspectionIntent, mutationToken: string, signal?: AbortSignal): Promise<string | null> {
  if (!validIntent(intent)) throw new Error('The inspection recovery reference is invalid.')
  signal?.throwIfAborted()
  const requestSignal = signal ? AbortSignal.any([signal, AbortSignal.timeout(10_000)]) : AbortSignal.timeout(10_000)
  const response = await fetch('/api/v1/inspections/lookup', {
    method: 'POST', cache: 'no-store', signal: requestSignal,
    headers: { 'Content-Type': 'application/json', 'X-ProofLoop-Token': mutationToken },
    body: JSON.stringify(intent),
  })
  const data = await response.json() as { found?: boolean; job?: Job | null; error?: { message?: string } }
  if (!response.ok)
    throw new Error(data.error?.message || 'The original inspection could not be recovered. It was not resubmitted.')
  if (data.found === false && data.job === null) {
    // A point-in-time lookup cannot fence an original POST still in flight.
    // Keep the capability until an explicit new inspection changes the binding.
    return null
  }
  if (data.found !== true || !data.job || !jobId.test(data.job.id) || !jobStates.has(data.job.state) ||
      !sameBinding(data.job.binding, intent.binding))
    throw new Error('The recovery response does not match the original bound inspection. No inspection was resubmitted.')
  return data.job.id
}

export function finishUnadmittedRecovery(intent: InspectionIntent, replacementBinding: InspectionBinding): void {
  if (!validIntent(intent) || !validBinding(replacementBinding) || sameBinding(intent.binding, replacementBinding))
    throw new Error('Start a new inspection before capturing again; the original binding has not been replaced.')
  clearPending(intent)
  const current = pendingInspection()
  if (current?.schema_version === 2 && sameIntent(current, intent))
    throw new Error('The old inspection reference could not be cleared. Enable browser session storage before capturing again.')
}

export async function awaitInspection(id: string, signal?: AbortSignal, expected?: PendingInspection): Promise<void> {
  if (!jobId.test(id)) throw new Error('The inspection reference is invalid.')
  for (;;) {
    signal?.throwIfAborted()
    const response = await fetch(`/api/v1/jobs/${id}`, { cache: 'no-store', signal })
    const data = await response.json() as { job?: Job; error?: { message?: string } }
    if (!response.ok || !data.job || data.job.id !== id || !jobStates.has(data.job.state) ||
        (expected?.schema_version === 2 && !sameBinding(data.job.binding, expected.binding)))
      throw new Error(data.error?.message || 'The inspection status is unavailable. Its model calls will not be retried automatically.')
    if (!['queued', 'running'].includes(data.job.state)) {
      if (expected) clearPending(expected)
      if (data.job.state !== 'succeeded')
        throw new Error(`Inspection stopped (${data.job.error_code || data.job.state}). Review the current evidence and start a new inspection; model calls were not retried.`)
      return
    }
    await new Promise<void>((resolve, reject) => {
      const finish = () => { signal?.removeEventListener('abort', abort); resolve() }
      const timer = window.setTimeout(finish, 600)
      const abort = () => {
        window.clearTimeout(timer)
        signal?.removeEventListener('abort', abort)
        reject(new DOMException('Stopped watching inspection', 'AbortError'))
      }
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) abort()
    })
  }
}
