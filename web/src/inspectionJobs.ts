// Persist only an opaque job reference. Images and capture tokens stay in memory.
const storageKey = 'proofloop.inspection-job.v1'
const jobId = /^[0-9a-f]{32}$/
type Job = { id: string; state: string; error_code?: string | null }

export function pendingInspection(): string | null {
  try {
    const value = sessionStorage.getItem(storageKey)
    return value && jobId.test(value) ? value : null
  } catch {
    return null
  }
}

export function rememberInspection(id: string): void {
  if (!jobId.test(id)) throw new Error('The server returned an invalid inspection reference.')
  try { sessionStorage.setItem(storageKey, id) } catch { /* Polling still works without storage. */ }
}

export async function awaitInspection(id: string, signal?: AbortSignal): Promise<void> {
  if (!jobId.test(id)) throw new Error('The inspection reference is invalid.')
  for (;;) {
    signal?.throwIfAborted()
    const response = await fetch(`/api/v1/jobs/${id}`, { cache: 'no-store', signal })
    const data = await response.json() as { job?: Job; error?: { message?: string } }
    if (!response.ok || !data.job || data.job.id !== id)
      throw new Error(data.error?.message || 'The inspection status is unavailable. Its model calls will not be retried automatically.')
    if (!['queued', 'running'].includes(data.job.state)) {
      try { if (pendingInspection() === id) sessionStorage.removeItem(storageKey) } catch { /* Optional storage. */ }
      if (data.job.state !== 'succeeded')
        throw new Error(`Inspection stopped (${data.job.error_code || data.job.state}). Review the current evidence and start a new inspection; model calls were not retried.`)
      return
    }
    await new Promise<void>((resolve, reject) => {
      const finish = () => { signal?.removeEventListener('abort', abort); resolve() }
      const timer = window.setTimeout(finish, 600)
      const abort = () => { window.clearTimeout(timer); reject(new DOMException('Stopped watching inspection', 'AbortError')) }
      signal?.addEventListener('abort', abort, { once: true })
    })
  }
}
