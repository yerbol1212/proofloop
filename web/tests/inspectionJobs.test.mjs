import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { afterEach, beforeEach, test } from 'node:test'
import { awaitInspection, finishUnadmittedRecovery, pendingInspection, prepareInspectionSubmission, recoverInspection } from '../src/inspectionJobs.ts'

const storageKey = 'proofloop.inspection-intent.v2'
const legacyKey = 'proofloop.inspection-job.v1'
const binding = { session_id: 'interactive-unit', revision: 4, task_hash: 'a'.repeat(64), protocol_hash: 'b'.repeat(64) }
const inspection = { capture_token: 'private-capture-token', images: [
  { data_url: 'data:image/jpeg;base64,private-image-one' },
  { data_url: 'data:image/jpeg;base64,private-image-two' },
] }
const id = 'd'.repeat(32)
const originalFetch = globalThis.fetch
let values

function json(value, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } })
}
function job(state = 'queued', other = {}) {
  return { id, state, binding, ...other }
}

beforeEach(() => {
  values = new Map()
  Object.defineProperty(globalThis, 'sessionStorage', { configurable: true, value: {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  } })
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    setTimeout: (callback) => setTimeout(callback, 0), clearTimeout,
  } })
  globalThis.fetch = () => { throw new Error('Unexpected network request') }
})
afterEach(() => { globalThis.fetch = originalFetch })

test('before POST, persist only opaque credentials and hash the exact UTF-8 wire body', async () => {
  const submission = await prepareInspectionSubmission(binding, inspection)
  assert.deepEqual(pendingInspection(), submission.intent)
  assert.equal(submission.intent.schema_version, 2)
  assert.match(submission.intent.idempotency_key, /^[0-9a-f]{64}$/)
  assert.equal(createHash('sha256').update(submission.body, 'utf8').digest('hex'), submission.intent.submission_fingerprint)
  assert.deepEqual(JSON.parse(submission.body), { idempotency_key: submission.intent.idempotency_key, binding, inspection })
  const stored = values.get(storageKey)
  for (const privateValue of ['data:image', 'private-image', 'private-capture-token', 'csrf', 'mutation'])
    assert.equal(stored.includes(privateValue), false)
  assert.deepEqual(Object.keys(JSON.parse(stored)).sort(), ['binding', 'idempotency_key', 'schema_version', 'submission_fingerprint'])
})

test('lost202 plus reload recovers the same job with one admission and one dispatch', async () => {
  let admissions = 0
  let dispatches = 0
  let lookups = 0
  let polls = 0
  const submission = await prepareInspectionSubmission(binding, inspection)
  globalThis.fetch = async (url, options = {}) => {
    if (url === '/api/v1/inspections') {
      admissions++
      dispatches++
      assert.deepEqual(pendingInspection(), submission.intent)
      throw new TypeError('The202 response was lost after server admission')
    }
    if (url === '/api/v1/inspections/lookup') {
      lookups++
      assert.equal(options.method, 'POST')
      assert.equal(options.headers['X-ProofLoop-Token'], 'fresh-local-mutation-token')
      assert.deepEqual(JSON.parse(options.body), submission.intent)
      assert.equal(options.body.includes('private-image'), false)
      return json({ found: true, job: job() })
    }
    assert.equal(url, `/api/v1/jobs/${id}`)
    assert.equal(options.method, undefined)
    return json({ job: job(++polls === 1 ? 'running' : 'succeeded') })
  }
  await assert.rejects(fetch('/api/v1/inspections', { method: 'POST', body: submission.body }), /lost/)
  const reloaded = pendingInspection()
  const recovered = await recoverInspection(reloaded, 'fresh-local-mutation-token')
  assert.equal(recovered, id)
  await awaitInspection(recovered, undefined, reloaded)
  assert.equal(pendingInspection(), null)
  assert.deepEqual({ admissions, dispatches, lookups, polls }, { admissions: 1, dispatches: 1, lookups: 1, polls: 2 })
})

test('storage write failure stops the flow before any admission fetch', async () => {
  sessionStorage.setItem = () => { throw new Error('Storage denied') }
  let requests = 0
  globalThis.fetch = async () => { requests++; return json({}) }
  await assert.rejects(async () => {
    const submission = await prepareInspectionSubmission(binding, inspection)
    await fetch('/api/v1/inspections', { method: 'POST', body: submission.body })
  }, /No inspection was sent/)
  assert.equal(requests, 0)
})

test('storage readback failure also stops before dispatch', async () => {
  sessionStorage.setItem = (key) => values.set(key, 'not the persisted intent')
  await assert.rejects(prepareInspectionSubmission(binding, inspection), /No inspection was sent/)
})

test('concurrent preparation cannot overwrite a pending submission capability', async () => {
  const results = await Promise.allSettled([
    prepareInspectionSubmission(binding, inspection), prepareInspectionSubmission(binding, inspection),
  ])
  const successes = results.filter((result) => result.status === 'fulfilled')
  assert.equal(successes.length, 1)
  assert.deepEqual(pendingInspection(), successes[0].value.intent)
})

test('unadmitted lookup preserves the intent until an explicit new binding fences late admission', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  let requests = 0
  globalThis.fetch = async (url) => {
    requests++
    assert.equal(url, '/api/v1/inspections/lookup')
    return json({ found: false, job: null })
  }
  assert.equal(await recoverInspection(intent, 'fresh-token'), null)
  assert.deepEqual(pendingInspection(), intent)
  await assert.rejects(prepareInspectionSubmission(binding, inspection), /Recover the previous/)
  assert.throws(() => finishUnadmittedRecovery(intent, binding), /has not been replaced/)
  finishUnadmittedRecovery(intent, { ...binding, revision: binding.revision + 1 })
  assert.equal(pendingInspection(), null)
  assert.equal(requests, 1)
})

test('lookup transport failures retain intent and have no automatic retry', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  let requests = 0
  globalThis.fetch = async () => { requests++; throw new TypeError('Lookup connection interrupted') }
  await assert.rejects(recoverInspection(intent, 'fresh-token'), /interrupted/)
  assert.equal(requests, 1)
  assert.deepEqual(pendingInspection(), intent)
})

test('lookup rejects stale binding and mismatched server credentials without clearing intent', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  globalThis.fetch = async () => json({ found: true, job: job('queued', { binding: { ...binding, revision: 9 } }) })
  await assert.rejects(recoverInspection(intent, 'fresh-token'), /does not match/)
  globalThis.fetch = async () => json({ error: { message: 'Recovery credentials do not match' } }, 409)
  await assert.rejects(recoverInspection(intent, 'fresh-token'), /credentials do not match/)
  assert.deepEqual(pendingInspection(), intent)
})

test('unknown job states and changed polling bindings cannot clear pending intent', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  for (const changed of [job('invented'), job('succeeded', { binding: { ...binding, session_id: 'other' } })]) {
    globalThis.fetch = async () => json({ job: changed })
    await assert.rejects(awaitInspection(id, undefined, intent), /unavailable/)
    assert.deepEqual(pendingInspection(), intent)
  }
})

test('a terminal interrupted job clears the old reference and never retries inference', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  let requests = 0
  globalThis.fetch = async (url, options) => {
    requests++
    assert.equal(url, `/api/v1/jobs/${id}`)
    assert.equal(options.method, undefined)
    return json({ job: job('interrupted', { error_code: 'outcome_unknown' }) })
  }
  await assert.rejects(awaitInspection(id, undefined, intent), /not retried/)
  assert.equal(requests, 1)
  assert.equal(pendingInspection(), null)
})

test('a completed old poll cannot clear a newer opaque intent', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  const newer = { ...intent, idempotency_key: 'e'.repeat(64) }
  globalThis.fetch = async () => {
    values.set(storageKey, JSON.stringify(newer))
    return json({ job: job('succeeded') })
  }
  await awaitInspection(id, undefined, intent)
  assert.deepEqual(pendingInspection(), newer)
})

test('aborting before lookup causes no request and preserves the recovery capability', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(recoverInspection(intent, 'fresh-token', controller.signal), { name: 'AbortError' })
  assert.deepEqual(pendingInspection(), intent)
})

test('malformed or overbroad stored data never becomes a request', async () => {
  const { intent } = await prepareInspectionSubmission(binding, inspection)
  for (const value of ['not-json', JSON.stringify({ ...intent, schema_version: 3 }),
    JSON.stringify({ ...intent, capture_token: 'must-never-be-persisted' }),
    JSON.stringify({ ...intent, binding: { ...binding, revision: true } })]) {
    values.set(storageKey, value)
    assert.throws(pendingInspection, /invalid/)
  }
})

test('legacy saved job IDs can finish read-only polling after upgrade', async () => {
  values.set(legacyKey, id)
  const saved = pendingInspection()
  assert.deepEqual(saved, { schema_version: 1, job_id: id })
  globalThis.fetch = async (url) => {
    assert.equal(url, `/api/v1/jobs/${id}`)
    return json({ job: job('succeeded') })
  }
  await awaitInspection(id, undefined, saved)
  assert.equal(pendingInspection(), null)
})
