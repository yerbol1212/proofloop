import { useEffect, useRef, useState } from 'react'
import type { Snapshot } from './types'
import { Icon } from './icons'

export type TaskAction = (
  path: string,
  payload: Record<string, unknown>,
  destination?: 'task' | 'inspection',
) => Promise<boolean>
const label = (value: string) => value.replaceAll('_', ' ').replaceAll('-', ' ')
const coffeePreset = {
  title: 'Coffee glass placement',
  requirements: 'The unique transparent drinking glass must be in DISPENSE.',
  objects: ['glass'],
  regions: ['dispense', 'staging'],
  concepts: { glass: 'transparent drinking glass' },
} as const

export function TaskEditor({
  state,
  pending,
  onAction,
}: {
  state: Snapshot
  pending: boolean
  onAction: TaskAction
}) {
  const task = state.active_task
  const [title, setTitle] = useState(task?.title || 'Placement inspection')
  const [objects, setObjects] = useState(
    (task?.objects || ['fork', 'spoon']).join(', '),
  )
  const [regionNames, setRegionNames] = useState(
    task?.protocol.regions || ['left', 'right'],
  )
  const [presetSelected, setPresetSelected] = useState(false)
  const [requirements, setRequirements] = useState(
    task?.requirements ||
      'The unique fork must be in LEFT. The unique spoon must be in RIGHT.',
  )
  const [unique, setUnique] = useState(false)
  const [reviewed, setReviewed] = useState(false)
  const [concepts, setConcepts] = useState<Record<string, string>>(
    task?.concepts || {},
  )
  const [mode, setMode] = useState<'concept' | 'guided'>(
    task?.perception_mode || 'concept',
  )
  const [localError, setLocalError] = useState<string | null>(null)
  const previousDraft = useRef<string | null>(null)
  const draft = state.task_draft
  useEffect(() => {
    if (previousDraft.current !== (draft?.draft_hash || null)) {
      setReviewed(false)
      previousDraft.current = draft?.draft_hash || null
    }
  }, [draft?.draft_hash])
  const inventory = objects
    .split(',')
    .map((value) => value.trim().replaceAll(' ', '_'))
    .filter(Boolean)
  const prompts = Object.fromEntries(
    inventory.map((id) => [id, concepts[id] ?? label(id)]),
  )
  const regions = regionNames.map((value) => value.trim())
  const authoredPreset =
    presetSelected &&
    title.trim() === coffeePreset.title &&
    requirements.trim() === coffeePreset.requirements &&
    inventory.join(',') === coffeePreset.objects.join(',') &&
    regions.join(',') === coffeePreset.regions.join(',') &&
    prompts.glass === coffeePreset.concepts.glass &&
    mode === 'guided'
  const changed = Boolean(
    draft &&
      (requirements.trim() !== draft.requirements ||
        title.trim() !== draft.title ||
        inventory.join(',') !== draft.objects.join(',') ||
        regions.join(',') !== draft.protocol.regions.join(',') ||
        inventory.some((id) => prompts[id] !== draft.concepts[id]) ||
        mode !== (draft.perception_mode || 'concept')),
  )
  const disabled = pending || state.busy
  async function createDraft() {
    setLocalError(null)
    setReviewed(false)
    if (
      inventory.length < 1 ||
      inventory.length > 6 ||
      new Set(inventory).size !== inventory.length ||
      inventory.some((id) => !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(id))
    ) {
      setLocalError(
        'Enter one to six distinct object names, separated by commas. Use letters, numbers, spaces, underscores or hyphens.',
      )
      return
    }
    if (
      regions.length !== 2 ||
      new Set(regions).size !== 2 ||
      regions.some((id) => !/^[a-z0-9][a-z0-9_.-]{0,63}$/.test(id))
    ) {
      setLocalError(
        'Give both regions distinct lowercase IDs, such as dispense and staging.',
      )
      return
    }
    if (
      Object.values(prompts).some(
        (prompt) => !prompt.trim() || prompt.length > 100,
      )
    ) {
      setLocalError(
        'Give every target a short visual description, up to 100 characters.',
      )
      return
    }
    await onAction(
      '/api/task/draft',
      {
        title: title.trim(),
        requirements: requirements.trim(),
        objects: inventory,
        regions,
        concepts: prompts,
        perception_mode: mode,
        draft_source: authoredPreset ? 'authored_coffee_preset' : 'spark',
        confirm_unique_targets: true,
      },
      'task',
    )
  }
  async function activate() {
    if (!draft || changed || !reviewed || !unique) return
    await onAction(
      '/api/task/activate',
      {
        draft_id: draft.draft_id,
        draft_hash: draft.draft_hash,
        confirm_requirements: true,
        confirm_unique_targets: true,
      },
      'inspection',
    )
  }
  return (
    <div className="task-workbench">
      <section className="panel task-form" aria-labelledby="task-form-title">
        <h2 id="task-form-title">Describe the task</h2>
        <p className="task-intro">
          Describe one to six targets in two named regions. Review the exact
          placement checks before they become active.
        </p>
        <button
          className="button secondary full-width"
          disabled={disabled}
          onClick={() => {
            setTitle(coffeePreset.title)
            setObjects(coffeePreset.objects.join(', '))
            setRegionNames([...coffeePreset.regions])
            setRequirements(coffeePreset.requirements)
            setConcepts({ ...coffeePreset.concepts })
            setMode('guided')
            setPresetSelected(true)
            setUnique(false)
            setReviewed(false)
            setLocalError(null)
          }}
        >
          Use coffee video preset
        </button>
        <p className="field-hint">
          One glass, DISPENSE and STAGING. This authored placement gate uses the
          licensed coffee demonstration layout and needs your review.
        </p>
        <label className="task-field">
          Task name
          <input
            value={title}
            maxLength={100}
            disabled={disabled}
            onChange={(event) => setTitle(event.target.value)}
          />
        </label>
        <label className="task-field">
          Object names
          <span className="field-hint">
            One to six visually distinct targets, separated by commas.
          </span>
          <input
            value={objects}
            disabled={disabled}
            onChange={(event) => {
              setObjects(event.target.value)
              setUnique(false)
            }}
            placeholder="fork, spoon"
          />
        </label>
        {regionNames.map((region, index) => (
          <label className="task-field" key={index}>
            {index === 0 ? 'First region ID' : 'Second region ID'}
            <input
              value={region}
              maxLength={64}
              disabled={disabled}
              onChange={(event) => {
                setRegionNames((previous) =>
                  previous.map((value, item) =>
                    item === index ? event.target.value : value,
                  ),
                )
                setReviewed(false)
              }}
              placeholder={index === 0 ? 'dispense' : 'staging'}
            />
          </label>
        ))}
        <p className="field-hint">
          Use two distinct lowercase IDs. New names reset the starting
          rectangles; approve their placement against both fresh views.
        </p>
        <label className="task-field">
          Required placement
          <textarea
            rows={5}
            maxLength={6000}
            value={requirements}
            disabled={disabled}
            onChange={(event) => setRequirements(event.target.value)}
            placeholder="The unique fork must be in LEFT. The unique spoon must be in RIGHT."
          />
        </label>
        <details className="prompt-mapping">
          <summary>
            {mode === 'guided'
              ? 'Review Spark locator descriptions'
              : 'Review SAM object prompts'}
          </summary>
          <p>
            {mode === 'guided'
              ? 'Spark uses these descriptions to propose target boxes. SAM receives those boxes to refine boundaries.'
              : 'Short object names can behave differently. These prompts identify your targets.'}{' '}
            They do not change the placement requirements.
          </p>
          {[...new Set(inventory)].slice(0, 6).map((id) => (
            <label className="task-field" key={id}>
              {label(id)} →{' '}
              {mode === 'guided' ? 'Spark locator description' : 'SAM prompt'}
              <input
                value={prompts[id]}
                maxLength={100}
                disabled={disabled}
                onChange={(event) => {
                  setConcepts((previous) => ({
                    ...previous,
                    [id]: event.target.value,
                  }))
                  setReviewed(false)
                }}
              />
            </label>
          ))}
        </details>
        <label className="task-field">
          Perception mode
          <select
            value={mode}
            disabled={disabled}
            onChange={(event) => {
              setMode(event.target.value as 'concept' | 'guided')
              setReviewed(false)
            }}
          >
            <option value="concept">SAM concept prompts</option>
            <option value="guided">Spark-guided SAM</option>
          </select>
          <span className="field-hint">
            Guided mode adds a Spark request to locate targets, then asks SAM to
            refine their boundaries. It uses more inference and can still return
            unknown. Keep every target fully visible in both views. Guided
            uploads require upright JPEGs up to 1280 pixels.
          </span>
          <span className="field-hint">
            Evidence is task-specific: concept prompts worked in the owned
            fork/spoon demo, but matched 0/8 labels in the historical coffee
            pilot (all UNKNOWN). On four fresh coffee pairs, guided v2 matched
            3/4 labels versus Spark alone’s 4/4. These small samples do not
            establish a universally better mode.
          </span>
        </label>
        <div className="confirmations">
          <label>
            <input
              type="checkbox"
              checked={unique}
              disabled={disabled}
              onChange={(event) => setUnique(event.target.checked)}
            />
            <span>
              I will use exactly one visually distinct target of each named
              class in the controlled scene.
            </span>
          </label>
        </div>
        <p className="task-scope">
          Supported now: observable placement. Counts, attachment, hidden
          contents and step order need different evidence and are not compiled
          here.
        </p>
        {localError || state.draft_error ? (
          <div className="notice notice-error" role="alert">
            <Icon name="info" />
            <span>{localError || state.draft_error}</span>
          </div>
        ) : null}
        {!state.api_ready ? (
          <div className="notice notice-error">
            Configure the local Meta key before requesting a draft.
          </div>
        ) : null}
        <button
          className="button primary full-width"
          disabled={
            disabled ||
            !unique ||
            !requirements.trim() ||
            !title.trim() ||
            (!authoredPreset && !state.api_ready)
          }
          onClick={() => void createDraft()}
        >
          {state.busy_operation === 'task_draft' ? (
            <span className="spinner" />
          ) : (
            <Icon name="plus" />
          )}
          {state.busy_operation === 'task_draft'
            ? authoredPreset
              ? 'Preparing authored checks…'
              : 'Spark is drafting checks…'
            : authoredPreset
              ? 'Review authored coffee checks'
              : 'Draft checks with Spark'}
        </button>
        <p className="capture-footnote">
          {authoredPreset
            ? 'This fixed authored preset makes no model request. '
            : 'One bounded Spark request. '}
          Creating a draft leaves the active task and its evidence unchanged.
        </p>
      </section>
      <section
        className="panel task-review"
        aria-labelledby="task-review-title"
      >
        <h2 id="task-review-title">Review executable checks</h2>
        {draft ? (
          <>
            <p className="draft-origin">
              {draft.origin === 'authored_frozen_coffee_preset'
                ? 'Authored coffee preset · no compilation inference'
                : draft.origin === 'live_meta_api'
                  ? draft.model
                  : 'Offline test draft'}{' '}
              · Awaiting approval
            </p>
            <p className="task-scope">
              Perception:{' '}
              {(draft.perception_mode || 'concept') === 'guided'
                ? `Spark-guided SAM${draft.perception_version === 'normalized_boxes_v2' ? ' v2' : ''}. Both views must support full target visibility. Object identity is proposed by Spark; SAM refines the boundary.`
                : 'SAM concept prompts.'}
            </p>
            <h3>{draft.title}</h3>
            <p className="task-requirements">{draft.requirements}</p>
            <p className="task-scope">
              Reviewed region IDs:{' '}
              {draft.protocol.regions.map((id) => id.toUpperCase()).join(' / ')}
              .
              {draft.calibration_note
                ? ` ${draft.calibration_note}`
                : ' Approve both rectangles against your actual camera view.'}
            </p>
            <ol className="task-checks">
              {draft.protocol.checks.map((check) => (
                <li key={check.id}>
                  <Icon name="image" />
                  <span>
                    <strong>{label(check.object_id)}</strong> must be in{' '}
                    <strong>{check.region_id.toUpperCase()}</strong>
                    <small>
                      {(draft.perception_mode || 'concept') === 'guided'
                        ? 'Spark locator description'
                        : 'SAM prompt'}
                      : “{draft.concepts[check.object_id]}”
                    </small>
                    <small>
                      Requires one unique target, stable positive detections and
                      fresh evidence.
                    </small>
                  </span>
                </li>
              ))}
            </ol>
            <p className="contract-id">
              Contract {draft.protocol_hash.slice(0, 12)}
            </p>
            {changed ? (
              <div className="notice">
                Your inputs changed. Request a new draft before activating.
              </div>
            ) : null}
            <div className="confirmations">
              <label>
                <input
                  type="checkbox"
                  checked={reviewed}
                  disabled={disabled || changed || !unique}
                  onChange={(event) => setReviewed(event.target.checked)}
                />
                <span>
                  I reviewed every check against my requirements and approve
                  this task.
                </span>
              </label>
            </div>
            <button
              className="button primary full-width"
              disabled={disabled || changed || !unique || !reviewed}
              onClick={() => void activate()}
            >
              <Icon name="check" />
              Activate reviewed task
            </button>
            <p className="capture-footnote">
              Activation clears the old result and requires two new views. A
              model proposal never certifies the scene.
            </p>
          </>
        ) : (
          <div className="empty-task">
            <Icon name="image" size={36} />
            <p>Your draft will appear here.</p>
            <span>
              Check the object, region and meaning of every requirement before
              activation.
            </span>
          </div>
        )}
        {task ? (
          <div className="active-contract">
            <h3>Active task</h3>
            <strong>{task.title}</strong>
            <p>{task.requirements}</p>
            <small>
              {task.reviewed
                ? 'Reviewed contract'
                : 'Authored starting contract'}{' '}
              · {task.protocol_hash.slice(0, 12)}
            </small>
          </div>
        ) : null}
      </section>
    </div>
  )
}
