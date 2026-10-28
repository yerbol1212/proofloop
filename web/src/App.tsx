import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties, PointerEvent } from 'react'
import { Icon } from './icons'
import { TaskEditor } from './TaskEditor'
import { VideoLab } from './VideoLab'
import { RecoveryPanel } from './RecoveryPanel'
import { AcquisitionPanel } from './AcquisitionPanel'
import { WorkflowPanel } from './WorkflowPanel'
import type {
  EvidenceImage,
  InspectPayload,
  LocalPhoto,
  PlacementCheck,
  Rect,
  Regions,
  Snapshot,
  Source,
  Status,
} from './types'

const statusNames: Record<Status, string> = {
  pass: 'Pass',
  fail: 'Fail',
  unknown: 'Unknown',
}
const objectLabel = (value: string) =>
  value
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/^./, (letter) => letter.toUpperCase())
const placementSummary = (state: Snapshot) =>
  state.checks
    .map(
      (check) =>
        `${objectLabel(check.object_id)} in ${check.expected_region.toUpperCase()}`,
    )
    .join('. ') + '.'

function StatusLabel({ status }: { status: Status }) {
  return (
    <span className={`status-label ${status}`}>
      <Icon
        name={
          status === 'pass' ? 'check' : status === 'fail' ? 'close' : 'info'
        }
        size={18}
      />
      {statusNames[status]}
    </span>
  )
}

function Notice({
  children,
  error = false,
}: {
  children: React.ReactNode
  error?: boolean
}) {
  return (
    <div
      className={`notice ${error ? 'notice-error' : ''}`}
      role={error ? 'alert' : undefined}
    >
      <Icon name="info" size={22} />
      <span>{children}</span>
    </div>
  )
}

function rectStyle(rect: Rect): CSSProperties {
  return {
    left: `${rect[0] * 100}%`,
    top: `${rect[1] * 100}%`,
    width: `${(rect[2] - rect[0]) * 100}%`,
    height: `${(rect[3] - rect[1]) * 100}%`,
  }
}

function regionError(regions: Regions): string | null {
  for (const name of Object.keys(regions)) {
    const rect = regions[name]
    if (
      rect.some((value) => !Number.isFinite(value) || value < 0 || value > 1) ||
      rect[2] - rect[0] < 0.02 ||
      rect[3] - rect[1] < 0.02
    )
      return 'Give both areas a width and height of at least 2%.'
  }
  const [a, b] = Object.values(regions)
  if (!a || !b || Object.keys(regions).length !== 2)
    return 'Review both named areas.'
  if (
    Math.min(a[2], b[2]) > Math.max(a[0], b[0]) &&
    Math.min(a[3], b[3]) > Math.max(a[1], b[1])
  )
    return 'The named areas must not overlap. Leave a gap between their boundaries.'
  return null
}

function PhotoView({
  image,
  regions,
  index,
  recorded,
  editing,
  onRegion,
  activeRegion,
  boxes = true,
}: {
  image: EvidenceImage | LocalPhoto
  regions: Regions
  index: number
  recorded?: boolean
  editing?: boolean
  onRegion?: (name: string, rect: Rect) => void
  activeRegion?: string | null
  boxes?: boolean
}) {
  const [drag, setDrag] = useState<{
    start: [number, number]
    current: [number, number]
  } | null>(null)
  const surfaceRef = useRef<HTMLDivElement>(null)
  const source = 'data_url' in image ? image.data_url : image.url
  const point = (event: PointerEvent<HTMLDivElement>): [number, number] => {
    const box = event.currentTarget.getBoundingClientRect()
    return [
      Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
      Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)),
    ]
  }
  const finish = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag || !activeRegion) return
    const end = point(event)
    const rect: Rect = [
      Math.min(drag.start[0], end[0]),
      Math.min(drag.start[1], end[1]),
      Math.max(drag.start[0], end[0]),
      Math.max(drag.start[1], end[1]),
    ]
    if (rect[2] - rect[0] >= 0.02 && rect[3] - rect[1] >= 0.02)
      onRegion?.(activeRegion, rect)
    setDrag(null)
  }
  return (
    <figure className="photo-view">
      <div
        className={`photo-frame ${editing && activeRegion ? 'can-draw' : ''}`}
      >
        <div
          ref={surfaceRef}
          className="image-surface"
          style={
            {
              aspectRatio: `${image.width}/${image.height}`,
              '--photo-ratio': image.width / image.height,
            } as CSSProperties
          }
          onPointerDown={(event) => {
            if (!editing || !activeRegion) return
            event.preventDefault()
            event.currentTarget.setPointerCapture(event.pointerId)
            const start = point(event)
            setDrag({ start, current: start })
          }}
          onPointerMove={(event) => {
            if (drag) setDrag({ ...drag, current: point(event) })
          }}
          onPointerUp={finish}
          onPointerCancel={() => setDrag(null)}
        >
          <img
            src={source}
            width={image.width}
            height={image.height}
            alt={`View ${index + 1} of the task arrangement`}
            draggable={false}
          />
          {Object.keys(regions).map((name, regionIndex) => (
            <div
              key={name}
              className={`region-outline ${regionIndex === 0 ? 'left' : 'right'}`}
              style={rectStyle(regions[name])}
            >
              <span>{name.toUpperCase()}</span>
            </div>
          ))}
          {boxes && 'boxes' in image
            ? image.boxes.map((box, n) => (
                <div
                  key={`${box.object_id}-${n}`}
                  className={`detection-box ${box.object_id}`}
                  style={rectStyle([box.x1, box.y1, box.x2, box.y2])}
                  title={`SAM detection: ${box.object_id}`}
                />
              ))
            : null}
          {drag && activeRegion ? (
            <div
              className={`region-outline drawing ${Object.keys(regions).indexOf(activeRegion) === 0 ? 'left' : 'right'}`}
              style={rectStyle([
                Math.min(drag.start[0], drag.current[0]),
                Math.min(drag.start[1], drag.current[1]),
                Math.max(drag.start[0], drag.current[0]),
                Math.max(drag.start[1], drag.current[1]),
              ])}
            />
          ) : null}
        </div>
      </div>
      <figcaption>
        <strong>View {index + 1}</strong>
        {recorded ? (
          <span> · {index === 0 ? '16.0' : '18.0'} s</span>
        ) : 'name' in image ? (
          <span className="photo-filename"> · {image.name}</span>
        ) : null}
      </figcaption>
    </figure>
  )
}

function EvidencePanel({ state }: { state: Snapshot }) {
  return (
    <section className="panel evidence-panel" aria-labelledby="evidence-title">
      <div className="panel-heading">
        <h2 id="evidence-title">Current evidence</h2>
        <span className="source-label">
          <Icon name="image" />
          {state.source_kind === 'recorded'
            ? 'Recorded reference'
            : state.source_kind === 'camera'
              ? 'Camera capture'
              : state.source_kind === 'upload'
                ? 'Uploaded photos'
                : 'No evidence'}
        </span>
      </div>
      {state.images.length > 0 ? (
        <div className="photo-grid">
          {state.images.map((image, index) => (
            <PhotoView
              key={image.id}
              image={image}
              regions={state.regions}
              index={index}
              recorded={state.source_kind === 'recorded'}
            />
          ))}
        </div>
      ) : (
        <div className="empty-evidence">
          <Icon name="camera" size={42} />
          <p>Two new views are required.</p>
        </div>
      )}
      {state.busy ? (
        <div className="checking-line" role="status">
          <span className="spinner" />
          Inspecting both views with Meta SAM…
        </div>
      ) : null}
    </section>
  )
}

function CheckRow({ check }: { check: PlacementCheck }) {
  const name = objectLabel(check.object_id)
  return (
    <div className="check-row">
      <Icon
        name={
          check.object_id === 'fork'
            ? 'fork'
            : check.object_id === 'spoon'
              ? 'spoon'
              : 'image'
        }
        size={40}
      />
      <div className="check-content">
        <div className="check-title">
          <h3>{name}</h3>
          <StatusLabel status={check.status} />
        </div>
        <p>
          Observed{' '}
          <strong className={check.status}>
            {check.observed_region
              ? check.observed_region.toUpperCase()
              : 'UNKNOWN'}
          </strong>
          <span className="check-divider"> · </span>Required{' '}
          {check.expected_region.toUpperCase()}
        </p>
      </div>
    </div>
  )
}

function needsInspection(state: Snapshot): boolean {
  return Boolean(
    state.inspection_request &&
      (!state.proposal ||
        state.checks.some((check) => check.status === 'unknown')),
  )
}

function RepairSteps({
  state,
  capture = false,
}: {
  state: Snapshot
  capture?: boolean
}) {
  if (state.inspection_request && needsInspection(state))
    return (
      <div className="repair-content">
        <h2>
          <Icon name="camera" size={26} />
          Next inspection
        </h2>
        <p className="inspection-request-title">
          {state.inspection_request.title}
        </p>
        {state.acquisition ? (
          <AcquisitionPanel plan={state.acquisition} />
        ) : (
          <ol className="repair-steps">
            {state.inspection_request.instructions.map((step, index) => (
              <li key={step}>
                <span>{index + 1}</span>
                <p>{step}</p>
              </li>
            ))}
          </ol>
        )}
      </div>
    )
  const steps = state.proposal?.steps?.length
    ? state.proposal.steps.map((step) =>
        step.startsWith('Keep both targets visible') ||
        step.startsWith('Keep every target visible')
          ? 'Keep all targets visible and hands clear.'
          : step,
      )
    : state.checks.map(
        (check) =>
          `${objectLabel(check.object_id)} must be in ${check.expected_region.toUpperCase()}.`,
      )
  // The proposal is a plan, never evidence that a physical action happened.
  return (
    <div className="repair-content">
      <h2>
        <Icon name="wrench" size={26} />
        {capture
          ? 'Placement to capture'
          : state.proposal
            ? 'Suggested repair'
            : 'Required placement'}
      </h2>
      <ol className="repair-steps">
        {steps.map((step, index) => (
          <li key={step}>
            <span>{index + 1}</span>
            <p>{step}</p>
          </li>
        ))}
      </ol>
    </div>
  )
}

function Inspector({
  state,
  onRepair,
  pending,
  capture = false,
}: {
  state: Snapshot
  onRepair: () => void
  pending: boolean
  capture?: boolean
}) {
  return (
    <aside className="panel inspector">
      {!capture ? (
        <>
          <h2>Placement checks</h2>
          <div className="check-list">
            {state.checks.map((check) => (
              <CheckRow key={check.check_id} check={check} />
            ))}
          </div>
          <div className="inspector-divider" />
        </>
      ) : null}
      {state.phase === 'verified' ? (
        <>
          <div className="verified-summary">
            <span className="verified-mark">
              <Icon name="check" size={32} />
            </span>
            <h2>Placement verified.</h2>
            <p>All required target placements were observed.</p>
          </div>
          <button
            className="button primary full-width"
            onClick={onRepair}
            disabled={pending}
          >
            <Icon name="refresh" />
            Inspect after any movement
          </button>
          <Notice>
            This result describes the captured views. Moving an item requires a
            new inspection.
          </Notice>
        </>
      ) : state.busy ? (
        <Notice>Checking fresh evidence. No correction is verified yet.</Notice>
      ) : (
        <>
          <RepairSteps state={state} capture={capture} />
          {!capture ? (
            <>
              <button
                className="button primary full-width"
                onClick={onRepair}
                disabled={
                  pending ||
                  Boolean(
                    state.acquisition && state.acquisition.status !== 'ready',
                  )
                }
              >
                {pending ? (
                  <span className="spinner" />
                ) : (
                  <Icon
                    name={
                      state.status === 'fail' && !needsInspection(state)
                        ? 'play'
                        : 'camera'
                    }
                  />
                )}
                {state.acquisition && state.acquisition.status !== 'ready'
                  ? 'Evidence follow-up stopped'
                  : needsInspection(state)
                    ? 'Capture requested views'
                    : state.status === 'fail'
                      ? 'Start repair'
                      : 'Capture new evidence'}
              </button>
              <Notice>
                <strong>Hint</strong>
                <br />
                Verification needs two new views.
              </Notice>
            </>
          ) : null}
        </>
      )}
    </aside>
  )
}

async function filePhoto(file: File): Promise<LocalPhoto> {
  if (!['image/jpeg', 'image/png'].includes(file.type))
    throw new Error(
      'Choose JPEG or PNG photos. For iPhone HEIC photos, export them as JPEG first.',
    )
  if (file.size > 5 * 1024 * 1024)
    throw new Error(
      'Each photo must be smaller than 5 MB. Export a smaller JPEG without replacing it with an old view.',
    )
  if (file.type === 'image/jpeg')
    validateJpegOrientation(await file.arrayBuffer())
  const data_url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () =>
      reject(new Error('This photo could not be read. Choose it again.'))
    reader.readAsDataURL(file)
  })
  const dimensions = await new Promise<{ width: number; height: number }>(
    (resolve, reject) => {
      const image = new Image()
      image.onload = () =>
        resolve({ width: image.naturalWidth, height: image.naturalHeight })
      image.onerror = () =>
        reject(
          new Error(
            'This photo could not be opened. Export a valid JPEG or PNG.',
          ),
        )
      image.src = data_url
    },
  )
  if (
    dimensions.width < 64 ||
    dimensions.height < 64 ||
    dimensions.width > 4096 ||
    dimensions.height > 4096 ||
    dimensions.width * dimensions.height > 16_000_000
  )
    throw new Error(
      'Use photos between 64 and 4096 pixels on each side, at most 16 megapixels. Export a smaller JPEG if needed.',
    )
  return { data_url, ...dimensions, name: file.name }
}

function validateJpegOrientation(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer)
  const data = new DataView(buffer)
  let offset = 2
  while (offset + 4 <= bytes.length && bytes[offset] === 0xff) {
    while (bytes[offset] === 0xff) offset++
    const marker = bytes[offset++]
    if (marker === 0xda || marker === 0xd9) return
    if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) continue
    if (offset + 2 > bytes.length) return
    const length = data.getUint16(offset)
    if (length < 2 || offset + length > bytes.length) return
    if (
      marker === 0xe1 &&
      length >= 16 &&
      bytes[offset + 2] === 0x45 &&
      bytes[offset + 3] === 0x78 &&
      bytes[offset + 4] === 0x69 &&
      bytes[offset + 5] === 0x66 &&
      bytes[offset + 6] === 0 &&
      bytes[offset + 7] === 0
    ) {
      const tiff = offset + 8
      const little = bytes[tiff] === 0x49 && bytes[tiff + 1] === 0x49
      if (!little && !(bytes[tiff] === 0x4d && bytes[tiff + 1] === 0x4d)) return
      const directory = tiff + data.getUint32(tiff + 4, little)
      if (directory + 2 > offset + length || directory < tiff + 8) return
      const count = data.getUint16(directory, little)
      for (let index = 0; index < count; index++) {
        const entry = directory + 2 + index * 12
        if (entry + 12 > offset + length) return
        if (
          data.getUint16(entry, little) === 0x112 &&
          data.getUint16(entry + 2, little) === 3 &&
          data.getUint32(entry + 4, little) === 1 &&
          data.getUint16(entry + 8, little) !== 1
        )
          throw new Error(
            'This JPEG uses rotation metadata. Export an upright JPEG or PNG with orientation removed, or take a screenshot of each new photo before uploading.',
          )
      }
    }
    offset += length
  }
}

function CapturePanel({
  state,
  onInspect,
  pending,
}: {
  state: Snapshot
  onInspect: (payload: InspectPayload) => Promise<void>
  pending: boolean
}) {
  const [photos, setPhotos] = useState<(LocalPhoto | null)[]>([null, null])
  const [source, setSource] = useState<Source>('upload')
  const [regions, setRegions] = useState<Regions>(() =>
    structuredClone(state.regions),
  )
  const [activeRegion, setActiveRegion] = useState<string | null>(null)
  const [sceneConfirmed, setSceneConfirmed] = useState(false)
  const [freshConfirmed, setFreshConfirmed] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraOpening, setCameraOpening] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const cameraGeneration = useRef(0)
  const openingRef = useRef(false)
  const fileGeneration = useRef(0)
  const mounted = useRef(true)
  const fileRef = useRef<HTMLInputElement>(null)
  const slotRefs = [
    useRef<HTMLInputElement>(null),
    useRef<HTMLInputElement>(null),
  ]
  const disabled = pending || capturing || loadingFiles || cameraOpening
  const geometryError = regionError(regions)
  const sameDimensions =
    photos.every(Boolean) &&
    photos[0]!.width === photos[1]!.width &&
    photos[0]!.height === photos[1]!.height
  const cameraExpired =
    source === 'camera' &&
    photos[0]?.captured_at_ms !== undefined &&
    Date.now() - photos[0].captured_at_ms > 30_000
  const canInspect = Boolean(
    state.capture_token &&
      state.api_ready &&
      photos.every(Boolean) &&
      sameDimensions &&
      sceneConfirmed &&
      (source === 'camera' || freshConfirmed) &&
      !geometryError &&
      !cameraExpired &&
      !disabled,
  )
  const stopCamera = useCallback(() => {
    cameraGeneration.current++
    openingRef.current = false
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraOn(false)
    setCameraOpening(false)
  }, [])
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      cameraGeneration.current++
      fileGeneration.current++
      openingRef.current = false
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])
  useEffect(() => {
    if (cameraOn && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      void videoRef.current.play().catch(() => {
        setError('The camera preview could not start. Try enabling it again.')
        stopCamera()
      })
    }
  }, [cameraOn, stopCamera])
  async function startCamera() {
    if (openingRef.current || disabled) return
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        'Camera access is unavailable here. Open this app on localhost, or upload two new photos.',
      )
      return
    }
    const generation = ++cameraGeneration.current
    fileGeneration.current++
    openingRef.current = true
    setCameraOpening(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      if (!mounted.current || generation !== cameraGeneration.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = stream
      setSource('camera')
      setPhotos([null, null])
      setSceneConfirmed(false)
      setFreshConfirmed(false)
      setCameraOn(true)
    } catch {
      if (mounted.current && generation === cameraGeneration.current)
        setError(
          'Camera access was not granted. Allow it in your browser, or upload two new photos.',
        )
    } finally {
      if (mounted.current && generation === cameraGeneration.current) {
        openingRef.current = false
        setCameraOpening(false)
      }
    }
  }
  function cameraPhoto(index: number): LocalPhoto {
    const video = videoRef.current
    if (!video || !video.videoWidth || video.readyState < 2)
      throw new Error('Wait for the camera preview, then try capturing again.')
    const scale = Math.min(
      1,
      1280 / Math.max(video.videoWidth, video.videoHeight),
    )
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(video.videoWidth * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    const context = canvas.getContext('2d')
    if (!context)
      throw new Error(
        'Your browser could not capture the camera image. Upload two new photos instead.',
      )
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    return {
      data_url: canvas.toDataURL('image/jpeg', 0.92),
      width: canvas.width,
      height: canvas.height,
      name: `Camera view ${index + 1}`,
      captured_at_ms: Date.now(),
    }
  }
  async function capturePair() {
    setCapturing(true)
    setError(null)
    setSceneConfirmed(false)
    try {
      const first = cameraPhoto(0)
      setPhotos([first, null])
      await new Promise((resolve) => window.setTimeout(resolve, 2000))
      if (!mounted.current) return
      const second = cameraPhoto(1)
      if (first.data_url === second.data_url)
        throw new Error(
          'The camera returned the same image twice. Keep the scene visible and capture again.',
        )
      setPhotos([first, second])
      stopCamera()
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : 'The camera capture failed.',
      )
      setPhotos([null, null])
    } finally {
      if (mounted.current) setCapturing(false)
    }
  }
  async function loadFiles(files: FileList | null, slot?: number) {
    if (!files?.length) return
    const generation = ++fileGeneration.current
    setError(null)
    setLoadingFiles(true)
    stopCamera()
    setSource('upload')
    setSceneConfirmed(false)
    setFreshConfirmed(false)
    if (slot === undefined) setPhotos([null, null])
    else
      setPhotos((previous) =>
        previous.map((photo, index) => (index === slot ? null : photo)),
      )
    try {
      if (slot === undefined && files.length !== 2)
        throw new Error(
          'Choose exactly two fresh photos taken at least one second apart.',
        )
      const selected = await Promise.all(Array.from(files).map(filePhoto))
      if (
        state.active_task?.perception_mode === 'guided' &&
        selected.some(
          (photo) =>
            !photo.data_url.startsWith('data:image/jpeg;') ||
            Math.max(photo.width, photo.height) > 1280,
        )
      )
        throw new Error(
          'This reviewed task uses guided perception. Upload upright JPEG photos with no rotation metadata and a longest side of 1280 pixels or less.',
        )
      if (!mounted.current || generation !== fileGeneration.current) return
      if (slot === undefined) setPhotos(selected)
      else
        setPhotos((previous) =>
          previous.map((photo, index) =>
            index === slot ? selected[0] : photo,
          ),
        )
    } catch (caught) {
      if (mounted.current && generation === fileGeneration.current)
        setError(
          caught instanceof Error
            ? caught.message
            : 'The photos could not be opened.',
        )
    } finally {
      if (mounted.current && generation === fileGeneration.current)
        setLoadingFiles(false)
    }
  }
  function updateRegion(name: string, value: Rect) {
    setRegions((previous) => ({ ...previous, [name]: value }))
    setSceneConfirmed(false)
  }
  async function submit() {
    if (!canInspect || !state.capture_token) return
    setError(null)
    stopCamera()
    await onInspect({
      capture_token: state.capture_token,
      source,
      images: photos.map((photo) => ({
        data_url: photo!.data_url,
        ...(photo!.captured_at_ms === undefined
          ? {}
          : { captured_at_ms: photo!.captured_at_ms }),
      })),
      regions,
      confirm_regions: true,
      confirm_unique_targets: true,
      confirm_fresh_capture: true,
      confirm_pair_spacing: true,
    })
  }
  return (
    <section className="panel capture-panel" aria-labelledby="capture-title">
      <div className="panel-heading">
        <div>
          <h2 id="capture-title">New evidence</h2>
          <p className="panel-intro">
            Upload two photos or use camera controls.
          </p>
        </div>
      </div>
      <input
        ref={fileRef}
        className="visually-hidden"
        aria-label="Upload two fresh photos"
        type="file"
        accept="image/jpeg,image/png"
        multiple
        onChange={(event) => {
          void loadFiles(event.target.files)
          event.target.value = ''
        }}
      />
      <div className="capture-actions">
        <button
          className="button secondary"
          disabled={disabled}
          onClick={() => fileRef.current?.click()}
        >
          <Icon name="upload" />
          Upload two photos
        </button>
        <button
          className="button secondary"
          disabled={disabled}
          onClick={() => {
            if (cameraOn) stopCamera()
            else void startCamera()
          }}
        >
          {cameraOpening ? (
            <span className="spinner" />
          ) : (
            <Icon name="camera" />
          )}
          {cameraOpening
            ? 'Opening camera…'
            : cameraOn
              ? 'Turn camera off'
              : 'Use camera'}
        </button>
      </div>
      {cameraOn ? (
        <div className="camera-area">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            aria-label="Live camera preview"
          />
          <div className="camera-footer">
            <p>Keep the camera still and all targets visible.</p>
            <button
              className="button primary"
              disabled={disabled}
              onClick={() => void capturePair()}
            >
              {capturing ? (
                <span className="spinner" />
              ) : (
                <Icon name="camera" />
              )}
              {capturing ? 'Hold still for view 2…' : 'Capture two views'}
            </button>
          </div>
        </div>
      ) : null}
      <div className="photo-grid capture-grid">
        {photos.map((photo, index) => (
          <div key={index} className="capture-slot">
            <input
              ref={slotRefs[index]}
              className="visually-hidden"
              aria-label={`Upload view ${index + 1}`}
              type="file"
              accept="image/jpeg,image/png"
              onChange={(event) => {
                void loadFiles(event.target.files, index)
                event.target.value = ''
              }}
            />
            {photo ? (
              <>
                <PhotoView
                  image={photo}
                  index={index}
                  regions={regions}
                  editing={index === 0 && !disabled}
                  activeRegion={activeRegion}
                  onRegion={updateRegion}
                  boxes={false}
                />
                <button
                  className="text-button"
                  disabled={disabled}
                  onClick={() => slotRefs[index].current?.click()}
                >
                  Replace view {index + 1}
                </button>
              </>
            ) : (
              <>
                <button
                  className="photo-placeholder"
                  disabled={disabled}
                  onClick={() => slotRefs[index].current?.click()}
                >
                  <Icon name="camera" size={34} />
                  <span>
                    {capturing && index === 1
                      ? 'Hold still…'
                      : 'Choose a photo'}
                  </span>
                </button>
                <p className="placeholder-label">View {index + 1}</p>
              </>
            )}
          </div>
        ))}
      </div>
      {photos.every(Boolean) && !sameDimensions ? (
        <Notice error>
          Both views need the same dimensions and orientation. Use the same
          camera settings for each.
        </Notice>
      ) : null}
      <section className="calibration" aria-labelledby="calibration-title">
        <h3 id="calibration-title">
          Set{' '}
          {Object.keys(regions)
            .map((name) => name.toUpperCase())
            .join(' and ')}
        </h3>
        <p>Mark the two areas on your first photo.</p>
        <div className="region-actions">
          {Object.keys(regions).map((name, regionIndex) => (
            <button
              key={name}
              className={`region-button ${regionIndex === 0 ? 'left' : 'right'} ${activeRegion === name ? 'selected' : ''}`}
              disabled={!photos[0] || disabled}
              aria-pressed={activeRegion === name}
              onClick={() =>
                setActiveRegion(activeRegion === name ? null : name)
              }
            >
              <span className="region-swatch" />
              <span>Set {name.toUpperCase()} area</span>
              <Icon name="chevron" />
            </button>
          ))}
        </div>
        {activeRegion ? (
          <p className="draw-help" role="status">
            Drag on View 1 to mark {activeRegion.toUpperCase()}. The whole
            object must fit inside its area.
          </p>
        ) : null}
        <details className="numeric-regions">
          <summary>Adjust areas with numbers</summary>
          <p>Percentages of the first photo. Keep a gap between the areas.</p>
          {Object.keys(regions).map((name) => (
            <fieldset key={name}>
              <legend>{name.toUpperCase()}</legend>
              <div className="coordinate-fields">
                {['Left edge', 'Top edge', 'Right edge', 'Bottom edge'].map(
                  (label, index) => (
                    <label key={label}>
                      {label}
                      <input
                        type="number"
                        min="0"
                        max="100"
                        step="0.5"
                        value={Math.round(regions[name][index] * 1000) / 10}
                        disabled={disabled}
                        onChange={(event) => {
                          const updated = [...regions[name]] as Rect
                          updated[index] = event.target.valueAsNumber / 100
                          updateRegion(name, updated)
                        }}
                      />
                    </label>
                  ),
                )}
              </div>
            </fieldset>
          ))}
        </details>
        {geometryError ? (
          <p className="field-error" role="alert">
            {geometryError}
          </p>
        ) : null}
      </section>
      <div className="confirmations">
        <label>
          <input
            type="checkbox"
            checked={sceneConfirmed}
            disabled={
              disabled || !photos.every(Boolean) || Boolean(geometryError)
            }
            onChange={(event) => setSceneConfirmed(event.target.checked)}
          />
          <span>
            The marked areas match both views, with one target of each named
            class.
          </span>
        </label>
        {source === 'upload' ? (
          <label>
            <input
              type="checkbox"
              checked={freshConfirmed}
              disabled={disabled || !photos.every(Boolean)}
              onChange={(event) => setFreshConfirmed(event.target.checked)}
            />
            <span>
              I took both photos after starting this inspection, at least one
              second apart.
            </span>
          </label>
        ) : (
          <p className="capture-timing">
            Camera views are captured two seconds apart. Capture again if more
            than 30 seconds pass.
          </p>
        )}
      </div>
      {cameraExpired ? (
        <Notice>
          These camera views have expired. Use camera to capture two new views,
          then verify promptly.
        </Notice>
      ) : null}
      {error ? <Notice error>{error}</Notice> : null}
      {!state.api_ready ? (
        <Notice error>
          Meta API credentials are unavailable to the local server. Add
          META_API_KEY to the project’s .env.local file, then retry.
        </Notice>
      ) : null}
      <button
        className="button primary full-width verify-button"
        disabled={!canInspect}
        onClick={() => void submit()}
      >
        {pending ? <span className="spinner" /> : <Icon name="play" />}
        {pending ? 'Checking new evidence…' : 'Check new evidence'}
      </button>
      <p className="capture-footnote">
        <Icon name="info" size={19} />
        Upload photos from your phone on this computer. Upright JPEG or PNG, up
        to 5 MB each. Export HEIC or rotated photos first.
      </p>
    </section>
  )
}

function EvidenceLog({ state }: { state: Snapshot }) {
  return (
    <section className="panel log-panel">
      <h2>Evidence log</h2>
      <p className="log-note">{state.source_note}</p>
      <dl className="evidence-facts">
        {state.active_task ? (
          <>
            <div>
              <dt>Active task</dt>
              <dd>
                {state.active_task.title} ·{' '}
                {state.active_task.protocol_hash.slice(0, 12)}
              </dd>
            </div>
            <div>
              <dt>Perception</dt>
              <dd>
                {state.active_task.perception_mode === 'guided'
                  ? `Spark-guided SAM${state.active_task.perception_version === 'normalized_boxes_v2' ? ' v2' : ''}`
                  : 'SAM concept prompts'}
              </dd>
            </div>
            <div>
              <dt>Review provenance</dt>
              <dd>
                {state.active_task.review_origin
                  ? state.active_task.review_origin.replaceAll('_', ' ')
                  : 'Authored default; not reviewed'}
              </dd>
            </div>
          </>
        ) : null}
        <div>
          <dt>Source</dt>
          <dd>
            {state.source_kind === 'recorded'
              ? 'Historical recorded reference'
              : state.source_kind === 'camera'
                ? 'Camera · client-reported capture times'
                : state.source_kind === 'upload'
                  ? 'Uploads · operator-declared freshness and spacing'
                  : 'Awaiting new evidence'}
          </dd>
        </div>
        <div>
          <dt>Revision</dt>
          <dd>{state.revision}</dd>
        </div>
        <div>
          <dt>Current result</dt>
          <dd>{statusNames[state.status]}</dd>
        </div>
        {state.verified_at_ms ? (
          <div>
            <dt>Last passing inspection</dt>
            <dd>
              {new Date(state.verified_at_ms).toLocaleTimeString()}{' '}
              {state.stale ? '· expired' : '· captured observation'}
            </dd>
          </div>
        ) : null}
      </dl>
      {state.inference_origin ? (
        <p className="inference-origin">
          Model source:{' '}
          {state.inference_origin === 'injected_client_not_verified_live'
            ? 'Injected test client · not verified live Meta inference'
            : state.inference_origin.replaceAll('_', ' ')}
        </p>
      ) : null}
      <h3>Placement reasons</h3>
      <ul className="reason-list">
        {state.checks.map((check) => (
          <li key={check.check_id}>
            <StatusLabel status={check.status} />
            <div>
              <strong>{objectLabel(check.object_id)}</strong>
              <p>{check.reason}</p>
            </div>
          </li>
        ))}
      </ul>
      <RecoveryPanel
        receipt={state.recovery || null}
        activeProtocolHash={state.active_task?.protocol_hash}
        activeTaskHash={state.active_task?.task_hash}
      />
      {state.acquisition ? <AcquisitionPanel plan={state.acquisition} /> : null}
      <h3>Session trace</h3>
      <ol className="trace-list">
        {state.trace.map((entry) => (
          <li key={entry.id}>
            <span className={`trace-dot ${entry.status}`} />
            <div>
              <p>{entry.label}</p>
              <span>Revision {entry.revision}</span>
            </div>
          </li>
        ))}
      </ol>
      {!state.recovery ? (
        <p className="log-footnote">
          A repair proposal is hypothetical. Only new model observations can
          support a passing correction. Upload timing is declared by the
          operator; this app cannot prove when a photo was taken.
        </p>
      ) : null}
    </section>
  )
}

function ProgressRail({ state }: { state: Snapshot }) {
  const hasObserved =
    state.trace.some((entry) =>
      [
        'seed_loaded',
        'recorded_reference_loaded',
        'inspection_complete',
      ].includes(entry.event),
    ) || state.source_kind === 'recorded'
  const proposed = Boolean(state.proposal) || state.recovery?.kind === 'repair'
  const inspectionOnly =
    needsInspection(state) || state.recovery?.kind === 'inspection'
  const fresh = state.source_kind === 'camera' || state.source_kind === 'upload'
  const verified = state.phase === 'verified'
  return (
    <footer className="progress-rail" aria-label="Inspection progress">
      <ol>
        {[
          { name: 'Observed', done: hasObserved },
          {
            name: inspectionOnly ? 'Evidence requested' : 'Repair proposed',
            done: inspectionOnly
              ? state.trace.some((entry) => entry.event === 'repair_started')
              : proposed,
          },
          { name: 'New capture', done: fresh },
          { name: 'Verification', done: verified },
        ].map((step, index) => (
          <li key={step.name} className={step.done ? 'complete' : ''}>
            <span className="progress-circle">
              {step.done ? <Icon name="check" size={18} /> : index + 1}
            </span>
            <span>{step.name}</span>
          </li>
        ))}
      </ol>
    </footer>
  )
}

function heading(state: Snapshot): [string, string] {
  if (state.busy_operation === 'task_draft')
    return [
      'Drafting task checks.',
      'The active task and its evidence are unchanged while Spark drafts.',
    ]
  if (state.busy)
    return [
      'Checking the new evidence.',
      'Keep the arrangement still while the new evidence is inspected.',
    ]
  if (needsInspection(state))
    return ['More evidence is needed.', state.inspection_request!.title]
  if (state.phase === 'awaiting_capture')
    return ['Capture the correction.', placementSummary(state)]
  if (state.phase === 'verified')
    return [
      'Placement verified.',
      'All required target placements were observed.',
    ]
  if (state.phase === 'mistake') {
    const failures = state.checks.filter((check) => check.status === 'fail')
    return failures.length === 1
      ? [
          `${objectLabel(failures[0].object_id)} in the wrong area.`,
          `Move the ${objectLabel(failures[0].object_id).toLowerCase()} to ${failures[0].expected_region.toUpperCase()}, then show two new views.`,
        ]
      : [
          'Items in the wrong areas.',
          'Return each target to its required area, then show two new views.',
        ]
  }
  if (state.phase === 'error')
    return [
      'The correction is unverified.',
      'Start a new inspection and show two clear new views.',
    ]
  if (state.phase === 'needs_evidence')
    return [
      'Inspect your arrangement.',
      placementSummary(state) + ' Start with two new views.',
    ]
  return [
    'New evidence is needed.',
    state.stale
      ? 'The captured result has expired. Inspect again after any movement.'
      : 'Keep all targets visible, then capture two clear views.',
  ]
}

export function App() {
  const [state, setState] = useState<Snapshot | null>(null)
  const [tab, setTab] = useState<
    'inspection' | 'log' | 'task' | 'lab' | 'workflow'
  >('inspection')
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const csrfRef = useRef('')
  const currentRef = useRef<Snapshot | null>(null)
  const requestSequence = useRef(0)
  const appliedSequence = useRef(0)
  const live = useRef(true)
  const applyState = useCallback((next: Snapshot, sequence: number) => {
    if (
      !live.current ||
      sequence < appliedSequence.current ||
      (currentRef.current && next.revision < currentRef.current.revision)
    )
      return
    appliedSequence.current = sequence
    csrfRef.current = next.csrf_token || csrfRef.current
    currentRef.current = next
    setState(next)
    setConnectionError(null)
  }, [])
  const refresh = useCallback(async () => {
    const sequence = ++requestSequence.current
    try {
      const response = await fetch('/api/state', { cache: 'no-store' })
      if (!response.ok) throw new Error()
      applyState((await response.json()) as Snapshot, sequence)
    } catch {
      if (live.current && sequence >= appliedSequence.current)
        setConnectionError(
          'The local ProofLoop server is unavailable. Keep it running, then retry the connection.',
        )
    }
  }, [applyState])
  useEffect(() => {
    live.current = true
    void refresh()
    const timer = window.setInterval(() => {
      void refresh()
    }, 2000)
    return () => {
      live.current = false
      window.clearInterval(timer)
    }
  }, [refresh])
  async function action(
    path: string,
    payload?: InspectPayload | Record<string, unknown>,
    destination: 'inspection' | 'task' | 'workflow' = 'inspection',
  ): Promise<boolean> {
    if (pending || currentRef.current?.busy || connectionError) return false
    setPending(true)
    setActionError(null)
    try {
      const response = await fetch(path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-ProofLoop-Token': csrfRef.current,
        },
        body: JSON.stringify(payload ?? {}),
      })
      const result = (await response.json()) as Snapshot & {
        error?: { message?: string }
      }
      if (!response.ok)
        throw new Error(
          result.error?.message ||
            'The request could not complete. Retry with a new inspection.',
        )
      applyState(result, ++requestSequence.current)
      setTab(destination)
      return true
    } catch (caught) {
      setActionError(
        caught instanceof Error
          ? caught.message
          : 'The request could not complete.',
      )
      await refresh()
      return false
    } finally {
      if (live.current) setPending(false)
    }
  }
  const viewState =
    state && connectionError
      ? {
          ...state,
          status: 'unknown' as const,
          phase:
            state.phase === 'awaiting_capture'
              ? ('awaiting_capture' as const)
              : ('unknown' as const),
          stale: true,
          source_note:
            'The local server is disconnected. The current inspection is unverified until it reconnects.',
          checks: state.checks.map((check) => ({
            ...check,
            status: 'unknown' as const,
            reason:
              'The local server is disconnected; no current verification is available.',
          })),
        }
      : state
  const capture = viewState?.phase === 'awaiting_capture'
  const blocked = pending || Boolean(connectionError) || Boolean(state?.busy)
  const titles = connectionError
    ? [
        'Inspection is unverified.',
        'Reconnect to the local server before continuing.',
      ]
    : viewState
      ? heading(viewState)
      : [
          'Loading your inspection…',
          'Connecting to the local ProofLoop server.',
        ]
  return (
    <div className="app-shell">
      <header className="app-header">
        <a
          className="wordmark"
          href="#inspection"
          onClick={() => setTab('inspection')}
        >
          ProofLoop
        </a>
        <nav aria-label="Workbench">
          <button
            className={tab === 'task' ? 'active' : ''}
            onClick={() => setTab('task')}
            aria-current={tab === 'task' ? 'page' : undefined}
          >
            Task
          </button>
          <button
            className={tab === 'inspection' ? 'active' : ''}
            onClick={() => setTab('inspection')}
            aria-current={tab === 'inspection' ? 'page' : undefined}
          >
            Inspection
          </button>
          <button
            className={tab === 'lab' ? 'active' : ''}
            onClick={() => setTab('lab')}
            aria-current={tab === 'lab' ? 'page' : undefined}
          >
            Video lab
          </button>
          <button
            className={tab === 'workflow' ? 'active' : ''}
            onClick={() => setTab('workflow')}
            aria-current={tab === 'workflow' ? 'page' : undefined}
          >
            Workflow
          </button>
          <button
            className={tab === 'log' ? 'active' : ''}
            onClick={() => setTab('log')}
            aria-current={tab === 'log' ? 'page' : undefined}
          >
            Evidence log
          </button>
        </nav>
        <button
          className="button new-inspection"
          disabled={
            !state ||
            blocked ||
            state.busy ||
            Boolean(state.acquisition && state.acquisition.status !== 'ready')
          }
          onClick={() => void action('/api/new')}
        >
          <Icon name="plus" size={22} />
          New inspection
        </button>
      </header>
      <main id="inspection">
        <div className="page-heading">
          <h1>
            {tab === 'lab'
              ? 'Test against the real world.'
              : tab === 'workflow'
                ? 'Check before continuing.'
                : tab === 'task'
                  ? 'Define what complete means.'
                  : tab === 'log'
                    ? 'Every result has a trail.'
                    : titles[0]}
          </h1>
          <p>
            {tab === 'lab'
              ? 'Compare model results with frozen labels on licensed robot videos.'
              : tab === 'workflow'
                ? 'A visible preparation checkpoint and a separate decision to continue.'
                : tab === 'task'
                  ? 'Describe requirements, review Spark’s checks, then inspect against the approved task.'
                  : tab === 'log'
                    ? 'Inspect the observations, declared capture conditions, and decisions.'
                    : titles[1]}
          </p>
        </div>
        {connectionError ? (
          <div className="connection-notice">
            <Notice error>{connectionError}</Notice>
            <button className="button secondary" onClick={() => void refresh()}>
              Retry connection
            </button>
          </div>
        ) : null}
        {actionError || state?.error ? (
          <Notice error>{actionError || state?.error}</Notice>
        ) : null}
        {viewState ? (
          tab === 'lab' ? (
            <VideoLab />
          ) : tab === 'workflow' ? (
            <WorkflowPanel
              workflow={viewState.workflow || null}
              pending={blocked}
              onAction={action}
            />
          ) : tab === 'task' ? (
            <TaskEditor
              key={
                viewState.active_task?.task_hash ||
                viewState.active_task?.protocol_hash ||
                'default'
              }
              state={viewState}
              pending={blocked}
              onAction={action}
            />
          ) : tab === 'log' ? (
            <EvidenceLog state={viewState} />
          ) : (
            <>
              {capture ? (
                <div className="capture-banner">
                  <Notice>Old evidence cleared. Two new views required.</Notice>
                </div>
              ) : null}
              <div
                className={`workbench ${capture ? 'capture-workbench' : ''}`}
              >
                {capture ? (
                  <CapturePanel
                    key={`${viewState.session_id}-${viewState.revision}`}
                    state={viewState}
                    pending={blocked}
                    onInspect={async (payload) => {
                      await action('/api/inspect', payload)
                    }}
                  />
                ) : (
                  <EvidencePanel state={viewState} />
                )}
                <Inspector
                  state={viewState}
                  pending={blocked}
                  capture={capture}
                  onRepair={() => void action('/api/repair')}
                />
              </div>
            </>
          )
        ) : (
          <div className="loading-panel" role="status">
            {!connectionError ? (
              <>
                <span className="spinner" />
                Loading inspection…
              </>
            ) : (
              'Waiting for the local server.'
            )}
          </div>
        )}
      </main>
      {viewState && tab === 'inspection' ? (
        <ProgressRail state={viewState} />
      ) : null}
    </div>
  )
}
