import type { WorkflowDTO } from './WorkflowPanel'
export type Status = 'pass' | 'fail' | 'unknown'
export type Rect = [number, number, number, number]
export type Regions = Record<string, Rect>
export type Source = 'camera' | 'upload'
export interface EvidenceImage {
  id: string
  url: string
  width: number
  height: number
  boxes: { object_id: string; x1: number; y1: number; x2: number; y2: number }[]
}
export interface PlacementCheck {
  check_id: string
  object_id: string
  status: Status
  expected_region: string
  observed_region: string | null
  reason: string
  confidence: number | null
}
export interface Snapshot {
  session_id: string
  phase:
    | 'mistake'
    | 'awaiting_capture'
    | 'checking'
    | 'verified'
    | 'unknown'
    | 'needs_evidence'
    | 'error'
  status: Status
  revision: number
  source_kind: 'recorded' | Source | 'none'
  images: EvidenceImage[]
  checks: PlacementCheck[]
  proposal: { title: string; steps: string[]; hypothetical: boolean } | null
  regions: Regions
  capture_token: string | null
  trace: {
    id: string
    event: string
    label: string
    status: Status
    revision: number
  }[]
  busy: boolean
  error: string | null
  api_ready: boolean
  source_note: string
  inference_origin: string | null
  verified_at_ms: number | null
  evidence_age_ms: number | null
  stale: boolean
  pending_token_issued_at_ms: number | null
  csrf_token: string
  active_task?: ActiveTask
  task_draft?: TaskDraft | null
  draft_error?: string | null
  busy_operation?: 'task_draft' | 'inspection' | null
  inspection_request?: {
    title: string
    instructions: string[]
    object_ids: string[]
    reason: string
    status?: AcquisitionPlan['status']
    attempts_used?: number
    attempts_remaining?: number
    max_attempts?: number
    requests?: AcquisitionPlan['requests']
    stop_reason?: string | null
  } | null
  acquisition?: AcquisitionPlan | null
  workflow?: WorkflowDTO | null
  recovery?: RecoveryReceipt | null
}
export interface RecoveryReceipt {
  receipt_id: string
  receipt_hash: string
  kind: 'repair' | 'inspection'
  outcome:
    | 'awaiting_fresh_evidence'
    | 'fresh_goal_verified_after_requested_repair'
    | 'fresh_goal_verified_after_inspection'
    | 'still_failed'
    | 'needs_evidence'
    | 'error'
    | 'cancelled_task_change'
  before_revision: number
  after_revision: number | null
  before_status: Status
  after_status: Status | null
  before_checks: {
    check_id: string
    object_id?: string
    status: Status
    reason?: string
  }[]
  after_checks: {
    check_id: string
    object_id?: string
    status: Status
    reason?: string
  }[]
  plan_steps: string[]
  task_hash: string | null
  protocol_hash: string
  verification_note: string
  physical_action_execution_verified: false
}
export interface TaskCheck {
  id: string
  kind: string
  object_id: string
  region_id: string
  unique_target?: boolean
}
export interface TaskProtocol {
  id: string
  version: string
  objects: string[]
  regions: string[]
  checks: TaskCheck[]
}
export interface ActiveTask {
  title: string
  objects: string[]
  regions?: string[]
  requirements: string
  protocol_hash: string
  task_hash?: string | null
  protocol: TaskProtocol
  concepts: Record<string, string>
  origin: string
  reviewed: boolean
  human_reviewed: boolean
  review_origin: string
  model?: string
  perception_mode?: 'concept' | 'guided'
  perception_note?: string
  perception_version?: string
}
export interface TaskDraft {
  draft_id: string
  draft_hash: string
  title: string
  requirements: string
  objects: string[]
  regions: string[]
  protocol: TaskProtocol
  protocol_hash: string
  concepts: Record<string, string>
  origin: string
  requires_human_approval: boolean
  approved: boolean
  model: string | null
  suggested_regions?: Regions
  calibration_note?: string
  claim_limit: string
  perception_mode?: 'concept' | 'guided'
  perception_note?: string
  perception_version?: string
}
export interface AcquisitionPlan {
  schema_version: number
  status:
    | 'ready'
    | 'not_needed'
    | 'budget_exhausted'
    | 'timeline_exhausted'
    | 'unsupported_contract'
  inspection_only: true
  physical_action_allowed: false
  attempts_used: number
  max_attempts: number
  attempts_remaining: number
  unknown_check_ids: string[]
  observed_fail_check_ids: string[]
  requests: {
    check_id: string
    object_id: string
    object_label: string
    region_id: string
    region_label: string
    gap_codes: string[]
    view_indices: number[]
    reason: string
    instructions: string[]
  }[]
  stop_reason: string | null
}
export interface LocalPhoto {
  data_url: string
  width: number
  height: number
  name: string
  captured_at_ms?: number
}
export interface InspectPayload {
  capture_token: string
  source: Source
  images: { data_url: string; captured_at_ms?: number }[]
  regions: Regions
  confirm_regions: true
  confirm_unique_targets: true
  confirm_fresh_capture: true
  confirm_pair_spacing: true
}
