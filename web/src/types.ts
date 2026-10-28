export type Status = 'pass' | 'fail' | 'unknown'
export type Rect = [number, number, number, number]
export type Regions = { left: Rect; right: Rect }
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
  phase: 'mistake' | 'awaiting_capture' | 'checking' | 'verified' | 'unknown' | 'needs_evidence' | 'error'
  status: Status
  revision: number
  source_kind: 'recorded' | Source | 'none'
  images: EvidenceImage[]
  checks: PlacementCheck[]
  proposal: { title: string; steps: string[]; hypothetical: boolean } | null
  regions: Regions
  capture_token: string | null
  trace: { id: string; event: string; label: string; status: Status; revision: number }[]
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
