"""Owned symbolic fixtures for a checker diagnostic, never camera evidence.

These fixtures supply accurate facts by construction.  They isolate the state
checker from perception and must not be reported as a Meta-model benchmark.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


EVIDENCE_ORIGIN = "synthetic_oracle"


@dataclass(frozen=True)
class FixtureEvent:
    """A runtime input or an assertion boundary in a symbolic episode."""

    kind: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    description: str
    protocol: dict[str, Any]
    events: tuple[FixtureEvent, ...]
    expected_status: str
    family: str


def protocol_fixture(*, temporal: bool = False, screwdriver_region: str = "left") -> dict[str, Any]:
    """Return a newly allocated, human-authored specification fixture."""

    protocol: dict[str, Any] = {
        "id": "synthetic-repair-kit",
        "version": "1.1" if screwdriver_region != "left" else "1.0",
        "objects": ["divider", "screwdriver", "pliers"],
        "regions": ["tray", "left", "right"],
        "checks": [
            {
                "id": "divider-visible",
                "kind": "visible",
                "object_id": "divider",
                "region_id": "tray",
                "expected": True,
                "max_age_ms": 30000,
                "min_confidence": 0.8,
                "min_stable_samples": 2,
            },
            {
                "id": "screwdriver-location",
                "kind": "count",
                "object_id": "screwdriver",
                "region_id": screwdriver_region,
                "expected": 1,
                "max_age_ms": 30000,
                "min_confidence": 0.8,
                "min_stable_samples": 2,
            },
            {
                "id": "pliers-location",
                "kind": "count",
                "object_id": "pliers",
                "region_id": "right",
                "expected": 1,
                "max_age_ms": 30000,
                "min_confidence": 0.8,
                "min_stable_samples": 2,
            },
        ],
        "steps": [
            {"id": "place-divider", "requires": []},
            {"id": "place-screwdriver", "requires": ["place-divider"]},
            {"id": "place-pliers", "requires": ["place-divider"]},
        ],
        "checkpoints": [
            {
                "id": "open-check",
                "check_ids": ["divider-visible", "screwdriver-location", "pliers-location"],
            }
        ],
    }
    if temporal:
        protocol["checks"].append(
            {
                "id": "inspect-before-completion",
                "kind": "before",
                "checkpoint_id": "open-check",
                "event_id": "completion-declared",
            }
        )
    return protocol


def _fact(case_id: str, object_id: str, region_id: str, value: int | bool,
          *, at: int = 1000, **overrides: Any) -> FixtureEvent:
    arguments: dict[str, Any] = {
        "kind": "visible" if object_id == "divider" else "count",
        "object_id": object_id,
        "region_id": region_id,
        "value": value,
        "evidence_id": f"fixture://proofloop/{case_id}/{object_id}-{region_id}-{at}",
        "timestamp_ms": at,
        "confidence": 1.0,
        "fully_visible": True,
        "stable_samples": 2,
        "absence_validated": True,
    }
    arguments.update(overrides)
    return FixtureEvent("observe", arguments)


def _state(case_id: str, *, screwdriver_count: int = 1, at: int = 1000,
           reverse: bool = False, screwdriver_region: str = "left") -> tuple[FixtureEvent, ...]:
    contents = [
        _fact(case_id, "screwdriver", screwdriver_region, screwdriver_count, at=at + 10),
        _fact(case_id, "pliers", "right", 1, at=at + 20),
    ]
    if reverse:
        contents.reverse()
        # Reverse the observation order without assigning decreasing timestamps.
        contents = [FixtureEvent(event.kind, dict(event.arguments, timestamp_ms=at + 10 * (i + 1)))
                    for i, event in enumerate(contents)]
    return (_fact(case_id, "divider", "tray", True, at=at), *contents)


def _expect(status: str, label: str, *, at: int = 1100) -> FixtureEvent:
    return FixtureEvent("expect", {"status": status, "label": label, "now_ms": at})


def diagnostic_cases() -> tuple[DiagnosticCase, ...]:
    """Independent synthetic episodes; the labels never enter the checker."""

    base = protocol_fixture()
    cases: list[DiagnosticCase] = []

    def add(case_id: str, description: str, events: tuple[FixtureEvent, ...],
            expected: str, family: str, *, protocol: dict[str, Any] | None = None) -> None:
        cases.append(DiagnosticCase(case_id, description, deepcopy(protocol or base),
                                    events, expected, family))

    name = "correct"
    add(name, "A supported complete state has current, stable observations.",
        _state(name), "pass", "valid")

    name = "alternate-order"
    add(name, "The two independent contents facts arrive in the alternate order.",
        _state(name, reverse=True), "pass", "valid")

    name = "wrong-region"
    add(name, "A supported empty left region and a screwdriver in the right region contradict the specification.",
        (*_state(name, screwdriver_count=0), _fact(name, "screwdriver", "right", 1, at=1030)),
        "fail", "observable_deviation")

    name = "missing-visible-item"
    add(name, "Validated stable absence in a supported region is a contradiction.",
        _state(name, screwdriver_count=0), "fail", "observable_deviation")

    name = "missing-observation"
    add(name, "No screwdriver observation is supplied; absence cannot be inferred.",
        (_fact(name, "divider", "tray", True), _fact(name, "pliers", "right", 1, at=1020)),
        "unknown", "insufficient_evidence")

    name = "occluded-after-pass"
    add(name, "A hand-like occlusion invalidates previously supported contents facts.",
        (*_state(name), _expect("pass", "before-occlusion"),
         FixtureEvent("invalidate", {"reason": "occluded", "timestamp_ms": 1200})),
        "unknown", "insufficient_evidence")

    name = "low-confidence"
    add(name, "A low confidence screwdriver count cannot establish completion.",
        (*_state(name), _fact(name, "screwdriver", "left", 1, at=1050, confidence=0.4)),
        "unknown", "insufficient_evidence")

    name = "unsupported-negative"
    add(name, "Zero matches with an incomplete view are insufficient evidence of absence.",
        (*_state(name), _fact(name, "screwdriver", "left", 0, at=1050,
                            fully_visible=False, absence_validated=False)),
        "unknown", "insufficient_evidence")

    name = "unstable-negative"
    add(name, "An unsupported single empty detection must not become a deviation claim.",
        (*_state(name), _fact(name, "screwdriver", "left", 0, at=1050, stable_samples=1)),
        "unknown", "insufficient_evidence")

    name = "expired-observations"
    add(name, "Facts exceeding the specification's age limit cannot certify the current state.",
        (*_state(name), _expect("unknown", "expired", at=40000)),
        "unknown", "insufficient_evidence")

    name = "new-attempt-stale-facts"
    add(name, "Starting a new attempt cannot reuse the previous attempt's contents facts.",
        (*_state(name), _expect("pass", "prior-attempt"),
         FixtureEvent("start_attempt", {"timestamp_ms": 1200})),
        "unknown", "freshness")

    name = "repair-fresh-evidence"
    add(name, "A repair needs a new attempt and current observations before it can pass.",
        (*_state(name, screwdriver_count=0), _expect("fail", "original-deviation"),
         FixtureEvent("start_attempt", {"timestamp_ms": 1200}),
         _expect("unknown", "after-action-before-reobservation", at=1250),
         *_state(name, at=1300), _expect("pass", "fresh-repair-observation", at=1400)),
        "pass", "recovery")

    name = "repair-partial-refresh"
    add(name, "Refreshing other objects cannot refresh a stale screwdriver fact.",
        (*_state(name), _expect("pass", "original-state"),
         FixtureEvent("start_attempt", {"timestamp_ms": 1200}),
         _fact(name, "divider", "tray", True, at=1300),
         _fact(name, "pliers", "right", 1, at=1320)),
        "unknown", "freshness")

    temporal = protocol_fixture(temporal=True)
    name = "temporal-correct"
    add(name, "A confirmed checkpoint is observed before a non-mutating completion declaration.",
        (*_state(name), FixtureEvent("record_checkpoint", {"checkpoint_id": "open-check", "timestamp_ms": 1100}),
         FixtureEvent("record_event", {"event_id": "completion-declared", "timestamp_ms": 1200})),
        "pass", "temporal", protocol=temporal)

    name = "temporal-missing-checkpoint"
    add(name, "A completion declaration without the required earlier view is insufficient evidence.",
        (*_state(name), FixtureEvent("record_event", {"event_id": "completion-declared", "timestamp_ms": 1200})),
        "unknown", "temporal", protocol=temporal)

    name = "temporal-wrong-order"
    add(name, "An explicitly observed completion declaration followed by inspection violates the authored order.",
        (*_state(name), FixtureEvent("record_event", {"event_id": "completion-declared", "timestamp_ms": 1100,
                                                   "history_complete": True}),
         FixtureEvent("record_checkpoint", {"checkpoint_id": "open-check", "timestamp_ms": 1200})),
        "fail", "temporal", protocol=temporal)

    name = "edited-requirement"
    add(name, "A data-only confirmed specification edit moves the required compartment to the right.",
        _state(name, screwdriver_region="right"), "pass", "editable_specification",
        protocol=protocol_fixture(screwdriver_region="right"))
    return tuple(cases)
