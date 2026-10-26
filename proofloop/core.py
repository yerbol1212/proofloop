"""Declarative checks and a bounded, deterministic repair model.

This module does not call models or turn predicted actions into observations.
All timestamps are milliseconds on one source-media timeline, not wall time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import heapq
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Union


class ValidationError(ValueError):
    """An untrusted protocol, observation, or planning model is invalid."""


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0, maximum: float = math.inf) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValidationError(f"{label} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{label} must be a boolean")
    return value


def _fields(data: Mapping[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    if not isinstance(data, Mapping):
        raise ValidationError(f"{label} must be an object")
    extra, missing = set(data) - allowed, required - set(data)
    if extra or missing:
        raise ValidationError(f"{label}: unknown fields {sorted(extra)}; missing fields {sorted(missing)}")


def _ids(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValidationError(f"{label} must be a list")
    result = tuple(_id(value, label) for value in values)
    if len(set(result)) != len(result):
        raise ValidationError(f"{label} contains duplicate IDs")
    return result


@dataclass(frozen=True)
class Check:
    id: str
    kind: str
    object_id: str | None = None
    region_id: str | None = None
    expected: int | bool | None = None
    checkpoint_id: str | None = None
    event_id: str | None = None
    max_age_ms: int = 30_000
    min_confidence: float = 0.8
    min_stable_samples: int = 2
    unique_target: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Check:
        base = {"id", "kind"}
        if not isinstance(data, Mapping):
            raise ValidationError("check must be an object")
        kind = data.get("kind")
        if kind in ("count", "visible", "location"):
            required = base | {"object_id", "region_id"} | ({"unique_target"} if kind == "location" else {"expected"})
            _fields(data, required | {"max_age_ms", "min_confidence", "min_stable_samples"}, required, "check")
            expected = data.get("expected")
            if kind == "count":
                _integer(expected, "count expected")
            elif kind == "visible":
                _boolean(expected, "visible expected")
            elif _boolean(data["unique_target"], "unique_target") is not True:
                raise ValidationError("location requires an explicit approved unique_target=true contract")
            return cls(
                id=_id(data["id"], "check.id"), kind=kind,
                object_id=_id(data["object_id"], "object_id"),
                region_id=_id(data["region_id"], "region_id"), expected=expected,
                max_age_ms=_integer(data.get("max_age_ms", 30_000), "max_age_ms"),
                min_confidence=_number(data.get("min_confidence", 0.8), "min_confidence", 0, 1),
                min_stable_samples=_integer(data.get("min_stable_samples", 2), "min_stable_samples", 2),
                unique_target=kind == "location",
            )
        if kind == "before":
            required = base | {"checkpoint_id", "event_id"}
            _fields(data, required, required, "check")
            return cls(id=_id(data["id"], "check.id"), kind=kind,
                       checkpoint_id=_id(data["checkpoint_id"], "checkpoint_id"),
                       event_id=_id(data["event_id"], "event_id"))
        raise ValidationError(f"unsupported predicate kind: {kind!r}")

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "before":
            return {"id": self.id, "kind": self.kind, "checkpoint_id": self.checkpoint_id,
                    "event_id": self.event_id}
        result = {"id": self.id, "kind": self.kind, "object_id": self.object_id,
                "region_id": self.region_id,
                "max_age_ms": self.max_age_ms, "min_confidence": self.min_confidence,
                "min_stable_samples": self.min_stable_samples}
        result.update({"unique_target": self.unique_target} if self.kind == "location" else {"expected": self.expected})
        return result


@dataclass(frozen=True)
class Step:
    id: str
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Checkpoint:
    id: str
    check_ids: tuple[str, ...]


@dataclass(frozen=True)
class Protocol:
    id: str
    version: str
    objects: tuple[str, ...]
    regions: tuple[str, ...]
    checks: tuple[Check, ...]
    steps: tuple[Step, ...] = ()
    checkpoints: tuple[Checkpoint, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Protocol:
        required = {"id", "version", "objects", "regions", "checks"}
        _fields(data, required | {"steps", "checkpoints"}, required, "protocol")
        objects, regions = _ids(data["objects"], "objects"), _ids(data["regions"], "regions")
        if not objects or not regions or not isinstance(data["checks"], (list, tuple)) or not data["checks"]:
            raise ValidationError("protocol needs objects, regions, and at least one check")
        checks = tuple(Check.from_dict(check) for check in data["checks"])
        _ids([check.id for check in checks], "checks")
        for check in checks:
            if check.kind != "before" and (check.object_id not in objects or check.region_id not in regions):
                raise ValidationError(f"check {check.id}: unknown object or region")
        raw_steps, raw_checkpoints = data.get("steps", []), data.get("checkpoints", [])
        if not isinstance(raw_steps, (list, tuple)) or not isinstance(raw_checkpoints, (list, tuple)):
            raise ValidationError("steps and checkpoints must be lists")
        steps: list[Step] = []
        for item in raw_steps:
            _fields(item, {"id", "requires"}, {"id"}, "step")
            steps.append(Step(_id(item["id"], "step.id"), _ids(item.get("requires", []), "requires")))
        step_ids = set(_ids([step.id for step in steps], "steps"))
        if any(not set(step.requires) <= step_ids for step in steps):
            raise ValidationError("step refers to an unknown predecessor")
        completed: set[str] = set()
        while len(completed) < len(steps):
            ready = {step.id for step in steps if set(step.requires) <= completed} - completed
            if not ready:
                raise ValidationError("step prerequisite graph contains a cycle")
            completed |= ready
        checkpoints: list[Checkpoint] = []
        spatial_check_ids = {check.id for check in checks if check.kind != "before"}
        for item in raw_checkpoints:
            _fields(item, {"id", "check_ids"}, {"id", "check_ids"}, "checkpoint")
            check_ids = _ids(item["check_ids"], "check_ids")
            if not check_ids or not set(check_ids) <= spatial_check_ids:
                raise ValidationError("checkpoint needs existing spatial checks")
            checkpoints.append(Checkpoint(_id(item["id"], "checkpoint.id"), check_ids))
        checkpoint_ids = set(_ids([checkpoint.id for checkpoint in checkpoints], "checkpoints"))
        if any(check.kind == "before" and check.checkpoint_id not in checkpoint_ids for check in checks):
            raise ValidationError("before predicate refers to an unknown checkpoint")
        return cls(_id(data["id"], "protocol.id"), _id(data["version"], "protocol.version"),
                   objects, regions, checks, tuple(steps), tuple(checkpoints))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "objects": list(self.objects),
                "regions": list(self.regions), "checks": [check.to_dict() for check in self.checks],
                "steps": [{"id": step.id, "requires": list(step.requires)} for step in self.steps],
                "checkpoints": [{"id": point.id, "check_ids": list(point.check_ids)} for point in self.checkpoints]}

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def allowed_next_steps(self, completed: Iterable[str]) -> tuple[str, ...]:
        done = set(completed)
        if not done <= {step.id for step in self.steps}:
            raise ValidationError("completed steps contain an unknown ID")
        return tuple(step.id for step in self.steps if step.id not in done and set(step.requires) <= done)


@dataclass(frozen=True)
class Observation:
    """One source-supported fact in a specification's current evidence epoch.

    For location, ``region_id`` identifies the required region of the check,
    while ``value`` names the positively observed region of its declared unique
    target. A missing target is None, never an inferred location or absence.
    """

    session_id: str
    spec_hash: str
    attempt_id: int
    revision: int
    kind: str
    object_id: str
    region_id: str
    value: int | bool | str | None
    evidence_id: str
    timestamp_ms: int
    confidence: float = 1.0
    fully_visible: bool = False
    stable_samples: int = 1
    absence_validated: bool = False

    def __post_init__(self) -> None:
        for key in ("session_id", "spec_hash", "object_id", "region_id", "evidence_id"):
            _id(getattr(self, key), key)
        for key in ("attempt_id", "revision", "timestamp_ms"):
            _integer(getattr(self, key), key)
        _integer(self.stable_samples, "stable_samples", 1)
        _number(self.confidence, "confidence", 0, 1)
        _boolean(self.fully_visible, "fully_visible")
        _boolean(self.absence_validated, "absence_validated")
        if self.kind not in ("count", "visible", "location"):
            raise ValidationError("observation has an unsupported kind")
        if self.value is not None:
            if self.kind == "count":
                _integer(self.value, "count value")
            elif self.kind == "visible":
                _boolean(self.value, "visible value")
            else:
                _id(self.value, "location value")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Observation:
        required = {"session_id", "spec_hash", "attempt_id", "revision", "kind", "object_id",
                    "region_id", "value", "evidence_id", "timestamp_ms"}
        _fields(data, required | {"confidence", "fully_visible", "stable_samples", "absence_validated"},
                required, "observation")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: Status
    reason: str
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "status": self.status.value,
                "reason": self.reason, "evidence_ids": list(self.evidence_ids)}


def aggregate_status(results: Iterable[CheckResult]) -> Status:
    statuses = [result.status for result in results]
    if Status.FAIL in statuses:
        return Status.FAIL
    if statuses and all(status == Status.PASS for status in statuses):
        return Status.PASS
    return Status.UNKNOWN


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    attempt_id: int
    revision: int
    timestamp_ms: int
    status: Status
    results: tuple[CheckResult, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    attempt_id: int
    revision: int
    timestamp_ms: int
    evidence_id: str
    history_complete: bool = False


@dataclass
class EvidenceStore:
    protocol: Protocol
    session_id: str
    attempt_id: int = field(default=0, init=False)
    revision: int = field(default=0, init=False)
    current_time_ms: int = field(default=0, init=False)
    observations: list[Observation] = field(default_factory=list, init=False)
    checkpoint_records: list[CheckpointRecord] = field(default_factory=list, init=False)
    events: list[EventRecord] = field(default_factory=list, init=False)
    invalidations: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _id(self.session_id, "session_id")

    def advance_time(self, timestamp_ms: int) -> None:
        _integer(timestamp_ms, "timestamp_ms")
        if timestamp_ms < self.current_time_ms:
            raise ValidationError("source time cannot move backwards; create another session for another clip")
        self.current_time_ms = timestamp_ms

    def add(self, observation: Observation) -> None:
        if observation.session_id != self.session_id or observation.spec_hash != self.protocol.content_hash:
            raise ValidationError("observation belongs to a different session or specification")
        if observation.attempt_id != self.attempt_id or observation.revision != self.revision:
            raise ValidationError("observation belongs to a stale attempt or invalidated revision")
        if observation.object_id not in self.protocol.objects or observation.region_id not in self.protocol.regions:
            raise ValidationError("observation refers to an unknown object or region")
        if observation.kind == "location" and observation.value is not None and observation.value not in self.protocol.regions:
            raise ValidationError("location observation refers to an unknown observed region")
        self.advance_time(observation.timestamp_ms)
        self.observations.append(observation)

    def observe(self, *, kind: str, object_id: str, region_id: str, value: int | bool | str | None,
                evidence_id: str, timestamp_ms: int, confidence: float = 1.0,
                fully_visible: bool = False, stable_samples: int = 1,
                absence_validated: bool = False) -> Observation:
        observation = Observation(self.session_id, self.protocol.content_hash, self.attempt_id,
                                  self.revision, kind, object_id, region_id, value, evidence_id,
                                  timestamp_ms, confidence, fully_visible, stable_samples, absence_validated)
        self.add(observation)
        return observation

    def invalidate(self, reason: str, timestamp_ms: int) -> None:
        """Invalidate all current facts/checkpoints after motion, mutation, or occlusion."""
        _id(reason, "reason")
        self.advance_time(timestamp_ms)
        self.revision += 1
        self.invalidations.append({"attempt_id": self.attempt_id, "revision": self.revision,
                                   "timestamp_ms": timestamp_ms, "reason": reason})

    def start_attempt(self, timestamp_ms: int) -> int:
        self.advance_time(timestamp_ms)
        self.attempt_id += 1
        self.revision = 0
        return self.attempt_id

    def record_checkpoint(self, checkpoint_id: str, *, timestamp_ms: int,
                          evidence_id: str | None = None) -> CheckpointRecord:
        checkpoint = next((point for point in self.protocol.checkpoints if point.id == checkpoint_id), None)
        if checkpoint is None:
            raise ValidationError("unknown checkpoint")
        if evidence_id is not None:
            _id(evidence_id, "evidence_id")
        self.advance_time(timestamp_ms)
        results = tuple(result for result in evaluate(self.protocol, self, now_ms=timestamp_ms)
                        if result.check_id in checkpoint.check_ids)
        evidence_ids = tuple(dict.fromkeys(evidence for result in results for evidence in result.evidence_ids))
        if evidence_id is not None:
            evidence_ids += (_id(evidence_id, "evidence_id"),)
        record = CheckpointRecord(checkpoint_id, self.attempt_id, self.revision, timestamp_ms,
                                  aggregate_status(results), results, evidence_ids)
        self.checkpoint_records.append(record)
        return record

    def record_event(self, event_id: str, *, timestamp_ms: int, evidence_id: str,
                     history_complete: bool = False) -> EventRecord:
        if event_id not in {check.event_id for check in self.protocol.checks if check.kind == "before"}:
            raise ValidationError("event is not declared by a before predicate")
        _id(evidence_id, "evidence_id")
        _boolean(history_complete, "history_complete")
        self.advance_time(timestamp_ms)
        event = EventRecord(event_id, self.attempt_id, self.revision, timestamp_ms,
                            evidence_id, history_complete)
        self.events.append(event)
        return event


def _spatial_result(check: Check, store: EvidenceStore, now_ms: int) -> CheckResult:
    matching = [observation for observation in store.observations
                if observation.kind == check.kind and observation.object_id == check.object_id
                and observation.region_id == check.region_id and observation.timestamp_ms <= now_ms
                and observation.attempt_id == store.attempt_id and observation.revision == store.revision]
    if not matching:
        return CheckResult(check.id, Status.UNKNOWN, "No current evidence for this attempt and revision")
    observation = matching[-1]
    evidence = (observation.evidence_id,)
    if now_ms - observation.timestamp_ms > check.max_age_ms:
        return CheckResult(check.id, Status.UNKNOWN, "Evidence is stale", evidence)
    if observation.value is None:
        return CheckResult(check.id, Status.UNKNOWN, "Observation is unavailable or ambiguous", evidence)
    if observation.confidence < check.min_confidence:
        return CheckResult(check.id, Status.UNKNOWN, "Observation confidence is below the approved threshold", evidence)
    if observation.stable_samples < check.min_stable_samples:
        return CheckResult(check.id, Status.UNKNOWN, "Insufficient repeated stable observations", evidence)
    if check.kind == "location":
        if check.unique_target is not True:
            return CheckResult(check.id, Status.UNKNOWN, "Location requires an approved unique-target scene contract", evidence)
        status = Status.PASS if observation.value == check.region_id else Status.FAIL
        reason = ("Declared unique target is positively located in its required region" if status == Status.PASS
                  else "Declared unique target is positively located in another approved region")
        return CheckResult(check.id, status, reason, evidence)
    complete = observation.fully_visible and observation.absence_validated
    if check.kind == "visible":
        if observation.value is False and not complete:
            return CheckResult(check.id, Status.UNKNOWN, "No detection does not prove absence", evidence)
    elif not complete:
        # A supported lower bound can prove there are too many objects; it cannot
        # prove an exact count, or the absence of a required object.
        if observation.value > check.expected:
            return CheckResult(check.id, Status.FAIL, "Observed count already exceeds the allowed count", evidence)
        return CheckResult(check.id, Status.UNKNOWN, "Exact counts require a fully supported visible region", evidence)
    status = Status.PASS if observation.value == check.expected else Status.FAIL
    reason = "Supported observation satisfies the predicate" if status == Status.PASS else "Supported observation contradicts the predicate"
    return CheckResult(check.id, status, reason, evidence)


def _before_result(check: Check, store: EvidenceStore, now_ms: int) -> CheckResult:
    events = [event for event in store.events if event.event_id == check.event_id
              and event.attempt_id == store.attempt_id and event.revision == store.revision
              and event.timestamp_ms <= now_ms]
    if not events:
        return CheckResult(check.id, Status.UNKNOWN, "No current observed completion event")
    event = events[-1]
    checkpoints = [point for point in store.checkpoint_records if point.checkpoint_id == check.checkpoint_id
                   and point.attempt_id == event.attempt_id and point.revision == event.revision
                   and point.timestamp_ms < event.timestamp_ms]
    if not checkpoints:
        status = Status.FAIL if event.history_complete else Status.UNKNOWN
        reason = ("Complete observed history proves the required earlier checkpoint did not occur"
                  if event.history_complete else "Missing earlier checkpoint evidence cannot prove an ordering violation")
        return CheckResult(check.id, status, reason, (event.evidence_id,))
    checkpoint = checkpoints[-1]
    # Never fall back to an older passing checkpoint after a failed/unknown recheck.
    if checkpoint.status == Status.PASS:
        spatial_checks = {item.id: item for item in store.protocol.checks if item.kind != "before"}
        for result in checkpoint.results:
            predicate = spatial_checks[result.check_id]
            supporting = [observation for observation in store.observations
                          if observation.evidence_id in result.evidence_ids
                          and observation.kind == predicate.kind and observation.object_id == predicate.object_id
                          and observation.region_id == predicate.region_id
                          and observation.attempt_id == checkpoint.attempt_id and observation.revision == checkpoint.revision
                          and observation.timestamp_ms <= checkpoint.timestamp_ms]
            if not supporting or event.timestamp_ms - supporting[-1].timestamp_ms > predicate.max_age_ms:
                return CheckResult(check.id, Status.UNKNOWN, "Checkpoint evidence was stale at the completion event",
                                   checkpoint.evidence_ids + (event.evidence_id,))
    return CheckResult(check.id, checkpoint.status,
                       "Latest earlier checkpoint was " + checkpoint.status.value,
                       checkpoint.evidence_ids + (event.evidence_id,))


def evaluate(protocol: Protocol, store: EvidenceStore, *, now_ms: int | None = None) -> tuple[CheckResult, ...]:
    if protocol.content_hash != store.protocol.content_hash:
        raise ValidationError("cannot evaluate evidence against another specification version")
    at = store.current_time_ms if now_ms is None else _integer(now_ms, "now_ms")
    if at < store.current_time_ms:
        raise ValidationError("historical evaluation requires a recorded result, not the current mutable store")
    return tuple(_before_result(check, store, at) if check.kind == "before"
                 else _spatial_result(check, store, at) for check in protocol.checks)


Scalar = Union[str, int, bool]


def _facts(data: Any, label: str) -> dict[str, Scalar]:
    if not isinstance(data, Mapping):
        raise ValidationError(f"{label} must be a fact object")
    result = {}
    for key, value in data.items():
        _id(key, label + " key")
        if type(value) not in (str, int, bool):
            raise ValidationError(f"{label} values must be strings, integers, or booleans")
        result[key] = value
    return result


@dataclass(frozen=True)
class RepairAction:
    id: str
    kind: str
    preconditions: Mapping[str, Scalar]
    effects: Mapping[str, Scalar]
    cost: float = 1.0
    reveals: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepairAction:
        required = {"id", "kind"}
        _fields(data, required | {"preconditions", "effects", "cost", "reveals", "invalidates"}, required, "action")
        kind = data["kind"]
        effects = _facts(data.get("effects", {}), "effects")
        reveals = _ids(data.get("reveals", []), "reveals")
        invalidates = _ids(data.get("invalidates", []), "invalidates")
        if kind == "physical":
            if not effects or reveals:
                raise ValidationError("physical actions need effects and cannot reveal observations")
        elif kind == "inspect":
            if effects or invalidates or not reveals:
                raise ValidationError("inspection must reveal facts without changing world state")
        else:
            raise ValidationError("action kind must be physical or inspect")
        return cls(_id(data["id"], "action.id"), kind,
                   _facts(data.get("preconditions", {}), "preconditions"), effects,
                   _number(data.get("cost", 1), "cost"), reveals, invalidates)


@dataclass(frozen=True)
class RepairPlan:
    action_ids: tuple[str, ...]
    total_cost: float
    states_expanded: int
    final_world: Mapping[str, Scalar]
    final_known: Mapping[str, Scalar]

    def to_dict(self) -> dict[str, Any]:
        return {"action_ids": list(self.action_ids), "total_cost": self.total_cost,
                "states_expanded": self.states_expanded, "predicted_final_world": dict(self.final_world),
                "predicted_final_known": dict(self.final_known), "hypothetical": True,
                "execution_requires_fresh_observation": True,
                "optimality_scope": "minimum cost within the supplied finite deterministic action model"}


def _same(left: Scalar, right: Scalar) -> bool:
    # Python considers False == 0; declarative fact types must remain distinct.
    return type(left) is type(right) and left == right


def _satisfies(facts: Mapping[str, Scalar], required: Mapping[str, Scalar]) -> bool:
    return all(key in facts and _same(facts[key], value) for key, value in required.items())


def _state_key(world: Mapping[str, Scalar], known: Mapping[str, Scalar]) -> str:
    return json.dumps([world, known], sort_keys=True, separators=(",", ":"))


def plan_repair(initial_world: Mapping[str, Scalar], known: Mapping[str, Scalar],
                actions: Iterable[RepairAction], goal: Mapping[str, Scalar], *,
                max_states: int = 4096) -> RepairPlan | None:
    """Dijkstra search over physical state *and* knowledge in a tiny authored model.

    ``initial_world`` is a hypothetical deterministic state, not permission to
    act on unobserved facts. Preconditions and the goal require known facts.
    Physical effects invalidate knowledge; an inspection predicts a subsequent
    observation but never writes to an EvidenceStore. Runtime must perform and
    verify each action, stopping/replanning if reality differs from this model.
    """
    world, knowledge, target = _facts(initial_world, "initial_world"), _facts(known, "known"), _facts(goal, "goal")
    _integer(max_states, "max_states", 1)
    if not world or not target or not set(target) <= set(world):
        raise ValidationError("planner needs a nonempty world and a goal on existing facts")
    if not _satisfies(world, knowledge):
        raise ValidationError("known facts must be a consistent subset of the hypothetical world")
    moves = tuple(actions)
    _ids([action.id for action in moves], "actions")
    for action in moves:
        # Revalidate even direct dataclass construction before using model effects.
        RepairAction.from_dict({"id": action.id, "kind": action.kind, "preconditions": action.preconditions,
                                "effects": action.effects, "cost": action.cost,
                                "reveals": list(action.reveals), "invalidates": list(action.invalidates)})
        referenced = set(action.preconditions) | set(action.effects) | set(action.reveals) | set(action.invalidates)
        if not referenced <= set(world):
            raise ValidationError(f"action {action.id} refers to an unknown world fact")
    counter = itertools.count()
    queue: list[tuple[float, int, dict[str, Scalar], dict[str, Scalar], tuple[str, ...]]] = []
    heapq.heappush(queue, (0.0, next(counter), world, knowledge, ()))
    best = {_state_key(world, knowledge): 0.0}
    expanded = 0
    while queue:
        cost, _, current, informed, path = heapq.heappop(queue)
        if cost != best[_state_key(current, informed)]:
            continue
        if expanded >= max_states:
            raise ValidationError("repair state-space budget exceeded; no optimality result is available")
        expanded += 1
        if _satisfies(informed, target):
            return RepairPlan(path, cost, expanded, current, informed)
        for action in moves:
            if not _satisfies(informed, action.preconditions):
                continue
            next_world, next_known = dict(current), dict(informed)
            if action.kind == "physical":
                next_world.update(action.effects)
                for key in set(action.effects) | set(action.invalidates):
                    next_known.pop(key, None)
            else:
                for key in action.reveals:
                    next_known[key] = next_world[key]
            key, next_cost = _state_key(next_world, next_known), cost + action.cost
            if next_cost < best.get(key, math.inf):
                best[key] = next_cost
                heapq.heappush(queue, (next_cost, next(counter), next_world, next_known, path + (action.id,)))
    return None
