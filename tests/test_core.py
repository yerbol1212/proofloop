"""Behavioral safety and finite-model search tests; no models or network calls."""

import itertools
import unittest

from proofloop.core import (
    CheckResult, EvidenceStore, Observation, Protocol, RepairAction, Status,
    ValidationError, aggregate_status, evaluate, plan_repair,
)


def protocol_data(temporal=False):
    data = {
        "id": "repair-kit", "version": "1", "objects": ["divider", "driver", "pliers"],
        "regions": ["tray", "left", "right"],
        "checks": [
            {"id": "divider", "kind": "visible", "object_id": "divider", "region_id": "tray", "expected": True},
            {"id": "driver-left", "kind": "count", "object_id": "driver", "region_id": "left", "expected": 1},
            {"id": "pliers-right", "kind": "count", "object_id": "pliers", "region_id": "right", "expected": 1},
        ],
        "steps": [
            {"id": "divider", "requires": []}, {"id": "driver", "requires": ["divider"]},
            {"id": "pliers", "requires": ["divider"]}, {"id": "inspect", "requires": ["driver", "pliers"]},
        ],
        "checkpoints": [{"id": "ready", "check_ids": ["divider", "driver-left", "pliers-right"]}],
    }
    if temporal:
        data["checks"].append({"id": "ready-before-declare", "kind": "before",
                               "checkpoint_id": "ready", "event_id": "declare"})
    return data


def observe(store, object_id="driver", region_id="left", value=1, timestamp_ms=100, **options):
    defaults = {"kind": "count", "evidence_id": f"fixture://{object_id}/{timestamp_ms}",
                "fully_visible": True, "stable_samples": 2, "absence_validated": True}
    defaults.update(options)
    return store.observe(object_id=object_id, region_id=region_id, value=value,
                         timestamp_ms=timestamp_ms, **defaults)


def populate(store, timestamp_ms=100):
    observe(store, "divider", "tray", True, timestamp_ms, kind="visible")
    observe(store, "driver", "left", 1, timestamp_ms)
    observe(store, "pliers", "right", 1, timestamp_ms)


def by_id(store, check_id="driver-left", **options):
    return next(result for result in evaluate(store.protocol, store, **options) if result.check_id == check_id)


class ProtocolTests(unittest.TestCase):
    def test_independent_insertions_commute_but_prerequisite_is_preserved(self):
        protocol = Protocol.from_dict(protocol_data())
        self.assertEqual(protocol.allowed_next_steps([]), ("divider",))
        self.assertEqual(set(protocol.allowed_next_steps(["divider"])), {"driver", "pliers"})
        for order in (("driver", "pliers"), ("pliers", "driver")):
            completed = ["divider"]
            for step in order:
                self.assertIn(step, protocol.allowed_next_steps(completed))
                completed.append(step)
            self.assertEqual(protocol.allowed_next_steps(completed), ("inspect",))

    def test_approved_requirement_edit_changes_hash_and_executes_without_code_change(self):
        first = Protocol.from_dict(protocol_data())
        edited = protocol_data()
        edited["version"] = "2"
        edited["checks"][1]["region_id"] = "right"
        second = Protocol.from_dict(edited)
        self.assertNotEqual(first.content_hash, second.content_hash)
        store = EvidenceStore(second, "edited")
        observe(store, region_id="right")
        self.assertEqual(by_id(store).status, Status.PASS)
        with self.assertRaises(ValidationError):
            evaluate(first, store)
        self.assertEqual(first.content_hash, Protocol.from_dict(first.to_dict()).content_hash)

    def test_bad_references_cycles_counts_and_unsupported_code_are_rejected(self):
        examples = []
        wrong_object = protocol_data(); wrong_object["checks"][0]["object_id"] = "unknown"; examples.append(wrong_object)
        cycle = protocol_data(); cycle["steps"][0]["requires"] = ["inspect"]; examples.append(cycle)
        count = protocol_data(); count["checks"][1]["expected"] = -1; examples.append(count)
        boolean_count = protocol_data(); boolean_count["checks"][1]["expected"] = True; examples.append(boolean_count)
        arbitrary = protocol_data(); arbitrary["checks"][1]["kind"] = "python"; examples.append(arbitrary)
        missing = protocol_data(True); missing["checkpoints"] = []; examples.append(missing)
        recursive = protocol_data(True); recursive["checkpoints"][0]["check_ids"] = ["ready-before-declare"]; examples.append(recursive)
        malformed_list = protocol_data(); malformed_list["steps"] = None; examples.append(malformed_list)
        for invalid in examples:
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                Protocol.from_dict(invalid)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = Protocol.from_dict(protocol_data())
        self.store = EvidenceStore(self.protocol, "run-a")

    def test_empty_or_missing_evidence_is_unknown(self):
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.UNKNOWN)
        observe(self.store)
        self.assertEqual(by_id(self.store).status, Status.PASS)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.UNKNOWN)

    def test_supported_contradiction_dominates_pass_and_unknown(self):
        populate(self.store)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.PASS)
        observe(self.store, value=0, timestamp_ms=200)
        self.assertEqual(by_id(self.store).status, Status.FAIL)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.FAIL)
        self.assertEqual(by_id(self.store).evidence_ids, ("fixture://driver/200",))

    def test_no_detection_without_complete_supported_view_never_proves_absence(self):
        cases = [{"fully_visible": False}, {"stable_samples": 1}, {"absence_validated": False},
                 {"confidence": 0.2}, {"value": None}]
        for number, options in enumerate(cases):
            with self.subTest(options=options):
                value = options.pop("value", 0)
                observe(self.store, value=value, timestamp_ms=number + 1, **options)
                self.assertEqual(by_id(self.store).status, Status.UNKNOWN)

    def test_visible_false_does_not_pass_absence_predicate_without_support(self):
        data = protocol_data(); data["checks"][0]["expected"] = False
        store = EvidenceStore(Protocol.from_dict(data), "absence")
        observe(store, "divider", "tray", False, kind="visible", absence_validated=False)
        self.assertEqual(by_id(store, "divider").status, Status.UNKNOWN)
        observe(store, "divider", "tray", False, 200, kind="visible")
        self.assertEqual(by_id(store, "divider").status, Status.PASS)

    def test_positive_detection_can_prove_presence_without_proving_absence(self):
        observe(self.store, "divider", "tray", True, kind="visible", fully_visible=False, absence_validated=False)
        self.assertEqual(by_id(self.store, "divider").status, Status.PASS)
        observe(self.store, value=2, fully_visible=False, absence_validated=False)
        self.assertEqual(by_id(self.store).status, Status.FAIL)

    def test_new_ambiguous_observation_does_not_reuse_earlier_pass(self):
        observe(self.store)
        observe(self.store, value=None, timestamp_ms=200)
        self.assertEqual(by_id(self.store).status, Status.UNKNOWN)

    def test_source_age_occlusion_and_new_attempt_invalidate_contents(self):
        populate(self.store)
        self.assertEqual(by_id(self.store, now_ms=30_101).status, Status.UNKNOWN)
        self.store.invalidate(reason="hand could conceal removal", timestamp_ms=200)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.UNKNOWN)
        populate(self.store, 300)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.PASS)
        self.store.start_attempt(timestamp_ms=400)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.UNKNOWN)
        self.assertEqual(len(self.store.observations), 6)  # Trace is preserved.

    def test_old_revision_attempt_session_or_spec_cannot_be_injected(self):
        observation = observe(self.store)
        for key, value in (("session_id", "run-b"), ("spec_hash", "wrong"), ("attempt_id", 1), ("revision", 1)):
            raw = observation.to_dict(); raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                self.store.add(Observation.from_dict(raw))
        self.store.invalidate("movement", 200)
        with self.assertRaises(ValidationError):
            self.store.add(observation)

    def test_backdated_observations_and_replay_against_mutable_future_are_rejected(self):
        observe(self.store, timestamp_ms=200)
        with self.assertRaises(ValidationError):
            observe(self.store, timestamp_ms=100)
        with self.assertRaises(ValidationError):
            evaluate(self.protocol, self.store, now_ms=100)

    def test_invalid_invalidation_does_not_mutate_current_revision(self):
        populate(self.store)
        with self.assertRaises(ValidationError):
            self.store.invalidate("", 200)
        self.assertEqual(self.store.current_time_ms, 100)
        self.assertEqual(self.store.revision, 0)
        self.assertEqual(aggregate_status(evaluate(self.protocol, self.store)), Status.PASS)

    def test_conjunction_truth_table_and_empty_are_safe(self):
        for values in itertools.product(Status, repeat=3):
            expected = Status.FAIL if Status.FAIL in values else Status.PASS if all(v == Status.PASS for v in values) else Status.UNKNOWN
            self.assertEqual(aggregate_status(CheckResult(str(i), v, "fixture") for i, v in enumerate(values)), expected)
        self.assertEqual(aggregate_status([]), Status.UNKNOWN)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.protocol = Protocol.from_dict(protocol_data(True))
        self.store = EvidenceStore(self.protocol, "temporal")

    def result(self):
        return by_id(self.store, "ready-before-declare")

    def test_checkpoint_then_nonmutating_declaration_passes(self):
        populate(self.store)
        self.store.record_checkpoint("ready", timestamp_ms=150)
        self.store.record_event("declare", timestamp_ms=200, evidence_id="fixture://declare")
        self.assertEqual(self.result().status, Status.PASS)

    def test_absent_prior_checkpoint_is_unknown_unless_history_proves_violation(self):
        self.store.record_event("declare", timestamp_ms=100, evidence_id="fixture://declare")
        self.assertEqual(self.result().status, Status.UNKNOWN)
        self.store.start_attempt(200)
        self.store.record_event("declare", timestamp_ms=300, evidence_id="fixture://declare2", history_complete=True)
        self.assertEqual(self.result().status, Status.FAIL)
        populate(self.store, 400)
        self.store.record_checkpoint("ready", timestamp_ms=450)
        self.assertEqual(self.result().status, Status.FAIL)  # Later inspection cannot repair history.

    def test_stale_checkpoint_cannot_certify_a_much_later_completion(self):
        populate(self.store)
        self.store.record_checkpoint("ready", timestamp_ms=150)
        self.store.record_event("declare", timestamp_ms=30_101, evidence_id="fixture://late-declare")
        self.assertEqual(self.result().status, Status.UNKNOWN)

    def test_failed_or_unknown_latest_recheck_cannot_be_cleared_by_old_pass(self):
        for value in (0, None):
            with self.subTest(value=value):
                store = EvidenceStore(self.protocol, "recheck")
                populate(store)
                store.record_checkpoint("ready", timestamp_ms=110)
                observe(store, value=value, timestamp_ms=200)
                store.record_checkpoint("ready", timestamp_ms=210)
                store.record_event("declare", timestamp_ms=300, evidence_id="fixture://declare")
                expected = Status.FAIL if value == 0 else Status.UNKNOWN
                self.assertEqual(by_id(store, "ready-before-declare").status, expected)

    def test_mutation_and_new_attempt_require_new_checkpoint(self):
        populate(self.store)
        self.store.record_checkpoint("ready", timestamp_ms=110)
        self.store.invalidate("contents changed", 120)
        self.store.record_event("declare", timestamp_ms=130, evidence_id="fixture://declare")
        self.assertEqual(self.result().status, Status.UNKNOWN)
        self.store.start_attempt(140)
        populate(self.store, 150)
        self.store.record_event("declare", timestamp_ms=160, evidence_id="fixture://declare2")
        self.assertEqual(self.result().status, Status.UNKNOWN)
        self.assertEqual(self.store.checkpoint_records[0].status, Status.PASS)


class PlannerTests(unittest.TestCase):
    def action(self, **data):
        return RepairAction.from_dict(data)

    def test_physical_prediction_is_not_knowledge_inspection_does_not_change_world(self):
        physical = self.action(id="place", kind="physical", effects={"present": True})
        inspect = self.action(id="inspect", kind="inspect", reveals=["present"])
        world, known = {"present": False}, {"present": False}
        self.assertIsNone(plan_repair(world, known, [physical], {"present": True}))
        plan = plan_repair(world, known, [physical, inspect], {"present": True})
        self.assertEqual(plan.action_ids, ("place", "inspect"))
        self.assertEqual(plan.total_cost, 2)
        self.assertEqual(world, {"present": False})
        self.assertEqual(known, {"present": False})
        inspected = plan_repair({"present": True}, {}, [inspect], {"present": True})
        self.assertEqual(inspected.action_ids, ("inspect",))
        self.assertEqual(inspected.final_world, {"present": True})
        self.assertTrue(inspected.to_dict()["hypothetical"])

    def test_unknown_precondition_forces_inspection_and_legal_path(self):
        actions = [self.action(id="inspect", kind="inspect", reveals=["open", "present"]),
                   self.action(id="place", kind="physical", preconditions={"open": True}, effects={"present": True})]
        plan = plan_repair({"open": True, "present": False}, {}, actions, {"present": True})
        self.assertEqual(plan.action_ids, ("inspect", "place", "inspect"))
        self.assertIsNone(plan_repair({"open": False, "present": False}, {}, actions, {"present": True}))

    def test_shortest_weighted_path_matches_independent_exhaustive_paths(self):
        actions = [self.action(id="inspect", kind="inspect", reveals=["divider", "driver"], cost=0.5),
                   self.action(id="both", kind="physical", effects={"divider": True, "driver": True}, cost=5),
                   self.action(id="divider", kind="physical", effects={"divider": True}, cost=1),
                   self.action(id="driver", kind="physical", preconditions={"divider": True}, effects={"driver": True}, cost=1)]
        initial = {"divider": False, "driver": False}
        goal = {"divider": True, "driver": True}
        expected = None
        # Independent brute force enumerates every action sequence up to length 5.
        for length in range(6):
            for path in itertools.product(actions, repeat=length):
                world, known, cost = dict(initial), dict(initial), 0
                for action in path:
                    if any(key not in known or known[key] != value for key, value in action.preconditions.items()):
                        break
                    cost += action.cost
                    if action.kind == "physical":
                        world.update(action.effects)
                        for key in set(action.effects) | set(action.invalidates):
                            known.pop(key, None)
                    else:
                        known.update({key: world[key] for key in action.reveals})
                else:
                    if all(known.get(key) == value for key, value in goal.items()):
                        expected = cost if expected is None else min(expected, cost)
        plan = plan_repair(initial, initial, actions, goal)
        self.assertEqual(plan.total_cost, expected)
        self.assertEqual(plan.action_ids, ("divider", "inspect", "driver", "inspect"))
        self.assertEqual(plan.total_cost, 3)

    def test_negative_nan_cost_inspection_effects_and_unknown_references_rejected(self):
        invalid = [dict(id="negative", kind="physical", effects={"present": True}, cost=-1),
                   dict(id="nan", kind="physical", effects={"present": True}, cost=float("nan")),
                   dict(id="fake-inspect", kind="inspect", effects={"present": True}, reveals=["present"])]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                self.action(**data)
        action = self.action(id="unknown", kind="inspect", reveals=["other"])
        with self.assertRaises(ValidationError):
            plan_repair({"present": False}, {}, [action], {"present": True})

    def test_budget_exhaustion_does_not_return_unproven_minimum_and_bool_is_not_int(self):
        action = self.action(id="inspect", kind="inspect", reveals=["present"])
        with self.assertRaises(ValidationError):
            plan_repair({"present": True}, {}, [action], {"present": True}, max_states=1)
        with self.assertRaises(ValidationError):
            plan_repair({"present": False}, {"present": 0}, [action], {"present": True})


if __name__ == "__main__":
    unittest.main()
