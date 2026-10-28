"""Authored portability fixtures test contracts and budgets, never model quality."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from proofloop.compiler import COFFEE_PLACEMENT_PRESET, COFFEE_PRESET_REGIONS
from proofloop.core import ValidationError
from proofloop.guided import GUIDED_VERSION
from proofloop.interactive import DEFAULT_REGIONS
from proofloop.recovery import digest
from tests import test_interactive as interactive_fixtures

REGIONS = interactive_fixtures.REGIONS


class NamedCompileClient(interactive_fixtures.FakeCompileClient):
    def __init__(self, regions, placements):
        super().__init__(placements=placements)
        self.regions = regions

    def spark_json(self, *args, **kwargs):
        result = super().spark_json(*args, **kwargs)
        result["parsed"]["protocol"]["regions"] = list(self.regions)
        return result


class TaskPortabilityTests(unittest.TestCase):
    # Share only authored fixture helpers, not the parent's test methods.
    setUp = interactive_fixtures.InteractiveTests.setUp
    tearDown = interactive_fixtures.InteractiveTests.tearDown
    payload = interactive_fixtures.InteractiveTests.payload
    activate_payload = interactive_fixtures.InteractiveTests.activate_payload

    def install_named(self, regions=("ready", "buffer"), target="ready"):
        self.compiler = NamedCompileClient(regions, {"glass": target})
        self.engine._compile_client_factory = lambda: self.compiler
        return {"title": "Glass placement", "requirements": f"The unique glass must be in {target}.",
                "objects": ["glass"], "regions": list(regions), "confirm_unique_targets": True}

    def test_one_object_uses_two_calls_and_named_regions_flow_into_evidence(self):
        draft = self.engine.draft_task(self.install_named())["task_draft"]
        self.assertEqual(["ready", "buffer"], draft["regions"])
        self.assertIn('"allowed_regions": ["ready", "buffer"]', self.compiler.ledger[0]["prompt"])
        activated = self.engine.activate_task(self.activate_payload(draft))
        self.assertEqual(["ready", "buffer"], activated["active_task"]["regions"])
        self.assertEqual({"ready": DEFAULT_REGIONS["left"], "buffer": DEFAULT_REGIONS["right"]}, activated["regions"])
        payload = self.payload()
        payload["regions"] = {"ready": REGIONS["left"], "buffer": REGIONS["right"]}
        self.client.max_requests = 2
        state = self.engine.inspect_pair(payload)
        self.assertEqual("pass", state["status"])
        self.assertEqual(2, len(self.client.ledger))
        self.assertEqual("ready", state["checks"][0]["observed_region"])
        capture = next(self.engine._directory.glob("inspection-*/capture.json"))
        self.assertEqual({"ready", "buffer"}, set(json.loads(capture.read_text())["regions"]))

    def test_malformed_region_inventory_never_reaches_compiler(self):
        valid = self.install_named()
        for regions in ([], ["ready"], ["ready", "buffer", "other"], ["ready", "ready"],
                        ["READY", "buffer"], ["", "buffer"], [["nested"], "buffer"], None):
            with self.subTest(regions=regions), self.assertRaises(ValidationError):
                self.engine.draft_task(dict(valid, regions=regions))
        self.assertEqual([], self.compiler.ledger)

    def test_region_change_binds_hash_and_reinitializes_instead_of_relabeling_calibration(self):
        self.engine.inspect_pair(self.payload())
        self.engine._state["regions"] = {"left": [0, 0, 0.1, 0.1], "right": [0.9, 0.9, 1, 1]}
        draft = self.engine.draft_task(self.install_named())["task_draft"]
        state = self.engine.activate_task(self.activate_payload(draft))
        self.assertEqual("unknown", state["status"])
        self.assertEqual([], state["images"])
        self.assertEqual({"ready": DEFAULT_REGIONS["left"], "buffer": DEFAULT_REGIONS["right"]}, state["regions"])
        first_hash = state["active_task"]["task_hash"]
        other = self.engine.draft_task(self.install_named(("staged", "buffer"), "staged"))["task_draft"]
        self.assertNotEqual(first_hash, other["draft_hash"])
        with self.assertRaises(ValidationError):
            self.engine.activate_task(dict(self.activate_payload(other), regions=["ready", "buffer"]))

    def test_old_rectangle_keys_cannot_verify_new_named_contract(self):
        draft = self.engine.draft_task(self.install_named())["task_draft"]
        self.engine.activate_task(self.activate_payload(draft))
        with self.assertRaisesRegex(ValidationError, "exact region IDs"):
            self.engine.inspect_pair(self.payload())
        self.assertEqual([], self.client.ledger)

    def test_mutating_reviewed_region_labels_is_rejected(self):
        draft = self.engine.draft_task(self.install_named())["task_draft"]
        self.engine._state["task_draft"]["regions"] = ["changed", "buffer"]
        with self.assertRaisesRegex(ValidationError, "draft changed"):
            self.engine.activate_task(self.activate_payload(draft))
        self.assertEqual(["left", "right"], self.engine.snapshot()["active_task"]["regions"])

    def test_rehashed_region_inventory_must_match_executable_protocol(self):
        self.engine.draft_task(self.install_named())
        draft = self.engine._state["task_draft"]
        draft["regions"] = ["changed", "buffer"]
        draft["draft_hash"] = digest({key: value for key, value in draft.items() if key != "draft_hash"})
        with self.assertRaisesRegex(ValidationError, "region inventory"):
            self.engine.activate_task(self.activate_payload(draft))

    def preset_payload(self):
        return {**deepcopy(COFFEE_PLACEMENT_PRESET), "draft_source": "authored_coffee_preset",
                "confirm_unique_targets": True}

    def test_fixed_preset_requires_review_and_never_loads_a_model_client(self):
        factory = Mock(side_effect=AssertionError("The authored preset cannot use model inference"))
        self.engine._compile_client_factory = factory
        before = self.engine.snapshot()
        state = self.engine.draft_task(self.preset_payload())
        factory.assert_not_called()
        self.assertEqual(before["active_task"], state["active_task"])
        draft = state["task_draft"]
        self.assertIsNone(draft["model"])
        self.assertEqual("authored_frozen_coffee_preset", draft["origin"])
        self.assertEqual(GUIDED_VERSION, draft["perception_version"])
        self.assertEqual(COFFEE_PRESET_REGIONS, draft["suggested_regions"])
        with self.assertRaises(ValidationError):
            self.engine.activate_task(dict(self.activate_payload(draft), confirm_requirements=False))
        state = self.engine.activate_task(self.activate_payload(draft))
        self.assertEqual(draft["draft_hash"], state["active_task"]["task_hash"])
        self.assertEqual(COFFEE_PRESET_REGIONS, state["regions"])
        self.assertEqual("unknown", state["status"])
        self.assertEqual("authored_frozen_coffee_preset", state["active_task"]["origin"])
        approval = json.loads(next(self.engine._directory.glob("task-approval-*.json")).read_text())
        self.assertEqual(["dispense", "staging"], approval["regions"])
        self.assertEqual(GUIDED_VERSION, approval["perception_version"])

    def test_preset_cannot_bypass_compilation_for_changed_client_requirements(self):
        factory = Mock()
        self.engine._compile_client_factory = factory
        for field, value in (("requirements", "The glass must be in STAGING."), ("objects", ["cup"]),
                             ("regions", ["left", "right"]), ("concepts", {"glass": "red mug"}),
                             ("perception_mode", "concept"), ("title", "Changed gate"), ("draft_source", "invented")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.engine.draft_task(dict(self.preset_payload(), **{field: value}))
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
