"""Compiler contract tests use mocked responses and establish no live model quality."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from proofloop.compiler import CompilerError, authored_coffee_protocol, compile_protocol, compiler_schema
from proofloop.core import Protocol
from proofloop.meta import validate_json


def mocked_proposal():
    return {"protocol": {
        "id": "proofloop-task", "version": "draft-1", "objects": ["screwdriver", "bolt"],
        "regions": ["left", "right"], "checks": [
            {"id": "screwdriver-left", "kind": "count", "object_id": "screwdriver", "region_id": "left", "expected": 1},
            {"id": "bolt-right", "kind": "count", "object_id": "bolt", "region_id": "right", "expected": 2},
        ], "steps": [], "checkpoints": [],
    }, "unsupported_requirements": [], "clarifications_needed": []}


def mocked_client(proposal=None):
    client = Mock()
    client.spark_json.return_value = {
        "parsed": mocked_proposal() if proposal is None else proposal,
        "source": None, "usage": {"input_tokens": 12, "output_tokens": 34}, "latency_seconds": 0.1,
    }
    return client


class CompilerTests(unittest.TestCase):
    def compile(self, client):
        return compile_protocol(client, "Exactly one screwdriver in left and two bolts in right.",
                                objects=["screwdriver", "bolt"], regions=["left", "right"])

    def test_mocked_proposal_is_valid_but_never_approved(self):
        client = mocked_client()
        draft = self.compile(client)
        self.assertEqual(draft["source_origin"], "live_meta_api")
        self.assertTrue(draft["requires_human_approval"])
        self.assertFalse(draft["approved"])
        self.assertEqual(draft["review_status"], "awaiting_human_review")
        protocol = Protocol.from_dict(draft["protocol"])
        self.assertEqual(draft["content_hash"], protocol.content_hash)
        self.assertEqual(len(protocol.checks), 2)
        self.assertTrue(all(check.kind == "count" for check in protocol.checks))
        self.assertEqual(protocol.steps, ())
        self.assertEqual(client.spark_json.call_args.kwargs["name"], "proofloop_protocol_draft")
        self.assertIn("HUMAN REVIEW", client.spark_json.call_args.args[0])

    def test_schema_is_compatible_with_local_validator(self):
        schema = compiler_schema(["screwdriver", "bolt"], ["left", "right"])
        validate_json(mocked_proposal(), schema)
        self.assertNotIn("oneOf", json.dumps(schema))
        self.assertNotIn("anyOf", json.dumps(schema))

    def test_explicit_zero_count_is_preserved(self):
        proposal = mocked_proposal()
        proposal["protocol"]["checks"][0]["expected"] = 0
        draft = self.compile(mocked_client(proposal))
        self.assertEqual(draft["protocol"]["checks"][0]["expected"], 0)

    def test_optional_image_is_forwarded_without_becoming_observation(self):
        client = mocked_client()
        source = {"sha256": "0" * 64, "mime": "image/png", "bytes": 20}
        client.spark_json.return_value["source"] = source
        draft = compile_protocol(client, "One screwdriver in left.", objects=["screwdriver", "bolt"],
                                 regions=["left", "right"], image_path=Path("owned.png"))
        self.assertEqual(client.spark_json.call_args.kwargs["image_path"], Path("owned.png"))
        self.assertEqual(draft["input"]["image_source"], source)
        self.assertNotIn("observations", draft)

    def test_inventory_drift_and_duplicate_echo_are_rejected(self):
        for replacement in [["screwdriver", "hammer"], ["screwdriver", "screwdriver"], ["screwdriver"]]:
            proposal = mocked_proposal()
            proposal["protocol"]["objects"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(CompilerError):
                self.compile(mocked_client(proposal))
        proposal = mocked_proposal()
        proposal["protocol"]["regions"] = ["left", "left"]
        with self.assertRaises(CompilerError):
            self.compile(mocked_client(proposal))

    def test_unknown_check_refs_and_boolean_expected_rejected(self):
        for field, value in [("object_id", "invented"), ("region_id", "elsewhere"), ("expected", True),
                             ("expected", -1), ("expected", 17), ("kind", "visible")]:
            proposal = mocked_proposal()
            proposal["protocol"]["checks"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(CompilerError):
                self.compile(mocked_client(proposal))

    def test_empty_checks_duplicate_ids_and_conflicting_pairs_rejected(self):
        for mutation in ["empty", "duplicate_id", "duplicate_pair"]:
            proposal = mocked_proposal()
            if mutation == "empty":
                proposal["protocol"]["checks"] = []
            elif mutation == "duplicate_id":
                proposal["protocol"]["checks"][1]["id"] = "screwdriver-left"
            else:
                proposal["protocol"]["checks"][1].update(object_id="screwdriver", region_id="left")
            with self.subTest(mutation=mutation), self.assertRaises(CompilerError):
                self.compile(mocked_client(proposal))

    def test_any_inferred_order_or_cyclic_dag_rejected(self):
        for steps in [[{"id": "put-driver", "requires": []}],
                      [{"id": "a", "requires": ["b"]}, {"id": "b", "requires": ["a"]}],
                      [{"id": "a", "requires": ["unknown"]}]]:
            proposal = mocked_proposal()
            proposal["protocol"]["steps"] = steps
            with self.subTest(steps=steps), self.assertRaises(CompilerError):
                self.compile(mocked_client(proposal))

    def test_unsupported_or_ambiguous_requirements_block_acceptance(self):
        for key in ["unsupported_requirements", "clarifications_needed"]:
            proposal = mocked_proposal()
            proposal[key] = ["Order or exact region cannot be determined."]
            with self.subTest(key=key), self.assertRaises(CompilerError):
                self.compile(mocked_client(proposal))

    def test_inventory_order_is_canonicalized(self):
        ordinary = self.compile(mocked_client())
        proposal = mocked_proposal()
        proposal["protocol"]["objects"].reverse()
        proposal["protocol"]["regions"].reverse()
        reordered = self.compile(mocked_client(proposal))
        self.assertEqual(ordinary["content_hash"], reordered["content_hash"])

    def test_requirement_changes_produce_different_executable_hash(self):
        initial = self.compile(mocked_client())
        proposal = mocked_proposal()
        proposal["protocol"]["checks"][0]["region_id"] = "right"
        changed = self.compile(mocked_client(proposal))
        self.assertNotEqual(initial["content_hash"], changed["content_hash"])

    def test_invalid_input_stops_before_api(self):
        cases = [dict(requirements="", objects=["screwdriver"], regions=["left"]),
                 dict(requirements="x" * 8193, objects=["screwdriver"], regions=["left"]),
                 dict(requirements="One item", objects=[], regions=["left"]),
                 dict(requirements="One item", objects=["a", "a"], regions=["left"]),
                 dict(requirements="One item", objects=["with spaces"], regions=["left"])]
        for arguments in cases:
            client = mocked_client()
            with self.subTest(arguments=arguments), self.assertRaises(CompilerError):
                compile_protocol(client, **arguments)
            client.spark_json.assert_not_called()


class SpatialCompilerModeTests(unittest.TestCase):
    def test_one_target_named_regions_preserve_exact_ids_and_review_boundary(self):
        proposal = {"protocol": {"id": "proofloop-task", "version": "draft-1", "objects": ["glass"],
            "regions": ["dispense", "staging"], "checks": [{"id": "glass-at-dispense", "kind": "location",
            "object_id": "glass", "region_id": "dispense", "unique_target": True}], "steps": [], "checkpoints": []},
            "unsupported_requirements": [], "clarifications_needed": []}
        draft = compile_protocol(mocked_client(proposal), "The unique glass must be in dispense.", objects=["glass"],
                                 regions=["dispense", "staging"], predicate_kind="location", unique_objects=["glass"])
        self.assertEqual(["glass"], draft["protocol"]["objects"])
        self.assertEqual(["dispense", "staging"], draft["protocol"]["regions"])
        self.assertEqual("dispense", draft["protocol"]["checks"][0]["region_id"])
        self.assertFalse(draft["approved"])
        self.assertTrue(draft["requires_human_approval"])

    def test_authored_coffee_preset_copies_frozen_gate_without_model_or_approval(self):
        first = authored_coffee_protocol()
        self.assertIsNone(first["model"])
        self.assertEqual("authored_frozen_coffee_preset", first["source_origin"])
        self.assertFalse(first["approved"])
        self.assertTrue(first["requires_human_approval"])
        self.assertEqual(["glass"], first["protocol"]["objects"])
        self.assertEqual(["dispense", "staging"], first["protocol"]["regions"])
        check = first["protocol"]["checks"][0]
        self.assertEqual(("glass", "dispense", 2, 0.8, 30000),
                         (check["object_id"], check["region_id"], check["min_stable_samples"], check["min_confidence"], check["max_age_ms"]))
        original_hash = first["content_hash"]
        first["protocol"]["checks"][0]["region_id"] = "staging"
        first["suggested_regions"]["dispense"][0] = 0
        second = authored_coffee_protocol()
        self.assertEqual(original_hash, second["content_hash"])
        self.assertEqual("dispense", second["protocol"]["checks"][0]["region_id"])
        self.assertEqual(0.38, second["suggested_regions"]["dispense"][0])

    def proposal(self, kind="location"):
        proposal = mocked_proposal()
        for check in proposal["protocol"]["checks"]:
            check["kind"] = kind
            if kind == "location":
                del check["expected"]
                check["unique_target"] = True
            else:
                check["expected"] = True
        return proposal

    def compile_location(self, client, **updates):
        arguments = {"objects": ["screwdriver", "bolt"], "regions": ["left", "right"],
                     "predicate_kind": "location", "unique_objects": ["screwdriver", "bolt"]}
        arguments.update(updates)
        return compile_protocol(client, "The declared unique screwdriver goes in left and the declared unique bolt in right.",
                                **arguments)

    def test_location_mode_preserves_caller_contract_but_requires_human_review(self):
        client = mocked_client(self.proposal())
        draft = self.compile_location(client)
        task = Protocol.from_dict(draft["protocol"])
        self.assertEqual(["location", "location"], [check.kind for check in task.checks])
        self.assertTrue(all(check.unique_target for check in task.checks))
        self.assertEqual(["screwdriver", "bolt"], draft["input"]["confirmed_unique_objects"])
        self.assertFalse(draft["approved"])
        self.assertTrue(draft["requires_human_approval"])
        prompt = client.spark_json.call_args.args[0]
        self.assertIn("never you", prompt)
        validate_json(self.proposal(), compiler_schema(["screwdriver", "bolt"], ["left", "right"],
                                                     predicate_kind="location", unique_objects=["screwdriver", "bolt"]))

    def test_missing_invalid_or_extended_unique_contract_stops_before_inference(self):
        for unique in ([], ["invented"], ["bolt", "bolt"], "bolt", None):
            client = mocked_client(self.proposal())
            with self.subTest(unique=unique), self.assertRaises(CompilerError):
                self.compile_location(client, unique_objects=unique)
            client.spark_json.assert_not_called()
        client = mocked_client(self.proposal())
        with self.assertRaises(CompilerError):
            self.compile_location(client, predicate_kind="invented")
        client.spark_json.assert_not_called()

    def test_location_model_cannot_add_unconfirmed_object_or_remove_contract(self):
        for mutation in ("unconfirmed", "missing", "false", "count-field"):
            proposal = self.proposal()
            arguments = {}
            if mutation == "unconfirmed":
                arguments["unique_objects"] = ["screwdriver"]
            elif mutation == "missing":
                del proposal["protocol"]["checks"][0]["unique_target"]
            elif mutation == "false":
                proposal["protocol"]["checks"][0]["unique_target"] = False
            else:
                proposal["protocol"]["checks"][0]["expected"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(CompilerError):
                self.compile_location(mocked_client(proposal), **arguments)

    def test_unique_object_cannot_require_two_locations(self):
        proposal = self.proposal()
        proposal["protocol"]["checks"][1]["object_id"] = "screwdriver"
        with self.assertRaises(CompilerError):
            self.compile_location(mocked_client(proposal))

    def test_visibility_mode_accepts_explicit_true_or_false_without_uniqueness(self):
        proposal = self.proposal("visible")
        proposal["protocol"]["checks"][1]["expected"] = False
        client = mocked_client(proposal)
        draft = compile_protocol(client, "The screwdriver must be visible in left, and bolt absent from right.",
                                 objects=["screwdriver", "bolt"], regions=["left", "right"],
                                 predicate_kind="visible")
        self.assertEqual([True, False], [check["expected"] for check in draft["protocol"]["checks"]])
        self.assertEqual([], draft["input"]["confirmed_unique_objects"])
        self.assertNotIn("unique_target", draft["protocol"]["checks"][0])
        proposal["protocol"]["checks"][0]["expected"] = 1
        with self.assertRaises(CompilerError):
            compile_protocol(mocked_client(proposal), "Visible screwdriver", objects=["screwdriver", "bolt"],
                             regions=["left", "right"], predicate_kind="visible")

    def test_default_count_mode_cannot_silently_become_location(self):
        with self.assertRaises(CompilerError):
            compile_protocol(mocked_client(self.proposal()), "One screwdriver in left", objects=["screwdriver", "bolt"],
                             regions=["left", "right"], unique_objects=["screwdriver", "bolt"])


if __name__ == "__main__":
    unittest.main()
