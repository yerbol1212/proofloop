"""Compiler contract tests use mocked responses and establish no live model quality."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from proofloop.compiler import CompilerError, compile_protocol, compiler_schema
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


if __name__ == "__main__":
    unittest.main()
