"""Authored transport/geometry fixtures, never real model or accuracy results."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from proofloop import visibility_guard as guard
from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SPARK_MODEL
from tests.test_guided import jpeg_header


def contract():
    return {"target_id": "remote", "concept": "white remote control body",
            "attributes": {"white_body": "white rounded rectangular body", "battery_recess": "recessed battery compartment"},
            "exclusions": {"tape": "white tape strip", "gripper": "robot gripper"}}


def clear():
    return {"candidate_identity": "target", "direct_distinctive_evidence": True, "identity_ambiguous": False,
        "identity_observation": "Distinct rounded white body and rectangular recessed compartment.",
        "attributes": {"white_body": "visible_match", "battery_recess": "visible_match"},
        "exclusions": {"tape": "neighbor", "gripper": "neighbor"},
        "physical_visibility": {"classification": "clear", "visible_boundary": "complete",
            "visible_extremities": "complete", "external_occluder_hides_target": False,
            "frame_truncated": False, "crop_truncated": False, "full_frame_resolves_entire_extent": True,
            "blur_prevents_assessment": "no", "visible_terminal_landmarks": ["left rounded body end", "right rounded body end"],
            "occluder": "", "affected_boundary": "", "observation": "Both ends and the full body perimeter are visible; gripper is adjacent."}}


def provider(parsed=None, *, status="completed", usage=None):
    return {"model": SPARK_MODEL, "status": status,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(parsed or clear())}]}],
            "usage": usage or {"input_tokens": 71, "output_tokens": 53}}


def response(value):
    return BytesIO(json.dumps(value).encode())


class VisibilityGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "full-frame.jpg"
        self.source.write_bytes(jpeg_header(width=100, height=80, suffix=b"authored-not-photographic"))
        self.box = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}
        def crop(executable, source, target, box):
            suffix = guard.digest({"source": hashlib.sha256(source.read_bytes()).hexdigest(), "box": box}).encode()
            target.write_bytes(jpeg_header(width=box["x2"] - box["x1"] + 1,
                                          height=box["y2"] - box["y1"] + 1, suffix=suffix))
        for name, value in (("_crop", crop), ("_extractor_path", lambda: "authored-ffmpeg"),
                            ("_extractor_version", lambda executable: "authored-ffmpeg-version")):
            override = patch.object(guard, name, side_effect=value)
            override.start()
            self.addCleanup(override.stop)
        self.prepared = guard.prepare(self.source, self.box, contract(), self.root / "prepared",
                                      source_sha256="a" * 64, frame_index=17, track_id=1)

    def client(self, parsed=None, *, opener=None, **kwargs):
        return MetaClient("authored-not-a-real-key", max_requests=1,
                          opener=opener or (lambda *args, **kwargs: response(provider(parsed))), **kwargs)

    def rehash(self, value):
        value["prepared_hash"] = guard.digest({key: item for key, item in value.items() if key != "prepared_hash"})
        return value

    def test_preparation_binds_full_frame_and_deterministic_context_crop(self):
        value = self.prepared
        self.assertEqual(guard.VERSION, value["method_version"])
        self.assertEqual((100, 80), (value["frame_width"], value["frame_height"]))
        self.assertEqual({"x1": 12, "y1": 12, "x2": 48, "y2": 48}, value["crop_box"])
        self.assertEqual((37, 37), (value["crop_width"], value["crop_height"]))
        self.assertEqual({"x1": 8, "y1": 8, "x2": 28, "y2": 28}, value["candidate_in_crop"])
        self.assertEqual("actual_sam_video_track", value["anchor_origin"])
        self.assertEqual(["full_frame", "candidate_crop"], value["config"]["image_order"])
        guard.verify(value)

    def test_single_low_reasoning_request_registers_full_frame_then_crop(self):
        requests = []
        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return response(provider())
        result = guard.challenge(self.client(opener=opener), self.prepared)
        self.assertEqual("completed", result["status"])
        self.assertEqual("supported", result["evaluation"]["decision"])
        self.assertEqual(1, len(requests))
        payload = requests[0]
        self.assertEqual({"effort": "low"}, payload["reasoning"])
        self.assertEqual(4096, payload["max_output_tokens"])
        self.assertFalse(payload["store"])
        content = payload["input"][0]["content"]
        self.assertEqual(["input_text", "input_image", "input_image"], [item["type"] for item in content])
        self.assertEqual([self.prepared["frame_sha256"], self.prepared["crop_sha256"]],
                         [item["sha256"] for item in result["response"]["sources"]])
        self.assertIsNone(result["response"]["source"])
        self.assertEqual(71, result["usage"]["input_tokens"])
        self.assertEqual(53, result["usage"]["output_tokens"])

    def test_prompt_explicitly_separates_reference_free_identity_visibility_and_adjacent_objects(self):
        value = guard.prompt(self.prepared)
        for text in ("beyond its edges", "NOT a claim", "Recognizable does NOT mean clear", "normal_pose_not_visible",
                     "Neighbors alone", "gripper", "blur_prevents_assessment=no", "visible_terminal_landmarks"):
            self.assertIn(text, value)
        self.assertNotIn(str(self.source), value)
        self.assertNotIn("expected_label", value)
        self.assertNotIn("initial_frame", value)
        self.assertIn('"candidate_rectangle_inclusive_pixels": {"x1": 20', value)

    def test_normal_unseen_pose_attribute_does_not_reject_directly_identified_clear_target(self):
        parsed = clear()
        parsed["attributes"]["battery_recess"] = "normal_pose_not_visible"
        self.assertEqual("supported", guard.evaluate(parsed, contract())["decision"])
        parsed["direct_distinctive_evidence"] = False
        self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])

    def test_normal_pose_for_every_attribute_cannot_replace_a_visible_identity_anchor(self):
        parsed = clear()
        parsed["attributes"] = {key: "normal_pose_not_visible" for key in parsed["attributes"]}
        value = guard.evaluate(parsed, contract())
        self.assertEqual("unknown", value["decision"])
        self.assertTrue(value["fully_visible"])
        self.assertTrue(value["identity_ambiguous"])
        parsed["attributes"]["white_body"] = "visible_match"
        self.assertEqual("supported", guard.evaluate(parsed, contract())["decision"])

    def test_neighbor_exclusions_do_not_masquerade_as_target_identity_or_occlusion(self):
        for value in ("neighbor", "absent"):
            parsed = clear()
            parsed["exclusions"]["gripper"] = value
            self.assertEqual("supported", guard.evaluate(parsed, contract())["decision"])
        parsed["exclusions"]["gripper"] = "target"
        self.assertEqual("contradicted", guard.evaluate(parsed, contract())["decision"])

    def test_visible_attribute_mismatch_contradicts_but_uncertain_attribute_withholds(self):
        for value, expected in (("mismatch", "contradicted"), ("uncertain", "unknown")):
            parsed = clear()
            parsed["attributes"]["white_body"] = value
            self.assertEqual(expected, guard.evaluate(parsed, contract())["decision"])

    def test_identity_uncertainty_and_conflicting_exclusion_never_support(self):
        for key, value in (("candidate_identity", "uncertain"), ("identity_ambiguous", True),
                           ("direct_distinctive_evidence", False)):
            parsed = clear()
            parsed[key] = value
            self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])
        parsed = clear()
        parsed["candidate_identity"] = "other"
        self.assertEqual("contradicted", guard.evaluate(parsed, contract())["decision"])
        parsed = clear()
        parsed["exclusions"]["tape"] = "uncertain"
        self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])

    def test_recognizable_grip_occluded_target_remains_unknown(self):
        parsed = clear()
        parsed["physical_visibility"].update(classification="occluded", visible_boundary="interrupted",
            visible_extremities="hidden", external_occluder_hides_target=True,
            full_frame_resolves_entire_extent=False, occluder="robot gripper", affected_boundary="handle end")
        value = guard.evaluate(parsed, contract())
        self.assertEqual("unknown", value["decision"])
        self.assertFalse(value["fully_visible"])
        self.assertEqual([], value["consistency_issues"])

    def test_full_frame_can_resolve_a_crop_edge_without_relaxing_physical_occlusion(self):
        parsed = clear()
        parsed["physical_visibility"]["crop_truncated"] = True
        self.assertEqual("supported", guard.evaluate(parsed, contract())["decision"])
        parsed["physical_visibility"]["full_frame_resolves_entire_extent"] = False
        value = guard.evaluate(parsed, contract())
        self.assertEqual("unknown", value["decision"])
        self.assertIn("clear_visibility_conflicts_with_observed_extent", value["consistency_issues"])

    def test_every_clear_visibility_contradiction_fails_closed(self):
        changes = {"visible_boundary": "interrupted", "visible_extremities": "hidden",
            "external_occluder_hides_target": True, "frame_truncated": True,
            "full_frame_resolves_entire_extent": False, "blur_prevents_assessment": "yes",
            "occluder": "robot finger", "affected_boundary": "right edge"}
        for field, value in changes.items():
            parsed = clear()
            parsed["physical_visibility"][field] = value
            with self.subTest(field=field):
                evaluated = guard.evaluate(parsed, contract())
                self.assertEqual("unknown", evaluated["decision"])
                self.assertFalse(evaluated["fully_visible"])
                self.assertIn("clear_visibility_conflicts_with_observed_extent", evaluated["consistency_issues"])

    def test_unassessable_blur_or_boundaries_are_not_clear(self):
        for field in ("visible_boundary", "visible_extremities", "blur_prevents_assessment"):
            parsed = clear()
            parsed["physical_visibility"][field] = "uncertain"
            self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])

    def test_clear_requires_two_named_distinct_nonblank_terminal_landmarks(self):
        for names in ([], ["tip"], ["tip", " TIP "], [" ", "tip"]):
            parsed = clear()
            parsed["physical_visibility"]["visible_terminal_landmarks"] = names
            with self.subTest(names=names):
                self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])

    def test_occlusion_requires_named_occluder_and_affected_boundary(self):
        parsed = clear()
        parsed["physical_visibility"]["classification"] = "occluded"
        for change in ({}, {"external_occluder_hides_target": True},
                       {"external_occluder_hides_target": True, "occluder": "gripper"}):
            value = deepcopy(parsed)
            value["physical_visibility"].update(change)
            evaluated = guard.evaluate(value, contract())
            self.assertEqual("unknown", evaluated["decision"])
            self.assertIn("occlusion_lacks_named_occluder_and_affected_boundary", evaluated["consistency_issues"])

    def test_truncated_classification_requires_a_real_reported_image_edge(self):
        parsed = clear()
        parsed["physical_visibility"]["classification"] = "truncated"
        self.assertIn("truncation_lacks_an_image_edge", guard.evaluate(parsed, contract())["consistency_issues"])
        parsed["physical_visibility"]["frame_truncated"] = True
        self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])

    def test_physical_visibility_diagnostic_remains_independent_of_identity(self):
        for key, value in (("candidate_identity", "uncertain"), ("identity_ambiguous", True),
                           ("direct_distinctive_evidence", False)):
            parsed = clear()
            parsed[key] = value
            with self.subTest(key=key):
                evaluated = guard.evaluate(parsed, contract())
                self.assertTrue(evaluated["fully_visible"])
                self.assertNotEqual("supported", evaluated["decision"])

    def test_schema_rejects_extra_missing_fields_wrong_types_and_unbounded_observations(self):
        variants = [{**clear(), "invented": "expected target"}, {**clear(), "identity_ambiguous": "false"}]
        for field in clear():
            variants.append({key: value for key, value in clear().items() if key != field})
        for field, value in (("identity_observation", "x" * 241), ("identity_observation", " ")):
            variants.append({**clear(), field: value})
        parsed = clear()
        parsed["physical_visibility"]["observation"] = "x" * 241
        variants.append(parsed)
        for parsed in variants:
            with self.subTest(parsed=parsed), self.assertRaises(ValueError):
                guard.evaluate(parsed, contract())

    def test_contract_schema_bounds_and_names_are_enforced(self):
        for change in ({"target_id": "../remote"}, {"concept": ""}, {"attributes": {}}, {"exclusions": {}},
                       {"attributes": {"bad-name": "bad"}}, {"exclusions": {"item_" + str(i): "item" for i in range(7)}}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                guard.schema({**contract(), **change})

    def test_incomplete_responses_stay_unscored_and_retain_billed_tokens(self):
        result = guard.challenge(self.client(opener=lambda *args, **kwargs: response(provider(status="incomplete"))), self.prepared)
        self.assertEqual("unscored", result["status"])
        self.assertIsNone(result["evaluation"])
        self.assertEqual("incomplete_response", result["error"]["code"])
        self.assertEqual(53, result["usage"]["output_tokens"])
        self.assertEqual(1, result["usage"]["unsuccessful_spark_requests_with_reported_token_usage"])

    def test_schema_failure_is_unscored_instead_of_a_successful_unknown(self):
        result = guard.challenge(self.client({"missing": "fields"}), self.prepared)
        self.assertEqual("unscored", result["status"])
        self.assertIsNone(result["evaluation"])
        self.assertEqual("schema_validation_failed", result["error"]["code"])
        self.assertEqual(53, result["usage"]["output_tokens"])

    def test_transport_and_http_errors_never_retry(self):
        for kind in ("network", "http"):
            calls = []
            def opener(*args, **kwargs):
                calls.append(1)
                if kind == "network":
                    raise URLError("authored uncertainty")
                raise HTTPError("https://api.meta.ai/v1/responses", 429, "authored", {},
                                response({"error": {"code": "rate_limited"}, "usage": {"input_tokens": 3, "output_tokens": 5}}))
            result = guard.challenge(self.client(opener=opener), self.prepared)
            with self.subTest(kind=kind):
                self.assertEqual("unscored", result["status"])
                self.assertIsNone(result["evaluation"])
                self.assertEqual(1, len(calls))
                self.assertEqual(1, result["usage"]["request_attempts"])
                if kind == "http":
                    self.assertEqual(5, result["usage"]["output_tokens"])

    def test_wrong_or_reordered_registered_sources_model_and_status_fail_closed(self):
        for change in ("reverse", "missing", "hash", "single_source", "model", "status"):
            client = self.client()
            original = client.spark_json
            def altered(*args, **kwargs):
                value = original(*args, **kwargs)
                if change == "reverse":
                    value["sources"].reverse()
                elif change == "missing":
                    value["sources"].pop()
                elif change == "hash":
                    value["sources"][0]["sha256"] = "b" * 64
                elif change == "single_source":
                    value["source"] = value["sources"][0]
                elif change == "model":
                    value["response"]["model"] = "another-model"
                else:
                    value["response"]["status"] = "incomplete"
                return value
            with self.subTest(change=change), patch.object(client, "spark_json", side_effect=altered):
                result = guard.challenge(client, self.prepared)
                self.assertEqual("unscored", result["status"])
                self.assertEqual("local_validation_failed", result["error"]["code"])
                self.assertIsNone(result["evaluation"])

    def test_wrong_budget_retries_or_used_client_stops_before_network(self):
        for key, value in (("max_requests", 2), ("retries", 1), ("ledger", [{"used": True}])):
            client = self.client(opener=lambda *args, **kwargs: self.fail("invalid client reached HTTP"))
            setattr(client, key, value)
            with self.subTest(key=key), self.assertRaises(ValidationError):
                guard.challenge(client, self.prepared)

    def test_hash_provenance_changes_stop_before_http(self):
        for key, value in (("frame_index", 18), ("track_id", 2), ("source_sha256", "b" * 64),
                           ("prepared_hash", "c" * 64)):
            prepared = deepcopy(self.prepared)
            prepared[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                guard.challenge(self.client(opener=lambda *args, **kwargs: self.fail("changed hash reached HTTP")), prepared)

    def test_rehashed_invalid_provenance_geometry_extractor_and_order_are_rejected(self):
        changes = ({"frame_index": True}, {"track_id": -1}, {"source_sha256": "bad"},
            {"anchor_origin": "later_reference_box"}, {"frame_width": 101}, {"crop_height": 38},
            {"extractor": "changed"}, {"config": {**guard.CONFIG, "image_order": ["candidate_crop", "full_frame"]}},
            {"candidate_in_crop": {"x1": 9, "y1": 8, "x2": 28, "y2": 28}})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                guard.verify(self.rehash({**deepcopy(self.prepared), **change}))

    def test_rehashed_translated_geometry_cannot_reuse_crop_bytes(self):
        prepared = deepcopy(self.prepared)
        prepared["candidate_box"] = {"x1": 30, "y1": 30, "x2": 50, "y2": 50}
        prepared["crop_box"] = {"x1": 22, "y1": 22, "x2": 58, "y2": 58}
        with self.assertRaises(ValidationError):
            guard.verify(self.rehash(prepared))

    def test_source_or_crop_byte_changes_stop_before_http(self):
        for field in ("source_file", "crop_file"):
            path = Path(self.prepared[field])
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(self.client(opener=lambda *args, **kwargs: self.fail("changed bytes reached HTTP")), self.prepared)
            path.write_bytes(original)

    def test_source_change_during_inference_cannot_return_accepted_evidence(self):
        for field in ("source_file", "crop_file"):
            path = Path(self.prepared[field])
            original = path.read_bytes()
            def opener(*args, **kwargs):
                path.write_bytes(original + b"changed during request")
                return response(provider())
            client = self.client(opener=opener)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(client, self.prepared)
            self.assertEqual(53, client.usage_summary()["output_tokens"])
            path.write_bytes(original)

    def test_invalid_candidate_bounds_types_and_source_binding_do_not_create_output(self):
        variants = [{"candidate_box": {"x1": 20, "y1": 20, "x2": 100, "y2": 40}},
            {"candidate_box": {"x1": True, "y1": 20, "x2": 40, "y2": 40}},
            {"source_sha256": None}, {"frame_index": True}, {"track_id": -1}]
        for change in variants:
            args = {"candidate_box": self.box, "source_sha256": "a" * 64, "frame_index": 17, "track_id": 1, **change}
            output = self.root / "invalid"
            with self.subTest(change=change), self.assertRaises(ValidationError):
                guard.prepare(self.source, args.pop("candidate_box"), contract(), output, **args)
            self.assertFalse(output.exists())

    def test_public_summary_allowlists_classifications_hashes_and_counts(self):
        parsed = clear()
        parsed["physical_visibility"]["visible_terminal_landmarks"][0] = "/Users/private/provider-private-id"
        result = guard.challenge(self.client(parsed), self.prepared)
        result["usage"]["private_injected"] = "/Users/private"
        result["response"]["response"]["id"] = "provider-private-id"
        summary = guard.public_summary(result)
        encoded = json.dumps(summary)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("/Users/", encoded)
        self.assertNotIn("provider-private-id", encoded)
        self.assertNotIn("observation", encoded)
        self.assertEqual("supported", summary["classification"]["decision"])
        self.assertFalse(summary["physical_action_authorized"])
        self.assertEqual(71, summary["usage"]["input_tokens"])

    def test_public_projection_recomputes_completed_evaluation_and_refuses_changed_decision(self):
        result = guard.challenge(self.client(), self.prepared)
        result["evaluation"]["decision"] = "unknown"
        with self.assertRaises(ValidationError):
            guard.public_summary(result)

    def test_mutating_parsed_and_evaluation_together_cannot_rewrite_retained_provider_output(self):
        result = guard.challenge(self.client(), self.prepared)
        result["response"]["parsed"]["candidate_identity"] = "other"
        result["evaluation"] = guard.evaluate(result["response"]["parsed"], contract())
        with self.assertRaises(ValidationError):
            guard.public_summary(result)

    def test_mutating_retained_text_without_the_provider_output_is_rejected(self):
        result = guard.challenge(self.client(), self.prepared)
        result["response"]["text"] = "changed provider answer"
        with self.assertRaises(ValidationError):
            guard.public_summary(result)
        result["status"] = "unscored"
        with self.assertRaises(ValidationError):
            guard.public_summary(result)

    def test_unscored_summary_never_implies_a_physical_classification(self):
        result = guard.challenge(self.client({"missing": "fields"}), self.prepared)
        value = guard.public_summary(result)
        self.assertEqual("unscored", value["status"])
        self.assertIsNone(value["classification"])
        self.assertEqual(53, value["usage"]["output_tokens"])

    def test_assessment_and_evaluation_do_not_mutate_inputs(self):
        prepared, parsed = deepcopy(self.prepared), clear()
        expected = deepcopy(parsed)
        guard.evaluate(parsed, contract())
        result = guard.assess(self.client(parsed), prepared)
        self.assertEqual(self.prepared, prepared)
        self.assertEqual(expected, parsed)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))


if __name__ == "__main__":
    unittest.main()
