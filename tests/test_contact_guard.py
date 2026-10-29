"""Authored contract/transport fixtures; never model accuracy measurements."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from proofloop import contact_guard as guard
from proofloop import visibility_guard as native
from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SPARK_MODEL
from tests.test_guided import jpeg_header
from tests.test_visibility_guard import clear as native_clear, contract


def clear():
    result = native_clear()
    result["boundary_relations"] = {side: {"relation": "clear_background_gap", "neighbor": "robot gripper",
        "observation": "A visible dark background gap separates this target outline from the adjacent gripper."}
        for side in guard.SIDES}
    return result


def provider(parsed=None, *, status="completed"):
    return {"model": SPARK_MODEL, "status": status,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(parsed or clear())}]}],
        "usage": {"input_tokens": 91, "output_tokens": 83}}


def response(value):
    return BytesIO(json.dumps(value).encode())


def png_header(width, height, suffix=b""):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + suffix


class ContactGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "frame.jpg"
        self.source.write_bytes(jpeg_header(width=100, height=80, suffix=b"authored-not-photographic"))
        self.box = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}
        def crop(executable, source, target, box):
            suffix = guard.digest({"source": hashlib.sha256(source.read_bytes()).hexdigest(), "box": box}).encode()
            target.write_bytes(jpeg_header(width=box["x2"] - box["x1"] + 1,
                height=box["y2"] - box["y1"] + 1, suffix=suffix))
        def enlarge(executable, source, target):
            width, height = native._jpeg_dimensions(source.read_bytes())
            target.write_bytes(png_header(width * 3, height * 3, hashlib.sha256(source.read_bytes()).digest()))
        for module, name, replacement in ((native, "_crop", crop), (native, "_extractor_path", lambda: "authored-ffmpeg"),
                (native, "_extractor_version", lambda executable: "authored-ffmpeg-version"),
                (guard, "_enlarge", enlarge)):
            override = patch.object(module, name, side_effect=replacement)
            override.start()
            self.addCleanup(override.stop)
        self.prepared = guard.prepare(self.source, self.box, contract(), self.root / "prepared",
            source_sha256="a" * 64, frame_index=17, track_id=1)

    def client(self, parsed=None, *, opener=None):
        return MetaClient("authored-not-a-real-key", max_requests=1,
            opener=opener or (lambda *args, **kwargs: response(provider(parsed))))

    def rehash(self, value):
        value["prepared_hash"] = guard.digest({key: item for key, item in value.items() if key != "prepared_hash"})
        return value

    def test_preparation_binds_native_images_and_exact_3x_inclusive_geometry(self):
        prepared = self.prepared
        self.assertEqual(guard.VERSION, prepared["method_version"])
        self.assertEqual((37, 37), (prepared["crop_width"], prepared["crop_height"]))
        self.assertEqual((111, 111), (prepared["enlargement_width"], prepared["enlargement_height"]))
        self.assertEqual({"x1": 24, "y1": 24, "x2": 86, "y2": 86}, prepared["candidate_in_enlargement"])
        self.assertNotEqual(prepared["native_preparation_hash"], prepared["prepared_hash"])
        self.assertFalse(prepared["config"]["enlargement"]["adds_source_detail"])
        guard.verify(prepared)

    def test_one_low_4096_request_orders_exact_frame_crop_enlargement(self):
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
        self.assertEqual(["input_text", "input_image", "input_image", "input_image"],
            [item["type"] for item in payload["input"][0]["content"]])
        self.assertEqual([self.prepared[field] for field in ("frame_sha256", "crop_sha256", "enlargement_sha256")],
            [item["sha256"] for item in result["response"]["sources"]])
        self.assertEqual(83, result["usage"]["output_tokens"])

    def test_prompt_is_reference_free_and_does_not_invent_boundary_detail(self):
        value = guard.prompt(self.prepared)
        for text in ("NO new detail", "NOT the proposal rectangle", "white-on-white", "visible dark gap",
                     "not proof", "all four relations", "prior identity", "Do not invent a gap"):
            self.assertIn(text, value)
        self.assertNotIn(str(self.root), value)
        self.assertNotIn("expected_label", value)
        self.assertNotIn("episode_", value)

    def test_adjacent_gripper_with_visible_gap_can_support(self):
        result = guard.evaluate(clear(), contract())
        self.assertEqual("supported", result["decision"])
        self.assertTrue(result["fully_visible"])
        self.assertEqual([], result["consistency_issues"])

    def test_distinct_continuous_outline_does_not_need_a_background_gap(self):
        parsed = clear()
        parsed["boundary_relations"]["left"].update(relation="visible_target_boundary", neighbor="",
            observation="The body outline remains continuously visible along a contrasting surface.")
        self.assertEqual("supported", guard.evaluate(parsed, contract())["decision"])

    def test_unresolved_low_contrast_contact_never_becomes_clear_or_asserted_occlusion(self):
        parsed = clear()
        parsed["boundary_relations"]["right"].update(relation="unresolved_contact", neighbor="white robot gripper",
            observation="The two white surfaces meet without a directly resolvable outline.")
        result = guard.evaluate(parsed, contract())
        self.assertEqual("unknown", result["decision"])
        self.assertFalse(result["fully_visible"])
        self.assertFalse(result["physical_visibility"]["external_occluder_hides_target"])
        self.assertIn("clear_visibility_conflicts_with_boundary_relations", result["consistency_issues"])
        parsed["physical_visibility"].update(classification="uncertain", visible_boundary="uncertain",
            full_frame_resolves_entire_extent=False)
        self.assertEqual([], guard.evaluate(parsed, contract())["consistency_issues"])

    def test_every_unresolved_region_independently_withholds_support(self):
        for side in guard.SIDES:
            for relation in ("external_overlap", "unresolved_contact", "uncertain"):
                parsed = clear()
                parsed["boundary_relations"][side]["relation"] = relation
                with self.subTest(side=side, relation=relation):
                    result = guard.evaluate(parsed, contract())
                    self.assertEqual("unknown", result["decision"])
                    self.assertFalse(result["fully_visible"])

    def test_visible_external_gripping_overlap_still_blocks_recognizable_target(self):
        parsed = clear()
        parsed["boundary_relations"]["left"]["relation"] = "external_overlap"
        parsed["physical_visibility"].update(classification="occluded", visible_boundary="interrupted",
            visible_extremities="hidden", external_occluder_hides_target=True, full_frame_resolves_entire_extent=False,
            occluder="robot gripper", affected_boundary="left handle end")
        result = guard.evaluate(parsed, contract())
        self.assertEqual("unknown", result["decision"])
        self.assertEqual([], result["consistency_issues"])
        self.assertFalse(result["fully_visible"])

    def test_contact_requires_named_neighbor_and_consistent_physical_overlap(self):
        parsed = clear()
        parsed["boundary_relations"]["top"].update(relation="external_overlap", neighbor=" ")
        result = guard.evaluate(parsed, contract())
        self.assertIn("top_contact_lacks_named_neighbor", result["consistency_issues"])
        self.assertIn("top_overlap_conflicts_with_physical_visibility", result["consistency_issues"])

    def test_resolved_boundaries_do_not_override_blur_identity_or_occlusion_failure(self):
        for field, value in (("blur_prevents_assessment", "yes"), ("visible_extremities", "hidden"),
                ("frame_truncated", True), ("external_occluder_hides_target", True)):
            parsed = clear()
            parsed["physical_visibility"][field] = value
            with self.subTest(field=field):
                self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])
        parsed = clear()
        parsed["attributes"]["white_body"] = "mismatch"
        self.assertEqual("contradicted", guard.evaluate(parsed, contract())["decision"])
        parsed = clear()
        parsed["attributes"] = {key: "normal_pose_not_visible" for key in parsed["attributes"]}
        self.assertEqual("unknown", guard.evaluate(parsed, contract())["decision"])

    def test_schema_requires_each_bounded_boundary_and_nonblank_observation(self):
        variants = []
        for side in guard.SIDES:
            parsed = clear()
            del parsed["boundary_relations"][side]
            variants.append(parsed)
        for field, value in (("relation", "touching_but_probably_fine"), ("neighbor", "x" * 121),
                ("observation", "x" * 201), ("observation", " ")):
            parsed = clear()
            parsed["boundary_relations"]["right"][field] = value
            variants.append(parsed)
        parsed = clear()
        parsed["boundary_relations"]["extra"] = parsed["boundary_relations"]["right"]
        variants.append(parsed)
        for parsed in variants:
            with self.subTest(parsed=parsed), self.assertRaises(ValueError):
                guard.evaluate(parsed, contract())

    def test_physical_boundary_diagnostic_stays_independent_of_identity(self):
        parsed = clear()
        parsed["direct_distinctive_evidence"] = False
        result = guard.evaluate(parsed, contract())
        self.assertEqual("unknown", result["decision"])
        self.assertTrue(result["fully_visible"])
        self.assertTrue(result["all_projected_boundary_regions_resolved"])

    def test_incomplete_schema_and_network_failures_stay_unscored_no_retry(self):
        for kind in ("incomplete", "schema", "network"):
            calls = []
            def opener(*args, **kwargs):
                calls.append(1)
                if kind == "network":
                    raise URLError("authored uncertainty")
                return response(provider({"missing": "fields"} if kind == "schema" else None,
                    status="incomplete" if kind == "incomplete" else "completed"))
            result = guard.challenge(self.client(opener=opener), self.prepared)
            with self.subTest(kind=kind):
                self.assertEqual("unscored", result["status"])
                self.assertIsNone(result["evaluation"])
                self.assertEqual(1, len(calls))
                self.assertEqual(1, result["usage"]["request_attempts"])
                if kind != "network":
                    self.assertEqual(83, result["usage"]["output_tokens"])

    def test_used_retrying_or_multirequest_client_stops_before_network(self):
        for key, value in (("max_requests", 2), ("retries", 1), ("ledger", [{"used": True}])):
            client = self.client(opener=lambda *args, **kwargs: self.fail("invalid client reached HTTP"))
            setattr(client, key, value)
            with self.subTest(key=key), self.assertRaises(ValidationError):
                guard.challenge(client, self.prepared)

    def test_changed_native_or_enlarged_pixels_stop_before_network(self):
        for field in ("source_file", "crop_file", "enlargement_file"):
            path = Path(self.prepared[field])
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(self.client(opener=lambda *args, **kwargs: self.fail("changed bytes reached HTTP")), self.prepared)
            path.write_bytes(original)

    def test_rehashing_changed_enlargement_bytes_cannot_forge_derivation(self):
        value = deepcopy(self.prepared)
        path = Path(value["enlargement_file"])
        path.write_bytes(path.read_bytes() + b"not the registered source crop")
        value["enlargement_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValidationError):
            guard.verify(self.rehash(value))

    def test_enlargement_geometry_config_hash_and_native_binding_fail_closed(self):
        variants = [{"enlargement_width": 112}, {"native_preparation_hash": "b" * 64},
            {"config": {**guard.CONFIG, "image_order": ["crop_enlargement", "candidate_crop", "full_frame"]}},
            {"candidate_in_enlargement": {"x1": 24, "y1": 24, "x2": 84, "y2": 84}},
            {"frame_index": 18}, {"candidate_box": {"x1": 21, "y1": 20, "x2": 40, "y2": 40}}]
        for change in variants:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                guard.verify(self.rehash({**deepcopy(self.prepared), **change}))

    def test_all_three_registered_sources_and_completed_model_are_required(self):
        for change in ("order", "missing", "hash", "model", "status", "single_source"):
            client = self.client()
            original = client.spark_json
            def altered(*args, **kwargs):
                value = original(*args, **kwargs)
                if change == "order":
                    value["sources"].reverse()
                elif change == "missing":
                    value["sources"].pop()
                elif change == "hash":
                    value["sources"][2]["sha256"] = "b" * 64
                elif change == "single_source":
                    value["source"] = value["sources"][0]
                elif change == "model":
                    value["response"]["model"] = "other"
                else:
                    value["response"]["status"] = "incomplete"
                return value
            with self.subTest(change=change), patch.object(client, "spark_json", side_effect=altered):
                result = guard.challenge(client, self.prepared)
                self.assertEqual("unscored", result["status"])
                self.assertEqual("local_validation_failed", result["error"]["code"])

    def test_changes_during_request_cannot_return_accepted_evidence(self):
        path = Path(self.prepared["enlargement_file"])
        def opener(*args, **kwargs):
            path.write_bytes(path.read_bytes() + b"changed during request")
            return response(provider())
        client = self.client(opener=opener)
        with self.assertRaises(ValidationError):
            guard.challenge(client, self.prepared)
        self.assertEqual(83, client.usage_summary()["output_tokens"])

    def test_public_summary_hashes_all_model_named_boundary_evidence(self):
        parsed = clear()
        for boundary in parsed["boundary_relations"].values():
            boundary.update(neighbor="/Users/private/provider-secret-id", observation="/Users/private/provider-secret-id")
        parsed["physical_visibility"]["visible_terminal_landmarks"][0] = "/Users/private/provider-secret-id"
        result = guard.challenge(self.client(parsed), self.prepared)
        result["usage"]["private"] = "/Users/private/provider-secret-id"
        result["response"]["response"]["id"] = "provider-secret-id"
        public = guard.public_summary(result)
        encoded = json.dumps(public)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("/Users/", encoded)
        self.assertNotIn("provider-secret-id", encoded)
        boundary = public["classification"]["boundary_relations"]["right"]
        self.assertEqual("clear_background_gap", boundary["relation"])
        self.assertNotIn("observation", boundary)
        self.assertEqual(64, len(boundary["observation_sha256"]))
        self.assertFalse(public["physical_action_authorized"])

    def test_mutable_classification_and_parsed_copies_cannot_rewrite_provider_output(self):
        result = guard.challenge(self.client(), self.prepared)
        result["evaluation"]["decision"] = "unknown"
        with self.assertRaises(ValidationError):
            guard.public_summary(result)
        result = guard.challenge(self.client(), self.prepared)
        result["response"]["parsed"]["boundary_relations"]["right"]["relation"] = "unresolved_contact"
        result["evaluation"] = guard.evaluate(result["response"]["parsed"], contract())
        with self.assertRaises(ValidationError):
            guard.public_summary(result)

    def test_unscored_summary_has_no_classification_and_inputs_stay_unchanged(self):
        prepared, parsed = deepcopy(self.prepared), clear()
        expected = deepcopy(parsed)
        guard.evaluate(parsed, contract())
        result = guard.challenge(self.client({"missing": "fields"}), prepared)
        public = guard.public_summary(result)
        self.assertEqual("unscored", public["status"])
        self.assertIsNone(public["classification"])
        self.assertEqual(83, public["usage"]["output_tokens"])
        self.assertEqual(self.prepared, prepared)
        self.assertEqual(expected, parsed)


class ActualNearestEnlargementTests(unittest.TestCase):
    def test_enlargement_repeats_each_decoded_source_pixel_in_3x3_blocks(self):
        try:
            executable = native._extractor_path()
        except ValidationError:
            self.skipTest("FFmpeg unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            colors = [b"\xff\x00\x00", b"\x00\xff\x00", b"\x00\x00\xff", b"\xff\xff\xff"]
            source, target = root / "authored.ppm", root / "nearest.png"
            source.write_bytes(b"P6\n2 2\n255\n" + b"".join(colors))
            guard._enlarge(executable, source, target)
            self.assertEqual((6, 6), guard._png_dimensions(target.read_bytes()))
            output = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(target),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"],
                check=True, capture_output=True, timeout=20).stdout
            expected = b"".join(colors[(y // 3) * 2 + x // 3] for y in range(6) for x in range(6))
            self.assertEqual(expected, output)


if __name__ == "__main__":
    unittest.main()
