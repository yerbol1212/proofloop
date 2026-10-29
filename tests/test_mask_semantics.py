"""Authored composition controls, not semantic model-quality measurements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import struct
import tempfile
import unittest
from unittest.mock import patch
import base64

from proofloop import benchmark, mask_semantics as module
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL, parse_sam_boxes
from tests.test_benchmark import contract
from tests.test_guided import jpeg_header


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded(records):
    result = deepcopy(records)
    for record in result:
        raw = b"\x01" * 25
        record["mask"].update(decoded=True, raster_format="base64_row_major_uint8_0_or_1",
                              raster_base64=base64.b64encode(raw).decode(), foreground_pixels=25,
                              raster_sha256=hashlib.sha256(raw).hexdigest())
    return result


def supported():
    return {"frames": [dict({field: "supported" for field in module.FIELDS},
                            frame_index=i, full_scene_match_count="one") for i in range(2)]}


class Client:
    def __init__(self, mutate=None, error=None):
        self.retries, self.max_requests, self.ledger, self.calls = 0, 1, [], []
        self.mutate, self.error = mutate, error

    def spark_json(self, prompt, schema, **options):
        self.calls.append((prompt, schema, options))
        usage = {"input_tokens": 100, "output_tokens": 50}
        if self.error:
            status, response, code, http = self.error
            entry = {"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1, "status": status}
            if response is not None and "usage" in response:
                entry["usage"] = deepcopy(response["usage"])
            if http:
                entry.update(http_status=http, error_code=code)
            self.ledger.append(entry)
            raise MetaAPIError("Authored private failure", code=code, status=http, response=deepcopy(response))
        value = supported()
        if self.mutate:
            self.mutate(value)
        text = json.dumps(value)
        result = {"response": {"status": "completed", "model": SPARK_MODEL,
                               "output_text": text, "usage": deepcopy(usage), "private_data": "do-not-export"},
                  "source": None, "sources": [{"sha256": sha(path)} for path in options["image_paths"]],
                  "parsed": value, "text": text, "usage": deepcopy(usage)}
        self.ledger.append({"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
                            "status": "ok", "usage": usage})
        return result

    def usage_summary(self):
        replay = MetaClient("authored-not-a-key", retries=0, max_requests=1)
        replay.ledger = deepcopy(self.ledger)
        return replay.usage_summary()


class MaskSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "inputs/local/authored"
        folder.mkdir(parents=True)
        (folder / "video.mp4").write_text("authored-video")
        source = contract()
        source["cases"] = source["cases"][:1]
        self.case = source["cases"][0]
        self.paths = []
        for i, frame in enumerate(self.case["frames"]):
            path = self.root / frame["file"]
            path.write_bytes(jpeg_header(suffix=str(i).encode()))
            frame["sha256"] = sha(path)
            self.paths.append(path)
        self.envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
        self.sam = {}
        for path in self.paths:
            text = ("<0f>1<|box;x1=10;y1=10;x2=14;y2=14;w=100;h=80;c=0.95|>"
                    "<|mask;x=0;y=0;c=0.95;data=5,5,!!!!!(QO(0lu8?|>")
            self.sam[sha(path)] = {"glass": {"source": {"sha256": sha(path)},
                "response": {"status": "completed", "model": SAM_MODEL, "output_text": text},
                "text": text, "objects": decoded(parse_sam_boxes(text)),
                "mask_decoding": "official_meta_sam_parser_0.1.0"}}
        parsed = {"coordinate_system": "normalized_0_1000", "frames": [
            {"frame_index": i, "width": 100, "height": 80, "objects": [{"object_id": "glass",
             "identity_ambiguous": False, "fully_visible": True, "boundary_uncertain": False,
             "candidates": [{"x1": 100, "y1": 125, "x2": 150, "y2": 188}]}]} for i in range(2)]}
        self.locator = {"source": None, "sources": [{"sha256": sha(path)} for path in self.paths],
                        "response": {"status": "completed", "model": SPARK_MODEL, "output_text": json.dumps(parsed)},
                        "text": json.dumps(parsed), "parsed": parsed}
        self.addCleanup(patch.stopall)
        patch("proofloop.mask_semantics.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.tracking_geometry.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.semantic._extractor_path", return_value="fixture-ffmpeg").start()
        self.render_calls = []
        patch("proofloop.semantic.subprocess.run", side_effect=self.render).start()

    def render(self, args, **options):
        self.render_calls.append((args, options))
        if args[-1] == "-version":
            if hasattr(options.get("stdout"), "write"):
                options["stdout"].write(b"authored-ffmpeg-fixture\n")
            return subprocess.CompletedProcess(args, 0, stdout=b"authored-ffmpeg-fixture\n")
        if "-vf" in args:
            geometry = args[args.index("-vf") + 1]
            width, height = map(int, re.search(r"crop=w=(\d+):h=(\d+)", geometry).groups())
            data = Path(args[args.index("-i") + 1]).read_bytes()
            Path(args[-1]).write_bytes(jpeg_header(width, height, hashlib.sha256(data + geometry.encode()).digest()))
        else:
            Path(args[-1]).write_bytes(b"\x89PNG\r\n\x1a\n" + hashlib.sha256(options["input"]).digest())
        return subprocess.CompletedProcess(args, 0, stdout=b"")

    def prepare(self, suffix="prepared"):
        return module.prepare_pair(self.envelope, self.sam, self.locator,
                                   self.root / "artifacts/local" / suffix, root=self.root)

    def locator_change(self, mutate):
        mutate(self.locator["parsed"])
        text = json.dumps(self.locator["parsed"])
        self.locator["text"] = self.locator["response"]["output_text"] = text

    def test_correct_target_preserves_provider_objects_and_scores(self):
        prepared = self.prepare()
        report = module.challenge(Client(), prepared, root=self.root)
        self.assertEqual("completed", report["status"])
        self.assertEqual("supported", module.summarize(report, root=self.root)["decision"])
        self.assertEqual(self.sam, module.supported_results(report, root=self.root))
        self.assertTrue((Path(prepared["output_dir"]) / "challenge-report.json").is_file())

    def test_high_confidence_merged_foreground_veto_is_per_view(self):
        client = Client(lambda value: value["frames"][0].update(mask_target_only="contradicted"))
        report = module.challenge(client, self.prepare(), root=self.root)
        projection = module.supported_results(report, root=self.root)
        self.assertEqual({}, projection[sha(self.paths[0])])
        self.assertEqual(self.sam[sha(self.paths[1])], projection[sha(self.paths[1])])
        self.assertEqual("unknown", report["evaluation"]["decision"])
        scores = [row["glass"]["objects"][0]["confidence"] for row in self.sam.values()]
        self.assertEqual([.95, .95], scores)

    def test_every_semantic_uncertainty_withholds_without_invented_confidence(self):
        for field in module.FIELDS:
            report = module.challenge(Client(lambda value, f=field: value["frames"][0].update({f: "uncertain"})),
                                      self.prepare(field), root=self.root)
            self.assertEqual("unknown", report["evaluation"]["decision"])
            self.assertNotIn("confidence", json.dumps(report["evaluation"]))

    def test_duplicates_are_rejected_by_full_scene_count(self):
        report = module.challenge(Client(lambda value: value["frames"][1].update(full_scene_match_count="multiple")),
                                  self.prepare(), root=self.root)
        self.assertEqual("unknown", report["evaluation"]["decision"])

    def test_cached_occlusion_cannot_be_overridden_or_spend_spark(self):
        self.locator_change(lambda value: value["frames"][0]["objects"][0].update(fully_visible=False))
        prepared = self.prepare()
        self.assertEqual("withheld", prepared["status"])
        client = Client()
        report = module.challenge(client, prepared, root=self.root)
        self.assertEqual([], client.calls)
        self.assertEqual("withheld", module.summarize(report, root=self.root)["status"])
        self.assertTrue(all(not value for value in module.supported_results(report, root=self.root).values()))

    def test_cached_ambiguity_and_candidate_count_remain_vetoes(self):
        for kind in ("ambiguous", "missing", "multiple"):
            def mutate(value):
                obj = value["frames"][0]["objects"][0]
                if kind == "ambiguous":
                    obj["identity_ambiguous"] = True
                elif kind == "missing":
                    obj["candidates"] = []
                else:
                    obj["candidates"] *= 2
            original = deepcopy(self.locator)
            self.locator_change(mutate)
            self.assertEqual("withheld", self.prepare(kind)["status"])
            self.locator = original

    def test_empty_and_multiple_sam_proposals_cannot_choose_preferred_instance(self):
        for count in (0, 2):
            original = deepcopy(self.sam)
            result = self.sam[sha(self.paths[0])]["glass"]
            text = result["text"]
            if count == 0:
                text = ""
            else:
                text += text.replace("<0f>1", "<0f>2").replace("x1=10", "x1=20").replace("x2=14", "x2=24")
            result.update(text=text, objects=decoded(parse_sam_boxes(text)))
            result["response"]["output_text"] = text
            self.assertEqual("withheld", self.prepare(str(count))["status"])
            self.sam = original

    def test_low_sam_score_is_never_upgraded_or_challenged(self):
        for result in self.sam.values():
            value = result["glass"]
            text = value["text"].replace("0.95", "0.79")
            value.update(text=text, objects=decoded(parse_sam_boxes(text)))
            value["response"]["output_text"] = text
        self.assertEqual("withheld", self.prepare()["status"])

    def test_missing_wire_confidence_withholds_without_fabricating_score(self):
        original = self.sam[sha(self.paths[0])]["glass"]
        text = original["text"].replace(";c=0.95", "")
        original.update(text=text, objects=decoded(parse_sam_boxes(text)))
        original["response"]["output_text"] = text
        prepared = self.prepare()
        self.assertEqual("withheld", prepared["status"])
        self.assertIn("sam_score_unavailable", prepared["frames"][0]["withheld_reasons"])
        self.assertIsNone(prepared["sam_results"][sha(self.paths[0])]["glass"]["objects"][0]["confidence"])

    def test_binary_mask_maps_to_source_pixel_cells(self):
        self.prepare()
        raw = next(options["input"] for _, options in self.render_calls if "input" in options)
        self.assertTrue(raw.startswith(b"P5\n100 80\n255\n"))
        pixels = raw.split(b"\n", 3)[3]
        self.assertEqual(8000, len(pixels))
        self.assertEqual(25, pixels.count(255))
        self.assertEqual(255, pixels[10 * 100 + 10])
        self.assertEqual(0, pixels[9 * 100 + 10])
        self.assertEqual(0, pixels[10 * 100 + 15])

    def test_resampled_box_local_mask_uses_source_cell_centers(self):
        record = deepcopy(next(iter(self.sam.values()))["glass"]["objects"][0])
        record["box"].update(x1=10, x2=19, y1=10, y2=14)
        raw = bytes([0, 0, 1, 1, 1] * 5)
        def decode_partial(records):
            value = deepcopy(records)
            value[0]["mask"].update(raster_base64=base64.b64encode(raw).decode(),
                raster_sha256=hashlib.sha256(raw).hexdigest(), foreground_pixels=15)
            return value
        with patch("proofloop.tracking_geometry.decode_sam_masks", side_effect=decode_partial):
            module._mask_png("fixture-ffmpeg", self.root / "resampled.png", record, 100, 80)
        pixels = self.render_calls[-1][1]["input"].split(b"\n", 3)[3]
        self.assertEqual(30, pixels.count(255))
        self.assertEqual(0, pixels[10 * 100 + 13])
        self.assertEqual(255, pixels[10 * 100 + 14])
        self.assertEqual(255, pixels[14 * 100 + 19])
        self.assertEqual(0, pixels[15 * 100 + 19])

    def test_offsource_mask_coordinates_are_rejected_without_rendering(self):
        record = deepcopy(next(iter(self.sam.values()))["glass"]["objects"][0])
        for edge in (-1, 100):
            record["box"].update(x1=edge, x2=edge + 4)
            count = len(self.render_calls)
            with self.assertRaises(ValidationError):
                module._mask_png("fixture-ffmpeg", self.root / "invalid.png", record, 100, 80)
            self.assertEqual(count, len(self.render_calls))

    def test_media_processes_have_timeout_and_no_captured_error_stream(self):
        self.prepare()
        for args, options in self.render_calls:
            self.assertGreater(options["timeout"], 0)
            self.assertLessEqual(options["timeout"], 15)
            self.assertEqual(subprocess.DEVNULL, options["stderr"])
            if args[-1] != "-version":
                self.assertEqual(subprocess.DEVNULL, options["stdout"])

    def test_crop_or_mask_timeout_does_not_create_eligible_preparation(self):
        for mask in (False, True):
            def timeout(args, **options):
                if ("-vf" not in args and args[-1] != "-version") == mask and args[-1] != "-version":
                    raise subprocess.TimeoutExpired(args, options["timeout"])
                return self.render(args, **options)
            with patch("proofloop.mask_semantics.subprocess.run", side_effect=timeout), self.assertRaises(ValidationError):
                self.prepare("timeout" + str(mask))

    def test_prompt_contains_only_concepts_and_actual_derived_geometry(self):
        prepared = self.prepare()
        prompt = module.challenge_prompt(prepared)
        self.assertIn('"target_concepts": {"glass": "glass"}', prompt)
        self.assertIn("NOT additional photographs", prompt)
        for value in (self.envelope["contract"]["requirements"], '"annotation"', '"regions"', '"checks"', "dispense", "staging"):
            self.assertNotIn(value, prompt)
        client = Client()
        module.challenge(client, prepared, root=self.root)
        options = client.calls[0][2]
        self.assertEqual(6, len(options["image_paths"]))
        self.assertEqual("low", options["reasoning_effort"])
        self.assertEqual(4096, options["max_output_tokens"])

    def test_raw_sam_text_cannot_disagree_with_parsed_or_decoded_copy(self):
        for kind in ("score", "raster", "source"):
            original = deepcopy(self.sam)
            value = self.sam[sha(self.paths[0])]["glass"]
            if kind == "score":
                value["objects"][0]["confidence"] = .99
            elif kind == "raster":
                value["objects"][0]["mask"]["raster_sha256"] = "0" * 64
            else:
                value["source"]["sha256"] = "0" * 64
            with self.assertRaises(ValidationError):
                self.prepare(kind)
            self.sam = original

    def test_retained_locator_text_cannot_disagree_with_parsed_copy(self):
        self.locator["parsed"]["frames"][0]["objects"][0]["fully_visible"] = False
        with self.assertRaises(ValidationError):
            self.prepare()

    def test_corrupt_source_crop_or_mask_rejected_before_paid_call(self):
        for kind in ("source_file", "crop_file", "mask_file"):
            prepared = self.prepare(kind)
            path = Path(prepared["frames"][0][kind])
            old = path.read_bytes()
            path.write_bytes(old + b"tamper")
            client = Client()
            with self.assertRaises(ValidationError):
                module.challenge(client, prepared, root=self.root)
            self.assertEqual([], client.calls)
            path.write_bytes(old)

    def test_response_parsed_copy_tampering_cannot_change_veto(self):
        report = module.challenge(Client(), self.prepare(), root=self.root)
        report["challenger"]["parsed"]["frames"][0]["mask_target_only"] = "contradicted"
        with self.assertRaises(ValidationError):
            module.summarize(report, root=self.root)

    def test_duplicate_json_fields_and_nonfinite_encodings_rejected(self):
        for text in ('{"frames":[],"frames":[]}', '{"frames":NaN}'):
            report = module.challenge(Client(), self.prepare(hashlib.sha256(text.encode()).hexdigest()), root=self.root)
            report["challenger"]["response"]["output_text"] = text
            report["challenger"]["text"] = text
            with self.assertRaises(ValidationError):
                module.summarize(report, root=self.root)

    def test_failures_remain_unscored_with_private_body_and_token_cost(self):
        errors = [("network_error", None, "network_error", None),
                  ("http_error", {"error": {"code": "rate_limit", "message": "do-not-export"}}, "rate_limit", 429),
                  ("invalid_response", {"status": "incomplete", "model": SPARK_MODEL,
                    "usage": {"input_tokens": 100, "output_tokens": 4096}}, "incomplete_response", None)]
        for i, error in enumerate(errors):
            client = Client(error=error)
            report = module.challenge(client, self.prepare("error" + str(i)), root=self.root)
            summary = module.summarize(report, root=self.root)
            self.assertEqual("unscored", summary["status"])
            self.assertIsNone(summary["decision"])
            self.assertEqual(1, summary["usage"]["request_attempts"])
            self.assertTrue(summary["usage"]["unreported_charge_possible"])
            self.assertEqual(1, len(client.calls))
            self.assertNotIn("do-not-export", json.dumps(summary))
            self.assertTrue(all(not value for value in module.supported_results(report, root=self.root).values()))
            if i == 2:
                self.assertEqual(4096, summary["usage"]["output_tokens"])
                self.assertGreater(summary["usage"]["estimated_usd"], 0)

    def test_ledger_cannot_be_erased_to_claim_free_failed_attempt(self):
        report = module.challenge(Client(error=("network_error", None, "network_error", None)),
                                  self.prepare(), root=self.root)
        report["usage"]["ledger"] = []
        with self.assertRaises(ValidationError):
            module.summarize(report, root=self.root)

    def test_entire_usage_replacement_cannot_erase_known_network_attempt(self):
        report = module.challenge(Client(error=("network_error", None, "network_error", None)),
                                  self.prepare(), root=self.root)
        report["usage"] = module._usage([])
        with self.assertRaisesRegex(ValidationError, "recorded attempt"):
            module.summarize(report, root=self.root)

    def test_locally_rejected_billed_response_cannot_be_replayed_as_zero_tokens(self):
        report = module.challenge(Client(lambda value: value["frames"][0].update(identity="bogus")),
                                  self.prepare(), root=self.root)
        self.assertEqual("unscored", report["status"])
        self.assertEqual("local_validation_failed", report["error"]["code"])
        self.assertEqual({"input_tokens": 100, "output_tokens": 50}, report["error"]["response"]["usage"])
        report["usage"] = module._usage([{"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
            "status": "ok", "usage": {"input_tokens": 0, "output_tokens": 0}}])
        with self.assertRaisesRegex(ValidationError, "token usage"):
            module.summarize(report, root=self.root)

    def test_local_rejection_cannot_substitute_a_different_error_response(self):
        report = module.challenge(Client(lambda value: value["frames"][0].update(identity="bogus")),
                                  self.prepare(), root=self.root)
        response = deepcopy(report["error"]["response"])
        response["output_text"] = "different retained provider output"
        report["error"]["response"], report["error"]["response_receipt"] = module._retain_error(response, report["prepared"])
        with self.assertRaisesRegex(ValidationError, "actual challenger response"):
            module.summarize(report, root=self.root)

    def test_local_pre_transport_error_retains_explicit_charge_uncertainty(self):
        client = Client()
        client.spark_json = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("authored local failure"))
        report = module.challenge(client, self.prepare(), root=self.root)
        summary = module.summarize(report, root=self.root)
        self.assertEqual("unscored", summary["status"])
        self.assertTrue(summary["usage"]["accounting_incomplete"])
        self.assertTrue(summary["usage"]["unreported_charge_possible"])
        self.assertIsNone(summary["usage"]["estimated_usd"])
        report["usage"] = module._usage([])
        with self.assertRaises(ValidationError):
            module.summarize(report, root=self.root)

    def test_spatial_control_origin_is_truthful_and_bound_separately(self):
        prepared = module.prepare_pair(self.envelope, self.sam, self.locator,
            self.root / "artifacts/local/spatial-control", root=self.root,
            proposal_origin="retained_spatial_sam_control")
        report = module.challenge(Client(), prepared, root=self.root)
        summary = module.summarize(report, root=self.root)
        self.assertEqual("retained_spatial_sam_control", summary["anchor_origin"])
        prepared["anchor_origin"] = "sam_concept_proposal"
        with self.assertRaises(ValidationError):
            module.validate_prepared(prepared, root=self.root)

    def test_unsupported_proposal_origin_rejected_without_media_work(self):
        with self.assertRaises(ValidationError):
            module.prepare_pair(self.envelope, self.sam, self.locator,
                self.root / "artifacts/local/invalid-origin", root=self.root, proposal_origin="authored_candidate")
        self.assertEqual([], self.render_calls)

    def test_oversized_failure_body_has_bounded_receipt_and_keeps_tokens(self):
        response = {"status": "incomplete", "model": SPARK_MODEL,
                    "usage": {"input_tokens": 100, "output_tokens": 4096},
                    "private_padding": "x" * module.MAX_ERROR_BYTES}
        client = Client(error=("invalid_response", response, "incomplete_response", None))
        report = module.challenge(client, self.prepare(), root=self.root)
        summary = module.summarize(report, root=self.root)
        self.assertIsNone(report["error"]["response"])
        self.assertFalse(summary["failed_response_retention"]["retained"])
        self.assertGreater(summary["failed_response_retention"]["bytes"], module.MAX_ERROR_BYTES)
        self.assertEqual(4096, summary["usage"]["output_tokens"])
        self.assertLess((Path(report["prepared"]["output_dir"]) / "challenge-report.json").stat().st_size, 100000)

    def test_completed_assessment_cannot_be_relabelled_unscored(self):
        report = module.challenge(Client(), self.prepare(), root=self.root)
        report.update(status="unscored", evaluation=None,
                      error={"code": "local_validation_failed", "http_status": None,
                             "response": deepcopy(report["challenger"]["response"])})
        with self.assertRaises(ValidationError):
            module.summarize(report, root=self.root)

    def test_six_attachments_remain_exact_and_in_order(self):
        report = module.challenge(Client(), self.prepare(), root=self.root)
        report["challenger"]["sources"].reverse()
        with self.assertRaises(ValidationError):
            module.summarize(report, root=self.root)

    def test_single_call_marker_blocks_repeat_spend(self):
        prepared = self.prepare()
        module.challenge(Client(), prepared, root=self.root)
        client = Client()
        with self.assertRaises(FileExistsError):
            module.challenge(client, prepared, root=self.root)
        self.assertEqual([], client.calls)

    def test_nonfresh_client_rejected_before_call(self):
        prepared = self.prepare()
        for mutation in (lambda c: setattr(c, "max_requests", 2), lambda c: setattr(c, "retries", 1),
                         lambda c: c.ledger.append({})):
            client = Client()
            mutation(client)
            with self.assertRaises(ValidationError):
                module.challenge(client, prepared, root=self.root)
            self.assertEqual([], client.calls)


class MaskSemanticsRealCodecTests(unittest.TestCase):
    def test_actual_ffmpeg_and_official_parser_reproduce_full_source_mask(self):
        try:
            import meta_sam_parser  # noqa: F401
        except (ImportError, TypeError, SyntaxError):
            self.skipTest("Optional official SAM parser requires Python >=3.10")
        executable = module.semantic._extractor_path()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = contract()
            source["cases"] = source["cases"][:1]
            folder = root / "inputs/local/authored"
            folder.mkdir(parents=True)
            (folder / "video.mp4").write_text("authored-video")
            sam, paths = {}, []
            for i, frame in enumerate(source["cases"][0]["frames"]):
                path = root / frame["file"]
                subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                f"color=c={'red' if i == 0 else 'blue'}:s=100x80", "-frames:v", "1",
                                "-threads", "1", "-map_metadata", "-1", str(path)], check=True, capture_output=True)
                paths.append(path)
                frame["sha256"] = sha(path)
                text = ("<0f>1<|box;x1=10;y1=10;x2=14;y2=14;w=100;h=80;c=0.95|>"
                        "<|mask;x=0;y=0;c=0.95;data=5,5,!!!!!(QO(0lu8?|>")
                sam[sha(path)] = {"glass": {"source": {"sha256": sha(path)},
                    "response": {"status": "completed", "model": SAM_MODEL, "output_text": text},
                    "text": text, "objects": module.decode_sam_masks(parse_sam_boxes(text)),
                    "mask_decoding": "official_meta_sam_parser_0.1.0"}}
            parsed = {"coordinate_system": "normalized_0_1000", "frames": [
                {"frame_index": i, "width": 100, "height": 80, "objects": [{"object_id": "glass",
                 "identity_ambiguous": False, "fully_visible": True, "boundary_uncertain": False,
                 "candidates": [{"x1": 100, "y1": 125, "x2": 150, "y2": 188}]}]} for i in range(2)]}
            locator = {"source": None, "sources": [{"sha256": sha(path)} for path in paths],
                       "response": {"status": "completed", "model": SPARK_MODEL, "output_text": json.dumps(parsed)},
                       "text": json.dumps(parsed), "parsed": parsed}
            envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
            prepared = module.prepare_pair(envelope, sam, locator, root / "artifacts/local/prepared", root=root)
            self.assertEqual("eligible", prepared["status"])
            for frame in prepared["frames"]:
                png = Path(frame["mask_file"]).read_bytes()
                self.assertEqual(b"\x89PNG\r\n\x1a\n", png[:8])
                self.assertEqual((100, 80), struct.unpack(">II", png[16:24]))
            report = module.challenge(Client(), prepared, root=root)
            self.assertEqual("supported", module.summarize(report, root=root)["decision"])


if __name__ == "__main__":
    unittest.main()
