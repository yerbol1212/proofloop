"""Authored adapter fixtures only; no licensed recording or live API inferences."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.benchmark import (approve_manifest, baseline_prompt, comparison_metrics, evaluate_guided, evaluate_sam,
                                 evaluate_spark, manifest_hash, public_summary, request_requirements,
                                 run_benchmark, validate_manifest, verify_media)
from proofloop.core import Protocol, ValidationError
from proofloop.meta import MetaAPIError, SAM_MODEL, SPARK_MODEL


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def task(kind="location", age=30000):
    check = {"id": "glass-dispense", "kind": kind, "object_id": "glass", "region_id": "dispense",
             "max_age_ms": age}
    check.update({"unique_target": True} if kind == "location" else {"expected": True})
    return Protocol.from_dict({"id": "authored-recorded-test", "version": "1", "objects": ["glass"],
                               "regions": ["dispense", "staging"], "checks": [check]})


def contract(*, kind="location", age=30000):
    cases = []
    for identifier, times, status in (("initial", (500, 1500), "fail"), ("final", (20000, 21000), "pass")):
        cases.append({"id": identifier, "source_id": "authored-source", "role": "benchmark",
                      "regions": {"dispense": [0, 0, .45, 1], "staging": [.55, 0, 1, 1]},
                      "frames": [{"file": f"inputs/local/authored/{identifier}-{time}.jpg",
                                  "sha256": digest(f"{identifier}-{time}"), "source_timestamp_ms": time}
                                 for time in times], "evaluation_timestamp_ms": times[-1],
                      "annotation": {"status": status, "checks": {"glass-dispense": status},
                                     "reason": "Authored test label; not a real visual review.",
                                     "origin": "source_frames_reviewed_before_inference"}})
    return {"schema_version": "1.0", "evidence_origin": "external_recorded_video", "dataset_id": "authored-test",
            "sources": [{"id": "authored-source", "publisher": "Authored unit-test fixture",
                         "page_url": "https://example.com/test", "media_url": "https://example.com/test.mp4",
                         "license": {"name": "authored fixture", "url": "https://example.com/license",
                                     "attribution": "Unit-test author"},
                         "file": "inputs/local/authored/video.mp4", "sha256": digest("authored-video"),
                         "duration_ms": 22000}], "requirements": "A declared unique glass belongs in dispense.",
            "protocol": task(kind, age).to_dict(), "concepts": {"glass": "glass"}, "cases": cases}


def approved(value=None):
    value = value or contract()
    return approve_manifest(value, manifest_hash(value), confirm_unique_targets=True)


def result(frame_hash, region="dispense", *, empty=False, confidence=.98):
    x1, x2 = (10, 30) if region == "dispense" else (60, 80)
    raster = b"\x01" * 6
    return {"source": {"sha256": frame_hash, "mime": "image/jpeg", "bytes": 100},
            "response": {"status": "completed", "model": SAM_MODEL, "id": "authored-sam-shape"},
            "mask_decoding": "official_meta_sam_parser_0.1.0", "objects": [] if empty else [
                {"frame": 0, "object_id": 0, "confidence": confidence,
                 "box": {"x1": x1, "y1": 10, "x2": x2, "y2": 30, "width": 100, "height": 100},
                 "mask": {"decoded": True, "width": 2, "height": 3,
                          "raster_format": "base64_row_major_uint8_0_or_1",
                          "raster_base64": base64.b64encode(raster).decode(),
                          "foreground_pixels": 6, "raster_sha256": hashlib.sha256(raster).hexdigest()}}]}


def sam_results(envelope, *, empty=False):
    return {case["id"]: {frame["sha256"]: {"glass": result(frame["sha256"],
                                "staging" if case["id"] == "initial" else "dispense", empty=empty)}
                         for frame in case["frames"]} for case in envelope["contract"]["cases"]}


def spark_result(case, status="pass"):
    return {"sources": [{"sha256": frame["sha256"]} for frame in case["frames"]],
            "response": {"status": "completed", "model": SPARK_MODEL},
            "parsed": {"checks": [{"check_id": "glass-dispense", "status": status,
                                    "reason": "Authored baseline test fixture"}]}}


def media_fixture(root, envelope):
    value = deepcopy(envelope["contract"])
    records = value["sources"] + [frame for case in value["cases"] for frame in case["frames"]]
    for index, record in enumerate(records):
        data = b"authored-video-transport" if index == 0 else b"\xff\xd8\xff" + f"authored-image-{index}".encode()
        path = root / record["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
    return approved(value)


class RecordedBenchmarkTests(unittest.TestCase):
    def test_manifest_freezes_all_inputs_and_explicit_request_budget(self):
        envelope = approved()
        summary = validate_manifest(envelope, verify_files=False)
        self.assertEqual(2, summary["benchmark_cases"])
        self.assertEqual({"sam": 4, "spark": 2}, request_requirements(envelope))
        self.assertFalse(summary["media_hashes_checked"])
        changed = deepcopy(envelope)
        changed["contract"]["cases"][0]["regions"]["dispense"][2] = .4
        with self.assertRaises(ValidationError):
            validate_manifest(changed, verify_files=False)

    def test_assistant_review_is_not_labeled_human_review(self):
        value = contract()
        with self.assertRaises(ValidationError):
            approve_manifest(value, manifest_hash(value), confirm_unique_targets=True,
                             review_origin="assistant_under_user_authorization")
        envelope = approve_manifest(value, manifest_hash(value), confirm_unique_targets=True,
                                    review_origin="assistant_under_user_authorization", authorization_reference="User authorized tests")
        self.assertFalse(envelope["approval"]["human_reviewed_exact_contract"])

    def test_location_requires_declared_uniqueness(self):
        value = contract()
        with self.assertRaises(ValidationError):
            approve_manifest(value, manifest_hash(value))

    def test_labels_are_predeclared_complete_and_consistent(self):
        for mutation in (lambda c: c["cases"][0]["annotation"].update(origin="sam_prediction"),
                         lambda c: c["cases"][0]["annotation"]["checks"].clear(),
                         lambda c: c["cases"][0]["annotation"].update(status="pass")):
            value = contract()
            mutation(value)
            with self.assertRaises(ValidationError):
                approved(value)

    def test_illustrative_case_cannot_sneak_into_accuracy(self):
        value = contract()
        value["cases"][0].update(role="illustrative", annotation=None)
        envelope = approved(value)
        metrics = comparison_metrics(envelope, [{"case_id": "initial", "status": "pass"},
                                                 {"case_id": "final", "status": "pass"}])
        self.assertEqual(1, metrics["eligible_cases"])
        self.assertEqual(1, metrics["illustrative_cases_excluded"])
        self.assertEqual(0, metrics["incorrect_passes"])
        value["cases"][0]["annotation"] = contract()["cases"][0]["annotation"]
        with self.assertRaises(ValidationError):
            approved(value)

    def test_duplicate_or_subsecond_sources_are_rejected(self):
        for mutation in (lambda c: c["cases"][0]["frames"][1].update(sha256=c["cases"][0]["frames"][0]["sha256"]),
                         lambda c: c["cases"][1]["frames"][0].update(sha256=c["cases"][0]["frames"][0]["sha256"]),
                         lambda c: c["cases"][0]["frames"][1].update(source_timestamp_ms=600)):
            value = contract()
            mutation(value)
            with self.assertRaises(ValidationError):
                approved(value)

    def test_invalid_time_calibration_license_and_unsupported_exact_counts_rejected(self):
        for mutation in (lambda c: c["cases"][0]["regions"].update(staging=[.3, 0, 1, 1]),
                         lambda c: c["sources"][0]["license"].update(url="file:///private/license"),
                         lambda c: c["cases"][0].update(evaluation_timestamp_ms=1000),
                         lambda c: c["cases"][0]["frames"][0].update(source_timestamp_ms=23000),
                         lambda c: c["protocol"]["checks"][0].update(kind="count", expected=1)):
            value = contract()
            mutation(value)
            with self.assertRaises(ValidationError):
                approved(value)

    def test_sam_positive_location_uses_independent_cases(self):
        envelope = approved()
        cases = evaluate_sam(envelope, sam_results(envelope), inference_origin="authored_fixture")
        self.assertEqual(["fail", "pass"], [case["status"] for case in cases])
        self.assertEqual("authored_fixture", cases[0]["inference_origin"])
        self.assertEqual(1.0, comparison_metrics(envelope, cases)["accuracy"])

    def test_missing_zero_or_ambiguous_evidence_is_unknown(self):
        envelope = approved()
        for raw in ({}, sam_results(envelope, empty=True)):
            self.assertEqual(["unknown", "unknown"], [case["status"] for case in evaluate_sam(envelope, raw)])
        raw = sam_results(envelope)
        case = envelope["contract"]["cases"][1]
        for frame in case["frames"]:
            detection = raw[case["id"]][frame["sha256"]]["glass"]["objects"][0]
            raw[case["id"]][frame["sha256"]]["glass"]["objects"].append({**detection, "object_id": 1})
        self.assertEqual("unknown", evaluate_sam(envelope, raw)[1]["status"])

    def test_stale_required_support_is_unknown(self):
        envelope = approved(contract(age=500))
        self.assertEqual(["unknown", "unknown"], [case["status"] for case in evaluate_sam(envelope, sam_results(envelope))])

    def test_sam_source_model_and_decoding_are_binding(self):
        envelope = approved()
        for field, value in (("source", {"sha256": "0" * 64}),
                             ("response", {"status": "completed", "model": "other-model"}),
                             ("mask_decoding", "not-official")):
            raw = sam_results(envelope)
            first = next(iter(raw["initial"].values()))["glass"]
            first[field] = value
            with self.assertRaises(ValidationError):
                evaluate_sam(envelope, raw)

    def test_baseline_has_same_images_requirements_but_no_ground_truth_or_sam(self):
        envelope = approved()
        case = envelope["contract"]["cases"][0]
        prompt = baseline_prompt(envelope, case)
        self.assertIn(envelope["contract"]["requirements"], prompt)
        self.assertIn("UNKNOWN", prompt)
        self.assertNotIn("annotation", prompt)
        self.assertNotIn(case["annotation"]["reason"], prompt)
        raw = {case["id"]: spark_result(case, "fail")}
        self.assertEqual(["fail", "unknown"], [r["status"] for r in evaluate_spark(envelope, raw)])
        raw[case["id"]]["sources"].reverse()
        with self.assertRaises(ValidationError):
            evaluate_spark(envelope, raw)

    def test_baseline_duplicate_check_ids_rejected(self):
        value = contract()
        value["protocol"]["checks"].append({**value["protocol"]["checks"][0], "id": "other-check"})
        for case in value["cases"]:
            case["annotation"]["checks"]["other-check"] = case["annotation"]["status"]
        envelope = approved(value)
        case = envelope["contract"]["cases"][0]
        raw = spark_result(case)
        raw["parsed"]["checks"] *= 2
        with self.assertRaises(ValidationError):
            evaluate_spark(envelope, {case["id"]: raw})

    def test_metrics_penalize_all_unknown_and_report_incorrect_passes(self):
        envelope = approved()
        unknown = comparison_metrics(envelope, [{"case_id": c["id"], "status": "unknown"}
                                                  for c in envelope["contract"]["cases"]])
        self.assertEqual(0.0, unknown["accuracy"])
        self.assertEqual(0.0, unknown["correct_acceptance_rate"])
        self.assertEqual(1.0, unknown["unknown_rate"])
        passes = comparison_metrics(envelope, [{"case_id": c["id"], "status": "pass"}
                                                 for c in envelope["contract"]["cases"]])
        self.assertEqual(1, passes["incorrect_passes"])
        self.assertEqual(1.0, passes["incorrect_pass_rate"])
        missing = comparison_metrics(envelope, [])
        self.assertEqual(2, missing["missing_predictions"])
        self.assertEqual(0.0, missing["accuracy"])
        self.assertEqual(0, missing["unknown_results"])

    def test_media_hashes_and_local_boundary_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope = media_fixture(root, approved())
            self.assertEqual(5, len(verify_media(envelope, project_root=root)))
            frame = envelope["contract"]["cases"][0]["frames"][0]
            (root / frame["file"]).write_bytes(b"changed")
            with self.assertRaises(ValidationError):
                verify_media(envelope, project_root=root)
            envelope = media_fixture(root, approved())
            value = deepcopy(envelope["contract"])
            value["cases"][0]["frames"][0]["file"] = ".env.local"
            with self.assertRaises(ValidationError):
                verify_media(approved(value), project_root=root)

    def test_explicit_budget_prevents_factory_or_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):
                run_benchmark(lambda _: self.fail("Factory must not run"), approved(), Path(directory), max_requests=5)

    def test_live_runner_failure_saves_error_without_fabricating_inference(self):
        class FailureClient:
            retries = 0
            def segment_image(self, *args, **kwargs):
                raise MetaAPIError("secret provider diagnostics", code="test_failure")
            def usage_summary(self):
                return {"request_attempts": 1, "estimated_usd": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope = media_fixture(root, approved())
            output = root / "artifacts/local/test-run"
            report = run_benchmark(lambda _: FailureClient(), envelope, output, project_root=root, max_requests=6)
            self.assertEqual("error", report["status"])
            self.assertNotIn("cases", report["conditions"]["sam"])
            self.assertNotIn("secret provider", json.dumps(report))
            self.assertTrue((output / "approved-manifest.json").is_file())
            self.assertEqual("error", json.loads((output / "report.json").read_text())["status"])

    def test_evaluation_role_rejected_before_media_access_or_paid_calls(self):
        for role in ("heldout", "", False, [], {"unexpected": "value"}):
            with self.subTest(role=role), self.assertRaisesRegex(ValidationError, "Evaluation role"):
                run_benchmark(lambda _: self.fail("Invalid role must not create a client"), approved(), Path("unused"),
                              max_requests=6, evaluation_role=role)

    def test_evaluation_role_persists_even_when_first_request_fails(self):
        class FailureClient:
            retries = 0
            def segment_image(self, *args, **kwargs):
                raise MetaAPIError("Authored failure", code="fixture_error")
            def usage_summary(self):
                return {"request_attempts": 1, "estimated_usd": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope = media_fixture(root, approved())
            for role in ("development_reprocessing", "new_source_holdback"):
                output = root / "artifacts/local" / role
                report = run_benchmark(lambda _: FailureClient(), envelope, output, project_root=root,
                                       max_requests=6, evaluation_role=role)
                self.assertEqual(role, report["evaluation_role"])
                self.assertEqual(role, json.loads((output / "report.json").read_text())["evaluation_role"])
                self.assertEqual(role, public_summary(envelope, report)["evaluation_role"])

    def test_export_preserves_explicit_role_and_version_without_guessing_legacy(self):
        envelope = approved()
        report = {"contract_hash": envelope["contract_hash"], "status": "completed", "conditions": {"guided": {
            "status": "completed", "observation_version": "normalized_boxes_v2", "cases": [], "metrics": {}}}}
        legacy = public_summary(envelope, report)
        self.assertNotIn("evaluation_role", legacy)
        self.assertEqual("normalized_boxes_v2", legacy["conditions"]["guided"]["observation_version"])
        report["evaluation_role"] = "development_reprocessing"
        self.assertEqual("development_reprocessing", public_summary(envelope, report)["evaluation_role"])
        report["conditions"]["guided"]["observation_version"] = "PRIVATE_SENTINEL"
        self.assertNotIn("observation_version", public_summary(envelope, report)["conditions"]["guided"])
        report["evaluation_role"] = "arbitrary_holdout_claim"
        with self.assertRaisesRegex(ValidationError, "Evaluation role"):
            public_summary(envelope, report)

    def test_guided_budget_is_optional_and_an_upper_bound(self):
        envelope = approved()
        self.assertEqual({"sam": 4, "spark": 2}, request_requirements(envelope))
        self.assertEqual({"guided": 6}, request_requirements(envelope, ("guided",)))
        self.assertEqual({"sam": 4, "spark": 2, "guided": 6},
                         request_requirements(envelope, ("sam", "spark", "guided")))
        with self.assertRaises(ValidationError):
            run_benchmark(lambda _: self.fail("No client before budget review"), envelope, Path("unused"),
                          conditions=("guided",), max_requests=5)

    def test_guided_actual_responses_need_identity_provenance_and_exact_sources(self):
        envelope = approved()
        raw = sam_results(envelope)
        with self.assertRaises(ValidationError):
            evaluate_guided(envelope, raw)
        for case_results in raw.values():
            for image_results in case_results.values():
                for result_value in image_results.values():
                    result_value["guided_provenance"] = {"method": "spark_box_guided_sam", "locator_model": SPARK_MODEL}
        cases = evaluate_guided(envelope, raw, inference_origin="authored_fixture")
        self.assertEqual(["fail", "pass"], [case["status"] for case in cases])
        self.assertIn("does not independently prove", cases[0]["method_note"])
        next(iter(raw["initial"].values()))["glass"]["source"]["sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            evaluate_guided(envelope, raw)

    def test_guided_missing_localization_creates_no_fake_sam_result(self):
        class LocatorOnlyClient:
            retries = 0
            def __init__(self, budget):
                self.calls = 0
            def usage_summary(self):
                return {"request_attempts": self.calls, "estimated_usd": .001 * self.calls}
        def no_candidates(client, images, concepts):
            client.calls += 1  # One authored locator request; deliberately no SAM calls.
            hashes = [hashlib.sha256(image.read_bytes()).hexdigest() for image in images]
            return {"results": {source_hash: {} for source_hash in hashes}, "locator": {"authored_fixture": True},
                    "unavailable": [{"object_id": "glass", "reason": "Locator found no candidate"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope = media_fixture(root, approved())
            with patch("proofloop.guided.guided_segment_pair", side_effect=no_candidates):
                report = run_benchmark(LocatorOnlyClient, envelope, root / "artifacts/local/guided-run",
                                       project_root=root, max_requests=6, conditions=("guided",))
            condition = report["conditions"]["guided"]
            self.assertEqual("completed", report["status"])
            self.assertEqual(6, condition["planned_requests"])
            self.assertEqual(2, condition["usage"]["request_attempts"])
            self.assertEqual(["unknown", "unknown"], [case["status"] for case in condition["cases"]])
            self.assertEqual([], condition["cases"][0]["grounding"]["evidence"]["sources"])
            self.assertTrue((root / "artifacts/local/guided-run/guided/initial.json").is_file())
            exported = public_summary(envelope, report)
            self.assertIn("guided", exported["conditions"])
            self.assertIn("Spark", exported["conditions"]["guided"]["method_note"])
            self.assertNotIn("locator", json.dumps(exported))

    def test_run_rejects_media_mutated_during_request(self):
        class MutatingClient:
            retries = 0
            def segment_image(self, image, *args, **kwargs):
                frame_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                image.write_bytes(b"changed after transport")
                return result(frame_hash)
            def usage_summary(self):
                return {"request_attempts": 1, "estimated_usd": .0025}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope = media_fixture(root, approved())
            report = run_benchmark(lambda _: MutatingClient(), envelope, root / "artifacts/local/mutated-run",
                                   project_root=root, max_requests=6)
            self.assertEqual("error", report["status"])
            self.assertEqual("ValidationError", report["conditions"]["sam"]["error_type"])
            self.assertNotIn("spark", report["conditions"])
            self.assertNotIn("cases", report["conditions"]["sam"])

    def test_public_export_filters_arbitrary_private_fields(self):
        envelope = approved()
        envelope["contract"]["sources"][0]["license"]["extra_private"] = "PRIVATE_SENTINEL"
        envelope["contract_hash"] = manifest_hash(envelope["contract"])
        report = {"contract_hash": envelope["contract_hash"], "status": "completed",
                  "inference_origin": "live_meta_api", "limits": ["PRIVATE_SENTINEL"],
                  "conditions": {"sam": {"status": "completed", "metrics": {"extra_private": "PRIVATE_SENTINEL"},
                                           "usage": {"requests": ["PRIVATE_SENTINEL"]},
                                           "cases": [{"case_id": "initial", "status": "fail", "raw": "PRIVATE_SENTINEL"}]}}}
        summary = public_summary(envelope, report)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(summary))
        self.assertEqual({"name", "url", "attribution"}, set(summary["sources"][0]["license"]))

    def test_runner_uses_same_exact_pair_and_separate_condition_usage(self):
        class FixtureClient:
            retries = 0
            def __init__(self, budget):
                self.budget, self.calls = budget, 0
            def segment_image(self, image, prompt, **kwargs):
                self.calls += 1
                frame_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                return result(frame_hash, "staging" if "initial" in image.name else "dispense")
            def spark_json(self, prompt, schema, *, image_paths, **kwargs):
                self.calls += 1
                return {"sources": [{"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in image_paths],
                        "response": {"status": "completed", "model": SPARK_MODEL},
                        "parsed": {"checks": [{"check_id": "glass-dispense", "status": "pass", "reason": "Fixture"}]}}
            def usage_summary(self):
                return {"request_attempts": self.calls, "estimated_usd": .01 * self.calls}
        clients = []
        def factory(budget):
            client = FixtureClient(budget)
            clients.append(client)
            return client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope = media_fixture(root, approved())
            report = run_benchmark(factory, envelope, root / "artifacts/local/complete-run", project_root=root, max_requests=6)
            self.assertEqual("completed", report["status"])
            self.assertEqual([4, 2], [client.calls for client in clients])
            self.assertEqual(1, report["conditions"]["spark"]["metrics"]["incorrect_passes"])
            exported = public_summary(envelope, report)
            self.assertNotIn("raster", json.dumps(exported))
            self.assertNotIn("inputs/local", json.dumps(exported))
            self.assertNotIn("authored-sam-shape", json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
