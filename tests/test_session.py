"""Session contract and correction-loop tests with authored SAM-shaped results.

These fixtures exercise state transitions and provenance; they make no model
requests and do not establish performance on a physical phone recording.
"""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest

from proofloop.core import Protocol
from proofloop.meta import MetaAPIError, SAM_MODEL
from proofloop.session import (
    approve_contract,
    contract_hash,
    evaluate_session,
    run_session,
    session_contract,
    verify_capture_files,
)


STAGE_IDS = ("initial_correct", "wrong_region", "occluded", "corrected")
REGIONS = {"left": [0.05, 0.05, 0.45, 0.95], "right": [0.55, 0.05, 0.95, 0.95]}
CONCEPTS = {"screwdriver": "screwdriver", "pliers": "pliers"}


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def protocol(*, max_age_ms=30000):
    return Protocol.from_dict({
        "id": "authored-phone-session-test", "version": "1",
        "objects": ["screwdriver", "pliers"], "regions": ["left", "right"],
        "checks": [
            {"id": "screwdriver-left", "kind": "location", "object_id": "screwdriver",
             "region_id": "left", "unique_target": True, "max_age_ms": max_age_ms},
            {"id": "pliers-right", "kind": "location", "object_id": "pliers",
             "region_id": "right", "unique_target": True, "max_age_ms": max_age_ms},
        ],
    })


def capture_manifest():
    stages = []
    for identifier, start, end, offsets in (
        ("initial_correct", 0, 8, (3, 5)),
        ("wrong_region", 13, 21, (16, 18)),
        ("occluded", 26, 34, (29, 31)),
        ("corrected", 39, 47, (42, 44)),
    ):
        frames = [
            {"file": f"{identifier}-{index}.jpg", "sha256": digest(f"{identifier}-{index}"),
             "source_timestamp_ms": offset * 1000, "source_timestamp_seconds": offset,
             "requested_seconds": offset, "width": 100, "height": 100, "bytes": 123}
            for index, offset in enumerate(offsets, 1)
        ]
        stages.append({"id": identifier, "declared_label": "Authored test annotation, not model truth",
                       "start_seconds": start, "end_seconds": end,
                       "sample_seconds": list(offsets), "frames": frames,
                       "distinct_frame_hashes": True})
    return {
        "schema_version": "1.0", "experiment": "phone_capture_preflight",
        "evidence_origin": "synthetic_test_fixture",
        "origin_verification": "caller_declared_not_independently_verified",
        "ground_truth_origin": "human_declared_not_model_inferred",
        "offsets_confirmed": True, "inference_performed": False,
        "physical_repair_verified": False, "status": "extracted_pending_visual_review",
        "source": {"file": "../authored-fixture.mp4", "sha256": digest("authored-clip"),
                   "duration_seconds": 48, "timestamp_basis": "authored fixture PTS"},
        "stages": stages,
    }


def contract(*, manifest=None, task=None):
    return session_contract(capture_manifest() if manifest is None else manifest,
                            protocol() if task is None else task,
                            deepcopy(REGIONS), dict(CONCEPTS),
                            manifest_file="inputs/local/phone/authored/manifest.json")


def approved(value=None):
    value = contract() if value is None else value
    return approve_contract(value, contract_hash(value), confirm_unique_targets=True)


def detection(region):
    x1, x2 = (10, 30) if region == "left" else (60, 80)
    raster = bytes([1] * 6)
    return {
        "frame": 0, "object_id": 0, "confidence": 0.97,
        "box": {"x1": x1, "y1": 10, "x2": x2, "y2": 80, "width": 100, "height": 100},
        "mask": {"decoded": True, "width": 2, "height": 3,
                 "raster_format": "base64_row_major_uint8_0_or_1",
                 "raster_base64": base64.b64encode(raster).decode(),
                 "foreground_pixels": 6, "raster_sha256": hashlib.sha256(raster).hexdigest()},
    }


def results_for(manifest=None):
    manifest = capture_manifest() if manifest is None else manifest
    results = {}
    for stage in manifest["stages"]:
        stage_id = stage["id"]
        results[stage_id] = {}
        for frame in stage["frames"]:
            by_object = {}
            for object_id in CONCEPTS:
                region = "left" if object_id == "screwdriver" else "right"
                if stage_id == "wrong_region":
                    region = "right"
                by_object[object_id] = {
                    "source": {"sha256": frame["sha256"], "mime": "image/jpeg", "bytes": 123},
                    "objects": [] if stage_id == "occluded" else [detection(region)],
                    "response": {"status": "completed", "model": SAM_MODEL,
                                 "id": f"authored-{stage_id}-{frame['source_timestamp_ms']}-{object_id}"},
                    "mask_decoding": "official_meta_sam_parser_0.1.0",
                }
            results[stage_id][frame["sha256"]] = by_object
    return results


def stage_entries(report):
    return {item["stage_id"]: item for item in report["trace"] if item["event"] == "stage"}


def checks_by_id(stage):
    checks = stage["checks"]
    return checks if isinstance(checks, dict) else {item["check_id"]: item for item in checks}


def local_capture_fixture(project_root):
    """Write transport fixtures, not playable video or model-ready photographs."""
    directory = project_root / "inputs/local/phone/authored"
    directory.mkdir(parents=True)
    manifest = capture_manifest()
    source = directory.parent / "authored-fixture.mp4"
    source_bytes = b"authored transport fixture; not an actual encoded phone clip"
    source.write_bytes(source_bytes)
    manifest["source"].update(sha256=hashlib.sha256(source_bytes).hexdigest(), bytes=len(source_bytes))
    for stage in manifest["stages"]:
        for frame in stage["frames"]:
            data = (f"authored transport fixture {stage['id']} {frame['file']}").encode()
            (directory / frame["file"]).write_bytes(data)
            frame.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    envelope = approved(contract(manifest=manifest))
    return envelope, directory


class AuthoredSamClient:
    """A local fake with the same bounded-call/usage surface as MetaClient."""

    def __init__(self, manifest, *, fail_on_attempt=None, on_attempt=None, transform_result=None):
        self.max_requests = 16
        self.retries = 0
        self.ledger = []
        self.fail_on_attempt = fail_on_attempt
        self.on_attempt = on_attempt
        self.transform_result = transform_result
        self.results = {}
        for stage_results in results_for(manifest).values():
            self.results.update(stage_results)

    def segment_image(self, path, concept, *, decode_masks):
        if len(self.ledger) >= self.max_requests:
            raise AssertionError("Session exceeded the mock request cap")
        source_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.ledger.append({"concept": concept, "source_sha256": source_digest,
                            "decode_masks": decode_masks})
        if len(self.ledger) == self.fail_on_attempt:
            raise MetaAPIError("Authored transport failure", code="network_error")
        result = deepcopy(self.results[source_digest][concept])
        if self.on_attempt is not None:
            self.on_attempt(len(self.ledger), Path(path))
        return self.transform_result(result) if self.transform_result is not None else result

    def usage_summary(self):
        return {"request_attempts": len(self.ledger), "estimated_usd": 0,
                "estimate_origin": "authored_mock_no_api_calls"}


class SessionContractTests(unittest.TestCase):
    def test_contract_copies_approved_inputs(self):
        manifest, regions, concepts = capture_manifest(), deepcopy(REGIONS), dict(CONCEPTS)
        value = session_contract(manifest, protocol(), regions, concepts, manifest_file="manifest.json")
        before = deepcopy(value)
        manifest["stages"][0]["frames"][0]["sha256"] = digest("changed")
        regions["left"][0] = 0.2
        concepts["screwdriver"] = "different concept"
        self.assertEqual(before, value)
        self.assertEqual("1.0", value["schema_version"])
        self.assertEqual("manifest.json", value["capture_manifest_file"])

    def test_contract_hash_ignores_dictionary_insertion_order(self):
        value = contract()
        reversed_top_level = dict(reversed(list(value.items())))
        self.assertEqual(contract_hash(value), contract_hash(reversed_top_level))

    def test_approval_requires_current_digest_and_unique_target_confirmation(self):
        value = contract()
        with self.assertRaises(ValueError):
            approve_contract(value, digest("wrong digest"), confirm_unique_targets=True)
        with self.assertRaises(ValueError):
            approve_contract(value, contract_hash(value), confirm_unique_targets=False)
        envelope = approved(value)
        self.assertEqual(contract_hash(value), envelope["contract_hash"])
        self.assertIs(True, envelope["approval"]["approved"])
        self.assertEqual("local_operator", envelope["approval"]["origin"])
        self.assertIs(True, envelope["approval"]["unique_targets_confirmed"])
        self.assertIs(True, envelope["approval"]["human_reviewed_exact_contract"])

    def test_assistant_review_records_authority_without_claiming_exact_human_review(self):
        value = contract()
        reference = "Direct user instruction: check the uploaded clip and proceed with next steps."
        envelope = approve_contract(value, contract_hash(value), confirm_unique_targets=True,
                                    review_origin="assistant_under_user_authorization",
                                    authorization_reference=reference)
        report = evaluate_session(envelope, results_for(), inference_origin="authored_test_fixture")
        self.assertEqual({"origin": "assistant_under_user_authorization",
                          "human_reviewed_exact_contract": False,
                          "authorization_reference": reference}, report["approval_provenance"])
        self.assertTrue(report["recorded_observation_loop_supported"])

    def test_assistant_review_requires_explicit_authorization_reference(self):
        value = contract()
        for reference in (None, "", " \n ", 123, "x" * 2001):
            with self.subTest(reference_type=type(reference).__name__), self.assertRaises(ValueError):
                approve_contract(value, contract_hash(value), confirm_unique_targets=True,
                                 review_origin="assistant_under_user_authorization",
                                 authorization_reference=reference)

    def test_contradictory_or_missing_review_provenance_is_rejected(self):
        value = contract()
        assistant = approve_contract(value, contract_hash(value), confirm_unique_targets=True,
                                     review_origin="assistant_under_user_authorization",
                                     authorization_reference="Direct user instruction to proceed")
        malformed = []
        for original in (approved(), assistant):
            changed = deepcopy(original)
            changed["approval"]["human_reviewed_exact_contract"] = not changed["approval"]["human_reviewed_exact_contract"]
            malformed.append(changed)
        for field in ("authorization_reference", "human_reviewed_exact_contract"):
            changed = deepcopy(assistant)
            del changed["approval"][field]
            malformed.append(changed)
        changed = deepcopy(assistant)
        changed["approval"]["origin"] = "unspecified_reviewer"
        malformed.append(changed)
        for index, changed in enumerate(malformed):
            with self.subTest(case=index), self.assertRaises(ValueError):
                evaluate_session(changed, {})

    def test_assistant_review_cannot_authorize_a_different_contract(self):
        value = contract()
        envelope = approve_contract(value, contract_hash(value), confirm_unique_targets=True,
                                    review_origin="assistant_under_user_authorization",
                                    authorization_reference="Direct user instruction to proceed")
        envelope["contract"]["regions"]["left"][0] = 0.08
        with self.assertRaises(ValueError):
            evaluate_session(envelope, results_for())

    def test_report_only_exposes_whitelisted_approval_provenance(self):
        envelope = approved()
        envelope["approval"]["private_notes"] = "private operator notes"
        report = evaluate_session(envelope, results_for())
        self.assertEqual({"origin": "local_operator", "human_reviewed_exact_contract": True},
                         report["approval_provenance"])
        self.assertNotIn("private operator notes", json.dumps(report))

    def test_unapproved_envelope_is_rejected(self):
        envelope = approved()
        for field in ("approved", "unique_targets_confirmed"):
            changed = deepcopy(envelope)
            changed["approval"][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_session(changed, {})

    def test_post_approval_calibration_concept_source_and_protocol_edits_are_rejected(self):
        mutations = (
            lambda value: value["regions"]["left"].__setitem__(0, 0.08),
            lambda value: value["concepts"].__setitem__("screwdriver", "red handled screwdriver"),
            lambda value: value["capture"]["source"].__setitem__("sha256", digest("another clip")),
            lambda value: value["protocol"]["checks"][0].__setitem__("max_age_ms", 500),
        )
        for mutation in mutations:
            envelope = approved()
            mutation(envelope["contract"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluate_session(envelope, {})

    def test_duplicate_source_hash_cannot_be_reassigned_to_corrected_stage(self):
        manifest = capture_manifest()
        manifest["stages"][-1]["frames"][0]["sha256"] = manifest["stages"][0]["frames"][0]["sha256"]
        with self.assertRaises(ValueError):
            approved(contract(manifest=manifest))

    def test_frames_must_be_ordered_in_their_window_and_one_second_apart(self):
        for timestamps in ((5000, 3000), (3000, 3500), (3000, 8000)):
            manifest = capture_manifest()
            for frame, timestamp in zip(manifest["stages"][0]["frames"], timestamps):
                frame["source_timestamp_ms"] = timestamp
                frame["source_timestamp_seconds"] = timestamp / 1000
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                approved(contract(manifest=manifest))

    def test_stages_must_keep_required_order(self):
        manifest = capture_manifest()
        manifest["stages"][1], manifest["stages"][2] = manifest["stages"][2], manifest["stages"][1]
        with self.assertRaises(ValueError):
            approved(contract(manifest=manifest))

    def test_overlapping_calibration_is_rejected(self):
        value = contract()
        value["regions"]["left"] = [0, 0, 0.7, 1]
        with self.assertRaises(ValueError):
            approved(value)

    def test_every_inventory_object_needs_one_location_check_and_concept(self):
        for missing in ("check", "concept"):
            value = contract()
            if missing == "check":
                value["protocol"]["checks"].pop()
            else:
                del value["concepts"]["pliers"]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                approved(value)


class CorrectionSessionTests(unittest.TestCase):
    def test_supported_loop_passes_fails_abstains_and_rechecks(self):
        report = evaluate_session(approved(), results_for(), inference_origin="authored_test_fixture")
        stages = stage_entries(report)
        self.assertEqual(list(STAGE_IDS), list(stages))
        self.assertEqual(["pass", "fail", "unknown", "pass"],
                         [stages[stage_id]["status"] for stage_id in STAGE_IDS])
        self.assertEqual("fail", checks_by_id(stages["wrong_region"])["screwdriver-left"]["status"])
        self.assertEqual("pass", checks_by_id(stages["wrong_region"])["pliers-right"]["status"])
        self.assertEqual("authored_test_fixture", report["inference_origin"])

    def test_every_transition_has_an_immediate_unknown_invalidation(self):
        report = evaluate_session(approved(), results_for())
        invalidations = [item for item in report["trace"] if item["event"] == "invalidation"]
        self.assertGreaterEqual(len(invalidations), 3)
        self.assertTrue(all(item["status"] == "unknown" for item in invalidations))

    def test_repair_proposal_does_not_itself_produce_pass(self):
        report = evaluate_session(approved(), results_for())
        trace = report["trace"]
        proposals = [item for item in trace if item["event"] == "repair_proposal"]
        self.assertTrue(proposals)
        plans = [item["plan"] for item in proposals if item.get("plan") is not None]
        self.assertTrue(plans, "The supported wrong placement should produce a bounded repair plan")
        plan = plans[0]
        self.assertIs(True, plan["hypothetical"])
        self.assertIs(True, plan["execution_requires_fresh_observation"])
        self.assertEqual(["move-screwdriver-right-to-left", "inspect-screwdriver"], plan["action_ids"])
        wrong = stage_entries(report)["wrong_region"]
        self.assertEqual("fail", wrong["status"])
        self.assertEqual("unknown", stage_entries(report)["occluded"]["status"])

    def test_missing_corrected_results_cannot_reuse_initial_pass(self):
        results = results_for()
        del results["corrected"]
        report = evaluate_session(approved(), results)
        self.assertEqual("pass", stage_entries(report)["initial_correct"]["status"])
        self.assertEqual("unknown", stage_entries(report)["corrected"]["status"])

    def test_one_corrected_frame_is_insufficient_even_after_initial_pass(self):
        results = results_for()
        results["corrected"].pop(next(iter(results["corrected"])))
        report = evaluate_session(approved(), results)
        self.assertEqual("unknown", stage_entries(report)["corrected"]["status"])

    def test_latest_ambiguous_corrected_frame_cannot_fall_back_to_an_old_pass(self):
        results = results_for()
        last_frame = list(results["corrected"])[-1]
        results["corrected"][last_frame]["screwdriver"]["objects"] = []
        report = evaluate_session(approved(), results)
        self.assertEqual("unknown", stage_entries(report)["corrected"]["status"])

    def test_frame_response_source_digest_must_match_selected_frame(self):
        results = results_for()
        first_frame = next(iter(results["corrected"]))
        results["corrected"][first_frame]["screwdriver"]["source"]["sha256"] = digest("old source")
        with self.assertRaises(ValueError):
            evaluate_session(approved(), results)

    def test_declared_labels_are_not_ground_truth_for_the_evaluator(self):
        results = results_for()
        for by_object in results["wrong_region"].values():
            by_object["screwdriver"]["objects"] = [detection("left")]
        report = evaluate_session(approved(), results)
        self.assertEqual("pass", stage_entries(report)["wrong_region"]["status"])

    def test_stale_frame_support_is_checked_at_stage_end_not_response_time(self):
        report = evaluate_session(approved(contract(task=protocol(max_age_ms=1000))), results_for())
        self.assertEqual("unknown", stage_entries(report)["initial_correct"]["status"])

    def test_caller_inputs_are_not_changed_by_evaluation(self):
        envelope, results = approved(), results_for()
        original_envelope, original_results = deepcopy(envelope), deepcopy(results)
        evaluate_session(envelope, results)
        self.assertEqual(original_envelope, envelope)
        self.assertEqual(original_results, results)

    def test_wrong_provider_incomplete_or_undecoded_results_are_rejected(self):
        for field, value in (("model", "unapproved-provider-model"),
                             ("status", "incomplete"), ("mask_decoding", "skipped_explicitly")):
            results = results_for()
            first_frame = next(iter(results["initial_correct"]))
            result = results["initial_correct"][first_frame]["screwdriver"]
            if field == "mask_decoding":
                result[field] = value
            else:
                result["response"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_session(approved(), results)


class SessionFileIntegrationTests(unittest.TestCase):
    def test_verification_binds_eight_actual_files_to_approved_digests(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            envelope, directory = local_capture_fixture(project_root)
            files = verify_capture_files(envelope, project_root=project_root)
            self.assertEqual(8, len(files))
            for source_digest, path in files.items():
                self.assertEqual(directory.resolve(), path.parent)
                self.assertEqual(source_digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_unapproved_stale_or_changed_files_stop_before_any_segment_call(self):
        for scenario in ("unapproved", "stale_approval", "changed_manifest", "changed_source", "changed_frame"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                project_root = Path(temporary)
                envelope, directory = local_capture_fixture(project_root)
                client = AuthoredSamClient(envelope["contract"]["capture"])
                if scenario == "unapproved":
                    envelope["approval"]["approved"] = False
                elif scenario == "stale_approval":
                    envelope["contract"]["concepts"]["screwdriver"] = "changed concept"
                elif scenario == "changed_manifest":
                    changed = deepcopy(envelope["contract"]["capture"])
                    changed["source"]["duration_seconds"] = 49
                    (directory / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                elif scenario == "changed_source":
                    (directory.parent / "authored-fixture.mp4").write_bytes(b"modified original source")
                else:
                    frame = envelope["contract"]["capture"]["stages"][0]["frames"][0]
                    (directory / frame["file"]).write_bytes(b"modified selected image")
                with self.assertRaises(ValueError):
                    run_session(client, envelope, project_root / "artifacts/local/test-run",
                                project_root=project_root)
                self.assertEqual([], client.ledger)

    def test_full_run_makes_sixteen_mock_calls_and_saves_private_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            envelope, _ = local_capture_fixture(project_root)
            client = AuthoredSamClient(envelope["contract"]["capture"])
            output = project_root / "artifacts/local/test-run"
            report = run_session(client, envelope, output, project_root=project_root,
                                 inference_origin="authored_mock_no_api_calls")
            self.assertEqual(16, len(client.ledger))
            self.assertTrue(all(item["decode_masks"] is True for item in client.ledger))
            self.assertEqual(16, report["new_requests"])
            self.assertEqual([], report["failures"])
            self.assertEqual("authored_mock_no_api_calls", report["inference_origin"])
            self.assertEqual("synthetic_test_fixture", report["evidence_origin"])
            self.assertIs(True, report["recorded_observation_loop_supported"])
            self.assertEqual(report, json.loads((output / "report.json").read_text(encoding="utf-8")))
            self.assertEqual(envelope, json.loads((output / "approval.json").read_text(encoding="utf-8")))
            self.assertEqual(18, len(list(output.glob("*.json"))))
            self.assertEqual(0o700, stat.S_IMODE(output.stat().st_mode))
            self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.glob("*.json")))

    def test_existing_output_and_uncapped_or_retrying_clients_stop_before_calls(self):
        for scenario in ("existing_output", "uncapped", "retrying"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                project_root = Path(temporary)
                envelope, _ = local_capture_fixture(project_root)
                client = AuthoredSamClient(envelope["contract"]["capture"])
                output = project_root / "artifacts/local/test-run"
                if scenario == "existing_output":
                    output.mkdir(parents=True)
                elif scenario == "uncapped":
                    client.max_requests = 17
                else:
                    client.retries = 1
                with self.assertRaises(ValueError):
                    run_session(client, envelope, output, project_root=project_root)
                self.assertEqual([], client.ledger)

    def test_first_network_failure_stops_and_missing_stages_remain_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            envelope, _ = local_capture_fixture(project_root)
            client = AuthoredSamClient(envelope["contract"]["capture"], fail_on_attempt=1)
            output = project_root / "artifacts/local/test-run"
            report = run_session(client, envelope, output, project_root=project_root,
                                 inference_origin="authored_mock_no_api_calls")
            self.assertEqual(1, len(client.ledger))
            self.assertEqual(1, report["new_requests"])
            self.assertEqual(1, len(report["failures"]))
            self.assertEqual("MetaAPIError", report["failures"][0]["error_type"])
            self.assertTrue(all(status == "unknown" for status in report["stage_statuses"].values()))
            self.assertIs(False, report["recorded_observation_loop_supported"])
            self.assertTrue((output / "report.json").is_file())

    def test_first_unbound_result_stops_before_further_paid_attempts(self):
        for scenario in ("wrong_model", "wrong_source", "incomplete", "undecoded"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                project_root = Path(temporary)
                envelope, _ = local_capture_fixture(project_root)

                def transform(result):
                    if scenario == "wrong_model":
                        result["response"]["model"] = "unapproved-model"
                    elif scenario == "wrong_source":
                        result["source"]["sha256"] = digest("different input image")
                    elif scenario == "incomplete":
                        result["response"]["status"] = "incomplete"
                    else:
                        result["mask_decoding"] = "skipped_explicitly"
                    return result

                client = AuthoredSamClient(envelope["contract"]["capture"], transform_result=transform)
                output = project_root / "artifacts/local/test-run"
                report = run_session(client, envelope, output, project_root=project_root,
                                     inference_origin="authored_mock_no_api_calls")
                self.assertEqual(1, len(client.ledger))
                self.assertEqual(1, report["new_requests"])
                self.assertTrue(report["failures"])
                self.assertTrue(all(status == "unknown" for status in report["stage_statuses"].values()))
                self.assertIs(False, report["recorded_observation_loop_supported"])
                self.assertTrue((output / "report.json").is_file())
                self.assertFalse((output / "initial_correct-0-0.json").exists(),
                                 "A rejected result must not be stored as an accepted frame response")

    def test_midrun_changed_next_frame_is_rejected_before_requesting_that_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            envelope, directory = local_capture_fixture(project_root)
            changed_frame = envelope["contract"]["capture"]["stages"][0]["frames"][1]

            def change_next_frame(attempt, _path):
                if attempt == 1:
                    (directory / changed_frame["file"]).write_bytes(b"changed while the first request was pending")

            client = AuthoredSamClient(envelope["contract"]["capture"], on_attempt=change_next_frame)
            output = project_root / "artifacts/local/test-run"
            report = run_session(client, envelope, output, project_root=project_root,
                                 inference_origin="authored_mock_no_api_calls")
            self.assertEqual(2, len(client.ledger), "Only the unchanged first frame may receive its two concept requests")
            self.assertTrue(report["failures"])
            self.assertIs(False, report["gates"]["source_files_still_match"])
            self.assertIs(False, report["recorded_observation_loop_supported"])
            self.assertTrue(all(status == "unknown" for status in report["stage_statuses"].values()))
            self.assertTrue((output / "report.json").is_file())

    def test_source_changed_by_last_response_cannot_certify_supported_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            envelope, directory = local_capture_fixture(project_root)

            def change_source_on_last_response(attempt, _path):
                if attempt == 16:
                    (directory.parent / "authored-fixture.mp4").write_bytes(b"source changed before final verification")

            client = AuthoredSamClient(envelope["contract"]["capture"], on_attempt=change_source_on_last_response)
            output = project_root / "artifacts/local/test-run"
            report = run_session(client, envelope, output, project_root=project_root,
                                 inference_origin="authored_mock_no_api_calls")
            self.assertEqual(16, len(client.ledger))
            self.assertIs(False, report["gates"]["source_files_still_match"])
            self.assertIs(False, report["recorded_observation_loop_supported"])
            self.assertTrue((output / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
