"""Authored orchestration/transport controls; no paid or model-quality tests."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from proofloop import benchmark, concept_verified_experiment as experiment, mask_semantics, meta
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL, parse_sam_boxes
from tests.test_anchor_padding_experiment import original
from tests import test_mask_semantics as semantic_fixtures

SparkClient, decoded = semantic_fixtures.Client, semantic_fixtures.decoded


def locator_text(wrapper, _paths):
    text = json.dumps(wrapper["parsed"])
    wrapper["text"] = wrapper["response"]["output_text"] = text
    wrapper["source"] = None


class Client(SparkClient):
    def __init__(self, budget=1, *, score=.95, sam_error=None, mutate=None, error=None):
        super().__init__(mutate=mutate, error=error)
        self.max_requests = budget
        self.score, self.sam_error = score, sam_error

    def segment_image(self, path, prompt, **options):
        self.calls.append(("sam", path, prompt, options))
        if self.sam_error is not None:
            code = "network_error" if self.sam_error.status is None else "authored_http"
            entry = {"model": SAM_MODEL, "endpoint": "responses", "attempt": 1,
                     "status": "network_error" if self.sam_error.status is None else "http_error"}
            if self.sam_error.status is not None:
                entry.update(http_status=self.sam_error.status, error_code=code)
            self.ledger.append(entry)
            raise self.sam_error
        text = ("<0f>1<|box;x1=10;y1=10;x2=14;y2=14;w=100;h=80;c=" + str(self.score) + "|>"
                "<|mask;x=0;y=0;c=" + str(self.score) + ";data=5,5,!!!!!(QO(0lu8?|>")
        self.ledger.append({"model": SAM_MODEL, "endpoint": "responses", "attempt": 1, "status": "ok"})
        return {"source": {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()},
                "response": {"status": "completed", "model": SAM_MODEL, "output_text": text,
                             "private_note": "PRIVATE_PROVIDER_TEXT"},
                "text": text, "objects": decoded(parse_sam_boxes(text)),
                "mask_decoding": "official_meta_sam_parser_0.1.0"}


class ConceptVerifiedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pilot = original(self.root, transform=locator_text)
        self.output = self.root / "artifacts/local/concept-experiment"
        self.addCleanup(patch.stopall)
        patch("proofloop.meta._sam_decoder", return_value=(object, decoded, ValueError)).start()
        patch("proofloop.meta.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.mask_semantics.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.tracking_geometry.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.semantic._extractor_path", return_value="authored-ffmpeg-fixture").start()
        self.render_calls = []
        patch("proofloop.semantic.subprocess.run", side_effect=lambda *args, **kwargs:
              semantic_fixtures.MaskSemanticsTests.render(self, *args, **kwargs)).start()

    def freeze(self, **kwargs):
        return experiment.freeze([self.pilot], self.output, root=self.root, **kwargs)

    def run_pilot(self, factory=Client):
        return experiment.run(factory, self.output, root=self.root)

    def test_freeze_preserves_all_nine_cases_eighteen_frames_and_twenty_seven_cap(self):
        second = original(self.root, "second", transform=locator_text)
        third = original(self.root, "third", transform=locator_text)
        plan = experiment.freeze([self.pilot, second, third], self.output, root=self.root)
        self.assertEqual(9, len(plan["cases"]))
        self.assertEqual({"sam": 18, "spark": 9, "total": 27}, plan["request_caps"])
        self.assertIn("concept_verified_experiment.py", plan["implementation_sha256"])
        self.assertIn("mask_semantics.py", plan["implementation_sha256"])
        result = experiment.summarize(self.output, root=self.root)
        self.assertEqual("not_started", result["status"])
        self.assertEqual(9, sum(row["metrics"]["missing_predictions"] for row in result["metrics"]))
        self.assertTrue(all(row["prediction"] is None for row in result["results"]))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_one_shot_concept_calls_then_eligible_challenges_use_originals_and_exact_budgets(self):
        self.freeze()
        clients = []
        def factory(budget):
            self.assertEqual(1, budget)
            client = Client(budget)
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual("completed", result["status"])
        self.assertEqual((9, 6, 3), tuple(result["usage"][key] for key in ("request_attempts", "sam_requests", "spark_requests")))
        for client in clients[:6]:
            kind, path, prompt, options = client.calls[0]
            self.assertEqual("sam", kind)
            self.assertEqual("glass", prompt)
            self.assertIn("inputs/local/first", str(path))
            self.assertTrue(options["decode_masks"])
        for client in clients[6:]:
            _prompt, _schema, options = client.calls[0]
            self.assertEqual(6, len(options["image_paths"]))
            self.assertEqual("low", options["reasoning_effort"])
            self.assertEqual(4096, options["max_output_tokens"])
        self.assertTrue(all(row["prediction"] == "pass" for row in result["results"]))
        self.assertEqual(result, experiment.summarize(self.output, root=self.root))
        with self.assertRaises(ValidationError): self.run_pilot()

    def test_cached_occlusion_veto_keeps_sam_diagnostic_coverage_but_blocks_all_spark(self):
        def occluded(wrapper, paths):
            for frame in wrapper["parsed"]["frames"]:
                frame["objects"][0]["fully_visible"] = False
            locator_text(wrapper, paths)
        self.pilot = original(self.root, "occluded", transform=occluded)
        self.freeze()
        clients = []
        def factory(budget):
            clients.append(Client(budget))
            return clients[-1]
        result = self.run_pilot(factory)
        self.assertEqual(6, len(clients))
        self.assertEqual(6, result["usage"]["sam_requests"])
        self.assertEqual(0, result["usage"]["spark_requests"])
        self.assertTrue(all(row["prediction"] == "unknown" for row in result["results"]))
        self.assertIn("cached_not_fully_visible", result["results"][0]["semantic_challenge"]["frames"][0]["reasons"])

    def test_unchanged_point_eight_threshold_withholds_low_score_before_spark(self):
        self.freeze()
        result = self.run_pilot(lambda budget: Client(budget, score=.79))
        self.assertEqual(6, result["usage"]["request_attempts"])
        self.assertTrue(all(row["prediction"] == "unknown" for row in result["results"]))
        self.assertEqual(0, result["usage"]["spark_requests"])
        self.assertEqual([.79], result["results"][0]["frames"][0]["scores"])

    def test_zero_and_multiple_concept_detections_complete_diagnostics_but_cannot_choose_target(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.output = self.root / "artifacts/local" / ("candidates-" + str(count))
                self.freeze()
                def factory(budget):
                    client = Client(budget)
                    actual = client.segment_image
                    def segment(*args, **kwargs):
                        result = actual(*args, **kwargs)
                        text = "" if count == 0 else result["text"] + "," + result["text"][4:].replace("1<|box;", "2<|box;", 1)
                        result["text"] = result["response"]["output_text"] = text
                        result["objects"] = decoded(parse_sam_boxes(text))
                        return result
                    client.segment_image = segment
                    return client
                result = self.run_pilot(factory)
                self.assertEqual("completed", result["status"])
                self.assertEqual(6, result["usage"]["request_attempts"])
                self.assertTrue(all(row["prediction"] == "unknown" for row in result["results"]))
                self.assertTrue(all(row["frames"][0]["object_count"] == count for row in result["results"]))

    def test_mask_semantic_merged_distractor_veto_cannot_upscore_or_clear_two_view_gate(self):
        self.freeze()
        result = self.run_pilot(lambda budget: Client(budget,
            mutate=lambda parsed: parsed["frames"][0].update(mask_target_only="contradicted")))
        self.assertEqual(9, result["usage"]["request_attempts"])
        self.assertTrue(all(row["prediction"] == "unknown" for row in result["results"]))
        for row in result["results"]:
            self.assertEqual([.95], row["frames"][0]["scores"])
            self.assertEqual("unknown", row["semantic_challenge"]["assessment"]["decision"])

    def test_sam_paid_network_failure_is_missing_not_unknown_and_other_parts_continue(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = Client(budget, sam_error=MetaAPIError("PRIVATE_PROVIDER_TEXT", code="network_error") if not clients else None)
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual("incomplete", result["status"])
        self.assertEqual(8, len(clients))
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertIsNone(result["results"][0]["prediction"])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.assertEqual(1, result["metrics"][0]["metrics"]["missing_predictions"])

    def test_one_local_pair_preparation_error_preserves_paid_evidence_and_later_pairs_continue(self):
        self.freeze()
        actual = mask_semantics.prepare_pair
        calls = []
        def prepare(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise ValidationError("Authored image renderer failure")
            return actual(*args, **kwargs)
        with patch.object(mask_semantics, "prepare_pair", side_effect=prepare):
            result = self.run_pilot()
        self.assertEqual(3, len(calls))
        self.assertEqual("incomplete", result["status"])
        self.assertEqual(8, result["usage"]["request_attempts"])
        self.assertIsNone(result["results"][0]["prediction"])
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertEqual("ValidationError", result["results"][0]["preparation_error"]["error_type"])
        self.assertEqual(["pass", "pass"], [row["prediction"] for row in result["results"][1:]])

    def test_missing_official_parser_prevents_paid_admission_and_run_marker(self):
        self.freeze()
        with patch("proofloop.meta._sam_decoder", side_effect=MetaAPIError("Authored missing parser")):
            with self.assertRaises(MetaAPIError): self.run_pilot(lambda _: self.fail("No paid admission"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_every_http_stop_status_blocks_all_remaining_stages(self):
        for code in (401, 402, 403, 429):
            with self.subTest(code=code):
                self.output = self.root / "artifacts/local" / ("stop-" + str(code))
                self.freeze()
                clients = []
                def factory(budget):
                    client = Client(budget, sam_error=MetaAPIError("private", code="authored_http", status=code))
                    clients.append(client)
                    return client
                result = self.run_pilot(factory)
                self.assertEqual(1, len(clients))
                self.assertEqual(code, result["admission_stop_code"])
                self.assertEqual(0, result["usage"]["spark_requests"])
                self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])
                self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_actual_http_error_string_code_is_not_mistaken_for_status(self):
        self.freeze()
        clients = []
        def factory(budget):
            def opener(*args, **kwargs):
                raise HTTPError("https://api.meta.ai/v1/responses", 402, "authored", {},
                    io.BytesIO(b'{"error":{"code":"billing_issue","message":"PRIVATE_PROVIDER_TEXT"}}'))
            client = MetaClient("authored-not-a-key", max_requests=budget, retries=0, opener=opener)
            client.segment_image = lambda *args, **kwargs: client._request("POST", "responses", {"model": SAM_MODEL})
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(1, len(clients))
        self.assertEqual(402, result["admission_stop_code"])
        self.assertTrue(result["results"][0]["frames"][0]["provider_response_preserved"])
        self.assertNotIn("PRIVATE_PROVIDER_TEXT", json.dumps(result))

    def test_failed_spark_remains_missing_with_bound_private_response_and_global_stop(self):
        self.freeze()
        clients = []
        response = {"error": {"code": "rate_limited", "message": "PRIVATE_PROVIDER_TEXT"}}
        def factory(budget):
            client = Client(budget, error=("http_error", response, "rate_limited", 429))
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(7, len(clients))
        self.assertEqual(429, result["admission_stop_code"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])
        self.assertIsNone(result["results"][0]["prediction"])
        self.assertEqual("unscored", result["results"][0]["semantic_challenge"]["assessment"]["status"])

    def test_incomplete_spark_output_is_missing_and_later_pairs_still_complete(self):
        self.freeze()
        clients = []
        response = {"model": SPARK_MODEL, "status": "incomplete", "usage": {"input_tokens": 8, "output_tokens": 12}}
        def factory(budget):
            client = Client(budget, error=("invalid_response", response, "incomplete_response", None) if len(clients) == 6 else None)
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(9, len(clients))
        self.assertIsNone(result["admission_stop_code"])
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertIsNone(result["results"][0]["prediction"])
        self.assertEqual(["pass", "pass"], [row["prediction"] for row in result["results"][1:]])

    def test_failed_child_spark_accounting_cannot_be_erased_only_in_outer_report(self):
        self.freeze()
        clients = []
        response = {"model": SPARK_MODEL, "status": "incomplete", "usage": {"input_tokens": 1000, "output_tokens": 4000}}
        def factory(budget):
            client = Client(budget, error=("invalid_response", response, "incomplete_response", None) if len(clients) == 6 else None)
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(9, result["usage"]["request_attempts"])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        original_bytes = path.read_bytes()
        state = json.loads(original_bytes)
        key = next(key for key, value in state["attempts"].items() if key.startswith("spark/") and value["status"] == "error")
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                state["attempts"][key]["usage"] = experiment.accounting._ledger_usage([], incomplete=incomplete)
                benchmark._save(path, state)
                with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_reused_wrong_budget_retry_or_dirty_client_is_never_dispatched(self):
        for mutation in (lambda client: setattr(client, "max_requests", 2),
                         lambda client: setattr(client, "retries", 1),
                         lambda client: client.ledger.append({"model": SAM_MODEL, "status": "ok"})):
            with self.subTest(mutation=mutation):
                self.output = self.root / "artifacts/local" / ("admission-" + str(id(mutation)))
                self.freeze()
                clients = []
                def factory(budget):
                    client = Client(budget)
                    mutation(client)
                    clients.append(client)
                    return client
                result = self.run_pilot(factory)
                self.assertTrue(all(not client.calls for client in clients))
                self.assertEqual(0, result["usage"]["request_attempts"])
                self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_same_client_cannot_be_reused_for_the_next_paid_part(self):
        self.freeze()
        client = Client()
        result = self.run_pilot(lambda _: client)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(1, result["usage"]["request_attempts"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])

    def test_approval_media_locator_or_code_tamper_prevents_first_admission(self):
        plan = self.freeze()
        plan_path = self.output / "plan.json"
        data = plan_path.read_bytes()
        value = json.loads(data)
        value["cases"][0]["envelope"]["approval"]["authorization_reference"] = "altered"
        benchmark._save(plan_path, value)
        with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No admission"))
        plan_path.write_bytes(data)
        with patch.object(experiment, "_implementation", return_value={}):
            with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No admission"))
        frame = plan["cases"][0]["envelope"]["contract"]["cases"][0]["frames"][0]
        (self.root / frame["file"]).write_bytes(b"corrupted source")
        with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No admission"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_sam_response_text_or_ledger_tamper_rejects_saved_result(self):
        self.freeze()
        self.run_pilot()
        (self.output / "summary.json").unlink()
        state_file = self.output / "run-state.json"
        state_bytes = state_file.read_bytes()
        state = json.loads(state_bytes)
        first = next(iter(state["attempts"].values()))
        first["usage"]["request_attempts"] = 0
        benchmark._save(state_file, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)
        state_file.write_bytes(state_bytes)
        state = json.loads(state_bytes)
        key = next(key for key in state["attempts"] if key.startswith("sam/"))
        identifier = key.split("/")[1]
        part = self.output / "parts" / identifier / "sam-0.json"
        value = json.loads(part.read_text())
        value["objects"][0]["confidence"] = 1.0
        benchmark._save(part, value)
        state["attempts"][key]["response_sha256"] = benchmark._file_hash(part)
        benchmark._save(state_file, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_unplanned_response_or_forged_summary_is_rejected(self):
        self.freeze()
        self.run_pilot()
        summary_path = self.output / "summary.json"
        value = json.loads(summary_path.read_text())
        value["results"][0]["prediction"] = "unknown"
        benchmark._save(summary_path, value)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)
        summary_path.unlink()
        benchmark._save(self.output / "parts" / "unplanned.json", {})
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_missing_accounting_flags_or_changed_response_token_usage_invalidates_replay(self):
        self.freeze()
        self.run_pilot()
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        original_bytes = path.read_bytes()
        state = json.loads(original_bytes)
        first = next(iter(state["attempts"].values()))
        first["usage"].pop("accounting_incomplete")
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)
        path.write_bytes(original_bytes)
        state = json.loads(original_bytes)
        key = next(key for key in state["attempts"] if key.startswith("sam/"))
        identifier = key.split("/")[1]
        response_file = self.output / "parts" / identifier / "sam-0.json"
        response = json.loads(response_file.read_text())
        response["response"]["usage"] = {"input_tokens": 10000, "output_tokens": 10000}
        benchmark._save(response_file, response)
        state["attempts"][key]["response_sha256"] = benchmark._file_hash(response_file)
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_network_error_cannot_erase_entire_ledger_to_claim_known_zero_paid_work(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = Client(budget, sam_error=MetaAPIError("Authored lost response", code="network_error") if not clients else None)
            clients.append(client)
            return client
        self.run_pilot(factory)
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        key = next(key for key in state["attempts"] if key.startswith("sam/"))
        state["attempts"][key]["usage"] = experiment.accounting._ledger_usage([])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_dispatched_error_with_no_retained_ledger_is_uncertain_not_known_zero(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = Client(budget)
            if not clients:
                def fail(*args, **kwargs):
                    raise MetaAPIError("Authored lost child ledger", code="network_error")
                client.segment_image = fail
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.assertTrue(result["usage"]["request_attempts_incomplete"])
        self.assertTrue(result["results"][0]["frames"][0]["usage"]["accounting_incomplete"])

    def test_failed_sam_parse_retains_paid_success_and_rejects_network_ledger_relabeling(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = Client(budget)
            if not clients:
                actual = client.segment_image
                def invalid(*args, **kwargs):
                    value = actual(*args, **kwargs)
                    value["objects"][0]["confidence"] = .99
                    return value
                client.segment_image = invalid
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(8, result["usage"]["request_attempts"])
        self.assertIsNone(result["results"][0]["prediction"])
        self.assertFalse(result["usage"]["unreported_charge_possible"])
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        key = next(key for key in state["attempts"] if key.startswith("sam/"))
        state["attempts"][key]["usage"] = experiment.accounting._ledger_usage([
            {"model": SAM_MODEL, "endpoint": "responses", "attempt": 1, "status": "network_error"}])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_failed_sam_provider_body_tokens_cannot_disagree_with_failed_attempt_ledger(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = Client(budget)
            if not clients:
                response = {"model": SAM_MODEL, "status": "completed", "usage": {"input_tokens": 0, "output_tokens": 0}, "output_text": "invalid SAM"}
                def invalid(*args, **kwargs):
                    client.ledger.append({"model": SAM_MODEL, "endpoint": "responses", "attempt": 1,
                                          "status": "ok", "usage": deepcopy(response["usage"])})
                    raise MetaAPIError("Authored parser rejection", code="sam_parse_failed", response=response)
                client.segment_image = invalid
            clients.append(client)
            return client
        self.run_pilot(factory)
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        key = next(key for key in state["attempts"] if key.startswith("sam/"))
        state["attempts"][key]["usage"] = experiment.accounting._ledger_usage([
            {"model": SAM_MODEL, "endpoint": "responses", "attempt": 1, "status": "ok",
             "usage": {"input_tokens": 100, "output_tokens": 100}}])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_public_projection_excludes_credentials_raw_text_masks_private_paths_and_authorization(self):
        self.freeze()
        result = self.run_pilot()
        text = json.dumps(result)
        for forbidden in ("PRIVATE_AUTHORIZATION", "PRIVATE_PROVIDER_TEXT", "inputs/local", "artifacts/local",
                          "encoded_payload", "raster_base64", "output_text", "do-not-export", "authored-not-a-key"):
            self.assertNotIn(forbidden, text)
        self.assertEqual("post_outcome_exposed_development", result["evaluation_role"])
        self.assertFalse(result["physical_action_execution_verified"])


if __name__ == "__main__":
    unittest.main()
