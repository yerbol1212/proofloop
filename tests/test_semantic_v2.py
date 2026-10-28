"""Authored adapter/ledger fixtures, never semantic model-quality evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, SPARK_MODEL
from proofloop.semantic import challenge_prompt, challenge_schema
from proofloop.semantic_v2 import (METHOD_CONFIG, SEMANTIC_V2_VERSION,
                                  challenge_candidate_pair_v2, public_challenge_summary_v2,
                                  validate_public_challenge_summary_v2, verify_prepared_v2)
from tests import test_semantic as fixtures

Client, jpeg = fixtures.Client, fixtures.jpeg


class ErrorClient(Client):
    def __init__(self, kind="incomplete"):
        super().__init__()
        self.kind = kind

    def spark_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        usage = {"input_tokens": 100, "output_tokens": 4096}
        response = {"id": "private-error-provider-id", "model": SPARK_MODEL, "status": "incomplete",
                    "usage": usage, "incomplete_details": {"reason": "max_output_tokens"},
                    "private_body": "should-never-be-public"}
        entry = {"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
                 "status": "invalid_response", "usage": deepcopy(usage)}
        code, status = "incomplete_response", None
        if self.kind == "network":
            response = None
            entry = {"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
                     "status": "network_error", "error_type": "TimeoutError"}
            code = "network_error"
        elif self.kind == "http":
            response = {"error": {"code": "rate_limit", "message": "private-http-body"}}
            entry = {"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
                     "status": "http_error", "http_status": 429, "error_code": "rate_limit"}
            code, status = "rate_limit", 429
        elif self.kind == "invalid_json":
            response = None
            entry = {"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
                     "status": "invalid_json"}
            code = "invalid_json"
        elif self.kind == "schema":
            response.update(status="completed", output_text="not JSON")
            entry["status"] = "ok"
            code = "schema_validation_failed"
        self.ledger.append(entry)
        raise MetaAPIError("Authored request failure", code=code, status=status, response=response)


class SemanticV2Tests(unittest.TestCase):
    setUp = fixtures.SemanticChallengeTests.setUp
    crop = fixtures.SemanticChallengeTests.crop

    def test_low_reasoning_exactly_one_call_preserves_prompt_schema_and_sources(self):
        client = Client()
        report = challenge_candidate_pair_v2(client, self.prepared)
        self.assertEqual(SEMANTIC_V2_VERSION, report["method_version"])
        self.assertEqual("completed", report["status"])
        self.assertEqual("supported", report["evaluation"]["decision"])
        self.assertEqual(1, len(client.calls))
        prompt, schema, options = client.calls[0]
        self.assertEqual(challenge_prompt(self.prepared), prompt)
        self.assertEqual(challenge_schema(self.prepared["contract"]), schema)
        self.assertEqual("low", options["reasoning_effort"])
        self.assertEqual(4096, options["max_output_tokens"])
        self.assertEqual(METHOD_CONFIG["response_schema_name"], options["name"])
        self.assertEqual([Path(frame["source_file"]) for frame in self.prepared["frames"]]
                         + [Path(frame["crop_file"]) for frame in self.prepared["frames"]], options["image_paths"])

    def test_complete_public_summary_replays_existing_gate_without_acceptance_change(self):
        report = challenge_candidate_pair_v2(Client(), self.prepared)
        summary = public_challenge_summary_v2(report)
        validate_public_challenge_summary_v2(summary)
        self.assertEqual("supported", summary["challenge"]["decision"])
        self.assertIsNone(summary["failure"])
        self.assertFalse(summary["changes_guided_v2_acceptance"])
        self.assertEqual(summary["challenge"]["usage"], summary["usage"])

    def test_original_preparation_and_contract_are_not_mutated(self):
        before = deepcopy(self.prepared)
        challenge_candidate_pair_v2(Client(), self.prepared)
        self.assertEqual(before, self.prepared)

    def test_semantic_conflict_is_completed_unknown_not_api_failure(self):
        def conflict(result, _):
            result["parsed"]["frames"][0]["candidate_relation"] = "excluded"
        summary = public_challenge_summary_v2(challenge_candidate_pair_v2(Client(conflict), self.prepared))
        self.assertEqual("completed", summary["status"])
        self.assertEqual("unknown", summary["challenge"]["decision"])
        self.assertIsNone(summary["failure"])

    def test_incomplete_response_retained_unscored_with_full_billable_tokens(self):
        client = ErrorClient()
        report = challenge_candidate_pair_v2(client, self.prepared)
        summary = public_challenge_summary_v2(report)
        self.assertEqual(1, len(client.calls))
        self.assertEqual("unscored", report["status"])
        self.assertIsNone(report["evaluation"])
        self.assertIsNone(report["challenger"])
        self.assertEqual("private-error-provider-id", report["error"]["response"]["id"])
        self.assertEqual(4096, summary["usage"]["output_tokens"])
        self.assertAlmostEqual(0.017533, summary["usage"]["estimated_usd"], places=6)
        self.assertIsNone(summary["challenge"])
        self.assertFalse(summary["failure"]["scored"])
        self.assertTrue(summary["failure"]["raw_response_retained"])
        self.assertTrue(summary["failure"]["token_usage_reported"])

    def test_unreported_network_failure_not_retried_or_counted_as_free_evidence(self):
        client = ErrorClient("network")
        summary = public_challenge_summary_v2(challenge_candidate_pair_v2(client, self.prepared))
        self.assertEqual(1, len(client.calls))
        self.assertEqual("unscored", summary["status"])
        self.assertFalse(summary["failure"]["token_usage_reported"])
        self.assertFalse(summary["failure"]["raw_response_retained"])
        self.assertIn("unreported charges remain unknown", summary["limitations"])

    def test_http_and_invalid_json_failures_retain_bounded_attempt(self):
        for kind in ("http", "invalid_json"):
            with self.subTest(kind=kind):
                client = ErrorClient(kind)
                summary = public_challenge_summary_v2(challenge_candidate_pair_v2(client, self.prepared))
                self.assertEqual("unscored", summary["status"])
                self.assertEqual(1, summary["usage"]["request_attempts"])
                self.assertIsNone(summary["challenge"])

    def test_schema_failure_keeps_actual_completed_response_unscored(self):
        report = challenge_candidate_pair_v2(ErrorClient("schema"), self.prepared)
        summary = public_challenge_summary_v2(report)
        self.assertEqual("unscored", summary["status"])
        self.assertEqual("completed", summary["failure"]["response_status"])
        self.assertEqual("schema_validation_failed", summary["failure"]["code"])

    def test_valid_json_answer_cannot_be_relabelled_as_schema_failure(self):
        report = challenge_candidate_pair_v2(ErrorClient("schema"), self.prepared)
        from tests.test_semantic import supported
        report["error"]["response"]["output_text"] = json.dumps(supported())
        with self.assertRaises(ValidationError):
            public_challenge_summary_v2(report)

    def test_nonfresh_client_rejected_before_call(self):
        for mutation in (lambda c: setattr(c, "retries", 1), lambda c: setattr(c, "max_requests", 2),
                         lambda c: c.ledger.append({"model": SPARK_MODEL})):
            client = Client()
            mutation(client)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                challenge_candidate_pair_v2(client, self.prepared)
            self.assertEqual([], client.calls)

    def test_source_changed_before_request_is_rejected_without_spend(self):
        self.paths[0].write_bytes(jpeg(suffix=b"changed-before"))
        client = Client()
        with self.assertRaises(ValidationError):
            challenge_candidate_pair_v2(client, self.prepared)
        self.assertEqual([], client.calls)

    def test_source_changed_after_request_cannot_gain_support(self):
        def change(_, paths):
            Path(paths[0]).write_bytes(jpeg(suffix=b"changed-after"))
        report = challenge_candidate_pair_v2(Client(change), self.prepared)
        self.assertEqual("unscored", report["status"])
        self.assertIsNone(report["evaluation"])
        self.assertEqual("source_evidence_changed", report["error"]["code"])
        self.assertIsNotNone(report["challenger"]["response"])
        with self.assertRaises(ValidationError):
            public_challenge_summary_v2(report)

    def test_response_attachment_model_and_status_must_bind(self):
        for mutation in (lambda r, _: r["sources"].reverse(),
                         lambda r, _: r["response"].update(model="other-model"),
                         lambda r, _: r["response"].update(status="incomplete")):
            with self.subTest(mutation=mutation):
                report = challenge_candidate_pair_v2(Client(mutation), self.prepared)
                self.assertEqual("unscored", report["status"])
                self.assertIsNone(report["evaluation"])
                self.assertEqual("local_validation_failed", report["error"]["code"])
                public_challenge_summary_v2(report)

    def test_misordered_ids_retained_as_unscored_local_failure(self):
        def mutation(result, _):
            result["parsed"]["frames"][0]["required_attributes"].reverse()
        report = challenge_candidate_pair_v2(Client(mutation), self.prepared)
        self.assertEqual("unscored", report["status"])
        self.assertIsNone(report["evaluation"])
        self.assertIsNotNone(report["challenger"]["response"])
        public_challenge_summary_v2(report)

    def test_valid_assessment_cannot_be_relabelled_as_local_failure(self):
        report = challenge_candidate_pair_v2(Client(), self.prepared)
        report.update(status="unscored", evaluation=None, error={
            "exception_class": "ValidationError", "code": "local_validation_failed", "http_status": None,
            "response": deepcopy(report["challenger"]["response"])})
        with self.assertRaises(ValidationError):
            public_challenge_summary_v2(report)

    def test_modified_method_prompt_and_source_bindings_fail_replay(self):
        report = challenge_candidate_pair_v2(Client(), self.prepared)
        mutations = (lambda r: r["method_config"].update(reasoning_effort="high"),
                     lambda r: r["method_config"].update(automatic_retries=1),
                     lambda r: r["request_binding"].update(prompt_sha256="0" * 64),
                     lambda r: r["request_binding"]["source_sha256"].reverse(),
                     lambda r: r.update(extra="unreviewed"))
        for mutation in mutations:
            changed = deepcopy(report)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                public_challenge_summary_v2(changed)

    def test_completed_usage_totals_and_raw_response_must_match_ledger(self):
        report = challenge_candidate_pair_v2(Client(), self.prepared)
        for mutation in (lambda r: r["usage"].update(input_tokens=101),
                         lambda r: r["challenger"]["usage"].update(output_tokens=51),
                         lambda r: r["usage"]["requests"][0].update(status="invalid_response")):
            changed = deepcopy(report)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                public_challenge_summary_v2(changed)

    def test_failure_usage_body_status_and_scoring_cannot_be_modified(self):
        report = challenge_candidate_pair_v2(ErrorClient(), self.prepared)
        for mutation in (lambda r: r["usage"].update(estimated_usd=0),
                         lambda r: r["error"]["response"]["usage"].update(output_tokens=100),
                         lambda r: r["error"]["response"].update(status="completed"),
                         lambda r: r["error"]["response"].update(model="another-model"),
                         lambda r: r.update(evaluation={"decision": "unknown"}),
                         lambda r: r.update(status="completed"),
                         lambda r: r["error"].update(response=None)):
            changed = deepcopy(report)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                public_challenge_summary_v2(changed)

    def test_export_omits_private_response_ids_paths_and_messages(self):
        for client in (Client(), ErrorClient(), ErrorClient("http")):
            summary = public_challenge_summary_v2(challenge_candidate_pair_v2(client, self.prepared))
            encoded = json.dumps(summary)
            for private in (str(self.root), "private-provider-id", "private-error-provider-id",
                            "should-never-be-public", "private-http-body", "source_file", "crop_file"):
                self.assertNotIn(private, encoded)

    def test_public_summary_modifications_rejected(self):
        for client in (Client(), ErrorClient()):
            summary = public_challenge_summary_v2(challenge_candidate_pair_v2(client, self.prepared))
            for mutation in (lambda s: s.update(extra="private"),
                             lambda s: s["method_config"].update(max_output_tokens=8192),
                             lambda s: s["usage"].update(estimated_usd=float("nan")),
                             lambda s: s["request_binding"].update(source_sha256=["0" * 64]),
                             lambda s: s.update(changes_guided_v2_acceptance=True)):
                changed = deepcopy(summary)
                mutation(changed)
                with self.subTest(mutation=mutation, status=summary["status"]), self.assertRaises(ValidationError):
                    validate_public_challenge_summary_v2(changed)

    def authored(self):
        from proofloop.candidate_review import prepare_authored_candidate_pair
        from unittest.mock import patch
        with patch("proofloop.candidate_review._extractor_path", return_value="fixture-ffmpeg"):
            return prepare_authored_candidate_pair(self.paths,
                [{"x1": 10, "y1": 10, "x2": 30, "y2": 30}] * 2,
                fixtures.contract(), self.root / "authored-crops",
                review_note="Authored source/crop fixture; no model locator inferred.")

    def test_authored_candidate_dispatch_and_completed_projection_have_no_locator_claim(self):
        prepared = self.authored()
        verify_prepared_v2(prepared)
        report = challenge_candidate_pair_v2(Client(), prepared)
        summary = public_challenge_summary_v2(report)
        validate_public_challenge_summary_v2(summary)
        challenge = summary["challenge"]
        self.assertEqual("authored_candidate", challenge["anchor_origin"])
        self.assertEqual("authored_candidate_pair_v1", challenge["preparation_version"])
        self.assertEqual("supported", challenge["decision"])
        for frame in challenge["frames"]:
            self.assertNotIn("original_locator", frame)
            self.assertNotIn("original_candidate_count", frame)
        self.assertIn("manually authored", challenge["limitations"])
        self.assertNotIn("guided_locator_hash", json.dumps(challenge))

    def test_authored_failure_retains_prepared_origin_without_inventing_assessment(self):
        report = challenge_candidate_pair_v2(ErrorClient(), self.authored())
        self.assertEqual("authored_candidate", report["prepared"]["anchor_origin"])
        summary = public_challenge_summary_v2(report)
        self.assertIsNone(summary["challenge"])
        self.assertEqual("unscored", summary["status"])

    def test_authored_public_model_locator_injection_rejected(self):
        summary = public_challenge_summary_v2(challenge_candidate_pair_v2(Client(), self.authored()))
        for mutation in (lambda s: s["challenge"]["frames"][0].update(original_locator={}),
                         lambda s: s["challenge"].update(anchor_origin="saved_spark_locator"),
                         lambda s: s["challenge"]["author_review"].update(source_views_and_crops_reviewed=False),
                         lambda s: s["challenge"]["frames"][0]["assessment"].update(decision="unknown")):
            changed = deepcopy(summary)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                validate_public_challenge_summary_v2(changed)


if __name__ == "__main__":
    unittest.main()
