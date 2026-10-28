"""Authored aggregate replay controls; no model-quality claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SPARK_MODEL
from proofloop.semantic import SEMANTIC_VERSION
from proofloop.semantic_experiment import (PUBLIC_REPORT, _failed_usage, _freeze, build_public_experiment,
                                          replay_public_experiment)


def usage(input_tokens, output_tokens, status="ok"):
    client = MetaClient("authored-offline-aggregate-fixture", max_requests=1)
    client.ledger = [{"endpoint": "responses", "model": SPARK_MODEL, "attempt": 1, "status": status,
                      "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}]
    return client.usage_summary()


class SemanticExperimentReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "proofloop").mkdir()
        project = Path(__file__).resolve().parents[1]
        for name in ("semantic.py", "guided.py"):
            (self.root / "proofloop" / name).write_bytes((project / "proofloop" / name).read_bytes())
        semantic_hash = hashlib.sha256((self.root / "proofloop/semantic.py").read_bytes()).hexdigest()
        guided_hash = hashlib.sha256((self.root / "proofloop/guided.py").read_bytes()).hexdigest()
        ids = ["candy-e00-early-unsupported", "candy-e00-visible-collection", "coffee-e04-dispense", "coffee-e05-staging"]
        self.entries = [{"case_id": key, "source_family": "candy" if i < 2 else "coffee",
                         "source_id": "candy00" if i < 2 else "coffee" + str(i),
                         "source_timestamps_ms": [500, 1500], "expected_identity": "unknown" if i == 0 else "supported"}
                        for i, key in enumerate(ids)]
        self.freezes = {}
        for suffix, module_hash, parent, prior in (("01", "a" * 64, None, None),
                                                  ("02", "b" * 64, "1" * 64, "a" * 64),
                                                  ("03", semantic_hash, "2" * 64, "b" * 64)):
            self.freezes[suffix] = {"freeze_hash": suffix[-1] * 64, "parent_freeze_hash": parent,
                                   "prior_module_sha256": prior, "module_sha256": module_hash,
                                   "guided_v2_module_sha256": guided_hash, "cases": deepcopy(self.entries),
                                   "max_total_attempts_including_prior_incomplete": 5, "estimated_allocation_usd": 0.25}
        self.records = {}
        base = "artifacts/local/semantic-review-20261028-"
        for suffix, cap in (("01", 2048), ("02", 4096)):
            self.records[base + suffix + "/stopped-error.json"] = {
                "case_id": ids[0], "exception_class": "MetaAPIError", "usage": usage(100, cap, "invalid_response")}
        self.summaries = {}
        for i in (1, 2, 3):
            key = ids[i]
            directory = base + "03/" + key + "/"
            u = usage(100, 4096 if i == 2 else 500, "invalid_response" if i == 2 else "ok")
            attempt = {"attempt_index": i + 2, "case_id": key, "method_module_sha256": semantic_hash,
                       "output_token_cap": 4096, "status": "incomplete" if i == 2 else "completed",
                       "scored": i != 2, "usage": u}
            if i == 2:
                attempt.update(error_code="incomplete_response", response={"model": SPARK_MODEL,
                    "status": "incomplete", "usage": deepcopy(u["requests"][0]["usage"])})
            else:
                self.records[directory + "live-challenge.json"] = {"prepared": {"fixture_id": key}, "usage": u}
                self.summaries[key] = {"decision": "unknown" if i == 1 else "supported", "usage": {
                    field: u[field] for field in ("request_attempts", "input_tokens", "output_tokens", "estimated_usd")}}
            self.records[directory + "attempt.json"] = attempt
        total = sum(record["usage"]["estimated_usd"] for name, record in self.records.items()
                    if name.endswith("stopped-error.json") or name.endswith("attempt.json"))
        self.records[base + "03/batch-timing.json"] = {"estimated_total_usd": round(total, 6),
                                                      "completed_batch_wall_seconds": 60.0}
        self.addCleanup(patch.stopall)
        patch("proofloop.semantic_experiment._freeze", side_effect=lambda root, suffix: deepcopy(self.freezes[suffix])).start()
        patch("proofloop.semantic_experiment._read", side_effect=lambda root, path: deepcopy(self.records[str(path)])).start()
        patch("proofloop.semantic_experiment._reviewed_candidate", side_effect=lambda root, entry: {
            "fixture_id": entry["case_id"]}).start()
        patch("proofloop.semantic_experiment.public_challenge_summary", side_effect=lambda report: deepcopy(
            self.summaries[report["prepared"]["fixture_id"]])).start()

    def test_completed_and_unavailable_cases_remain_separate_and_all_attempts_costed(self):
        result = build_public_experiment(self.root)
        self.assertEqual(2, len(result["cases"]))
        self.assertEqual(2, len(result["unavailable_cases"]))
        self.assertEqual(0, result["metrics"]["scored_negative_pairs"])
        self.assertEqual(1, result["metrics"]["expected_identity_matches"])
        self.assertEqual(5, result["cost_summary"]["actual_requests"])
        self.assertEqual(3, result["cost_summary"]["unscored_incomplete_requests"])
        self.assertEqual([1, 2], result["unavailable_cases"][0]["attempt_indices"])
        self.assertNotIn("challenge", result["unavailable_cases"][0])
        self.assertFalse(result["unavailable_cases"][0]["raw_response_retained"])
        self.assertTrue(result["unavailable_cases"][1]["raw_response_retained"])
        self.assertEqual("observed_classification_and_usage_ledger_only", result["attempts"][0]["failure_evidence"])
        self.assertFalse(result["attempts"][0]["raw_response_retained"])
        self.assertTrue(result["attempts"][3]["raw_response_retained"])

    def test_public_aggregate_must_exactly_replay(self):
        result = build_public_experiment(self.root)
        self.records[PUBLIC_REPORT] = result
        self.assertEqual(result, replay_public_experiment(self.root))
        self.records[PUBLIC_REPORT]["cost_summary"]["estimated_usd"] += 0.001
        with self.assertRaises(ValidationError):
            replay_public_experiment(self.root)

    def test_failed_ledger_total_tampering_is_rejected_even_in_valid_range(self):
        ledger = usage(100, 2048, "invalid_response")
        ledger["estimated_usd"] += 0.001
        with self.assertRaises(ValidationError):
            _failed_usage(ledger)

    def test_failed_response_usage_must_match_its_ledger(self):
        path = "artifacts/local/semantic-review-20261028-03/coffee-e04-dispense/attempt.json"
        self.records[path]["response"]["usage"]["output_tokens"] = 4095
        with self.assertRaises(ValidationError):
            build_public_experiment(self.root)

    def test_changed_frozen_expected_label_is_rejected(self):
        for frozen in self.freezes.values():
            frozen["cases"][0]["expected_identity"] = "supported"
        with self.assertRaises(ValidationError):
            build_public_experiment(self.root)

    def test_changed_completed_prepared_candidate_is_rejected(self):
        path = "artifacts/local/semantic-review-20261028-03/candy-e00-visible-collection/live-challenge.json"
        self.records[path]["prepared"]["fixture_id"] = "other"
        with self.assertRaises(ValidationError):
            build_public_experiment(self.root)

    def test_changed_current_frozen_method_is_rejected(self):
        (self.root / "proofloop/semantic.py").write_text("changed fixture source")
        with self.assertRaises(ValidationError):
            build_public_experiment(self.root)

    def test_freeze_hash_tampering_is_rejected(self):
        from proofloop.semantic import _hash
        frozen = {"method_version": SEMANTIC_VERSION, "evaluation_role": "development_reprocessing",
                  "automatic_retries": 0, "cases": deepcopy(self.entries)}
        frozen["freeze_hash"] = _hash(frozen)
        self.records["artifacts/local/semantic-review-20261028-01/freeze-approved.json"] = frozen
        _freeze(self.root, "01")
        frozen["automatic_retries"] = 1
        self.records["artifacts/local/semantic-review-20261028-01/freeze-approved.json"] = frozen
        with self.assertRaises(ValidationError):
            _freeze(self.root, "01")


if __name__ == "__main__":
    unittest.main()
