"""Trial scheduling/replay tests use explicit adapter fixtures and no inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.semantic import _hash
from proofloop import semantic_trial as trial


class SemanticTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.started = []
        self.patches = [
            patch.object(trial.method, "verify_prepared_v2", self.verify),
            patch.object(trial.method, "request_binding", self.binding),
            patch.object(trial.method, "public_challenge_summary_v2", lambda report: deepcopy(report["public"])),
            patch.object(trial, "_implementations", return_value={"test_fixture": "a" * 64}),
        ]
        for mock in self.patches:
            mock.start()
            self.addCleanup(mock.stop)

    def verify(self, prepared):
        if hashlib.sha256(Path(prepared["source_file"]).read_bytes()).hexdigest() != prepared["source_sha256"]:
            raise ValidationError("Fixture source changed")

    def binding(self, prepared):
        return {"prepared_hash": prepared["prepared_hash"], "source_sha256": [prepared["source_sha256"]]}

    def entry(self, case_id, expected="supported", origin="saved_spark_locator"):
        folder = self.root / "inputs" / "local" / case_id
        folder.mkdir(parents=True)
        source = folder / "source.jpg"
        source.write_bytes(b"explicit adapter fixture " + case_id.encode())
        prepared = {"source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "anchor_origin": origin}
        prepared["prepared_hash"] = _hash(prepared)
        prepared_path = folder / "prepared.json"
        prepared_path.write_text(json.dumps(prepared))
        return {"case_id": case_id, "expected_identity": expected, "source_family": "fixture",
                "source_id": "shared-fixture-recording", "source_timestamps_ms": [100, 200],
                "anchor_origin": origin, "source_attribution": {
                    "page_url": "https://example.test/source", "license": "Authored test fixture"},
                "prepared_file": str(prepared_path.relative_to(self.root)),
                "prepared_hash": prepared["prepared_hash"]}

    def freeze(self, entries):
        return trial.freeze_trial(entries, "artifacts/local/trial", phase="completion", root=self.root)

    def client(self):
        value = SimpleNamespace(retries=0, max_requests=1, ledger=[])
        self.started.append(value)
        return value

    def attempt(self, client, prepared):
        case_id = Path(prepared["source_file"]).parent.name
        status, decision = self.outcomes[case_id]
        public = {"status": status, "challenge": {"decision": decision} if status == "completed" else None,
                  "usage": {"request_attempts": 1, "input_tokens": 10, "output_tokens": 20, "estimated_usd": .01}}
        client.ledger.append({"explicit_fixture": True})
        return {"prepared": deepcopy(prepared), "request_binding": self.binding(prepared), "public": public,
                "private_raw_response": "must never enter public DTO"}

    def run_trial(self):
        with patch.object(trial.method, "challenge_candidate_pair_v2", self.attempt):
            return trial.run_trial("artifacts/local/trial", self.client, root=self.root)

    def test_freeze_has_code_config_source_and_request_bindings(self):
        entry = self.entry("case-1")
        plan = self.freeze([entry])
        self.assertEqual(1, plan["max_requests"])
        self.assertEqual(trial.method.METHOD_CONFIG, plan["method_config"])
        self.assertEqual({"test_fixture": "a" * 64}, plan["implementation_sha256"])
        self.assertEqual(entry["prepared_hash"], plan["cases"][0]["request_binding"]["prepared_hash"])
        self.assertEqual(64, len(plan["cases"][0]["prepared_file_sha256"]))

    def test_unattempted_and_zero_denominators_are_not_predictions(self):
        self.freeze([self.entry("case-1")])
        summary = trial.public_trial_summary("artifacts/local/trial", root=self.root)
        self.assertEqual("not_attempted", summary["cases"][0]["attempt_status"])
        self.assertIsNone(summary["cases"][0]["matches_expected"])
        self.assertIsNone(summary["metrics"]["completion_rate"])
        self.assertIsNone(summary["metrics"]["scored_accuracy"])
        self.assertIsNone(summary["metrics"]["known_negative"]["unsupported_acceptance_rate"])
        self.assertEqual(0, summary["cost_summary"]["estimated_usd"])

    def test_partial_failure_continues_once_and_replays_without_calls(self):
        entries = [self.entry("positive"), self.entry("failure", "unknown"), self.entry("negative", "unknown"),
                   self.entry("false-support", "unknown", "authored_candidate")]
        self.freeze(entries)
        self.outcomes = {"positive": ("completed", "supported"), "failure": ("unscored", None),
                         "negative": ("completed", "unknown"), "false-support": ("completed", "supported")}
        summary = self.run_trial()
        self.assertEqual(4, len(self.started))
        self.assertEqual(4, len({id(client) for client in self.started}))
        self.assertEqual(3, summary["metrics"]["completed"])
        self.assertEqual(1, summary["metrics"]["unscored"])
        self.assertEqual(.75, summary["metrics"]["completion_rate"])
        self.assertEqual(2 / 3, summary["metrics"]["scored_accuracy"])
        self.assertEqual(1, summary["metrics"]["known_positive"]["retention_rate"])
        self.assertEqual(.5, summary["metrics"]["known_negative"]["unsupported_acceptance_rate"])
        self.assertEqual(.04, summary["cost_summary"]["estimated_usd"])
        self.assertNotIn("private_raw_response", json.dumps(summary))
        self.assertNotIn("prepared_file", json.dumps(summary))
        with patch.object(trial.method, "challenge_candidate_pair_v2", side_effect=AssertionError("Replay called API")):
            self.assertEqual(summary, trial.public_trial_summary("artifacts/local/trial", root=self.root))

    def test_rerun_is_rejected_before_any_new_client(self):
        self.freeze([self.entry("case-1")])
        self.outcomes = {"case-1": ("unscored", None)}
        self.run_trial()
        with self.assertRaises(ValidationError):
            self.run_trial()
        self.assertEqual(1, len(self.started))

    def test_changed_source_stops_before_dispatch(self):
        entry = self.entry("case-1")
        self.freeze([entry])
        (self.root / entry["prepared_file"]).with_name("source.jpg").write_bytes(b"changed")
        with self.assertRaises(ValidationError):
            self.run_trial()
        self.assertEqual([], self.started)

    def test_changed_code_and_frozen_config_are_rejected(self):
        self.freeze([self.entry("case-1")])
        with patch.object(trial, "_implementations", return_value={"test_fixture": "b" * 64}):
            with self.assertRaises(ValidationError):
                trial.public_trial_summary("artifacts/local/trial", root=self.root)
        path = self.root / "artifacts/local/trial/freeze.json"
        plan = json.loads(path.read_text())
        plan["method_config"]["automatic_retries"] = 1
        plan["freeze_hash"] = _hash({key: value for key, value in plan.items() if key != "freeze_hash"})
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValidationError):
            self.run_trial()

    def test_prepared_file_bytes_change_cannot_hide_behind_prepared_hash(self):
        entry = self.entry("case-1")
        self.freeze([entry])
        path = self.root / entry["prepared_file"]
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValidationError):
            self.run_trial()

    def test_authored_origin_cannot_claim_a_model_locator(self):
        entry = self.entry("case-1", origin="authored_candidate")
        entry["anchor_origin"] = "saved_spark_locator"
        with self.assertRaises(ValidationError):
            self.freeze([entry])

    def test_duplicate_preparation_and_bad_limits_stop_before_freeze(self):
        entry = self.entry("case-1")
        for entries in ([], [entry] * 9, [entry, dict(entry, case_id="another")]):
            with self.subTest(count=len(entries)):
                with self.assertRaises(ValidationError):
                    self.freeze(entries)

    def test_source_attribution_and_path_boundary_are_checked(self):
        entry = self.entry("case-1")
        for replacement in ({"source_attribution": {"page_url": "http://example.test", "license": "fixture"}},
                            {"prepared_file": "../outside.json"}, {"case_id": "freeze"},
                            {"source_timestamps_ms": [True, 200]}):
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValidationError):
                    self.freeze([dict(entry, **replacement)])

    def test_trial_factory_retry_or_existing_ledger_rejected(self):
        self.freeze([self.entry("case-1")])
        with self.assertRaises(ValidationError):
            trial.run_trial("artifacts/local/trial", lambda: SimpleNamespace(retries=1, max_requests=1, ledger=[]),
                            root=self.root)

    def test_unreviewed_saved_attempt_rejected_by_replay(self):
        self.freeze([self.entry("case-1")])
        (self.root / "artifacts/local/trial/extra.json").write_text("{}")
        with self.assertRaises(ValidationError):
            trial.public_trial_summary("artifacts/local/trial", root=self.root)


if __name__ == "__main__":
    unittest.main()
