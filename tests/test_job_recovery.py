"""Opaque lost-response recovery is read-only and never retries inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from proofloop.jobs import JobConflict, JobIntegrityError, JobStore, JobValidationError
from tests.test_jobs import BINDING


class SubmissionIntentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "jobs.sqlite3"
        self.store = JobStore(self.path)
        self.key = "c" * 64
        self.payload = {"binding": deepcopy(BINDING), "inspection": {
            "images": [{"data_url": "private-image-one"}, {"data_url": "private-image-two"}],
            "capture_token": "private-capture-token"}}
        self.wire = json.dumps({"idempotency_key": self.key, **self.payload}).encode()
        self.fingerprint = hashlib.sha256(self.wire).hexdigest()

    def submit(self, **kwargs):
        return self.store.submit("inspection", self.payload, idempotency_key=self.key,
            binding=BINDING, now_ms=10, submission_fingerprint=kwargs.get("fingerprint", self.fingerprint))

    def lookup(self, store=None, **kwargs):
        return (store or self.store).lookup_intent("inspection", idempotency_key=kwargs.get("key", self.key),
            binding=kwargs.get("binding", BINDING), submission_fingerprint=kwargs.get("fingerprint", self.fingerprint))

    def test_lost_202_and_reload_recovers_same_completed_job_with_one_dispatch(self):
        original = self.submit()  # The caller loses this acknowledgement.
        claimed = self.store.claim_next("worker", now_ms=11)
        token = claimed["lease"]["claim_token"]
        self.store.mark_dispatched(original["id"], token, now_ms=12)
        completed = self.store.complete(original["id"], token, {"revision": 5, "status": "unknown"}, now_ms=13)
        reopened = JobStore(self.path)
        self.assertEqual(completed, self.lookup(reopened))
        self.assertEqual(completed, self.lookup(reopened))
        self.assertEqual(1, len(reopened.list_jobs()))
        self.assertIsNone(reopened.claim_next("new-worker", now_ms=14))
        self.assertEqual(["submitted", "claimed", "dispatched", "succeeded"],
            [row["event_type"] for row in reopened.events(original["id"])])

    def test_queued_recovery_adds_no_event_claim_or_admission(self):
        job = self.submit()
        self.assertEqual(job, self.lookup())
        self.assertEqual(1, self.lookup()["event_count"])
        self.assertEqual(0, self.lookup()["attempt"])
        self.assertFalse(self.lookup()["dispatched"])

    def test_missing_intent_is_read_only_and_does_not_create_a_job(self):
        self.assertIsNone(self.lookup())
        self.assertEqual([], self.store.list_jobs())
        with sqlite3.connect(str(self.path)) as connection:
            self.assertEqual(0, connection.execute("SELECT count(*) FROM job_submission_intents").fetchone()[0])

    def test_stale_binding_and_wrong_original_fingerprint_are_rejected(self):
        self.submit()
        for field, value in (("revision", BINDING["revision"] + 1), ("session_id", "new-session"),
                ("task_hash", "e" * 64), ("protocol_hash", "e" * 64)):
            with self.subTest(field=field), self.assertRaises(JobConflict):
                self.lookup(binding={**BINDING, field: value})
        with self.assertRaises(JobConflict):
            self.lookup(fingerprint="f" * 64)
        self.assertIsNone(self.lookup(key="d" * 64))

    def test_exact_canonical_retry_preserves_original_wire_fingerprint(self):
        job = self.submit()
        other_wire = json.dumps({"inspection": self.payload["inspection"], "binding": BINDING,
            "idempotency_key": self.key}, sort_keys=True, separators=(",", ":")).encode()
        other_fingerprint = hashlib.sha256(other_wire).hexdigest()
        self.assertNotEqual(self.fingerprint, other_fingerprint)
        self.assertEqual(job, self.submit(fingerprint=other_fingerprint))
        self.assertEqual(job, self.lookup())
        with self.assertRaises(JobConflict):
            self.lookup(fingerprint=other_fingerprint)

    def test_legacy_submission_cannot_be_upgraded_by_a_later_retry(self):
        job = self.submit(fingerprint=None)
        self.assertEqual(job, self.submit())
        with self.assertRaises(JobConflict):
            self.lookup()
        self.assertEqual(job, self.store.lookup_submission("inspection", self.payload,
            idempotency_key=self.key, binding=BINDING))

    def test_migration_preserves_legacy_jobs_and_creates_no_recovery_claim(self):
        job = self.submit(fingerprint=None)
        with sqlite3.connect(str(self.path)) as connection:
            connection.executescript("DROP TRIGGER job_submission_intents_no_update; DROP TRIGGER job_submission_intents_no_delete; DROP TABLE job_submission_intents;")
        reopened = JobStore(self.path)
        self.assertEqual(job, reopened.get(job["id"]))
        self.assertTrue(reopened.verify(job["id"])["valid"])
        with self.assertRaises(JobConflict):
            self.lookup(reopened)

    def test_recovery_rejects_malformed_credentials_before_database_lookup(self):
        self.submit()
        for fingerprint in (None, True, "", "F" * 64, "f" * 63, "private/path"):
            with self.subTest(fingerprint=fingerprint), self.assertRaises(JobValidationError):
                self.lookup(fingerprint=fingerprint)
        for key in (None, True, "", "has space", "x" * 257):
            with self.subTest(key=key), self.assertRaises(JobValidationError):
                self.lookup(key=key)

    def test_capability_and_job_admission_commit_atomically(self):
        with patch.object(self.store, "_append", side_effect=RuntimeError("authored failure")):
            with self.assertRaises(RuntimeError):
                self.submit()
        self.assertIsNone(self.lookup())
        with sqlite3.connect(str(self.path)) as connection:
            self.assertEqual(0, connection.execute("SELECT count(*) FROM jobs").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT count(*) FROM job_submission_intents").fetchone()[0])

    def test_recovery_capability_is_immutable_and_its_integrity_is_verified(self):
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE job_submission_intents SET fingerprint=? WHERE job_id=?", ("f" * 64, job["id"]))
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM job_submission_intents WHERE job_id=?", (job["id"],))
            connection.execute("DROP TRIGGER job_submission_intents_no_update")
            connection.execute("UPDATE job_submission_intents SET fingerprint=? WHERE job_id=?", ("f" * 64, job["id"]))
        with self.assertRaises(JobIntegrityError):
            self.lookup()
        self.assertFalse(self.store.verify(job["id"])["valid"])

    def test_changed_payload_event_journal_or_key_binding_cannot_be_recovered(self):
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("UPDATE jobs SET payload_json='{}' WHERE id=?", (job["id"],))
        with self.assertRaises(JobIntegrityError):
            self.lookup()

    def test_public_projection_and_events_do_not_publish_recovery_credentials_or_media(self):
        job = self.submit()
        public = json.dumps({"job": self.lookup(), "events": self.store.events(job["id"])})
        for private in (self.key, self.fingerprint, "private-image-one", "private-image-two", "private-capture-token", "submission_fingerprint"):
            self.assertNotIn(private, public)
        with sqlite3.connect(str(self.path)) as connection:
            capability = connection.execute("SELECT * FROM job_submission_intents").fetchone()
        encoded = json.dumps(capability)
        self.assertNotIn(self.key, encoded)
        self.assertNotIn("private-image", encoded)
        self.assertNotIn("private-capture-token", encoded)


if __name__ == "__main__":
    unittest.main()
