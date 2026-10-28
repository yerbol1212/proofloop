"""Engine admission/atomic-write tests with authored transport and injected clients."""
from copy import deepcopy
import json
import os
import stat
import unittest
from unittest.mock import Mock, patch

from proofloop.core import ValidationError
from proofloop.verification_policy import inspection_binding
from tests import test_interactive as transport


class RuntimeAdmissionTests(unittest.TestCase):
    def setUp(self):
        # Reuse existing authored image/client helpers without inheriting and
        # accidentally re-running their complete TestCase collection.
        self.fixture = transport.InteractiveTests("test_missing_seed_starts_unknown_without_fake_mistake")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.engine = self.fixture.engine

    def activate(self, *, guided=False):
        self.fixture.install_compiler(placements={"fork": "left", "spoon": "right"})
        task = self.fixture.task_payload()
        task["perception_mode"] = "guided" if guided else "concept"
        draft = self.engine.draft_task(task)["task_draft"]
        self.engine.activate_task(self.fixture.activate_payload(draft))

    def target(self):
        target = self.engine._directory / "atomic-fixture.json"
        self.engine._write(target, {"revision": 1, "complete": True})
        return target

    def temporary_files(self, target):
        return list(target.parent.glob("." + target.name + ".*.tmp"))

    def test_atomic_success_replaces_only_after_complete_private_file_is_synced(self):
        target = self.target()
        original, replace, fsync = target.read_bytes(), os.replace, os.fsync
        events = []
        def synced(descriptor):
            events.append("fsync")
            return fsync(descriptor)
        def replaced(source, destination):
            self.assertEqual(["fsync"], events)
            self.assertEqual(original, target.read_bytes())
            self.assertEqual({"revision": 2, "complete": True}, json.loads(source.read_text()))
            self.assertEqual(0o600, stat.S_IMODE(source.stat().st_mode))
            events.append("replace")
            return replace(source, destination)
        with patch("proofloop.interactive.os.fsync", side_effect=synced), \
             patch("proofloop.interactive.os.replace", side_effect=replaced):
            self.engine._write(target, {"revision": 2, "complete": True})
        self.assertEqual(["fsync", "replace", "fsync"], events)
        self.assertEqual(2, json.loads(target.read_text())["revision"])
        self.assertEqual(0o600, stat.S_IMODE(target.stat().st_mode))
        self.assertEqual([], self.temporary_files(target))

    def test_serialization_failure_preserves_last_complete_file_without_creating_temp(self):
        target = self.target()
        previous = target.read_bytes()
        for bad in ({"unserializable": {1, 2}}, {"nonfinite": float("nan")}):
            with self.subTest(value=bad), patch("proofloop.interactive.os.replace") as replace:
                with self.assertRaises((TypeError, ValueError)):
                    self.engine._write(target, bad)
                replace.assert_not_called()
                self.assertEqual(previous, target.read_bytes())
                self.assertEqual([], self.temporary_files(target))

    def test_replace_failure_preserves_last_complete_file_and_removes_private_temp(self):
        target = self.target()
        previous = target.read_bytes()
        def refuse_replace(source, destination):
            self.assertEqual(0o600, stat.S_IMODE(source.stat().st_mode))
            self.assertEqual(2, json.loads(source.read_text())["revision"])
            raise OSError("Authored atomic replacement failure")
        with patch("proofloop.interactive.os.replace", side_effect=refuse_replace):
            with self.assertRaises(OSError):
                self.engine._write(target, {"revision": 2})
        self.assertEqual(previous, target.read_bytes())
        self.assertEqual([], self.temporary_files(target))

    def test_file_sync_failure_preserves_last_complete_file_and_removes_temp(self):
        target = self.target()
        previous = target.read_bytes()
        with patch("proofloop.interactive.os.fsync", side_effect=OSError("Authored failed file sync")), \
             patch("proofloop.interactive.os.replace") as replace:
            with self.assertRaises(OSError):
                self.engine._write(target, {"revision": 2})
            replace.assert_not_called()
        self.assertEqual(previous, target.read_bytes())
        self.assertEqual([], self.temporary_files(target))

    def test_wrong_binding_stops_before_token_consumption_evidence_writes_and_provider_setup(self):
        self.activate()
        payload = self.fixture.payload()
        before = self.engine.snapshot()
        binding = inspection_binding(before)
        files = {path.relative_to(self.engine._directory): path.read_bytes()
                 for path in self.engine._directory.rglob("*") if path.is_file()}
        for key, wrong in (("session_id", "interactive-another-session"), ("task_hash", "b" * 64),
                           ("protocol_hash", "c" * 64), ("revision", binding["revision"] + 1)):
            runtime = Mock(side_effect=AssertionError("Provider setup was reached"))
            with self.subTest(key=key), patch.object(self.engine, "_write") as write, \
                 patch.object(self.engine, "_validate_pair") as validate:
                with self.assertRaisesRegex(ValidationError, "changed"):
                    self.engine.inspect_pair(payload, expected_binding={**binding, key: wrong},
                                             runtime_client_factory=runtime)
                write.assert_not_called()
                validate.assert_not_called()
                runtime.assert_not_called()
            self.assertEqual(before["capture_token"], self.engine.snapshot()["capture_token"])
            self.assertEqual(before["revision"], self.engine.snapshot()["revision"])
            self.assertEqual([], self.fixture.client.ledger)
        after = {path.relative_to(self.engine._directory): path.read_bytes()
                 for path in self.engine._directory.rglob("*") if path.is_file()}
        self.assertEqual(files, after)

    def test_runtime_concept_factory_receives_fresh_exact_budget_and_zero_retry_policy(self):
        self.activate()
        payload = self.fixture.payload()
        binding = inspection_binding(self.engine.snapshot())
        self.engine._client_factory = None
        clients, arguments = [], []
        def factory(**kwargs):
            arguments.append(kwargs)
            client = transport.FakeClient()
            clients.append(client)
            return client
        result = self.engine.inspect_pair(payload, expected_binding=binding, runtime_client_factory=factory)
        self.assertEqual([{"max_requests": 4, "retries": 0, "timeout": 25}], arguments)
        self.assertEqual("pass", result["status"])
        self.assertEqual(4, len(clients[0].ledger))
        self.assertIsNone(result["capture_token"])
        # A second lifecycle receives a new client rather than silently reusing
        # a prior request ledger or a prior operation's allowance.
        result = self.engine.inspect_pair(self.fixture.payload(index=11), runtime_client_factory=factory)
        self.assertEqual("pass", result["status"])
        self.assertEqual(2, len(clients))
        self.assertIsNot(clients[0], clients[1])
        self.assertEqual([4, 4], [len(client.ledger) for client in clients])

    def test_runtime_guided_factory_receives_locator_plus_per_object_sam_allowance(self):
        self.activate(guided=True)
        payload = self.fixture.guided_payload()
        binding = inspection_binding(self.engine.snapshot())
        self.engine._client_factory = None
        calls = []
        def factory(**kwargs):
            calls.append(kwargs)
            return self.fixture.guided_client()
        result = self.engine.inspect_pair(payload, expected_binding=binding, runtime_client_factory=factory)
        self.assertEqual([{"max_requests": 5, "retries": 0, "timeout": 45}], calls)
        self.assertEqual("pass", result["status"])
        self.assertEqual(["spark", "sam", "sam", "sam", "sam"], [item[0] for item in self.fixture.client.calls])

    def test_explicit_offline_fixture_injection_takes_precedence_over_runtime_factory(self):
        self.activate()
        payload = self.fixture.payload()
        runtime = Mock(side_effect=AssertionError("The runtime client must not replace explicit fixture injection"))
        result = self.engine.inspect_pair(payload, expected_binding=inspection_binding(self.engine.snapshot()),
                                          runtime_client_factory=runtime)
        runtime.assert_not_called()
        self.assertEqual("pass", result["status"])
        self.assertEqual("injected_client_not_verified_live", result["inference_origin"])
        self.assertEqual(4, len(self.fixture.client.ledger))

    def test_reused_or_wrong_budget_runtime_client_is_rejected_before_provider_calls(self):
        self.activate()
        self.engine._client_factory = None
        for field, value in (("max_requests", 3), ("retries", 1), ("ledger", [{"already": "used"}])):
            payload = self.fixture.payload(index=21 + 10 * len(self.engine._seen_hashes))
            client = transport.FakeClient()
            setattr(client, field, value)
            original = deepcopy(client.ledger)
            with self.subTest(field=field):
                result = self.engine.inspect_pair(payload, runtime_client_factory=lambda **kwargs: client)
                self.assertEqual("error", result["phase"])
                self.assertEqual("unknown", result["status"])
                self.assertEqual(original, client.ledger)


if __name__ == "__main__":
    unittest.main()
