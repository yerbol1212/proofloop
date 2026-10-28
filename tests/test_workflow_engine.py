"""Authored gate transport/rehash fixtures, without observed action claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.interactive import InteractiveEngine
from tests.test_interactive import FakeClient, FakeCompileClient, REGIONS, image, picture


class WorkflowEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = FakeClient()
        self.engine = InteractiveEngine(self.root, client_factory=lambda: self.client,
                                        compile_client_factory=lambda: FakeCompileClient({'fork': 'left', 'spoon': 'right'}))
        draft = self.engine.draft_task({'requirements': 'Fork LEFT spoon RIGHT', 'objects': ['fork', 'spoon'],
                                        'confirm_unique_targets': True})['task_draft']
        self.engine.activate_task({'draft_id': draft['draft_id'], 'draft_hash': draft['draft_hash'],
                                  'confirm_requirements': True, 'confirm_unique_targets': True,
                                  'review_origin': 'assistant_under_user_authorization',
                                  'authorization_reference': 'User requested implementation and controlled testing.'})

    def inspect(self):
        state = self.engine.new_inspection()
        return self.engine.inspect_pair({'capture_token': state['capture_token'], 'source': 'upload',
            'images': [image(41), image(42)], 'regions': deepcopy(REGIONS), 'confirm_regions': True,
            'confirm_unique_targets': True, 'confirm_fresh_capture': True, 'confirm_pair_spacing': True})

    def binding(self):
        state = self.engine.snapshot()
        return {'task_hash': state['active_task']['task_hash'], 'revision': state['revision']}

    def declaration(self):
        return {**self.binding(), 'confirm_operator_continuation': True,
                'review_origin': 'assistant_under_user_authorization',
                'authorization_reference': 'Controlled software QA under the user implementation request.'}

    def test_gate_checks_replay_and_continuation_is_a_separate_sanitized_declaration(self):
        with patch('proofloop.interactive._now_ms', return_value=10000):
            self.assertEqual('pass', self.inspect()['status'])
            self.assertFalse(self.engine.snapshot()['workflow']['can_continue'])
            self.assertTrue(self.engine.check_workflow(self.binding())['workflow']['can_continue'])
            state = self.engine.continue_workflow(self.declaration())
        declared = state['workflow']['declaration']
        self.assertFalse(declared['human_declared'])
        self.assertFalse(declared['physical_action_verified'])
        self.assertEqual('declared', state['workflow']['phase'])
        for private in ('authorization_reference', str(self.root), 'artifacts/local', 'workflow_continuation'):
            self.assertNotIn(private, json.dumps(state))
        records = list(self.engine._directory.glob('workflow-*.json'))
        self.assertEqual(1, len(records))
        self.assertEqual(declared['record_hash'], json.loads(records[0].read_text())['receipt_hash'])
        self.assertEqual(0o600, records[0].stat().st_mode & 0o777)

    def test_media_changed_after_successful_check_cannot_continue(self):
        with patch('proofloop.interactive._now_ms', return_value=10000):
            self.inspect()
            self.engine.check_workflow(self.binding())
            source = self.root / self.engine._evidence_record['frames'][0]['file']
            source.write_bytes(picture(99))
            with self.assertRaises(ValidationError):
                self.engine.continue_workflow(self.declaration())
            self.assertFalse(self.engine.snapshot()['workflow']['can_continue'])
        self.assertEqual([], list(self.engine._directory.glob('workflow-*.json')))

    def test_fail_unknown_and_expired_evidence_block_declaration(self):
        for options in ({'wrong': True}, {'covered': True}, {}):
            with self.subTest(options=options):
                with tempfile.TemporaryDirectory() as directory:
                    original = self.engine
                    self.engine = InteractiveEngine(Path(directory), client_factory=lambda: FakeClient(**options))
                    self.engine._state['active_task'] = deepcopy(original._state['active_task'])
                    with patch('proofloop.interactive._now_ms', return_value=10000):
                        self.inspect()
                    with patch('proofloop.interactive._now_ms', return_value=50001):
                        self.assertFalse(self.engine.snapshot()['workflow']['can_continue'])
                        with self.assertRaises(ValidationError):
                            self.engine.continue_workflow(self.declaration())
                    self.engine = original

    def test_old_revision_or_injected_replay_flag_cannot_grant_continuation(self):
        with patch('proofloop.interactive._now_ms', return_value=10000):
            self.inspect()
            old = self.declaration()
            with self.assertRaises(ValidationError):
                self.engine.check_workflow({**self.binding(), 'valid': True})
            self.engine.check_workflow(self.binding())
            self.engine.new_inspection()
            with self.assertRaises(ValidationError):
                self.engine.continue_workflow(old)
            self.assertFalse(self.engine.snapshot()['workflow']['can_continue'])

    def test_recorded_reference_cannot_pass_current_preparation_gate(self):
        with patch('proofloop.interactive._now_ms', return_value=10000):
            self.inspect()
            self.engine._state['source_kind'] = 'recorded'
            with self.assertRaises(ValidationError):
                self.engine.check_workflow(self.binding())
            with self.assertRaises(ValidationError):
                self.engine.continue_workflow(self.declaration())

    def test_expiration_during_replay_cannot_authorize_continuation(self):
        clock = [10000]
        with patch('proofloop.interactive._now_ms', side_effect=lambda: clock[0]):
            self.inspect()
            payload = self.declaration()
            replay = self.engine._replay_workflow_evidence
            def slow_replay(state):
                result = replay(state)
                clock[0] = 50001
                return result
            with patch.object(self.engine, '_replay_workflow_evidence', side_effect=slow_replay):
                with self.assertRaises(ValidationError):
                    self.engine.continue_workflow(payload)
            self.assertEqual([], list(self.engine._directory.glob('workflow-*.json')))

