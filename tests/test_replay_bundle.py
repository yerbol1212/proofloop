from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop import replay_bundle as replay


class PortableReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bundle.json"
        self.bundle = replay.load_bundle()

    def write(self, bundle=None):
        data = json.dumps(bundle if bundle is not None else self.bundle, sort_keys=True).encode()
        self.path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def test_actual_portable_comparison_reproduces_all_six_published_methods_without_transport(self):
        with patch("proofloop.meta.MetaClient._request", side_effect=AssertionError("No model calls")) as transport:
            result = replay.replay()
        transport.assert_not_called()
        self.assertEqual(0, result["provider_requests"])
        for name, expected in (("exposed_episode_2", (3, 6, 0, 10, 0)), ("new_episode_3", (4, 7, 0, 9, 1))):
            metrics = result["cohorts"][name]["methods"]["v2_veto_current"]["metrics"]
            self.assertEqual(expected, tuple(metrics[key] for key in (
                "positives_retained", "positive_count", "unsupported_support_on_unknown", "unknown_count", "positive_unscored")))
        self.assertEqual(4, len(result["policies"]))
        self.assertFalse(result["physical_action_authorized"])

    def test_every_portable_supported_view_has_current_distinct_bound_risk_free_proofs(self):
        result = replay.replay()
        for item in result["policies"]:
            warnings = {row["frame_index"]: row for row in item["warning_measurements"]}
            for row in item["policy"]["frames"]:
                if row["status"] != "supported":
                    continue
                chain = row["confirmation_evidence"]
                self.assertEqual(row["frame_index"], chain[-1]["frame_index"])
                self.assertEqual(2, len({p["frame_sha256"] for p in chain}))
                self.assertGreaterEqual(chain[-1]["timestamp_ms"] - chain[0]["timestamp_ms"], 500)
                for index in range(chain[0]["frame_index"], row["frame_index"] + 1):
                    self.assertFalse(warnings[index]["veto"])

    def test_original_network_gap_remains_unscored_in_both_current_comparisons(self):
        methods = replay.replay()["cohorts"]["new_episode_3"]["methods"]
        for name in ("v2_current", "v2_veto_current"):
            row = next(x for x in methods[name]["frames"] if x["clip_id"] == "remote_3" and x["frame_index"] == 5)
            self.assertEqual("unscored", row["status"])

    def test_known_contacts_and_clear_false_warning_are_preserved(self):
        policies = {p["clip_id"]: p for p in replay.replay()["policies"]}
        for name, index in (("remote_new", 59), ("remote_3", 59), ("screwdriver_new", 5)):
            self.assertTrue(policies[name]["warning_measurements"][index]["veto"])
            self.assertEqual("unknown", policies[name]["policy"]["frames"][index]["status"])

    def test_modified_bytes_require_original_trusted_digest(self):
        self.write()
        with self.assertRaisesRegex(ValidationError, "trusted receipt"):
            replay.load_bundle(self.path)

    def test_symlink_and_oversized_bundle_rejected(self):
        digest = self.write()
        linked = self.path.with_name("link.json")
        linked.symlink_to(self.path)
        with self.assertRaises(ValidationError):
            replay.load_bundle(linked, expected_sha256=digest)
        with patch.object(replay, "MAX_BUNDLE_BYTES", 3), self.assertRaises(ValidationError):
            replay.load_bundle(self.path, expected_sha256=digest)

    def test_duplicate_keys_and_nonfinite_json_rejected_even_with_matching_digest(self):
        for data in (b'{"schema_version":"1.0","schema_version":"1.0"}', b'{"x":NaN}', b'{"x":1e1000}'):
            self.path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValidationError):
                replay.load_bundle(self.path, expected_sha256=hashlib.sha256(data).hexdigest())

    def test_frozen_implementation_hash_change_rejected(self):
        self.bundle["implementation_sha256"]["tracking_policy.py"] = "0" * 64
        digest = self.write()
        with self.assertRaisesRegex(ValidationError, "dependency"):
            replay.replay(self.path, expected_sha256=digest)

    def test_omitted_dependencies_and_extra_schema_fields_rejected(self):
        for case in ("dependency", "top", "clip"):
            changed = deepcopy(self.bundle)
            if case == "dependency":
                changed["implementation_sha256"].pop("tracking_policy.py")
            elif case == "top":
                changed["invented_authority"] = True
            else:
                changed["clips"][0]["provider_body"] = "private"
            digest = self.write(changed)
            with self.subTest(case=case), self.assertRaises(ValidationError):
                replay.replay(self.path, expected_sha256=digest)

    def test_unsupported_scope_or_threshold_rejected(self):
        for field, value in (("scope", "physical_action_verified"), ("config", {})):
            changed = deepcopy(self.bundle)
            changed[field] = value
            digest = self.write(changed)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                replay.replay(self.path, expected_sha256=digest)

    def test_source_swapped_warning_and_duplicate_frame_rejected(self):
        for change in ("source", "duplicate"):
            changed = deepcopy(self.bundle)
            warning = changed["clips"][0]["warning_measurements"]
            if change == "source":
                warning[0]["source_sha256"] = "0" * 64
            else:
                warning[1] = deepcopy(warning[0])
            digest = self.write(changed)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                replay.replay(self.path, expected_sha256=digest)

    def test_cached_distance_cannot_be_relabelled_as_safe(self):
        warning = self.bundle["clips"][0]["warning_measurements"][5]
        self.assertTrue(warning["veto"])
        warning["veto"] = False
        digest = self.write()
        with self.assertRaisesRegex(ValidationError, "positive geometry"):
            replay.replay(self.path, expected_sha256=digest)

    def test_published_score_mismatch_rejected(self):
        self.bundle["expected_methods"]["exposed_episode_2"]["v2_veto_current"]["metrics"]["positives_retained"] = 6
        digest = self.write()
        with self.assertRaisesRegex(ValidationError, "published comparison"):
            replay.replay(self.path, expected_sha256=digest)

    def test_bundle_is_private_payload_free(self):
        data = replay.default_bundle().read_bytes()
        for forbidden in (b"/Users/", b"artifacts/local/", b"inputs/local/", b"Bearer ", b"data:image", b"data:video", b"encoded_mask"):
            self.assertNotIn(forbidden, data)

    def test_public_inspector_projects_all_recomputed_checkpoint_decisions(self):
        report = replay.replay()
        view = report["inspector"]
        self.assertEqual("tracking_replay_inspector_v1", view["schema_version"])
        self.assertEqual(4, len(view["clips"]))
        self.assertEqual(32, sum(len(clip["checkpoints"]) for clip in view["clips"]))
        self.assertEqual(0, view["provider_requests"])
        self.assertFalse(view["video_pixels_included"])
        self.assertFalse(view["physical_action_authorized"])
        for clip in view["clips"]:
            cohort = report["cohorts"][clip["cohort"]]
            for name in replay.METHODS:
                scored = {row["frame_index"]: row for row in cohort["methods"][name]["frames"]
                          if row["clip_id"] == clip["clip_id"]}
                for checkpoint in clip["checkpoints"]:
                    self.assertEqual(scored[checkpoint["frame_index"]]["status"], checkpoint["methods"][name]["status"])

    def test_public_inspector_preserves_clear_false_warning_and_retention_cost(self):
        view = replay.replay()["inspector"]
        cohort = next(c for c in view["cohorts"] if c["id"] == "exposed_episode_2")
        self.assertEqual(4, cohort["methods"]["v2_current"]["positives_retained"])
        self.assertEqual(3, cohort["methods"]["v2_veto_current"]["positives_retained"])
        clip = next(c for c in view["clips"] if c["clip_id"] == "screwdriver_new")
        frame = next(f for f in clip["checkpoints"] if f["frame_index"] == 5)
        self.assertEqual("supported", frame["reference"]["expected"])
        self.assertEqual("supported", frame["methods"]["v2_current"]["status"])
        self.assertEqual("unknown", frame["methods"]["v2_veto_current"]["status"])
        near = next(c for c in frame["warning"]["comparisons"] if c["near"])
        self.assertEqual(3, near["distance_px"])
        self.assertAlmostEqual(.5444876551628113, near["reported_confidence"])

    def test_public_inspector_preserves_contact_unknown_and_network_unscored(self):
        clips = {c["clip_id"]: c for c in replay.replay()["inspector"]["clips"]}
        contact = next(f for f in clips["remote_new"]["checkpoints"] if f["frame_index"] == 59)
        self.assertTrue(contact["warning"]["veto"])
        self.assertEqual("unknown", contact["methods"]["v2_veto_current"]["status"])
        gap = next(f for f in clips["remote_3"]["checkpoints"] if f["frame_index"] == 5)
        self.assertEqual("unscored", gap["scoring_status"])
        for method in gap["methods"].values():
            self.assertEqual("unscored", method["status"])
            self.assertIn(method["policy_status"], ("supported", "unknown"))

    def test_reference_annotations_cannot_supply_inspector_decisions(self):
        with patch.object(replay, "_inspector_clip", wraps=replay._inspector_clip) as project:
            replay.replay()
        clip, policies = project.call_args_list[0].args
        before = replay._inspector_clip(clip, policies)
        changed = deepcopy(clip)
        for row in changed["references"]:
            row["expected"] = "unknown" if row["expected"] == "supported" else "supported"
            row["condition"] = "scoring_annotation_only"
        after = replay._inspector_clip(changed, policies)
        self.assertNotEqual(before["checkpoints"][0]["reference"], after["checkpoints"][0]["reference"])
        for old, new in zip(before["checkpoints"], after["checkpoints"]):
            self.assertEqual(old["methods"], new["methods"])
            self.assertEqual(old["warning"], new["warning"])

    def test_public_inspector_only_exposes_closed_receipts_and_current_bound_proofs(self):
        view = replay.replay()["inspector"]
        for clip in view["clips"]:
            for frame in clip["checkpoints"]:
                self.assertEqual({"frame_index", "timestamp_ms", "frame_sha256", "reference",
                                  "scoring_status", "methods", "warning"}, set(frame))
                for name in ("v2_current", "v2_veto_current"):
                    method = frame["methods"][name]
                    if method["status"] != "supported":
                        continue
                    self.assertTrue(method["confirmation_ends_at_current"])
                    proofs = method["confirmation_evidence"]
                    self.assertEqual(frame["frame_index"], proofs[-1]["frame_index"])
                    self.assertEqual(frame["frame_sha256"], proofs[-1]["frame_sha256"])
                    self.assertGreaterEqual(proofs[-1]["timestamp_ms"] - proofs[0]["timestamp_ms"], 500)
        data = json.dumps(view).encode()
        for forbidden in (b"/Users/", b"artifacts/local/", b"inputs/local/", b"Bearer ", b"data:image", b"encoded_mask"):
            self.assertNotIn(forbidden, data)


if __name__ == "__main__":
    unittest.main()
