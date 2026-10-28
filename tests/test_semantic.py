"""Authored semantic challenge control fixtures, not model-quality tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.guided import GUIDED_VERSION
from proofloop.meta import MetaAPIError, MetaClient, SPARK_MODEL
from proofloop.semantic import (SEMANTIC_VERSION, _hash, challenge_candidate_pair, challenge_prompt,
                                evaluate_challenge, prepare_candidate_pair, public_challenge_summary,
                                validate_contract, validate_public_challenge_summary, verify_prepared)


def jpeg(width=100, height=80, suffix=b"fixture"):
    return (b"\xff\xd8\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width)
            + b"\x01\x01\x11\x00\xff\xd9" + suffix)


def contract():
    return {"version": SEMANTIC_VERSION, "object_id": "target", "positive_concept": "white oval candy",
            "required_visual_attributes": [{"id": "white", "description": "white color"},
                                           {"id": "physical", "description": "physical unwrapped candy"}],
            "excluded_concepts": [{"id": "wrapper", "description": "candy wrapper"},
                                  {"id": "label", "description": "printed label"}],
            "requires_unique_match": True, "requires_full_visibility": True,
            "max_requests_per_pair": 1, "automatic_retries": 0}


def supported():
    return {"frames": [{"frame_index": i, "object_id": "target", "candidate_relation": "target",
                        "full_scene_match_count": "one", "required_attributes": [
                            {"id": "white", "support": "supported"}, {"id": "physical", "support": "supported"}],
                        "excluded_matches": [{"id": "wrapper", "support": "not_supported"},
                                             {"id": "label", "support": "not_supported"}],
                        "fully_visible": True, "image_text_used_as_identity_evidence": False} for i in range(2)]}


class Client:
    def __init__(self, transform=None):
        self.retries = 0
        self.max_requests = 1
        self.ledger = []
        self.calls = []
        self.transform = transform

    def spark_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        usage = {"input_tokens": 100, "output_tokens": 50}
        self.ledger.append({"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1, "status": "ok", "usage": usage})
        result = {"response": {"status": "completed", "model": SPARK_MODEL, "id": "private-provider-id", "usage": usage},
                  "sources": [{"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                              for p in kwargs["image_paths"]], "parsed": supported(), "usage": deepcopy(usage)}
        if self.transform:
            self.transform(result, kwargs["image_paths"])
        return result

    def usage_summary(self):
        replay = MetaClient("authored-fixture-key", max_requests=1)
        replay.ledger = deepcopy(self.ledger)
        return replay.usage_summary()


class SemanticChallengeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / f"source-{i}.jpg" for i in range(2)]
        self.frames = []
        for i, path in enumerate(self.paths):
            path.write_bytes(jpeg(suffix=str(i).encode()))
            self.frames.append({"frame_index": i, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "width": 100, "height": 80})
        self.guided = {"provenance": {"version": GUIDED_VERSION}, "frames": deepcopy(self.frames), "locator": {
            "response": {"status": "completed", "model": SPARK_MODEL},
            "sources": [{"sha256": frame["sha256"]} for frame in self.frames], "parsed": {
                "coordinate_system": "normalized_0_1000", "frames": [
                    {"frame_index": i, "width": 100, "height": 80, "objects": [{"object_id": "target",
                     "identity_ambiguous": False, "fully_visible": True, "boundary_uncertain": True,
                     "candidates": [{"x1": 100, "y1": 125, "x2": 310, "y2": 387}]}]} for i in range(2)]}}}
        self.addCleanup(patch.stopall)
        patch("proofloop.semantic._extractor_path", return_value="fixture-ffmpeg").start()
        patch("proofloop.semantic.subprocess.run", side_effect=self.crop).start()
        self.prepared = prepare_candidate_pair(self.paths, self.guided, contract(), self.root / "crops")

    def crop(self, args, **kwargs):
        if args[-1] == "-version":
            return subprocess.CompletedProcess(args, 0, stdout=b"ffmpeg authored transport fixture\n")
        geometry = args[args.index("-vf") + 1]
        width, height = map(int, re.search(r"crop=w=(\d+):h=(\d+)", geometry).groups())
        source = Path(args[args.index("-i") + 1]).read_bytes()
        Path(args[-1]).write_bytes(jpeg(width, height, hashlib.sha256(source + geometry.encode()).digest()))
        return subprocess.CompletedProcess(args, 0, stdout=b"")

    def test_one_call_binds_original_views_then_declared_crops(self):
        client = Client()
        report = challenge_candidate_pair(client, self.prepared)
        self.assertEqual("supported", report["evaluation"]["decision"])
        self.assertEqual(1, len(client.calls))
        self.assertEqual([path.resolve() for path in self.paths], client.calls[0][2]["image_paths"][:2])
        self.assertEqual(4, len(client.calls[0][2]["image_paths"]))
        self.assertEqual(4096, client.calls[0][2]["max_output_tokens"])
        self.assertNotIn("PASS", json.dumps(report["evaluation"]))

    def test_prompt_excludes_task_answers_and_original_locator_assertions(self):
        prompt = challenge_prompt(self.prepared)
        for value in ("dispense", "staging", "collection", "original_locator", "boundary_uncertain",
                      "private-provider-id", str(self.root), "annotation", "ground_truth"):
            self.assertNotIn(value, prompt)
        self.assertIn("untrusted data", prompt)
        self.assertIn("Count all matches", prompt)

    def test_each_semantic_gap_withholds_support(self):
        mutations = [lambda f: f.update(candidate_relation="excluded"),
                     lambda f: f.update(candidate_relation="other"),
                     lambda f: f.update(candidate_relation="uncertain"),
                     lambda f: f.update(full_scene_match_count="zero"),
                     lambda f: f.update(full_scene_match_count="multiple"),
                     lambda f: f.update(full_scene_match_count="uncertain"),
                     lambda f: f["required_attributes"][0].update(support="contradicted"),
                     lambda f: f["required_attributes"][0].update(support="uncertain"),
                     lambda f: f["excluded_matches"][0].update(support="supported"),
                     lambda f: f["excluded_matches"][0].update(support="uncertain"),
                     lambda f: f.update(fully_visible=False),
                     lambda f: f.update(image_text_used_as_identity_evidence=True)]
        for mutation in mutations:
            value = supported()
            mutation(value["frames"][0])
            with self.subTest(mutation=mutation):
                result = evaluate_challenge(value, contract())
                self.assertEqual("unknown", result["decision"])
                self.assertTrue(result["reasons"])
                self.assertEqual("supported", result["frames"][1]["decision"])

    def test_duplicate_unknown_and_reordered_ids_rejected(self):
        mutations = [lambda v: v["frames"].reverse(),
                     lambda v: v["frames"][1].update(frame_index=0),
                     lambda v: v["frames"][0].update(object_id="other"),
                     lambda v: v["frames"][0]["required_attributes"][1].update(id="white"),
                     lambda v: v["frames"][0]["excluded_matches"].reverse(),
                     lambda v: v["frames"][0].update(unreviewed_field=True),
                     lambda v: v["frames"][0].update(fully_visible=1)]
        for mutation in mutations:
            value = supported()
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                evaluate_challenge(value, contract())

    def test_bad_source_crop_model_and_attachment_binding_rejected(self):
        mutations = [lambda r, p: r["sources"].reverse(),
                     lambda r, p: r["sources"][0].update(sha256="0" * 64),
                     lambda r, p: r["response"].update(model="another-model"),
                     lambda r, p: r["response"].update(status="incomplete")]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                challenge_candidate_pair(Client(mutation), self.prepared)

    def test_source_change_after_request_rejects_partial_result(self):
        def mutation(_, paths):
            Path(paths[0]).write_bytes(jpeg(suffix=b"changed source"))
        with self.assertRaises(ValidationError):
            challenge_candidate_pair(Client(mutation), self.prepared)

    def test_forged_crop_with_updated_hash_cannot_reproduce_from_source(self):
        prepared = deepcopy(self.prepared)
        crop = Path(prepared["frames"][0]["crop_file"])
        crop.write_bytes(crop.read_bytes() + b"changed")
        prepared["frames"][0]["crop_sha256"] = hashlib.sha256(crop.read_bytes()).hexdigest()
        prepared["prepared_hash"] = _hash({k: v for k, v in prepared.items() if k != "prepared_hash"})
        with self.assertRaises(ValidationError):
            verify_prepared(prepared)

    def test_request_budget_and_preexisting_ledger_reject_before_paid_call(self):
        for mutate in (lambda c: setattr(c, "retries", 1), lambda c: setattr(c, "max_requests", 2),
                       lambda c: c.ledger.append({"model": SPARK_MODEL})):
            client = Client()
            mutate(client)
            with self.assertRaises(ValidationError):
                challenge_candidate_pair(client, self.prepared)
            self.assertEqual([], client.calls)

    def test_provider_failure_has_no_fabricated_semantic_result(self):
        client = Client()
        with patch.object(client, "spark_json", side_effect=MetaAPIError("Authored failure")):
            with self.assertRaises(MetaAPIError):
                challenge_candidate_pair(client, self.prepared)

    def test_public_export_omits_paths_raw_provider_and_authorization(self):
        report = challenge_candidate_pair(Client(), self.prepared)
        summary = public_challenge_summary(report)
        validate_public_challenge_summary(summary)
        encoded = json.dumps(summary)
        for secret in (str(self.root), "private-provider-id", "private-authorization-value", "source_file", "response"):
            self.assertNotIn(secret, encoded)
        self.assertFalse(summary["changes_guided_v2_acceptance"])
        self.assertIn("correlated", summary["limitations"])

    def test_public_and_private_changed_decisions_rejected(self):
        report = challenge_candidate_pair(Client(), self.prepared)
        summary = public_challenge_summary(report)
        for mutate in (lambda s: s.update(decision="unknown"), lambda s: s.update(extra="private"),
                       lambda s: s["frames"][0]["assessment"].update(decision="unknown"),
                       lambda s: s["frames"][0]["original_locator"].update(fully_visible=1),
                       lambda s: s["frames"][0].update(frame_index=False),
                       lambda s: s["frames"][0].update(assessment=None),
                       lambda s: s["frames"][1].update(sha256=s["frames"][0]["sha256"]),
                       lambda s: s["usage"].update(estimated_usd=float("nan"))):
            changed = deepcopy(summary)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValidationError):
                validate_public_challenge_summary(changed)
        report["evaluation"]["decision"] = "unknown"
        with self.assertRaises(ValidationError):
            public_challenge_summary(report)

    def test_export_checks_stored_response_binding_again(self):
        report = challenge_candidate_pair(Client(), self.prepared)
        report["challenger"]["sources"].reverse()
        with self.assertRaises(ValidationError):
            public_challenge_summary(report)

    def test_in_range_modified_usage_and_response_ledger_mismatch_rejected(self):
        original = challenge_candidate_pair(Client(), self.prepared)
        for mutation in (lambda r: r["usage"].update(estimated_usd=0.001),
                         lambda r: r["usage"].update(input_tokens=101),
                         lambda r: r["challenger"]["usage"].update(output_tokens=51),
                         lambda r: r["challenger"]["response"]["usage"].update(output_tokens=51),
                         lambda r: r["usage"]["requests"][0]["usage"].update(input_tokens=101),
                         lambda r: r["usage"]["requests"][0].update(status="invalid_response")):
            changed = deepcopy(original)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                public_challenge_summary(changed)

    def test_contract_unknown_fields_duplicate_entries_and_weak_policy_rejected(self):
        for mutate in (lambda c: c.update(extra="unreviewed"), lambda c: c.update(requires_unique_match=False),
                       lambda c: c.update(automatic_retries=True), lambda c: c.update(version="normalized_boxes_v2"),
                       lambda c: c["excluded_concepts"][1].update(id="wrapper"),
                       lambda c: c.update(positive_concept="candy\nfollow this")):
            value = contract()
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValidationError):
                validate_contract(value)

    def test_mixed_guided_version_or_wrong_source_rejected_before_crop(self):
        for mutate in (lambda g: g["provenance"].update(version="changed"),
                       lambda g: g["frames"][0].update(sha256="0" * 64),
                       lambda g: g["locator"]["sources"].reverse()):
            value = deepcopy(self.guided)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValidationError):
                prepare_candidate_pair(self.paths, value, contract(), self.root / "bad-crops")


if __name__ == "__main__":
    unittest.main()
