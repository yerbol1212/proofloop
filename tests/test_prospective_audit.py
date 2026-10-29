"""Raw-text and source integrity checks on authored offline pilot fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from proofloop import benchmark, prospective, prospective_audit as audit
from proofloop.core import ValidationError
from proofloop.meta import MetaClient
from tests import test_prospective as fixtures


class TextClient(fixtures.Client):
    def spark_json(self, *args, **kwargs):
        value = super().spark_json(*args, **kwargs)
        text = json.dumps(value["parsed"], sort_keys=True)
        value["response"]["output"] = [
            {"type":"reasoning","summary":[{"text":"PRIVATE_PROVIDER_REASONING"}]},
            {"type":"message","content":[{"type":"output_text","text":text}]}]
        value["text"] = text
        return value


class ProspectiveAuditTests(unittest.TestCase):
    def setUp(self):
        # Reuse the real frozen harness/media fixture setup, not a mocked audit.
        fixtures.ProspectiveTests.setUp(self)
        prospective.freeze(self.envelope,self.output,root=self.root)
        prospective.run(TextClient,self.output,root=self.root)

    def result(self):
        return audit.audit(self.output,root=self.root)

    def raw_path(self, condition="spark"):
        folder = self.output / "parts/initial" / condition / condition
        return folder / ("initial.json" if condition == "guided" else "results.json")

    def reseal(self, path):
        # Model a private-record edit whose enclosing byte hashes were updated:
        # semantic replay must reject a raw/parsed contradiction anyway.
        state_path = self.output / "run-state.json"
        state = json.loads(state_path.read_text())
        for key,entry in state["parts"].items():
            case,condition = key.split("/")
            folder = self.output / "parts" / case / condition
            if path.is_relative_to(folder):
                entry["artifact_sha256"][str(path.relative_to(folder))] = benchmark._file_hash(path)
                break
        benchmark._save(state_path,state)

    def mutate_raw(self, transform, condition="spark"):
        path = self.raw_path(condition)
        value = json.loads(path.read_text())
        transform(value["locator"] if condition == "guided" else value["initial"])
        benchmark._save(path,value)
        self.reseal(path)
        return path

    def test_complete_bound_texts_replay_without_credentials_or_inference(self):
        with patch.object(MetaClient,"_request",side_effect=AssertionError("No transport")), \
             patch.object(MetaClient,"from_env",side_effect=AssertionError("No credentials")):
            receipt = self.result()
            self.assertEqual(receipt,self.result())
        self.assertEqual("passed",receipt["status"])
        self.assertEqual(0,receipt["new_model_requests"])
        self.assertEqual(2,receipt["case_count"])
        self.assertEqual(1,receipt["source_recordings_count"])
        self.assertEqual(4,receipt["source_images_count"])
        self.assertEqual(4,receipt["completed_parts_checked"])
        self.assertEqual(2,receipt["direct_spark_texts_checked"])
        self.assertEqual(2,receipt["guided_locator_texts_checked"])
        self.assertEqual(4,receipt["provider_text_parsed_matches"])
        self.assertEqual(4,receipt["source_pair_bindings_checked"])
        body = {k:v for k,v in receipt.items() if k != "receipt_sha256"}
        self.assertEqual(benchmark.manifest_hash(body),receipt["receipt_sha256"])
        text = json.dumps(receipt)
        for private in ("PRIVATE_PROVIDER", "PRIVATE_AUTHORIZATION", "/Users/private", str(self.root),
                        "inputs/local", "artifacts/local", '"checks":', '"parsed":', "raster_base64"):
            self.assertNotIn(private,text)

    def test_resealed_direct_raw_text_disagrees_with_unmodified_parsed_prediction(self):
        def mismatch(wrapper):
            other = deepcopy(wrapper["parsed"])
            other["checks"][0]["status"] = "fail"
            text = json.dumps(other)
            wrapper["response"]["output"][-1]["content"][0]["text"] = text
            wrapper["text"] = text
        self.mutate_raw(mismatch)
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_resealed_guided_locator_raw_text_disagrees_with_parsed_geometry(self):
        def mismatch(wrapper):
            other = deepcopy(wrapper["parsed"])
            other["frames"][0]["objects"][0]["fully_visible"] = False
            text = json.dumps(other)
            wrapper["response"]["output"][-1]["content"][0]["text"] = text
            wrapper["text"] = text
        self.mutate_raw(mismatch,"guided")
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_duplicate_provider_json_fields_are_rejected_after_resealing(self):
        def duplicate(wrapper):
            text = json.dumps(wrapper["parsed"])
            text = '{"checks":[],' + text[1:]
            wrapper["response"]["output"][-1]["content"][0]["text"] = text
            wrapper["text"] = text
        self.mutate_raw(duplicate)
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_duplicate_saved_wrapper_fields_are_rejected_after_resealing(self):
        path = self.raw_path()
        text = path.read_text()
        duplicate = '"parsed":' + json.dumps(json.loads(text)["initial"]["parsed"]) + ',"parsed":'
        self.assertIn('"parsed":',text)
        path.write_text(text.replace('"parsed":',duplicate,1))
        self.reseal(path)
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_nonfinite_literals_and_overflowing_exponents_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    audit._json('{"untrusted_extra":' + value + '}')
        self.assertEqual({"finite":1e200},audit._json('{"finite":1e200}'))
        path = self.raw_path()
        text = path.read_text()
        path.write_text(text[:-2] + ',"overflow":1e400}\n')
        self.reseal(path)
        with self.assertRaises(ValidationError): self.result()

    def test_invalid_or_empty_provider_text_rejected_without_fabricated_json(self):
        self.mutate_raw(lambda w:w["response"].update(output=[]))
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_malformed_provider_json_is_not_replaced_by_valid_parsed_copy(self):
        def malformed(wrapper):
            wrapper["response"]["output"][-1]["content"][0]["text"] = '{"checks":'
            wrapper["text"] = '{"checks":'
        self.mutate_raw(malformed)
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_conflicting_convenience_text_cannot_hide_message_output(self):
        def conflict(wrapper):
            wrapper["response"]["output_text"] = wrapper["text"]
            other = deepcopy(wrapper["parsed"])
            other["checks"][0]["status"] = "fail"
            wrapper["response"]["output"][-1]["content"][0]["text"] = json.dumps(other)
        self.mutate_raw(conflict)
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_wrapper_text_disagreement_is_rejected(self):
        self.mutate_raw(lambda w:w.update(text="PRIVATE_UNBOUND_TEXT"))
        self.assertEqual("completed",prospective.summarize(self.output,root=self.root)["status"])
        with self.assertRaises(ValidationError): self.result()

    def test_missing_or_reordered_source_bindings_cannot_receive_a_receipt(self):
        self.mutate_raw(lambda w:w.update(sources=[]))
        with self.assertRaises(ValidationError): self.result()

    def test_source_attachment_order_cannot_be_reversed(self):
        self.mutate_raw(lambda w:w["sources"].reverse())
        with self.assertRaises(ValidationError): self.result()

    def test_incomplete_provider_status_cannot_receive_a_receipt(self):
        self.mutate_raw(lambda w:w["response"].update(status="incomplete"))
        with self.assertRaises(ValidationError): self.result()

    def test_changed_source_pixels_fail_fresh_replay(self):
        path = self.root / self.envelope["contract"]["cases"][0]["frames"][0]["file"]
        path.write_bytes(path.read_bytes() + b"modified source")
        with self.assertRaises(ValidationError): self.result()

    def test_unstarted_study_has_no_completed_bound_sources_to_attest(self):
        fresh = self.root / "artifacts/local/fresh-pilot"
        prospective.freeze(self.envelope,fresh,root=self.root)
        with self.assertRaises(ValidationError): audit.audit(fresh,root=self.root)


if __name__ == "__main__":
    unittest.main()
