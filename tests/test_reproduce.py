"""Public reproduction safeguards; generated transport fixtures, no network or API."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.reproduce import (_copy_bounded, _obtain, _private, approve, main, prepare, profile_plan)


class ReproductionTests(unittest.TestCase):
    def test_default_is_plan_only_with_no_network_or_model_calls(self):
        with patch("proofloop.reproduce.urlopen") as network, patch("proofloop.meta.MetaClient") as model:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main([]))
            value = json.loads(output.getvalue())
            self.assertEqual(0, value["model_requests"])
            self.assertEqual(4, value["frames"])
            self.assertEqual(9970951, value["download_bytes"])
            self.assertFalse(value["automatic_approval"])
            network.assert_not_called()
            model.assert_not_called()

    def test_acquisition_profile_fixes_ordered_disjoint_future_pairs_and_terminal(self):
        plan = profile_plan("coffee-acquisition")
        self.assertEqual(12, plan["frames"])
        self.assertEqual(29889192, plan["download_bytes"])
        e00 = [case["timestamps_ms"] for case in plan["fixed_cases"] if case["case_id"].startswith("coffee-e00")]
        self.assertEqual([[16500, 18500], [19100, 20100], [20700, 21700]], e00)
        for before, after in zip(e00, e00[1:]):
            self.assertGreater(after[0], before[-1])
            self.assertGreaterEqual(after[-1] - after[0], 1000)
        self.assertEqual([500, 21500], plan["fixed_cases"][-1]["timestamps_ms"])

    def test_partial_oversize_and_wrong_hash_downloads_are_not_published(self):
        data = b"authored bounded download fixture"
        for actual, size, digest in ((data, len(data) - 1, hashlib.sha256(data).hexdigest()),
                                     (data[:-1], len(data), hashlib.sha256(data).hexdigest()),
                                     (data, len(data), "0" * 64)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "source.mp4"
                with self.assertRaises(ValidationError):
                    _copy_bounded(io.BytesIO(actual), target, size, digest)
                self.assertFalse(target.exists())
                self.assertFalse(target.with_suffix(".mp4.part").exists())

    def test_exact_download_is_atomic_and_refuses_to_overwrite(self):
        data = b"authored byte-exact test fixture"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "source.mp4"
            _copy_bounded(io.BytesIO(data), target, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(data, target.read_bytes())
            with self.assertRaises(FileExistsError):
                _copy_bounded(io.BytesIO(data), target, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(data, target.read_bytes())

    def test_missing_sources_cannot_download_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp, patch("proofloop.reproduce.urlopen") as network:
            with self.assertRaises(ValidationError):
                _obtain({"bytes": 3, "sha256": "0" * 64}, Path(tmp) / "missing.mp4", False)
            network.assert_not_called()

    def test_modified_cached_media_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp, patch("proofloop.reproduce.urlopen") as network:
            target = Path(tmp) / "source.mp4"
            target.write_bytes(b"modified source")
            with self.assertRaises(ValidationError):
                _obtain({"bytes": 1, "sha256": "0" * 64}, target, True)
            self.assertEqual(b"modified source", target.read_bytes())
            network.assert_not_called()

    def test_output_cannot_escape_ignored_private_folders_or_follow_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/local").mkdir(parents=True)
            (root / "artifacts/local/escape").symlink_to(root)
            for target in (root / "public.json", root / "artifacts/local", root / "artifacts/local/escape/secret"):
                with self.assertRaises(ValidationError):
                    _private(target, root, "artifacts/local")

    def test_preparation_is_unapproved_and_approval_binds_reviewed_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def obtain(record, target, download):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"---\nlicense: mit\n---\n" if target.name == "dataset-card.md" else b"fixture video")

            def extract(source, requested, end, target):
                data = b"\xff\xd8\xff" + str(requested).encode()
                target.write_bytes(data)
                return {"file": target.name, "sha256": hashlib.sha256(data).hexdigest(),
                        "source_timestamp_ms": round(requested * 1000)}

            with patch("proofloop.reproduce._tool"), patch("proofloop.reproduce._obtain", side_effect=obtain), \
                    patch("proofloop.reproduce._extract_frame", side_effect=extract), \
                    patch("proofloop.meta.MetaClient") as model:
                result = prepare("coffee-placement", Path("artifacts/local/reproduction"), root=root)
                self.assertEqual("prepared_unapproved", result["status"])
                self.assertFalse((root / "artifacts/local/reproduction/manifest-approved.json").exists())
                self.assertTrue((root / "inputs/local/reproduction/coffee-placement/MIT-aloha-LICENSE.txt").exists())
                with self.assertRaises(ValidationError):
                    approve(Path("artifacts/local/reproduction"), "0" * 64, root=root)
                # Authored media does not have pinned actual source bytes: a
                # real approval must fail media verification even on this hash.
                with self.assertRaises(ValidationError):
                    approve(Path("artifacts/local/reproduction"), result["contract_hash"], root=root)
                model.assert_not_called()

    def test_preparation_refuses_existing_output_before_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch("proofloop.reproduce.urlopen") as network:
            root = Path(tmp)
            target = root / "artifacts/local/existing"
            target.mkdir(parents=True)
            with self.assertRaises(ValidationError):
                prepare("coffee-placement", target, download=True, root=root)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
