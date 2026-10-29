"""Authored preparation fixtures only; no source download, credentials or Meta API."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_new_source_pilot as pilot


def probe(count=6, *, step=1000):
    return {"streams": [{"width": 640, "height": 480, "time_base": "1/1000"}],
            "format": {"duration": str(count * step / 1000)},
            "frames": [{"media_type": "video", "width": 640, "height": 480,
                        "best_effort_timestamp": i * step,
                        "best_effort_timestamp_time": str(i * step / 1000)} for i in range(count)]}


class PreparationUnitTests(unittest.TestCase):
    def test_default_plan_has_no_network_writes_or_tool_processes(self):
        output = io.StringIO()
        with patch.object(pilot, "prepare") as prepare, patch.object(pilot, "download") as fetch, \
                patch.object(pilot, "run_tool") as tool, redirect_stdout(output):
            self.assertEqual(0, pilot.main([]))
        prepare.assert_not_called(); fetch.assert_not_called(); tool.assert_not_called()
        value = json.loads(output.getvalue())
        self.assertEqual("plan_only", value["mode"])
        self.assertFalse(value["writes"])
        self.assertEqual(0, value["provider_requests"])
        self.assertEqual(["bridge-scripted-0", "ur5-cloth-0", "ur5-cup-4"], [v["id"] for v in value["sources"]])

    def test_checked_download_rejects_size_and_hash_changes(self):
        for data, size, digest in ((b"authored", 7, pilot.sha256(b"authored")),
                                   (b"authored", 8, "0" * 64), (b"", 0, pilot.sha256(b""))):
            with self.assertRaises(pilot.PreparationError):
                pilot.checked_bytes(data, size, digest)
        self.assertEqual(b"authored", pilot.checked_bytes(b"authored", 8, pilot.sha256(b"authored")))

    def test_bounded_reader_rejects_empty_oversize_and_deadline(self):
        for data, limit, clock in ((b"", 4, lambda: 0), (b"12345", 4, lambda: 0),
                                   (b"1234", 4, lambda: 10)):
            with self.assertRaises(pilot.PreparationError):
                pilot.bounded_read(io.BytesIO(data), limit, 10, clock=clock)

    def test_metadata_download_stops_at_first_line(self):
        response = io.BytesIO(b'{"row":1}\n' + b"ignored" * 100)
        self.assertEqual(b'{"row":1}\n', pilot.bounded_read(response, 10, 10, first_line=True, clock=lambda: 0))
        self.assertEqual(10, response.tell())

    def test_reader_checks_monotonic_deadline_after_read(self):
        values = iter((0, 0, 11))
        with self.assertRaises(pilot.PreparationError):
            pilot.bounded_read(io.BytesIO(b"a\n"), 4, 10, clock=lambda: next(values))

    def test_download_and_redirects_reject_untrusted_destinations(self):
        for url in ("http://huggingface.co/a", "https://localhost/a", "https://huggingface.co.evil/a",
                    "https://user:secret@huggingface.co/a", "https://huggingface.co:444/a"):
            with self.subTest(url=url), self.assertRaises(pilot.PreparationError):
                pilot._https(url)
        pilot._https("https://cas-bridge.xethub.hf.co/authored")

    def test_fixed_sampling_and_tie_break_are_not_image_selected(self):
        self.assertEqual([(0, 1), (2, 3), (4, 5)], pilot.select_pairs([0, 1000, 2000, 3000, 4000, 5000]))
        self.assertEqual([(0, 5), (32, 37), (65, 70)], pilot.select_pairs([200 * i for i in range(71)]))
        self.assertEqual([(0, 5), (58, 63), (117, 122)], pilot.select_pairs([200 * i for i in range(123)]))

    def test_bad_or_short_timeline_cannot_duplicate_frames_to_fill_pairs(self):
        for times in ([0, 1, 2, 3, 4, 5], [0, 1000, 1000, 2000, 3000, 4000],
                      [1, 1000, 2000, 3000, 4000, 5000], [0, 1000, 2000, 3000, 4000]):
            with self.subTest(times=times), self.assertRaises(pilot.PreparationError):
                pilot.select_pairs(times)

    def test_bridge_capture_time_does_not_use_one_fps_playback(self):
        source = replace(pilot.SOURCES[0], frame_count=6)
        capture = [1000, 1002.169, 1004, 1006, 1008, 1010]
        metadata = json.dumps({"file_name": pilot.METADATA_FILENAME, "num_frames": 6,
                               "observation_data": {"time_stamp": [[v] for v in capture]}}).encode()
        basis, mapping, duration = pilot.timing_mapping(source, probe(), metadata)
        self.assertEqual("original_dataset_capture_timestamps", basis)
        self.assertEqual(1000, mapping[1]["encoded_pts_ms"])
        self.assertEqual(2169, mapping[1]["source_timestamp_ms"])
        self.assertEqual(1000, mapping[1]["encoded_pts"])
        self.assertEqual("1/1000", mapping[1]["encoded_time_base"])
        self.assertEqual(6000, duration)

    def test_bridge_metadata_is_mandatory_and_source_bound(self):
        source = replace(pilot.SOURCES[0], frame_count=6)
        with self.assertRaises(pilot.PreparationError):
            pilot.timing_mapping(source, probe())
        for filename, count, times in (("wrong.mp4", 6, list(range(6))),
                                        (pilot.METADATA_FILENAME, 5, list(range(6))),
                                        (pilot.METADATA_FILENAME, 6, [0, 1, 1, 3, 4, 5])):
            row = json.dumps({"file_name": filename, "num_frames": count, "observation_data": {"time_stamp": times}}).encode()
            with self.assertRaises(pilot.PreparationError):
                pilot.timing_mapping(source, probe(), row)

    def test_decoded_timing_requires_exact_frame_inventory_dimensions_and_pts(self):
        source = replace(pilot.SOURCES[1], frame_count=6)
        cases = []
        for change in (lambda v: v["streams"][0].update(width=320),
                       lambda v: v["streams"][0].update(time_base="0/1"),
                       lambda v: v["frames"][1].update(best_effort_timestamp=0),
                       lambda v: v["frames"][1].update(best_effort_timestamp_time="NaN"),
                       lambda v: v["streams"][0].update(tags={"rotate": "90"}),
                       lambda v: v["frames"].pop()):
            value = probe(); change(value); cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(pilot.PreparationError):
                pilot.timing_mapping(source, value)

    def test_output_is_new_private_local_directory_without_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            inside = root / "inputs/local"
            inside.mkdir(parents=True)
            existing = inside / "existing"; existing.mkdir()
            sentinel = existing / "private.txt"; sentinel.write_text("preserve")
            alias = inside / "alias"; alias.symlink_to(existing, target_is_directory=True)
            for output in (root / "outside", inside, existing, alias / "child"):
                with self.subTest(output=output), self.assertRaises(pilot.PreparationError):
                    pilot.private_destination(output, root)
            self.assertEqual("preserve", sentinel.read_text())
            self.assertEqual(inside / "new", pilot.private_destination("inputs/local/new", root))

    def test_tools_and_budget_checked_before_any_output_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(pilot, "discover_tools", side_effect=pilot.PreparationError("missing")), \
                    patch.object(pilot, "download") as fetch:
                with self.assertRaises(pilot.PreparationError):
                    pilot.prepare("inputs/local/new", root=root)
                fetch.assert_not_called()
            self.assertFalse((root / "inputs").exists())
            values = iter((0, 301))
            with self.assertRaises(pilot.PreparationError):
                pilot.prepare("inputs/local/new", root=root, tools={"ffmpeg": "f", "ffprobe": "p"}, clock=lambda: next(values))
            self.assertFalse((root / "inputs").exists())

    def test_media_process_time_and_output_bounds(self):
        with self.assertRaises(pilot.PreparationError):
            pilot.run_tool([sys.executable, "-c", "import time; time.sleep(2)"], timeout=.03)
        with self.assertRaises(pilot.PreparationError):
            pilot.run_tool([sys.executable, "-c", "print('x'*8192)"], limit=64)

    def test_mismatched_download_never_issues_completed_receipt_or_overwrites_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            def runner(command, **kwargs):
                return (Path(command[0]).name + " version authored\n").encode()
            with self.assertRaises(pilot.PreparationError):
                pilot.prepare("inputs/local/mismatch", root=root, sources=(pilot.SOURCES[1],),
                              fetch=lambda *a, **kw: b"authored wrong bytes", runner=runner,
                              tools={"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"})
            destination = root / "inputs/local/mismatch"
            self.assertFalse((destination / "preparation-complete.json").exists())
            self.assertEqual([], list(destination.iterdir()))
            with self.assertRaises(pilot.PreparationError):
                pilot.prepare("inputs/local/mismatch", root=root, sources=(pilot.SOURCES[1],),
                              fetch=lambda *a, **kw: self.fail("Must not reuse existing incomplete directory"),
                              runner=runner, tools={"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"})


class AuthoredMediaIntegrationTests(unittest.TestCase):
    def test_zero_network_preparation_preserves_source_pts_pairs_and_byte_receipts(self):
        try:
            tools = pilot.discover_tools()
        except pilot.PreparationError:
            self.skipTest("Optional FFmpeg/ffprobe not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            video_path = root / "authored-testsrc.mp4"
            pilot.run_tool([tools["ffmpeg"], "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc=size=640x480:rate=1:duration=6", "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(video_path)])
            video, card = video_path.read_bytes(), b"Authored card: not a downloaded dataset or license assertion\n"
            source = replace(pilot.SOURCES[1], bytes=len(video), sha256=pilot.sha256(video),
                             card_bytes=len(card), card_sha256=pilot.sha256(card), frame_count=6)
            fetches, commands = [], []
            def fetch(url, limit, deadline, **kwargs):
                fetches.append(url)
                return video if url == source.url else card
            def runner(command, **kwargs):
                commands.append(command)
                return pilot.run_tool(command, **kwargs)
            report = pilot.prepare("inputs/local/authored", root=root, sources=(source,), fetch=fetch, runner=runner, tools=tools)
            out = root / "inputs/local/authored"
            self.assertTrue((out / "preparation-complete.json").is_file())
            value = json.loads((out / "ur5-cloth-0-prepared.json").read_text())
            receipt_hash = value.pop("receipt_sha256")
            self.assertEqual(receipt_hash, pilot.sha256(pilot.encoded(value)))
            self.assertEqual(0, report["provider_requests"])
            self.assertEqual(2, len(fetches))
            self.assertEqual([[0, 1], [2, 3], [4, 5]], report["sources"][0]["pairs"])
            self.assertEqual(pilot.sha256(pilot.encoded(value["timing_mapping"])), value["timing_mapping_sha256"])
            hashes = []
            for pair in value["pairs"]:
                self.assertGreaterEqual(pair["frames"][1]["source_timestamp_ms"] - pair["frames"][0]["source_timestamp_ms"], 1000)
                for frame in pair["frames"]:
                    self.assertEqual(value["timing_mapping_sha256"], frame["timing_mapping_sha256"])
                    self.assertEqual(source.sha256, frame["source_sha256"])
                    self.assertEqual(frame["sha256"], pilot.sha256((root / frame["file"]).read_bytes()))
                    hashes.append(frame["sha256"])
            self.assertEqual(6, len(set(hashes)))
            extraction = [cmd for cmd in commands if "-vf" in cmd]
            self.assertEqual(6, len(extraction))
            self.assertTrue(all(cmd[cmd.index("-fps_mode") + 1] == "passthrough" and "-noautorotate" in cmd for cmd in extraction))
            with self.assertRaises(pilot.PreparationError):
                pilot.prepare("inputs/local/authored", root=root, sources=(source,), fetch=fetch, runner=runner, tools=tools)
            self.assertEqual(2, len(fetches))


if __name__ == "__main__":
    unittest.main()
