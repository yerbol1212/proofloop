"""Fixed replay behavior on authored saved-response fixtures; no network or API."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.benchmark import comparison_metrics, evaluate_guided, evaluate_spark
from proofloop.core import ValidationError
from proofloop.followup import DATASET_ID, FIXED_CASES, _estimate, replay_fixed_followup, response_paths
from proofloop.meta import SPARK_MODEL
from tests.test_benchmark import approved, contract, digest, media_fixture, result, spark_result


def fixture(root):
    value = contract()
    source = value["sources"][0]
    value["dataset_id"] = DATASET_ID
    value["sources"] = []
    for index in range(3):
        item = deepcopy(source)
        item.update(id=f"coffee-episode-{index:06d}", file=f"inputs/local/fixture/source-{index}.mp4")
        value["sources"].append(item)
    value["cases"] = []
    for identifier, source_id, times, expected in FIXED_CASES:
        value["cases"].append({"id": identifier, "source_id": source_id, "role": "benchmark",
            "regions": {"dispense": [0, 0, .45, 1], "staging": [.55, 0, 1, 1]},
            "frames": [{"file": f"inputs/local/fixture/{identifier}-{time}.jpg", "sha256": digest(f"{identifier}-{time}"),
                        "source_timestamp_ms": time} for time in times],
            "evaluation_timestamp_ms": times[-1], "annotation": {"status": expected,
                "checks": {"glass-dispense": expected}, "reason": "Authored unit test, not real recorded evidence.",
                "origin": "source_frames_reviewed_before_inference"}})
    envelope = media_fixture(root, approved(value))
    folder = root / "artifacts/local/fixed-run"
    report = {"contract_hash": envelope["contract_hash"], "status": "completed", "inference_origin": "authored_fixture",
              "evaluation_role": "development_reprocessing", "conditions": {}}
    pairs = {}
    for name in ("guided", "spark"):
        directory = folder / name
        directory.mkdir(parents=True)
        sidecar, ledger = {}, []
        for case in envelope["contract"]["cases"]:
            cid = case["id"]
            if name == "spark":
                pair = spark_result(case, case["annotation"]["status"])
                pair.update(usage={"input_tokens": 100, "output_tokens": 200}, latency_seconds=.1)
                sidecar[cid] = pair
                raw = [pair]
            else:
                # Only e00's second follow-up supports a grounded placement.
                masks = cid == "coffee-e00-attempt-2"
                results = {frame["sha256"]: {} for frame in case["frames"]}
                for frame in case["frames"]:
                    if masks:
                        item = result(frame["sha256"])
                        item.update(usage={"images_processed": 1}, latency_seconds=.1,
                                    guided_provenance={"method": "spark_box_guided_sam", "locator_model": SPARK_MODEL,
                                                       "version": "normalized_boxes_v2"})
                        results[frame["sha256"]]["glass"] = item
                frames = [{"frame_index": index, "width": 100, "height": 100, "sha256": frame["sha256"]}
                          for index, frame in enumerate(case["frames"])]
                locator = {"sources": [{"sha256": frame["sha256"]} for frame in case["frames"]],
                           "response": {"model": SPARK_MODEL, "status": "completed"},
                           "usage": {"input_tokens": 100, "output_tokens": 200}, "latency_seconds": .1,
                           "parsed": {"frames": [{"frame_index": index, "objects": [{"object_id": "glass",
                               "fully_visible": masks, "identity_ambiguous": False, "boundary_uncertain": False,
                               "candidates": [{"x1": 100, "y1": 100, "x2": 300, "y2": 300}]}]}
                               for index in (0, 1)]}}
                pair = {"results": results, "locator": locator, "frames": frames, "unavailable": [],
                        "request_limit": 3, "provenance": {"version": "normalized_boxes_v2"}}
                sidecar[cid] = results
                raw = [locator] + [item for by_object in results.values() for item in by_object.values()]
                pairs[tuple(frame["sha256"] for frame in frames)] = pair
            (directory / f"{cid}.json").write_text(json.dumps(pair))
            ledger.extend({"model": item["response"]["model"], "status": "ok", "usage": item["usage"],
                           "latency_seconds": item["latency_seconds"]} for item in raw)
        (directory / "results.json").write_text(json.dumps(sidecar))
        evaluator = evaluate_guided if name == "guided" else evaluate_spark
        cases = evaluator(envelope, sidecar, **({"observation_version": "normalized_boxes_v2"} if name == "guided" else {}))
        report["conditions"][name] = {"status": "completed", "cases": cases,
            "metrics": comparison_metrics(envelope, cases), "usage": {"requests": ledger, "estimated_usd": _estimate(ledger)},
            **({"observation_version": "normalized_boxes_v2"} if name == "guided" else {})}

    def saved_replay(client, images, concepts):
        # Geometry transport is covered in guided tests; this fixture has no
        # actual JPEG pixels. The fixed controller still consumes real adapter
        # data structures, grounded observations and validated report sidecars.
        return deepcopy(client.pair)
    return envelope, report, folder, saved_replay


class FixedFollowupTests(unittest.TestCase):
    def test_replays_in_order_stops_after_known_result_and_separates_unused_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envelope, report, folder, replay = fixture(root)
            with patch("proofloop.followup.guided_segment_pair", side_effect=replay):
                value = replay_fixed_followup(envelope, report, folder, root)
            guided, spark = value["conditions"]["guided"], value["conditions"]["spark"]
            self.assertEqual(2, guided["trajectories"][0]["followup_attempts_used"])
            self.assertEqual(["unknown", "unknown", "pass"], [s["status"] for s in guided["trajectories"][0]["steps"]])
            self.assertEqual("timeline_exhausted", guided["trajectories"][1]["stop_reason"])
            self.assertEqual(0, guided["trajectories"][2]["followup_attempts_used"])
            self.assertEqual(["coffee-e00-attempt-2"], spark["unused_counterfactual_case_ids"])
            self.assertEqual(5, spark["policy_consumed_request_count"])
            self.assertEqual(1, spark["unused_counterfactual_request_count"])
            self.assertFalse(value["physical_action_execution_verified"])

    def test_embedded_public_policy_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envelope, report, folder, replay = fixture(root)
            report["followup_policy_replay"] = {"conditions": {"guided": {"resolved_trajectories": 300}}}
            with patch("proofloop.followup.guided_segment_pair", side_effect=replay):
                value = replay_fixed_followup(envelope, report, folder, root)
            self.assertEqual(1, value["conditions"]["guided"]["resolved_trajectories"])
            self.assertEqual("offline_policy_replay_of_fixed_saved_authored_responses", value["method"])

    def test_supported_fail_stops_without_consuming_favorable_future_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envelope, report, folder, replay = fixture(root)
            path = folder / "spark/coffee-e00-initial.json"
            pair = json.loads(path.read_text())
            pair["parsed"]["checks"][0]["status"] = "fail"
            path.write_text(json.dumps(pair))
            results_path = folder / "spark/results.json"
            results = json.loads(results_path.read_text())
            results["coffee-e00-initial"] = pair
            results_path.write_text(json.dumps(results))
            decisions = evaluate_spark(envelope, results)
            report["conditions"]["spark"].update(cases=decisions, metrics=comparison_metrics(envelope, decisions))
            with patch("proofloop.followup.guided_segment_pair", side_effect=replay):
                value = replay_fixed_followup(envelope, report, folder, root)
            first = value["conditions"]["spark"]["trajectories"][0]
            self.assertEqual("fail", first["final_status"])
            self.assertEqual(0, first["followup_attempts_used"])
            self.assertEqual(1, len(first["steps"]))
            self.assertEqual(["coffee-e00-attempt-1", "coffee-e00-attempt-2"],
                             value["conditions"]["spark"]["unused_counterfactual_case_ids"])

    def test_tampered_per_case_result_or_ledger_is_rejected(self):
        for mutation in ("case", "ledger", "locator"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                envelope, report, folder, replay = fixture(root)
                if mutation == "ledger":
                    report["conditions"]["guided"]["usage"]["requests"][0]["usage"]["output_tokens"] += 1
                else:
                    path = folder / "guided/coffee-e00-initial.json"
                    value = json.loads(path.read_text())
                    if mutation == "case":
                        value["results"] = {}
                    else:
                        value["locator"]["usage"]["output_tokens"] += 1
                    path.write_text(json.dumps(value))
                with patch("proofloop.followup.guided_segment_pair", side_effect=replay):
                    with self.assertRaises(ValidationError):
                        replay_fixed_followup(envelope, report, folder, root)

    def test_changed_schedule_or_method_is_rejected(self):
        for mutation in ("time", "version"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                envelope, report, folder, replay = fixture(root)
                if mutation == "time":
                    envelope["contract"]["cases"][0]["frames"][0]["source_timestamp_ms"] -= 1
                else:
                    report["conditions"]["guided"]["observation_version"] = "unfrozen-version"
                with patch("proofloop.followup.guided_segment_pair", side_effect=replay):
                    with self.assertRaises(ValidationError):
                        replay_fixed_followup(envelope, report, folder, root)

    def test_cache_dependencies_cover_all_pair_sidecars(self):
        paths = response_paths(Path("artifacts/local/run"))
        self.assertEqual(14, len(paths))
        self.assertIn(Path("artifacts/local/run/guided/coffee-e01-attempt-1.json"), paths)
        self.assertIn(Path("artifacts/local/run/spark/results.json"), paths)


if __name__ == "__main__":
    unittest.main()
