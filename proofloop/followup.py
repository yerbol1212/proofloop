"""Replay the fixed future-evidence development schedule from saved responses.

Read-only local analysis: no model requests, inferred camera motion or physical
repair claim. A public trajectory summary is an output, never an authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .acquisition import build_acquisition_plan, check_source_pair
from .benchmark import ROOT, _approved, _local_path, _read, validate_report, verify_media
from .core import EvidenceStore, Observation, ValidationError
from .grounding import SamImageEvidence
from .guided import GUIDED_VERSION, guided_segment_pair
from .meta import SAM_MODEL, SPARK_MODEL

DATASET_ID = "proofloop-fixed-future-evidence-development-20261028"
GUIDED_SHA256 = "8f77c12f32911d28c3a17bf53d47f2bf30ac021bea0d68a20d6c1dd0487a3a23"
FIXED_CASES = (("coffee-e00-initial", "coffee-episode-000000", (16500, 18500), "unknown"),
               ("coffee-e00-attempt-1", "coffee-episode-000000", (19100, 20100), "pass"),
               ("coffee-e00-attempt-2", "coffee-episode-000000", (20700, 21700), "pass"),
               ("coffee-e01-initial", "coffee-episode-000001", (18500, 20000), "unknown"),
               ("coffee-e01-attempt-1", "coffee-episode-000001", (20500, 21500), "pass"),
               ("coffee-e02-terminal", "coffee-episode-000002", (500, 21500), "unknown"))
TRAJECTORIES = (("coffee-e00", ("coffee-e00-initial", "coffee-e00-attempt-1", "coffee-e00-attempt-2")),
                ("coffee-e01", ("coffee-e01-initial", "coffee-e01-attempt-1")),
                ("coffee-e02-terminal", ("coffee-e02-terminal",)))


def response_paths(run_directory: Path) -> tuple[Path, ...]:
    """Files whose content can affect replay; callers may include them in cache keys."""
    folder = Path(run_directory)
    return tuple([folder / name / "results.json" for name in ("guided", "spark")]
                 + [folder / name / f"{case_id}.json" for name in ("guided", "spark")
                    for case_id, _, _, _ in FIXED_CASES])


def _estimate(entries: list[Mapping[str, Any]]) -> float:
    inputs, outputs, images = 0, 0, 0
    for entry in entries:
        if entry.get("status") != "ok":
            raise ValidationError("Fixed follow-up replay requires completed request records")
        if entry.get("model") == SPARK_MODEL:
            usage = entry.get("usage")
            if (not isinstance(usage, Mapping) or any(type(usage.get(field)) is not int or usage[field] < 0
                                                     for field in ("input_tokens", "output_tokens"))):
                raise ValidationError("Fixed follow-up costs need reported bounded token usage")
            inputs += usage["input_tokens"]
            outputs += usage["output_tokens"]
        elif entry.get("model") == SAM_MODEL:
            images += 1
        else:
            raise ValidationError("Unexpected model in fixed follow-up request records")
    total = inputs * 1.25 / 1_000_000 + outputs * 4.25 / 1_000_000
    total += images * .0025
    return round(total, 6)


class _SavedClient:
    """Replay only saved, bound responses through the unchanged guided adapter."""
    retries, max_requests = 0, 3

    def __init__(self, pair: Mapping[str, Any]):
        self.pair, self.ledger = pair, []

    def spark_json(self, *_args, **_kwargs):
        return deepcopy(self.pair["locator"])

    def segment_image(self, path, _prompt, **_kwargs):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        results = self.pair["results"].get(digest, {})
        if set(results) != {"glass"}:
            raise ValidationError("Guided follow-up response coverage differs from its locator")
        return deepcopy(results["glass"])


def _schedule(envelope: Mapping[str, Any]):
    protocol = _approved(envelope)
    contract = envelope["contract"]
    if (contract["dataset_id"] != DATASET_ID or list(protocol.objects) != ["glass"]
            or set(protocol.regions) != {"dispense", "staging"}
            or len(protocol.checks) != 1 or protocol.checks[0].kind != "location"
            or len(contract["cases"]) != len(FIXED_CASES)):
        raise ValidationError("This replay supports only the declared fixed future-evidence schedule")
    for case, (identifier, source_id, times, expected) in zip(contract["cases"], FIXED_CASES):
        if (case["id"] != identifier or case["source_id"] != source_id
                or tuple(frame["source_timestamp_ms"] for frame in case["frames"]) != times
                or case["evaluation_timestamp_ms"] != times[-1]
                or case["annotation"]["status"] != expected):
            raise ValidationError("Fixed follow-up case, source, timestamps or reviewed label changed")
    return protocol


def replay_fixed_followup(envelope: Mapping[str, Any], report: Mapping[str, Any],
                          run_directory: Path, root: Path = ROOT) -> dict[str, Any]:
    """Recompute a public-safe policy DTO from frozen inputs and private sidecars.

    Every pair was tested once as a fixed diagnostic control. This function
    consumes only chronological pairs needed by the policy, stops on any known
    verdict and separates unused counterfactual request estimates.
    """
    root = Path(root).resolve()
    folder = _local_path(Path(run_directory), root, output=True)
    if hashlib.sha256(Path(__file__).with_name("guided.py").read_bytes()).hexdigest() != GUIDED_SHA256:
        raise ValidationError("The guided implementation differs from this experiment's frozen method")
    protocol = _schedule(envelope)
    if (report.get("status") != "completed" or set(report.get("conditions", {})) != {"guided", "spark"}
            or report["conditions"]["guided"].get("observation_version") != GUIDED_VERSION
            or report.get("evaluation_role") != "development_reprocessing"):
        raise ValidationError("Fixed follow-up replay needs completed development guided-v2 and Spark results")
    verify_media(envelope, project_root=root)
    sidecars = {name: _read(folder / name / "results.json") for name in ("guided", "spark")}
    validate_report(envelope, report, raw_results=sidecars)
    cases = {case["id"]: case for case in envelope["contract"]["cases"]}
    records = {name: {case["case_id"]: case for case in report["conditions"][name]["cases"]}
               for name in ("guided", "spark")}
    pairs, ledgers = {}, {}
    for name in ("guided", "spark"):
        cursor, per_case = 0, {}
        actual = report["conditions"][name].get("usage", {}).get("requests")
        if not isinstance(actual, list):
            raise ValidationError("Fixed follow-up request ledger is unavailable")
        for case_id, case in cases.items():
            pair = _read(folder / name / f"{case_id}.json")
            if name == "guided":
                if pair.get("results") != sidecars[name].get(case_id):
                    raise ValidationError("Guided per-case responses differ from the evaluated sidecar")
                images = [_local_path(frame["file"], root) for frame in case["frames"]]
                replayed = guided_segment_pair(_SavedClient(pair), images, envelope["contract"]["concepts"])
                if replayed != pair:
                    raise ValidationError("Saved guided locator and mask coverage do not replay identically")
                pairs[case_id] = pair
                raw_responses = [pair["locator"]] + [result for frame in pair["frames"]
                                 for result in pair["results"][frame["sha256"]].values()]
            else:
                if pair != sidecars[name].get(case_id):
                    raise ValidationError("Spark per-case response differs from the evaluated sidecar")
                raw_responses = [pair]
            entries = actual[cursor:cursor + len(raw_responses)]
            if (len(entries) != len(raw_responses) or any(
                    entry.get("model") != raw.get("response", {}).get("model")
                    or entry.get("usage") != raw.get("usage")
                    or entry.get("latency_seconds") != raw.get("latency_seconds")
                    for entry, raw in zip(entries, raw_responses))):
                raise ValidationError("Fixed follow-up cost ledger differs from saved response usage")
            per_case[case_id] = entries
            cursor += len(raw_responses)
        if (cursor != len(actual) or _estimate(actual) != report["conditions"][name]["usage"].get("estimated_usd")):
            raise ValidationError("Fixed follow-up request totals or estimates changed")
        ledgers[name] = per_case

    def plan_for(case_id, attempts_used, used, future_ids):
        record, case, pair = records["guided"][case_id], cases[case_id], pairs[case_id]
        observations = [Observation.from_dict(item) for item in record["grounding"]["observations"]]
        store = EvidenceStore(protocol, observations[0].session_id)
        for observation in observations:
            store.add(observation)
        samples = [SamImageEvidence(object_id, frame["source_timestamp_ms"], result, "external_recorded_video",
                                    envelope["contract"]["concepts"][object_id])
                   for frame in case["frames"]
                   for object_id, result in pair["results"].get(frame["sha256"], {}).items()]
        return build_acquisition_plan(protocol, store, grounding=record["grounding"], locator=pair["locator"],
            samples=samples, regions=case["regions"], attempts_used=attempts_used, max_attempts=2,
            used_source_hashes=used, source_mode="recorded", now_ms=case["evaluation_timestamp_ms"],
            available_timestamps_ms=[frame["source_timestamp_ms"] for cid in future_ids for frame in cases[cid]["frames"]])

    method = ("offline_policy_replay_of_fixed_saved_live_responses" if report.get("inference_origin") == "live_meta_api"
              else "offline_policy_replay_of_fixed_saved_authored_responses" if report.get("inference_origin") == "authored_fixture"
              else "offline_policy_replay_of_fixed_saved_unverified_responses")
    output = {"method": method, "maximum_followup_attempts": 2,
              "physical_action_execution_verified": False, "camera_response_to_request_verified": False, "conditions": {}}
    for name in ("guided", "spark"):
        trajectories, consumed = [], set()
        for trajectory_id, sequence in TRAJECTORIES:
            used, steps, followups, previous_plan = [], [], 0, None
            for index, case_id in enumerate(sequence):
                case, record = cases[case_id], records[name][case_id]
                hashes = [frame["sha256"] for frame in case["frames"]]
                times = [frame["source_timestamp_ms"] for frame in case["frames"]]
                if index:
                    check_source_pair(previous_plan, hashes, timestamps_ms=times)
                    followups += 1
                consumed.add(case_id)
                used.extend(hashes)
                future_ids = sequence[index + 1:]
                if name == "guided":
                    plan = plan_for(case_id, followups, used, future_ids)
                else:
                    # Spark supplies verdicts, not grounded masks. Its baseline
                    # controller uses only the same frozen chronological rules.
                    plan = {"status": "not_needed" if record["status"] != "unknown" else
                            "budget_exhausted" if followups >= 2 else "ready" if future_ids else "timeline_exhausted",
                            "source_policy": {"required_views": 2, "min_spacing_ms": 1000,
                                "all_sources_must_be_new": True, "after_timestamp_ms": case["evaluation_timestamp_ms"],
                                "next_timestamps_ms": [f["source_timestamp_ms"] for f in cases[future_ids[0]]["frames"]]
                                                      if future_ids else None,
                                "excluded_source_hashes": sorted(used)}}
                steps.append({"case_id": case_id, "status": record["status"], "source_timestamps_ms": times,
                    "source_hashes": hashes, "policy_status": plan["status"], "requests": plan.get("requests", []),
                    "source_policy": plan["source_policy"], "actual_request_count": len(ledgers[name][case_id]),
                    "estimated_usd": _estimate(ledgers[name][case_id])})
                previous_plan = plan
                if record["status"] != "unknown" or plan["status"] != "ready":
                    break
            trajectories.append({"id": trajectory_id, "followup_attempts_used": followups,
                                 "final_status": steps[-1]["status"], "stop_reason": steps[-1]["policy_status"], "steps": steps})
        unused = set(cases) - consumed
        consumed_entries = [entry for cid in consumed for entry in ledgers[name][cid]]
        unused_entries = [entry for cid in unused for entry in ledgers[name][cid]]
        output["conditions"][name] = {"trajectories": trajectories,
            "resolved_trajectories": sum(t["final_status"] != "unknown" for t in trajectories),
            "terminal_unknown_trajectories": sum(t["final_status"] == "unknown" for t in trajectories),
            "policy_consumed_request_count": len(consumed_entries), "policy_consumed_estimated_usd": _estimate(consumed_entries),
            "unused_counterfactual_case_ids": sorted(unused), "unused_counterfactual_request_count": len(unused_entries),
            "unused_counterfactual_estimated_usd": _estimate(unused_entries)}
    return output
