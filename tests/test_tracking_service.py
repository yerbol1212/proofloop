"""Authored durable diagnostic fixtures; all provider/media transport is injected."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from proofloop import tracking_service as service
from proofloop.core import ValidationError
from proofloop.inference_journal import AttemptJournal
from proofloop.runtime_control import RuntimeControl, RuntimeUnavailable
from proofloop.meta import MetaClient, SPARK_MODEL
from tests.test_guided import jpeg_header
from tests.test_visibility_guard import contract
from tests import test_contact_experiment as authored


def result(mode="cached_demo", source="a"*64, origin=None):
    return {"schema_version":service.VERSION,"mode":mode,
        "inference_origin":origin or ("cached_development_projection" if mode == "cached_demo" else "injected_callback"),
        "status":"unknown","frames":[{"frame_index":0,"timestamp_ms":0,"source_sha256":source,
            "frame_sha256":None,"status":"unknown","track_id":None,"box":None,"gap_codes":["candidate_missing"],
            "warning":"geometry_unavailable","confirmation_evidence":[]}],"request_attempts":0,
        "estimated_usd":0.0,"unreported_charge_possible":False,"trust_mode":"warning_only_not_qualification",
        "physical_action_authorized":False,"changes_live_placement_acceptance":False}


class TrackingServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = [1000]
        self.callback = Mock(return_value=result())
        self.runner = service.TrackingService(self.root,cached_callback=self.callback,clock_ms=lambda:self.now[0])
        self.addCleanup(self.runner.close)
        self.request = {"idempotency_key":"authored-key","mode":"cached_demo","fixture_id":"latest-tracking"}
        self.source = {"width":100,"height":100,"frame_count":3,"duration_ms":1200,"time_base":"1/1000",
            "frames":[{"frame_index":i,"pts":timestamp,"timestamp_ms":timestamp} for i,timestamp in enumerate((0,600,1000))]}

    def asset(self):
        with patch.object(service.video,"_probe",return_value=(deepcopy(self.source),"authored-probe")):
            return self.runner.register_asset(b"\0\0\0\x18ftyp" + b"authored-video-pixels")

    def real_request(self, grippers=False):
        asset = self.asset()
        seeds = [{"object_id":"remote","track_id":1,"frame_index":0,"box":{"x1":10,"y1":10,"x2":49,"y2":49}}]
        if grippers:
            seeds += [{"object_id":name,"track_id":i,"frame_index":0,"box":{"x1":x,"y1":20,"x2":x+9,"y2":50}}
                      for name,i,x in (("left_gripper",3,0),("right_gripper",4,80))]
        return {"idempotency_key":"real-key","mode":"real_meta_video","asset_id":asset["asset_id"],
            "reviewed":True,"contract":contract(),"seeds":seeds,"checkpoints":[0,1,2]}

    def test_cached_job_is_idempotent_durable_reloadable_and_zero_provider(self):
        first = self.runner.submit(self.request)
        self.assertEqual(first,self.runner.submit(self.request))
        self.assertTrue(self.runner.run_once())
        restarted = service.TrackingService(self.root,cached_callback=self.callback)
        job = restarted.get(first["job_id"])
        self.assertEqual("succeeded",job["state"])
        self.assertEqual(result(),job["result"])
        self.assertTrue(job["integrity"]["result_valid"])
        self.assertEqual(1,self.callback.call_count)
        self.assertFalse(self.runner.run_once())
        self.assertEqual(["queued","running","succeeded"],[row["event"] for row in self.runner.events(first["job_id"])])

    def test_conflicting_key_cannot_dispatch_changed_input(self):
        self.runner.submit(self.request)
        other = deepcopy(self.request); other["fixture_id"] = "unregistered"
        with self.assertRaises(ValidationError): self.runner.submit(other)
        self.runner.client_factory = Mock()
        other = self.real_request(); other["idempotency_key"] = self.request["idempotency_key"]
        with self.assertRaises(service.TrackingConflict): self.runner.submit(other)
        self.callback.assert_not_called()

    def test_asset_limits_signature_tokens_and_changed_pixels_fail_closed(self):
        for data,mime in ((b"short","video/mp4"),(b"0"*13,"video/mp4"),(b"1234ftyp"+b"a"*(service.MAX_ASSET_BYTES),"video/mp4"),
                          (b"1234ftyp0000","text/plain")):
            with self.subTest(mime=mime),self.assertRaises(ValidationError): self.runner.register_asset(data,mime)
        asset = self.asset()
        self.assertNotIn(str(self.root),str(asset))
        for identifier in ("../../.env.local","https://example.com/video.mp4","a"*64):
            with self.assertRaises(ValidationError): self.runner.asset(identifier)
        (self.runner.assets / (asset["asset_id"]+".mp4")).write_bytes(b"1234ftypchanged")
        with self.assertRaises(ValidationError): self.runner.asset(asset["asset_id"])

    def test_real_admission_is_reviewed_bounded_and_rejects_arbitrary_paths(self):
        self.runner.client_factory = Mock()
        value = self.real_request(True)
        for field,replacement in (("reviewed",False),("asset_id","/private/key"),("checkpoints",[1,2]),
                                   ("checkpoints",[0,0]),("checkpoints",[0,3])):
            changed = deepcopy(value); changed[field] = replacement
            with self.subTest(field=field),self.assertRaises(ValidationError): self.runner.submit(changed)
        changed = deepcopy(value); changed["seeds"].pop()
        with self.assertRaises(ValidationError): self.runner.submit(changed)
        changed = deepcopy(value); changed["source_url"] = "https://example.com/video.mp4"
        with self.assertRaises(ValidationError): self.runner.submit(changed)
        self.runner.client_factory.assert_not_called()

    def test_asset_receipt_binding_is_rechecked_before_dispatch(self):
        self.runner.callback = Mock()
        request = self.real_request()
        job = self.runner.submit(request)
        path = self.runner.assets / (request["asset_id"]+".json")
        metadata = service._read(path); metadata["source"]["duration_ms"] = 1199
        service._atomic(path,metadata)
        self.runner.run_once()
        self.assertEqual("failed",self.runner.get(job["job_id"])["state"])
        self.runner.callback.assert_not_called()

    def test_callback_failure_has_redacted_terminal_error_and_no_retry(self):
        self.callback.side_effect = RuntimeError("SECRET private provider message")
        job = self.runner.submit(self.request); self.runner.run_once()
        current = self.runner.get(job["job_id"])
        self.assertEqual("failed",current["state"])
        self.assertEqual("diagnostic_failed",current["error_code"])
        self.assertNotIn("SECRET",str(current)); self.assertFalse(self.runner.run_once())

    def test_expired_worker_cannot_complete_or_dispatch_again(self):
        def callback(payload,identifier,guard):
            self.now[0] += 40000
            with self.assertRaises(service.TrackingConflict): guard()
            return result()
        self.callback.side_effect = callback
        job = self.runner.submit(self.request); self.runner.run_once()
        self.assertEqual("interrupted",self.runner.get(job["job_id"])["state"])
        self.assertFalse(self.runner.run_once())
        self.assertEqual(1,self.callback.call_count)

    def test_shutdown_fences_subsequent_requests_and_publication(self):
        def callback(payload,identifier,guard):
            self.runner.stop.set()
            with self.assertRaises(service.TrackingConflict): guard()
            return result()
        self.callback.side_effect = callback
        job = self.runner.submit(self.request); self.runner.run_once()
        current = self.runner.get(job["job_id"])
        self.assertEqual("interrupted",current["state"])
        self.assertEqual("outcome_unknown",current["error_code"])
        self.assertIsNone(current["result"])

    def test_competing_workers_claim_once(self):
        entered,release = threading.Event(),threading.Event()
        def callback(*args): entered.set(); release.wait(2); return result()
        self.callback.side_effect = callback
        self.runner.submit(self.request)
        worker = threading.Thread(target=self.runner.run_once); worker.start()
        self.assertTrue(entered.wait(2))
        second = service.TrackingService(self.root,cached_callback=self.callback,clock_ms=lambda:self.now[0])
        self.assertFalse(second.run_once())
        release.set(); worker.join(2)
        self.assertEqual(1,self.callback.call_count)

    def test_corrupt_result_suppresses_all_public_result_bytes(self):
        job = self.runner.submit(self.request); self.runner.run_once()
        file = self.root / "results" / (job["job_id"]+".json")
        file.write_text('{"secret":"private"}')
        current = self.runner.get(job["job_id"])
        self.assertIsNone(current["result"]); self.assertFalse(current["integrity"]["result_valid"])
        self.assertEqual("result_integrity_error",current["error_code"])

    def test_payload_mutation_is_rejected_without_callback(self):
        job = self.runner.submit(self.request)
        with sqlite3.connect(self.runner.db_path) as db:
            db.execute("UPDATE diagnostic_jobs SET payload='{}' WHERE id=?",(job["job_id"],))
        self.runner.run_once(); self.callback.assert_not_called()
        self.assertEqual("failed",self.runner.get(job["job_id"])["state"])

    def test_result_cannot_claim_placement_authority_paths_or_paid_cached_inference(self):
        for field,value in (("physical_action_authorized",True),("changes_live_placement_acceptance",True),
                            ("status","pass"),("request_attempts",1)):
            changed = result(); changed[field] = value
            with self.subTest(field=field),self.assertRaises(ValidationError): service.public_result(changed)
        changed = result(); changed["frames"][0]["gap_codes"] = ["/Users/private/key"]
        with self.assertRaises(ValidationError): service.public_result(changed)

    def test_worker_lifecycle_and_capabilities_are_explicit(self):
        self.assertFalse(self.runner.capabilities()["worker_available"])
        self.runner.start(); self.assertTrue(self.runner.capabilities()["worker_available"])
        self.runner.close(); self.assertFalse(self.runner.capabilities()["worker_available"])

    def model_path(self, *, fail_video=False, spark_outcome="ok", spark_tokens=None):
        adapters = []
        def factory(**kwargs):
            adapter = Mock(ledger=[]); adapter.config = kwargs; adapters.append(adapter); return adapter
        self.runner.client_factory = factory
        request = self.real_request(True)
        asset_path, metadata = self.runner._asset(request["asset_id"])
        source = {**self.source,"sha256":metadata["sha256"],"bytes":metadata["bytes"]}
        def prepare(path,seeds,**kwargs):
            return {"clip_file":str(path),"source":deepcopy(source),"prepared_hash":service._digest(seeds),"seeds":deepcopy(seeds)}
        def segment(adapter,prepared):
            adapter.ledger.append({"authored_attempt":True})
            return {"status":"unscored" if fail_video else "completed","prepared":prepared,"frames":[] if fail_video else [
                {"frame_index":row["frame_index"],"timestamp_ms":row["timestamp_ms"],"candidates":[
                    {"track_id":seed["track_id"],"object_id":seed["object_id"],"box":[.1,.1,.5,.5],"confidence":.95,"mask_supported":True}
                    for seed in prepared["seeds"]]} for row in source["frames"]]}
        def extract(args,**kwargs):
            file = Path(args[-1]); file.write_bytes(jpeg_header(width=100,height=100,suffix=file.name.encode()))
        def prepare_guard(file,box,contract_value,output,**kwargs):
            return {"file":file,"box":box,"contract":contract_value,**kwargs}
        def challenge(adapter,prepared):
            ledger = {"endpoint":"responses","model":SPARK_MODEL,
                "status":"ok" if spark_outcome == "completed_unreported" else
                    "invalid_response" if spark_outcome == "incomplete_response" else spark_outcome}
            if spark_outcome == "ok":
                ledger["usage"] = {"input_tokens":100,"output_tokens":100}
            if spark_tokens is not None:
                ledger["usage"] = deepcopy(spark_tokens)
            adapter.ledger.append(ledger)
            frames = [{"file":str(self.root/f"frame-{i}.jpg"),"sha256":"b"*64} for i in range(3)]
            index = prepared["frame_index"]
            frames[index] = {"file":str(prepared["file"]),"sha256":hashlib.sha256(prepared["file"].read_bytes()).hexdigest()}
            entry = {"contract":contract(),"prepared":{"source":source},"frames":frames}
            report = authored.ContactExperimentTests.checked(self,entry,index)
            if spark_outcome != "ok":
                client = MetaClient("authored-offline-key",max_requests=1,retries=0)
                client.ledger = adapter.ledger
                report["usage"] = client.usage_summary()
                if spark_outcome != "completed_unreported":
                    report.update(status="unscored",evaluation=None,response=None,
                        error={"code":spark_outcome,"http_status":None,"response":None})
            else:
                report["usage"] = {"estimated_usd":.001,"spark_requests_with_unreported_token_usage":0}
            return report
        warning_rows = [{"frame_index":row["frame_index"],"timestamp_ms":row["timestamp_ms"],"source_sha256":source["sha256"],
            "target_track_id":1,"warning":"near_gripper_proposal" if row["frame_index"] == 1 else "no_near_proposal_observed",
            "veto":row["frame_index"] == 1,"additional_geometry_complete":True} for row in source["frames"]]
        with patch.object(service.video,"prepare_clip",side_effect=prepare), \
             patch.object(service.video,"segment_video",side_effect=segment), \
             patch.object(service.video,"public_tracking_summary",return_value={"cost":{"estimated_usd":None if fail_video else .01}}), \
             patch.object(service.warnings,"compare",return_value={"frames":warning_rows}), \
             patch.object(service.visibility,"_extractor_path",return_value="authored-ffmpeg"), \
             patch.object(service.subprocess,"run",side_effect=extract), \
             patch.object(service.visibility,"prepare",side_effect=prepare_guard), \
             patch.object(service.visibility,"challenge",side_effect=challenge), \
             patch.object(service.geometry,"resolve_distractor_conflicts",return_value={}), \
             patch.object(service.masks,"_filter",side_effect=lambda runtime,*a:(runtime,{})):
            job = self.runner.submit(request); self.runner.run_once()
        return self.runner.get(job["job_id"]),adapters

    def test_actual_adapter_composition_uses_fresh_clients_one_parent_budget_and_risk_resets(self):
        job,adapters = self.model_path()
        self.assertEqual("succeeded",job["state"])
        self.assertEqual(5,len(adapters))  # Two SAM videos and three bounded visibility calls.
        self.assertEqual(5,job["result"]["request_attempts"])
        self.assertEqual(.023,job["result"]["estimated_usd"])
        self.assertEqual(["unknown"]*3,[row["status"] for row in job["result"]["frames"]])
        self.assertEqual("near_gripper_proposal",job["result"]["frames"][1]["warning"])
        for adapter in adapters:
            self.assertEqual(job["job_id"],adapter.config["operation_id"])
            self.assertEqual((1,0),(adapter.config["max_requests"],adapter.config["retries"]))
        self.assertFalse(job["result"]["changes_live_placement_acceptance"])
        self.assertFalse(job["result"]["unreported_charge_possible"])

    def test_lost_spark_responses_preserve_charge_uncertainty_without_retry(self):
        job,adapters = self.model_path(spark_outcome="network_error")
        self.assertEqual("succeeded",job["state"])
        self.assertEqual(5,len(adapters))
        self.assertEqual(5,job["result"]["request_attempts"])
        self.assertEqual(.02,job["result"]["estimated_usd"])  # Known SAM subtotal only.
        self.assertTrue(job["result"]["unreported_charge_possible"])
        self.assertEqual(["unscored"]*3,[row["status"] for row in job["result"]["frames"]])
        for adapter in adapters:
            self.assertEqual((1,0),(adapter.config["max_requests"],adapter.config["retries"]))

    def test_malformed_spark_response_preserves_unreported_charge(self):
        job,adapters = self.model_path(spark_outcome="invalid_json")
        self.assertEqual("succeeded",job["state"])
        self.assertTrue(job["result"]["unreported_charge_possible"])
        self.assertEqual(5,len(adapters))

    def test_incomplete_spark_response_without_usage_is_not_assumed_free(self):
        job,_ = self.model_path(spark_outcome="incomplete_response")
        self.assertTrue(job["result"]["unreported_charge_possible"])
        self.assertEqual(.02,job["result"]["estimated_usd"])

    def test_completed_spark_response_missing_usage_keeps_uncertainty(self):
        job,_ = self.model_path(spark_outcome="completed_unreported")
        self.assertTrue(job["result"]["unreported_charge_possible"])
        self.assertEqual(.02,job["result"]["estimated_usd"])

    def test_reported_zero_tokens_are_distinct_from_absent_usage(self):
        job,_ = self.model_path(spark_outcome="incomplete_response",
            spark_tokens={"input_tokens":0,"output_tokens":0})
        self.assertFalse(job["result"]["unreported_charge_possible"])
        self.assertEqual(.02,job["result"]["estimated_usd"])

    def test_malformed_spark_token_usage_is_uncertain(self):
        job,_ = self.model_path(spark_outcome="completed_unreported",
            spark_tokens={"input_tokens":True,"output_tokens":-1})
        self.assertTrue(job["result"]["unreported_charge_possible"])

    def test_incomplete_video_keeps_all_frames_unscored_without_followup_model_calls(self):
        job,adapters = self.model_path(fail_video=True)
        self.assertEqual("succeeded",job["state"])
        self.assertEqual(1,len(adapters))
        self.assertTrue(job["result"]["unreported_charge_possible"])
        self.assertEqual(["unscored"]*3,[row["status"] for row in job["result"]["frames"]])

    def controlled(self, *, limit=1):
        self.control = RuntimeControl(AttemptJournal(self.root/"runtime/attempts.sqlite3",request_limit=limit),
                                     self.root/"runtime",minimum_free_bytes=0)
        runner = service.TrackingService(self.root/"controlled",runtime_control=self.control,cached_callback=self.callback)
        self.addCleanup(runner.close)
        return runner

    def test_control_binds_worker_and_rejects_new_real_admission_when_dead_or_exhausted(self):
        controlled = self.controlled()
        controlled.callback = Mock()
        self.runner = controlled
        request = self.real_request()
        self.assertFalse(self.control.readiness("shadow_tracking")["ready"])
        with self.assertRaises(RuntimeUnavailable): controlled.submit(request)
        controlled.start()
        self.control.journal.reserve("c"*32,"responses",None,operation_limit=1)
        with self.assertRaises(RuntimeUnavailable): controlled.submit(request)
        with sqlite3.connect(controlled.db_path) as db:
            self.assertEqual(0,db.execute("SELECT count(*) FROM diagnostic_jobs").fetchone()[0])

    def test_exact_historical_duplicate_returns_when_closed_and_exhausted(self):
        controlled = self.controlled(); controlled.callback = Mock()
        self.runner = controlled
        request = self.real_request()
        # A healthy probe is needed only for NEW paid work; no inference runs here.
        with patch.object(controlled,"readiness",return_value={"worker_available":True}):
            # Rebind the registered bound probe by patching its worker indicators.
            controlled.worker = Mock(is_alive=Mock(return_value=True))
            first = controlled.submit(request)
        controlled.stop.set()
        self.control.journal.reserve("d"*32,"responses",None,operation_limit=1)
        self.assertEqual(first,controlled.submit(request))
        changed = deepcopy(request); changed["idempotency_key"] = "new-request"
        with self.assertRaises(service.TrackingConflict): controlled.submit(changed)
        controlled.callback.assert_not_called()

    def test_cached_new_work_ignores_paid_budget_but_closed_worker_rejects_new_keys(self):
        controlled = self.controlled(); self.control.journal.reserve("e"*32,"responses",None,operation_limit=1)
        job = controlled.submit(self.request)
        controlled.run_once()
        self.assertEqual("succeeded",controlled.get(job["job_id"])["state"])
        controlled.stop.set()
        self.assertEqual(job["job_id"],controlled.submit(self.request)["job_id"])
        changed = {**self.request,"idempotency_key":"closed-new"}
        with self.assertRaises(service.TrackingConflict): controlled.submit(changed)

    def test_injected_callback_cannot_launder_live_model_origin(self):
        self.runner.callback = Mock()
        request = self.real_request()
        source = self.runner.asset(request["asset_id"])["sha256"]
        self.runner.callback.return_value = result("real_meta_video",source,"live_meta_api")
        job = self.runner.submit(request); self.runner.run_once()
        self.assertEqual("injected_callback",self.runner.get(job["job_id"])["result"]["inference_origin"])

    def test_custom_provider_factory_is_explicitly_injected(self):
        job, _ = self.model_path()
        self.assertEqual("injected_provider",job["result"]["inference_origin"])

    def test_atomic_asset_reservations_bound_concurrent_service_instances(self):
        second = service.TrackingService(self.root,cached_callback=self.callback)
        started,release = threading.Event(),threading.Event()
        count,lock,results = [0],threading.Lock(),[]
        def probe(path):
            with lock:
                count[0] += 1
                if count[0] == 2: started.set()
            release.wait(2)
            return deepcopy(self.source),"authored-probe"
        def upload(runner):
            results.append(runner.register_asset(b"1234ftypauthored-pixels"))
        with patch.object(service,"MAX_ASSETS",2),patch.object(service.video,"_probe",side_effect=probe):
            workers = [threading.Thread(target=upload,args=(runner,)) for runner in (self.runner,second)]
            for worker in workers: worker.start()
            self.assertTrue(started.wait(2))
            with self.assertRaises(ValidationError): self.runner.register_asset(b"1234ftypmore-authored-pixels")
            release.set()
            for worker in workers: worker.join(2)
        self.assertEqual(2,len(results))
        with sqlite3.connect(self.runner.db_path) as db:
            self.assertEqual(2,db.execute("SELECT count(*) FROM diagnostic_assets").fetchone()[0])

    def test_failed_asset_probe_cleans_reservation_without_deleting_known_asset(self):
        original = self.asset()
        original_path = self.runner.assets/(original["asset_id"]+".mp4")
        original_bytes = original_path.read_bytes()
        with patch.object(service.video,"_probe",side_effect=ValidationError("unsupported")),self.assertRaises(ValidationError):
            self.runner.register_asset(b"1234ftypbad-authored-video")
        with patch.object(service.secrets,"token_hex",return_value=original["asset_id"]),self.assertRaises(ValidationError):
            self.runner.register_asset(b"1234ftypcollision-video")
        self.assertEqual(original_bytes,original_path.read_bytes())
        with sqlite3.connect(self.runner.db_path) as db:
            self.assertEqual(1,db.execute("SELECT count(*) FROM diagnostic_assets").fetchone()[0])

    def test_missing_official_parser_rejects_new_production_job_before_queue_or_spend(self):
        controlled = self.controlled(); self.runner = controlled
        controlled.worker = Mock(is_alive=Mock(return_value=True))
        request = self.real_request()
        with patch.object(service.meta,"_sam_decoder",side_effect=service.meta.MetaAPIError(
                "private parser detail",code="sam_parser_unavailable")),self.assertRaisesRegex(ValidationError,"local media dependencies"):
            controlled.submit(request)
        with sqlite3.connect(controlled.db_path) as db:
            self.assertEqual(0,db.execute("SELECT count(*) FROM diagnostic_jobs").fetchone()[0])
        self.assertEqual(0,self.control.journal.summary()["total_reserved_attempts"])
        # Offline examples do not need the parser, even under the same control.
        with patch.object(service.meta,"_sam_decoder",side_effect=AssertionError("Cached mode must not inspect parser")):
            job = controlled.submit(self.request); controlled.run_once()
        self.assertEqual("succeeded",controlled.get(job["job_id"])["state"])

    def test_missing_media_executable_rejects_new_job_but_historical_dedup_needs_no_tools(self):
        controlled = self.controlled(); self.runner = controlled
        controlled.worker = Mock(is_alive=Mock(return_value=True))
        request = self.real_request()
        for missing in ("ffmpeg","ffprobe"):
            with patch.object(service.meta,"_sam_decoder",return_value=(object(),object(),object())), \
                    patch.object(service.visibility,"_extractor_path",return_value=str(self.root/"no-ffmpeg") if missing == "ffmpeg" else sys.executable), \
                    patch.object(service.video,"_probe_path",return_value=self.root/"no-ffprobe" if missing == "ffprobe" else Path(sys.executable)), \
                    self.assertRaisesRegex(ValidationError,"local media dependencies"):
                controlled.submit(request)
        with sqlite3.connect(controlled.db_path) as db:
            self.assertEqual(0,db.execute("SELECT count(*) FROM diagnostic_jobs").fetchone()[0])
        self.assertEqual(0,self.control.journal.summary()["total_reserved_attempts"])
        with patch.object(controlled,"_preflight",return_value=None):
            job = controlled.submit(request)
        with patch.object(controlled,"_preflight",side_effect=AssertionError("Historical read must not inspect tools")):
            self.assertEqual(job,controlled.submit(request))

    def test_cached_job_has_bounded_private_stage_traces_and_zero_provider_calls(self):
        controlled = self.controlled()
        projected = result()
        projected["frames"] = [{**deepcopy(projected["frames"][0]),"frame_index":index,"timestamp_ms":index*200,
            "gap_codes":["candidate_missing","private_prompt_body"],"warning":"near_gripper_proposal"}
            for index in range(80)]
        self.callback.return_value = projected
        with patch.object(service.meta.MetaClient,"_request",side_effect=AssertionError("Cached tracking must not call models")):
            job = controlled.submit(self.request)
            controlled.run_once()
        saved = controlled.get(job["job_id"])
        self.assertEqual(("succeeded","cached_demo",0),(saved["state"],saved["result"]["mode"],saved["result"]["request_attempts"]))
        traces = self.control.snapshot()["telemetry"]["recent_spans"]
        self.assertTrue({"admission","queue_wait","execution","verification","artifact_publication"}.issubset({row["stage"] for row in traces}))
        self.assertTrue(all(row["operation_id"]==job["job_id"] and row["operation_kind"]=="shadow_tracking" and row["model"]=="none" for row in traces))
        reasons = [row["code"] for row in traces if row["stage"]=="evidence_request"]
        self.assertEqual({"not_detected","unsupported_observation","boundary_unconfirmed"},set(reasons))
        self.assertEqual(3,len(reasons))
        for private in (str(self.root),"private_prompt_body","authored-video-pixels"):
            self.assertNotIn(private,json.dumps(traces))
        self.assertEqual(0,self.control.journal.summary()["total_reserved_attempts"])

    def test_trace_storage_failure_preserves_already_returned_diagnostic_and_charge_projection(self):
        controlled = self.controlled();self.runner = controlled
        controlled.worker = Mock(is_alive=Mock(return_value=True))
        request = self.real_request()
        returned = result("real_meta_video",controlled.asset(request["asset_id"])["sha256"])
        returned.update(request_attempts=1,estimated_usd=.01)
        controlled.callback = Mock(return_value=returned)
        job = controlled.submit(request)
        with patch.object(self.control.telemetry,"record",side_effect=OSError("private disk detail")), \
                patch.object(service.meta.MetaClient,"_request",side_effect=AssertionError("Authored callback must not infer")):
            controlled.run_once()
        saved = controlled.get(job["job_id"])
        self.assertEqual("succeeded",saved["state"])
        self.assertEqual((1,.01),(saved["result"]["request_attempts"],saved["result"]["estimated_usd"]))
        self.assertIn("telemetry_unavailable",self.control.readiness("shadow_tracking")["reason_codes"])
        self.assertEqual(1,controlled.callback.call_count)


if __name__ == "__main__": unittest.main()
