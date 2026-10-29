"""Small loopback-only client for external developer/agent tools.

Methods are explicit; no remote URL, arbitrary route, file reader or physical
executor is exposed. Timeout stops polling and never resubmits an operation.
"""
from __future__ import annotations

import base64
import http.client
import json
import re
import time


class ClientError(RuntimeError):
    pass


class ProofLoopClient:
    def __init__(self, port=8790, *, timeout=20):
        if type(port) is not int or not 1 <= port <= 65535 or type(timeout) not in (int, float) or not 0 < timeout <= 180:
            raise ValueError("Use a local port and a bounded request timeout")
        self.port, self.timeout = port, timeout

    def _request(self, method, path, payload=None, *, accept_not_ready=False):
        host = "127.0.0.1:" + str(self.port)
        headers = {"Host": host}
        if method == "POST":
            state = self._request("GET", "/api/state")
            token = state.get("csrf_token")
            if type(token) is not str or not 16 <= len(token) <= 128:
                raise ClientError("Local mutation capability is unavailable")
            headers.update({"Origin": "http://" + host, "X-ProofLoop-Token": token, "Content-Type": "application/json"})
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024 or response.getheader("Content-Type", "").split(";")[0] != "application/json":
                raise ClientError("The local response is invalid or excessive")
            result = json.loads(data)
            if type(result) is not dict:
                raise ClientError("The local response is invalid")
            if accept_not_ready and response.status == 503 and result.get("ready") is False and result.get("schema_version") == "1.0":
                return result
            if not 200 <= response.status < 300:
                code = result.get("error", {}).get("code")
                safe = code if type(code) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else "local_request_rejected"
                raise ClientError(safe)
            return result
        except (OSError, ValueError, http.client.HTTPException):
            raise ClientError("The local request did not complete; recover its original reference before submitting new work") from None
        finally:
            connection.close()

    def readiness(self):
        return self._request("GET", "/api/ready", accept_not_ready=True)

    def operations(self):
        return self._request("GET", "/api/v1/operations")

    def replay(self):
        return self._request("GET", "/api/v1/replay")

    def tracking_capabilities(self):
        return self._request("GET", "/api/v1/tracking/capabilities")

    def upload_tracking_video(self, data):
        if type(data) is not bytes or not 12 <= len(data) <= 8 * 1024 * 1024:
            raise ValueError("Supply bounded video bytes")
        return self._request("POST", "/api/v1/tracking/assets", {"media_type": "video/mp4", "data": base64.b64encode(data).decode()})["asset"]

    def submit_tracking(self, request):
        return self._request("POST", "/api/v1/tracking/jobs", request)["job"]

    def cached_tracking(self, idempotency_key):
        return self.submit_tracking({"mode": "cached_demo", "fixture_id": "latest-tracking", "idempotency_key": idempotency_key})

    def tracking_job(self, job_id):
        if type(job_id) is not str or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Use the opaque tracking job reference")
        return self._request("GET", "/api/v1/tracking/jobs/" + job_id)["job"]

    def wait_tracking(self, job_id, *, timeout=120):
        if type(timeout) not in (int, float) or not 0 <= timeout <= 3600:
            raise ValueError("Use a bounded polling duration")
        deadline = time.monotonic() + timeout
        while True:
            job = self.tracking_job(job_id)
            if job.get("state") not in {"queued", "running"}:
                return job
            if time.monotonic() >= deadline:
                raise ClientError("Polling timed out; retain the original tracking job reference and poll it again")
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
