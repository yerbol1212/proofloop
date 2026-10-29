"""Installed entry point for the local workbench and zero-inference replay."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="proofloop", description=__doc__)
    parser.add_argument("command", choices=("serve", "doctor", "replay", "demo", "track-demo"))
    args, rest = parser.parse_known_args(argv)
    if args.command in ("replay", "demo"):
        from .replay_bundle import main as replay
        return replay(rest)
    if args.command == "serve":
        from .server import main as serve
        return serve(rest)
    if args.command == "track-demo":
        from .client import ProofLoopClient, ClientError
        import secrets
        request = argparse.ArgumentParser(prog="proofloop track-demo")
        request.add_argument("--port", type=int, default=8790)
        request.add_argument("--wait-seconds", type=float, default=120)
        request.add_argument("--idempotency-key", default=None)
        settings = request.parse_args(rest)
        try:
            client = ProofLoopClient(settings.port)
            key = settings.idempotency_key or secrets.token_hex(16)
            # A lost acceptance response must not erase the caller's recovery
            # reference. Reusing this key retrieves the original cached job.
            print(json.dumps({"idempotency_key": key, "mode": "cached_demo", "provider_requests": 0}), flush=True)
            job = client.cached_tracking(key)
            print(json.dumps({"job_id": job["job_id"], "mode": "cached_demo", "provider_requests": 0}), flush=True)
            if settings.wait_seconds:
                terminal = client.wait_tracking(job["job_id"], timeout=settings.wait_seconds)
                print(json.dumps(terminal, sort_keys=True))
                if terminal.get("state") != "succeeded" or terminal.get("integrity", {}).get("result_valid") is not True:
                    return 2
            return 0
        except (ValueError, ClientError) as error:
            print(str(error))
            return 2
    checks = argparse.ArgumentParser(prog="proofloop doctor")
    checks.add_argument("--data-dir")
    checks.add_argument("--profile", choices=("core", "workbench"), default="workbench",
                        help="Check the complete offline workbench (default), or the Python-only core")
    settings = checks.parse_args(rest)
    from .installation import doctor
    report = doctor(settings.data_dir, profile=settings.profile)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
