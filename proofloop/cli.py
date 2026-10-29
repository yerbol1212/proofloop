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
            job = client.cached_tracking(settings.idempotency_key or secrets.token_hex(16))
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
    settings = checks.parse_args(rest)
    from .installation import doctor
    report = doctor(settings.data_dir)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["python_supported"] and report["offline_replay_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
