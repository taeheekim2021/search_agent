"""Finite offline-model validation supervisor; never a production OpenSearch service."""

import argparse
import base64
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path

import httpx

from search_agent.config import BGE_ID, QWEN_ID
from search_agent.media.ingest import atomic_json, load_manifest

WORK = Path("/validation-work")
ENGINE = "/opt/opensearch/bin/opensearch"
COMPUTE_SECONDS = 1500
MAX_EVIDENCE_BYTES = 1_000_000


def read_json(path: Path) -> dict:
    if path.stat().st_size > 200_000:
        raise ValueError("Input/report too large")
    return json.loads(path.read_text())


def read_report(path: Path, operation: str) -> dict:
    report = read_json(path)
    if operation == "search":
        for run in report.get("warm", []):
            run.pop("latency_ms_successes", None)
        report["smoke_scope"] = (
            "8 planned calls; no statistically reliable percentiles or relevance conclusions"
        )
    return report


def preflight(manifest: Path, queries: Path, env: dict[str, str]) -> dict:
    if env.get("VALIDATION_EXECUTE") != "1":
        raise ValueError("Explicit execution opt-in required")
    if env.get("VALIDATION_LOOPBACK_AUTH_APPROVED") != "1":
        raise ValueError("Loopback-only test authentication exception requires approval")
    if os.geteuid() == 0 or Path(".env").exists():
        raise ValueError("Non-root isolated image without dotenv required")
    if (
        env.get("K_SERVICE")
        or any(
            env.get(k) != v
            for k, v in {
                "CLOUD_RUN_TASK_COUNT": "1",
                "CLOUD_RUN_TASK_INDEX": "0",
                "CLOUD_RUN_TASK_ATTEMPT": "0",
            }.items()
        )
        or not env.get("CLOUD_RUN_JOB")
    ):
        raise ValueError("Only a first-attempt single-task Cloud Run Job is supported")
    for key in ("SEARCH_CODE_REVISION", "SEARCH_EMBEDDING__REVISION", "SEARCH_RERANKER__REVISION"):
        if not re.fullmatch(r"[0-9a-f]{40}", env.get(key, "")):
            raise ValueError("Immutable code/model revision required")
    for key in ("VALIDATION_MODEL_IMAGE", "VALIDATION_ENGINE_IMAGE"):
        if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", env.get(key, "")):
            raise ValueError("Verified base image digest required")
    cache = Path(env["SEARCH_HF_CACHE_DIR"])
    if not cache.is_absolute():
        raise ValueError("Absolute baked model cache path required")
    for model, key in (
        (QWEN_ID, "SEARCH_EMBEDDING__REVISION"),
        (BGE_ID, "SEARCH_RERANKER__REVISION"),
    ):
        snapshot = cache / ("models--" + model.replace("/", "--")) / "snapshots" / env[key]
        if not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
            raise ValueError("Required baked model snapshot unavailable")
    raw = read_json(manifest)
    parsed = load_manifest(manifest)
    if (
        not 1 <= len(parsed.entries) <= 20
        or any(e.rights_scope != "metadata" for e in parsed.entries)
        or len(raw.get("entries", [])) != len(parsed.entries)
    ):
        raise ValueError("Supply 1..20 reviewed public metadata-only entries")
    query_data = read_json(queries)
    items = query_data.get("queries", [])
    if not isinstance(items, list) or len(items) != 3:
        raise ValueError("Supply exactly 3 queries (8 total search calls)")
    if any(
        not isinstance(q, dict) or not isinstance(q.get("query"), str) or len(q["query"]) > 500
        for q in items
    ):
        raise ValueError("Invalid bounded query")
    return {"manifest": raw, "queries": query_data}


def child_environment(env: dict[str, str], work: Path, run_id: str) -> dict[str, str]:
    # Allowlist avoids inherited server credentials and configuration overrides.
    result = {k: env[k] for k in ("PATH", "PYTHONPATH", "LANG", "LD_LIBRARY_PATH") if k in env}
    result.update(
        {
            "TMPDIR": str(work),
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "SEARCH_MODE": "real",
            "SEARCH_BACKEND": "opensearch",
            "SEARCH_ENVIRONMENT": "development",
            "SEARCH_DIMENSIONS": "2560",
            "SEARCH_CANDIDATE_TOP_K": "5",
            "SEARCH_RESULT_TOP_N": "3",
            "SEARCH_CACHE_DIR": str(work / "embeddings"),
            "MEDIA_ROOT": str(work / "media"),
            "OPENSEARCH_URL": "http://127.0.0.1:9200",
            "OPENSEARCH_ALLOW_HTTP_LOCAL": "true",
            "OPENSEARCH_PRODUCTION": "false",
            "OPENSEARCH_REPLICAS": "0",
            "OPENSEARCH_SHARDS": "1",
            "OPENSEARCH_INDEX": "validation-v1-" + run_id,
            "OPENSEARCH_BULK_RETRIES": "0",
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
            "OPENSEARCH_JAVA_HOME": "/opt/opensearch/jdk",
            "OPENSEARCH_JAVA_OPTS": "-Xms512m -Xmx512m -XX:ActiveProcessorCount=2",
        }
    )
    for key in (
        "SEARCH_HF_CACHE_DIR",
        "SEARCH_CODE_REVISION",
        "SEARCH_EMBEDDING__REVISION",
        "SEARCH_RERANKER__REVISION",
    ):
        result[key] = env[key]
    for model in ("EMBEDDING", "RERANKER"):
        for key, value in (
            ("DTYPE", "bfloat16"),
            ("DEVICE", "cpu"),
            ("BATCH_SIZE", "1"),
            ("MAX_LENGTH", "128"),
        ):
            result[f"SEARCH_{model}__{key}"] = value
    return result


def stop(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    # Kill the process group even if its leader already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def export(name: str, evidence: dict) -> None:
    """Checksummed chunks for Cloud Logging; no credentials or new storage permissions."""
    data = json.dumps(evidence, ensure_ascii=False).encode()
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError("Evidence exceeds bounded export size")
    chunks = [data[i : i + 12_288] for i in range(0, len(data), 12_288)]
    common = {"artifact": name, "sha256": hashlib.sha256(data).hexdigest(), "parts": len(chunks)}
    for i, chunk in enumerate(chunks):
        print(
            json.dumps(
                {
                    "validation_evidence": "chunk",
                    **common,
                    "part": i,
                    "base64": base64.b64encode(chunk).decode(),
                }
            ),
            flush=True,
        )
    print(json.dumps({"validation_evidence": "complete", **common}), flush=True)


def run_stage(command: list[str], env: dict[str, str], deadline: float) -> None:
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if code:
            raise RuntimeError("Benchmark stage failed")
    finally:
        stop(process)


def wait_engine(process: subprocess.Popen, deadline: float) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("OpenSearch exited during startup")
            try:
                r = client.get("http://127.0.0.1:9200/_cluster/health")
                if r.status_code == 200 and r.json().get("status") in ("yellow", "green"):
                    version = client.get("http://127.0.0.1:9200/").json()["version"]["number"]
                    if version != "2.19.3":
                        raise ValueError("Unexpected engine version")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise TimeoutError("OpenSearch readiness deadline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("queries", type=Path)
    args = parser.parse_args()
    engine = None
    evidence: dict = {}
    run_id = uuid.uuid4().hex
    stage = "preflight"
    exit_code = 1
    started = time.monotonic()
    deadline = started + COMPUTE_SECONDS

    # SIGTERM/soft deadline unwind through cleanup/export; platform timeout is 1800s.
    def terminate(signum, frame):
        raise TimeoutError("Validation interrupted")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGALRM, terminate)
    signal.alarm(COMPUTE_SECONDS)
    try:
        inputs = preflight(args.manifest, args.queries, dict(os.environ))
        WORK.mkdir(parents=True, exist_ok=True)
        for directory in ("opensearch-data", "opensearch-logs", "tmp"):
            (WORK / directory).mkdir(exist_ok=True)
        evidence = {
            "run_id": run_id,
            "status": "started",
            "inputs": inputs,
            "code_revision": os.environ["SEARCH_CODE_REVISION"],
            "model_image": os.environ["VALIDATION_MODEL_IMAGE"],
            "engine_image": os.environ["VALIDATION_ENGINE_IMAGE"],
            "input_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            "scope": "ephemeral_loopback_real_models; not production performance or HTTP serving",
            "rlimit_nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
            "vm_max_map_count": Path("/proc/sys/vm/max_map_count").read_text().strip(),
            "reports": {},
        }
        # Emit baseline BEFORE spawning models or engine; verify retained logs after execution.
        stage = "baseline_export"
        export(f"validation/{run_id}/baseline.json", evidence)
        env = child_environment(dict(os.environ), WORK, run_id)
        stage = "engine_start"
        engine = subprocess.Popen(
            [ENGINE],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        wait_engine(engine, min(deadline, time.monotonic() + 120))
        for operation, source in (("index", args.manifest), ("search", args.queries)):
            stage = operation
            output = WORK / f"{operation}.json"
            command = [
                sys.executable,
                "-m",
                "search_agent.benchmark",
                operation,
                str(source),
                "--output",
                str(output),
                "--run-real",
            ]
            if operation == "search":
                command += [
                    "--corpus-manifest",
                    str(args.manifest),
                    "--concurrency",
                    "1",
                    "--repeats",
                    "2",
                    "--warmups",
                    "1",
                ]
            try:
                run_stage(
                    command,
                    env,
                    min(deadline, time.monotonic() + (720 if operation == "index" else 600)),
                )
            except Exception:
                # benchmark emits a report even for many nonzero outcomes; preserve it.
                with suppress(OSError, ValueError, TypeError, AttributeError):
                    evidence["reports"][operation] = read_report(output, operation)
                raise
            report = read_report(output, operation)
            if (
                report.get("code_revision") != env["SEARCH_CODE_REVISION"]
                or report.get("resolved_embedding_revision") != env["SEARCH_EMBEDDING__REVISION"]
            ):
                raise ValueError("Code/embedding provenance mismatch")
            if operation == "index" and (
                report.get("actual_models_loaded", {}).get("embedding") is not True
                or report.get("indexing", {}).get("documents_indexed")
                != len(inputs["manifest"]["entries"])
            ):
                raise ValueError("Real indexing acceptance failed")
            if operation == "search":
                if report.get("resolved_reranker_revision") != env["SEARCH_RERANKER__REVISION"]:
                    raise ValueError("Reranker provenance mismatch")
                calls = [report["cold"], *report["warmup"], *report["warm"][0]["samples"]]
                if len(calls) != 8 or not all(
                    c.get("ok") and c.get("model_inference_performed") for c in calls
                ):
                    raise ValueError("Eight real-model search calls required")
                for run in report["warm"]:
                    run.pop("latency_ms_successes", None)
                report["smoke_scope"] = (
                    "8 calls; no statistically reliable percentiles or relevance conclusions"
                )
            evidence["reports"][operation] = report
            if operation == "search" and report.get("actual_models_loaded") != {
                "embedding": True,
                "reranker": True,
            }:
                raise ValueError("Both real models must have loaded")
            stage = operation + "_export"
            export(f"validation/{run_id}/{operation}.json", evidence)
        evidence["status"] = "completed"
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 — no exception text or credentials in evidence
        evidence.update(status="failed", stage=stage, error_type=type(exc).__name__)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            stop(engine)
            evidence["cleanup"] = (
                "owned_child_process_groups_terminated; ephemeral_index_not_retained"
            )
        except Exception as exc:  # noqa: BLE001 — still attempt final evidence export
            evidence.update(
                status="failed", cleanup="failed", cleanup_error_type=type(exc).__name__
            )
            exit_code = 1
        evidence["elapsed_seconds"] = time.monotonic() - started
        try:
            if evidence.get("inputs"):
                atomic_json(WORK / "final.json", evidence)
                export(f"validation/{run_id}/final.json", evidence)
            else:
                exit_code = 1
        except Exception:  # noqa: BLE001
            exit_code = 1
            evidence["export_failed"] = True
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": evidence.get("status"),
                    "exit_code": exit_code,
                    "export_failed": evidence.get("export_failed", False),
                }
            )
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
