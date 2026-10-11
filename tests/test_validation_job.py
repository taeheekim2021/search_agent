"""Synthetic orchestration tests: no models, Cloud APIs, or OpenSearch started."""

import base64
import hashlib
import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from search_agent import validation_job as job
from search_agent.media.wikidata import convert


@pytest.fixture
def setup(tmp_path):
    env = {
        "VALIDATION_EXECUTE": "1",
        "VALIDATION_LOOPBACK_AUTH_APPROVED": "1",
        "CLOUD_RUN_JOB": "test",
        "CLOUD_RUN_TASK_COUNT": "1",
        "CLOUD_RUN_TASK_INDEX": "0",
        "CLOUD_RUN_TASK_ATTEMPT": "0",
        "SEARCH_HF_CACHE_DIR": str(tmp_path / "cache"),
        "SEARCH_CODE_REVISION": "a" * 40,
        "SEARCH_EMBEDDING__REVISION": "b" * 40,
        "SEARCH_RERANKER__REVISION": "c" * 40,
        "VALIDATION_MODEL_IMAGE": "example.invalid/model@sha256:" + "d" * 64,
        "VALIDATION_ENGINE_IMAGE": "example.invalid/engine@sha256:" + "e" * 64,
    }
    for model, rev in ((job.QWEN_ID, "b" * 40), (job.BGE_ID, "c" * 40)):
        p = (
            Path(env["SEARCH_HF_CACHE_DIR"])
            / ("models--" + model.replace("/", "--"))
            / "snapshots"
            / rev
        )
        p.mkdir(parents=True)
        (p / "config.json").write_text("{}")
        (p / "model.safetensors").write_text("synthetic placeholder; never loaded")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        convert(
            [
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q123"},
                    "itemLabel": {"value": "fixture"},
                }
            ],
            "2026-10-11T00:00:00Z",
            1,
        ).model_dump_json()
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps({"queries": [{"id": str(i), "query": "fixture"} for i in range(3)]})
    )
    return env, manifest, queries


@pytest.mark.parametrize(
    "key,value",
    [
        ("CLOUD_RUN_TASK_COUNT", "2"),
        ("CLOUD_RUN_TASK_ATTEMPT", "1"),
        ("CLOUD_RUN_TASK_INDEX", "1"),
        ("K_SERVICE", "public"),
        ("VALIDATION_EXECUTE", "0"),
        ("SEARCH_EMBEDDING__REVISION", "main"),
        ("VALIDATION_MODEL_IMAGE", "image:latest"),
    ],
)
def test_preflight_rejects_unbounded_or_unpinned(setup, key, value):
    env, manifest, queries = setup
    with pytest.raises(ValueError):
        job.preflight(manifest, queries, env | {key: value})


def test_inputs_and_offline_environment(setup, tmp_path):
    env, manifest, queries = setup
    assert job.preflight(manifest, queries, env)["manifest"]
    child = job.child_environment(
        env
        | {
            "HF_TOKEN": "secret",
            "OPENSEARCH_URL": "https://remote.invalid",
            "SEARCH_ADMIN_API_KEY": "secret",
        },
        tmp_path,
        "abc",
    )
    assert "HF_TOKEN" not in child and "SEARCH_ADMIN_API_KEY" not in child
    assert child["HF_HUB_OFFLINE"] == "1"
    assert child["OPENSEARCH_URL"] == "http://127.0.0.1:9200"
    assert child["SEARCH_RERANKER__BATCH_SIZE"] == "1"
    raw = json.loads(manifest.read_text())
    raw["entries"] *= 21
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        job.preflight(manifest, queries, env)


def test_stage_timeout_terminates_group(monkeypatch):
    process = Mock(pid=123)
    process.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 1), None, None]
    monkeypatch.setattr(job.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(job.os, "killpg", kill)
    with pytest.raises(subprocess.TimeoutExpired):
        job.run_stage(["synthetic"], {}, job.time.monotonic() + 1)
    assert kill.call_args_list[0].args == (123, signal.SIGTERM)
    assert kill.call_args_list[1].args == (123, signal.SIGKILL)


def test_export_chunks_checksum_and_size(capsys):
    evidence = {"status": "completed", "text": "가" * 15000}
    job.export("validation/test/final.json", evidence)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[-1]["validation_evidence"] == "complete"
    assert all(len(json.dumps(line).encode()) < 25000 for line in lines)
    raw = b"".join(base64.b64decode(line["base64"]) for line in lines[:-1])
    assert hashlib.sha256(raw).hexdigest() == lines[-1]["sha256"]
    assert json.loads(raw) == evidence
    with pytest.raises(ValueError):
        job.export("x", {"x": "x" * job.MAX_EVIDENCE_BYTES})


@pytest.mark.parametrize("failure", [None, "index", "search", "cleanup"])
def test_supervisor_exports_inputs_partial_results_and_cleanup(
    setup, tmp_path, monkeypatch, failure
):
    env, manifest, queries = setup
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(job, "WORK", tmp_path)
    monkeypatch.setattr(job.sys, "argv", ["job", str(manifest), str(queries)])
    monkeypatch.setattr(job.signal, "signal", Mock())
    monkeypatch.setattr(job.signal, "alarm", Mock())
    upload = Mock()
    monkeypatch.setattr(job, "export", upload)
    engine = Mock()
    monkeypatch.setattr(job.subprocess, "Popen", Mock(return_value=engine))
    monkeypatch.setattr(job, "wait_engine", Mock())
    stop = Mock(side_effect=RuntimeError("cleanup secret") if failure == "cleanup" else None)
    monkeypatch.setattr(job, "stop", stop)

    def stage(command, env, deadline):
        operation = command[3]
        if operation == failure:
            Path(command[command.index("--output") + 1]).write_text(json.dumps({"partial": True}))
            raise RuntimeError("sensitive exception must not escape")
        if operation == "search":
            assert command[command.index("--repeats") + 1] == "2"
            assert command[command.index("--warmups") + 1] == "1"
        report = {
            "actual_models_loaded": {"embedding": True, "reranker": True},
            "code_revision": env["SEARCH_CODE_REVISION"],
            "resolved_embedding_revision": env["SEARCH_EMBEDDING__REVISION"],
            "resolved_reranker_revision": env["SEARCH_RERANKER__REVISION"],
            "indexing": {"documents_indexed": 1},
        }
        call = {"ok": True, "model_inference_performed": True}
        report.update(cold=call, warmup=[call], warm=[{"samples": [call] * 6}])
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report))

    monkeypatch.setattr(job, "run_stage", stage)
    assert job.main() == (1 if failure else 0)
    assert upload.call_args_list[0].args[0].endswith("/baseline.json")
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["inputs"]["manifest"]
    assert final["status"] == ("failed" if failure else "completed")
    assert "sensitive exception" not in json.dumps(final)
    assert "cleanup secret" not in json.dumps(final)
    if failure in ("index", "search"):
        assert final["reports"][failure]["partial"] is True
    stop.assert_called_once_with(engine)
    assert upload.call_args.args[0].endswith("/final.json")


def test_reconstruct_shuffled_duplicate_chunks_and_reject_corruption(capsys):
    from search_agent.validation_evidence import reconstruct

    name = "validation/" + "a" * 32 + "/final.json"
    job.export(name, {"status": "completed", "metadata": "x" * 20000})
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    shuffled = [{"jsonPayload": row} for row in reversed(rows)]
    shuffled.append({"jsonPayload": rows[0]})
    assert json.loads(reconstruct(shuffled)[name])["status"] == "completed"
    with pytest.raises(ValueError, match="Incomplete"):
        reconstruct(rows[:-1])
    with pytest.raises(ValueError, match="checksum"):
        reconstruct([row | {"sha256": "0" * 64} for row in rows])
    with pytest.raises(ValueError, match="Unexpected artifact"):
        reconstruct([row | {"artifact": "../outside.json"} for row in rows])


def test_auth_exception_requires_explicit_approval(setup):
    env, manifest, queries = setup
    env.pop("VALIDATION_LOOPBACK_AUTH_APPROVED")
    with pytest.raises(ValueError, match="requires approval"):
        job.preflight(manifest, queries, env)
