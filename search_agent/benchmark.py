"""Opt-in real-model benchmark. Never downloads media or silently uses mock models."""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import BGE_ID, QWEN_ID, Settings
from .domain import Conditions, SearchRequest
from .media.config import MediaSettings, OpenSearchSettings
from .media.ingest import Ingestor, atomic_json, load_manifest
from .media.opensearch import OpenSearchStore
from .media.service import MediaSearchAgent
from .models import BGEReranker, QwenEmbedding


def load_json(path: Path) -> dict:
    if path.stat().st_size > 2_000_000:
        raise ValueError("Benchmark input exceeds 2 MB")
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected input object")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resources(start_cpu: float, elapsed: float) -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu = usage.ru_utime + usage.ru_stime - start_cpu
    return {
        "scope": "benchmark_and_model_process_only_not_OpenSearch",
        "peak_rss_scope": "process_lifetime",
        "cpu_seconds": cpu,
        "cpu_percent_of_one_core": 100 * cpu / elapsed if elapsed else None,
        "peak_rss_bytes": usage.ru_maxrss * (1024 if platform.system() == "Linux" else 1),
    }


def summarize(samples: list[dict], elapsed: float) -> dict:
    successful = [s["latency_ms"] for s in samples if s["ok"]]
    return {
        "requests": len(samples),
        "successful": len(successful),
        "failure_rate": 1 - len(successful) / len(samples) if samples else None,
        "wall_seconds": elapsed,
        "successful_requests_per_second": len(successful) / elapsed if elapsed else None,
        "latency_ms_successes": {
            f"p{p}": float(np.percentile(successful, p)) if successful else None
            for p in (50, 95, 99)
        },
        "model_inference_requests": sum(s.get("model_inference_performed", False) for s in samples),
    }


def query_once(agent, query: dict) -> dict:
    start = perf_counter()
    try:
        response = agent.search(SearchRequest(query=query["query"]))
        return {
            "query_id": query["id"],
            "ok": True,
            "latency_ms": (perf_counter() - start) * 1000,
            "model_inference_performed": response.model_inference_performed,
            "ids": [r.content_id for r in response.results],
        }
    except Exception as exc:  # noqa: BLE001 — never include upstream data/secret error text
        return {
            "query_id": query["id"],
            "ok": False,
            "latency_ms": (perf_counter() - start) * 1000,
            "error_type": type(exc).__name__,
        }


def run_queries(agent, queries: list[dict], concurrency: int, repeats: int) -> dict:
    if not 1 <= concurrency <= 64 or not 1 <= repeats <= 1000 or not queries:
        raise ValueError("Invalid benchmark workload bounds")
    work = queries * repeats
    if len(work) > 100_000:
        raise ValueError("Benchmark request limit exceeded")
    start, cpu = perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(lambda query: query_once(agent, query), work))
    elapsed = perf_counter() - start
    return {
        "concurrency": concurrency,
        "load_model": "closed_loop_threads",
        **summarize(samples, elapsed),
        "resources": resources(cpu.ru_utime + cpu.ru_stime, elapsed),
        "samples": samples,
    }


def quality(
    samples: list[dict], qrels: dict | None, query_hash: str, index_uuid: str, k: int
) -> dict:
    if qrels is None:
        return {"status": "not_evaluated", "reason": "No reviewed relevance judgments"}
    if (
        qrels.get("reviewed") is not True
        or qrels.get("queries_sha256") != query_hash
        or qrels.get("index_uuid") != index_uuid
        or not qrels.get("source")
    ):
        raise ValueError("Judgments must be reviewed and bound to query hash and index UUID")
    judgments = qrels["judgments"]
    observed = {s["query_id"]: s for s in samples}
    metrics = []
    for query_id, sample in observed.items():
        grades = judgments.get(query_id)
        if (
            not sample["ok"]
            or not isinstance(grades, dict)
            or not grades
            or any(type(v) is not int or not 0 <= v <= 3 for v in grades.values())
            or not any(grades.values())
        ):
            raise ValueError(
                "Each evaluated query needs successful retrieval and reviewed 0..3 grades"
            )
        ids = sample["ids"][:k]
        dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ids))
        ideal = sum(
            (2**g - 1) / math.log2(i + 2)
            for i, g in enumerate(sorted(grades.values(), reverse=True)[:k])
        )
        metrics.append(
            {
                "query_id": query_id,
                "ndcg": dcg / ideal,
                "recall_of_judged_relevant": sum(grades.get(key, 0) > 0 for key in set(ids))
                / sum(g > 0 for g in grades.values()),
            }
        )
    return {
        "status": "evaluated_against_supplied_judgments",
        "k": k,
        "unjudged_policy": "nonrelevant; incomplete judgments may bias metrics",
        "queries": metrics,
        "mean_ndcg": sum(m["ndcg"] for m in metrics) / len(metrics),
    }


def index_info(store: OpenSearchStore) -> dict:
    info = store.request("GET", f"/{store.index}/_settings")
    mapping = store.request("GET", f"/{store.index}/_mapping")
    if not info or not mapping or set(info) != {store.index} or set(mapping) != {store.index}:
        raise ValueError("Benchmark requires a concrete index snapshot")
    return {
        "name": store.index,
        "uuid": info[store.index]["settings"]["index"]["uuid"],
        "document_count": store.count(Conditions()),
        "mapping_meta": mapping[store.index]["mappings"]["_meta"],
        "write_blocked": info[store.index]["settings"]["index"].get("blocks", {}).get("write"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["index", "search"])
    parser.add_argument("input", type=Path, help="Metadata manifest or benchmark query JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument(
        "--corpus-manifest", type=Path, help="Required for search dataset provenance"
    )
    args = parser.parse_args()
    if not args.run_real:
        parser.error("--run-real is required; this may download weights and perform index writes")
    store = None
    try:
        settings = Settings()
        if (
            settings.mode != "real"
            or settings.backend != "opensearch"
            or settings.dimensions != 2560
        ):
            raise ValueError("Benchmark requires real OpenSearch and 2560 dimensions")
        if any(len(c.revision) != 40 for c in (settings.embedding, settings.reranker)):
            raise ValueError("Pin both model revisions before benchmarking")
        store = OpenSearchStore(OpenSearchSettings.for_search(settings))
        embedding = QwenEmbedding(settings.embedding, str(settings.hf_cache_dir), 2560)
        reranker = BGEReranker(settings.reranker, str(settings.hf_cache_dir))
        report: dict = {
            "created_at": datetime.now(UTC).isoformat(),
            "code_revision": os.getenv("SEARCH_CODE_REVISION", "unrecorded"),
            "opensearch_version": (store.request("GET", "/") or {})
            .get("version", {})
            .get("number"),
            "mode": "real_models_requested",
            "input_sha256": digest(args.input),
            "python": platform.python_version(),
            "embedding_model": QWEN_ID,
            "reranker_model": BGE_ID,
            "embedding_config": settings.embedding.model_dump(),
            "reranker_config": settings.reranker.model_dump(),
            "top_k": settings.candidate_top_k,
            "top_n": settings.result_top_n,
            "scope": "single_process_service_path; excludes HTTP ingress/load_balancer",
        }
        start, cpu = perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        if args.operation == "index":
            manifest = load_manifest(args.input)
            result = Ingestor(MediaSettings(), embedding=embedding, index=store).run(
                manifest, dry_run=False
            )
            elapsed = perf_counter() - start
            report["indexing"] = {
                "status": result["status"],
                "documents_requested": len(manifest.entries),
                "documents_indexed": result["indexed"],
                "seconds": elapsed,
                "documents_per_second": result["indexed"] / elapsed,
                "cache_policy": "existing_persistent_cache_not_purged",
                "includes_model_load": True,
                "media_downloaded": False,
                "failed": len(manifest.entries) - result["indexed"],
            }
            failed = result["status"] != "completed"
        else:
            if args.corpus_manifest is None:
                raise ValueError("--corpus-manifest required for dataset provenance")
            corpus = load_manifest(args.corpus_manifest)
            report["declared_corpus_sha256"] = digest(args.corpus_manifest)
            report["declared_corpus_document_count"] = len(corpus.entries)
            snapshot = store.snapshot()
            before = index_info(snapshot)
            if before["document_count"] != len(corpus.entries):
                raise ValueError("Declared corpus count does not match benchmark index")
            queries = load_json(args.input)["queries"]
            if not isinstance(queries, list) or not 1 <= len(queries) <= 500:
                raise ValueError("Supply 1..500 queries")
            if len({q["id"] for q in queries}) != len(queries):
                raise ValueError("Query IDs must be unique")
            for query in queries:
                SearchRequest(query=query["query"])
                if not isinstance(query["id"], str) or not 1 <= len(query["id"]) <= 100:
                    raise ValueError("Invalid query ID")
            levels = [int(c) for c in args.concurrency.split(",")]
            if (
                not levels
                or len(levels) > 8
                or any(not 1 <= c <= 64 for c in levels)
                or not 1 <= args.repeats <= 1000
                or not 0 <= args.warmups <= 100
            ):
                raise ValueError("Invalid benchmark bounds")
            agent = MediaSearchAgent(settings, snapshot, embedding, reranker)
            report["cold"] = {
                "definition": "first_request_new_adapters_existing_disk_and_OS_caches",
                **query_once(agent, queries[0]),
            }
            report["warmup"] = [
                query_once(agent, queries[i % len(queries)]) for i in range(args.warmups)
            ]
            runs = [run_queries(agent, queries, level, args.repeats) for level in levels]
            report["warm"] = runs
            after = index_info(snapshot)
            if before != after:
                raise ValueError("Index provenance/count changed during benchmark")
            report["index_before_after_equal"] = True
            report["index"] = after
            report["quality"] = quality(
                runs[0]["samples"],
                load_json(args.qrels) if args.qrels else None,
                report["input_sha256"],
                before["uuid"],
                settings.result_top_n,
            )
            failed = not report["cold"]["ok"] or any(run["failure_rate"] for run in runs)
        elapsed = perf_counter() - start
        report["resources"] = resources(cpu.ru_utime + cpu.ru_stime, elapsed)
        report["index"] = report.get("index") or index_info(store)
        report["resolved_embedding_revision"] = embedding.resolved_revision
        report["resolved_reranker_revision"] = reranker.resolved_revision
        report["actual_models_loaded"] = {
            "embedding": embedding.model is not None,
            "reranker": reranker.model is not None,
        }
        report["cluster_resource_metrics"] = (
            "not_collected; capture authorized monitoring separately"
        )
        atomic_json(args.output, report)
        print(
            json.dumps({"status": "failed" if failed else "completed", "output": str(args.output)})
        )
        return 1 if failed else 0
    except Exception as exc:  # noqa: BLE001 — sanitize credentials and metadata
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 2
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
