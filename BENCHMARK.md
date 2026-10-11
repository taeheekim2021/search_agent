# Bounded performance validation and deployment preparation

## Current status and budget gate

The user has set a **10,000 KRW/month total budget**, including consumption covered by
credits. No production corpus or OpenSearch endpoint/credentials exists yet. No cloud
resources, credentials, IAM, networking, traffic or deployments have been changed.
The provided Dockerfile/Compose files are review templates, not deployed infrastructure.
Actual Qwen+BGE+OpenSearch benchmark results do not exist yet. Synthetic tests validate
correctness and measurement arithmetic, not operational performance or ranking quality.

The preferred immediate option is an already-owned workstation or approved existing
machine: incremental cloud compute/storage cost is zero, although its performance is not
representative of Cloud Run. An optional future single-machine cloud validation session
is capped in the **plan** at one compute hour total: 15 minutes setup, 15 minutes indexing,
15 minutes benchmark and 15 minutes result export/stop. This is an allocation to verify,
not a claim that these models will complete within it. Stop on timeout rather than extend
spending automatically. Proposed sizing for evaluation is 4 vCPU / 32 GiB host memory
(app container limit 24 GiB plus local OpenSearch); it is unverified and not production HA.

Before any billable creation, fill and approve the entire cost worksheet using current
region/account-currency SKUs. Rates are deliberately **not guessed**:

| Component | Proposed quantity | Cost to verify in KRW |
| --- | --- | --- |
| Existing search service | Current month's actual + remaining forecast | Unknown; subtract first |
| On-demand validation VM | At most 1 hour; chosen region/machine required | Hourly rate × 1 |
| Persistent disk/model cache | Actual GB × retained fraction of month (up to 730 hours) | GB-month rate × GB × fraction |
| Images/artifact registry/snapshots | Actual retained GB-month | Rate × retention |
| Network/logging/builds/requests | Measured or conservative allowance | Must include |
| Tax/currency/contingency | Account billing terms | Must include |

The sum, **before credits**, must fit 10,000 KRW including existing service costs. Until
rates, retention and existing spend are verified, the new-cloud option is blocked.
Stopping containers does not stop VM billing; stopping the VM does not remove disk/image
charges. Preserve/export reports, stop the VM, verify its stopped state and inspect billing.
Removing disks/images is a separate approved cleanup action, never an automatic task here.
Do not assume free-tier eligibility or unused credits. Budget alerts and max-instance
settings are not hard spending caps. No always-on OpenSearch cluster is proposed at this cap.

Official references checked 2026-10-11:
[Cloud Run pricing](https://cloud.google.com/run/pricing),
[Compute pricing](https://cloud.google.com/products/compute/pricing),
[disk/storage pricing](https://cloud.google.com/compute/disks-image-pricing),
[stopped VM behavior](https://cloud.google.com/compute/docs/instances/stop-start-instance),
[budgets](https://cloud.google.com/billing/docs/how-to/budgets), and
[maximum instances](https://cloud.google.com/run/docs/configuring/max-instances).
No numerical region-specific total has been verified; no provisioning is authorized by this file.

## Public metadata starter

`scripts/collect_wikidata.py` issues one bounded public SPARQL request (100 records by
default, maximum 500). It selects animated-film entities (P31/Q202866), Korean labels with
English/QID fallback, source IDs/URLs, and retrieval timestamp. IDs deduplicate records;
the adjacent provenance JSON records the exact query, response/manifest SHA-256, licence
and retrieval time. A rerun can change with Wikidata: retain the manifest and provenance
for a reproducible dataset revision. No videos, images, linked pages or film synopses are
fetched. Descriptions are generated factual templates, not copyrighted summaries.

[Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing) releases structured
main-namespace data under CC0. That does **not** license the films or images. Records set
`rights_scope=metadata`; UI evidence identifies metadata-only licensing and acquisition
paths reject these records. `original_url` links to the entity page, not a playable video.
Age ratings and media language remain unknown. These records are not a representative
future production catalog or a child-suitability recommendation.

```sh
python scripts/collect_wikidata.py data/wikidata_starter.json --limit 100
python scripts/collect_wikidata.py data/wikidata_starter.json --limit 100 --execute
python -m search_agent.media.cli ingest data/wikidata_starter.json
```

Use a descriptive User-Agent, a bounded query and no parallel retry storm, per
[Wikidata access guidance](https://www.wikidata.org/wiki/Wikidata:Data_access) and
[Query Service limits](https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual).
A 429/timeout/error stops collection; respect Retry-After before a manual retry. Collection
was attempted in this workspace but failed with `ProxyError`; no Wikidata corpus is bundled
or claimed collected. The downloader and synthetic converter tests are included instead.

## Reproducible real-model benchmark

Use an approved machine/endpoint with enough memory and install `pip install -e '.[models]'`.
Keep the required Qwen/Qwen3-Embedding-4B (2560 dimensions) and BAAI/bge-reranker-v2-m3.
Pin both `SEARCH_*__REVISION` values to reviewed 40-character model commits. Set
`SEARCH_MODE=real`, `SEARCH_BACKEND=opensearch`, and `SEARCH_CODE_REVISION` to the tested git
commit. Use the TLS/auth/replica settings in OPENSEARCH.md for production, or its explicit
loopback-only exception for local development. Model/embedding cache state must be recorded.
No benchmark command is run automatically by tests or CI.

1. Collect and review the public starter metadata. Use a new versioned concrete index.
2. Run indexing in its own process, then inspect failures and counts:

   ```sh
   python -m search_agent.benchmark index data/wikidata_starter.json \
     --run-real --output artifacts/index-real.json
   ```

   This performs metadata-only real Qwen indexing. Throughput includes initial model load,
   cache lookup and bulk acknowledgement. Existing persistent vector/model caches are not
   purged; use a new approved cache directory to measure an uncached run. No resume skipping
   is used. BGE is not needed for indexing and is explicitly reported as not loaded.
3. Run a separate fresh process for search, against a retained immutable version:

   ```sh
   python -m search_agent.benchmark search data/benchmark_queries.example.json \
     --corpus-manifest data/wikidata_starter.json --run-real \
     --concurrency 1,2,4 --repeats 10 --warmups 3 --output artifacts/search-real.json
   ```

   Replace example queries with a reviewed representative workload before making capacity
   decisions. The report records query/corpus SHA-256, declared corpus count, index UUID,
   schema/embedding identity, server/model/code versions, resolved model commits, dimensions
   (in mapping provenance), configured TopK/TopN, cold first-request latency, warm p50/p95/p99,
   successful throughput, failure rate, inference counts and per-request IDs/latencies.
   The first request is process-cold for models, not machine-cold: disk/OS/engine caches may
   already be warm. Warmups and cold results are separated from warm percentile samples.
   Failed requests are excluded from success latency percentiles and included in failure
   rate. All-zero-hit runs do not establish inference performance; inspect actual-model-load
   and inference-count fields. No silent mock path exists in the benchmark CLI.

The harness runs the existing service path **in-process**, with closed-loop thread
concurrency. It does not measure HTTP ingress, authentication proxy, Cloud Run cold starts,
multiple replicas or network load-generator overhead. The existing agent lock serializes
model work within a process; concurrency measures queuing/throughput under that constraint.
For real deployment acceptance, run a separate authenticated external load test and report
those transport/deployment metrics separately rather than relabeling this harness.

Resource metrics cover the benchmark/model process: CPU seconds and percent of one core,
and process-lifetime peak RSS (not incremental per-phase RAM). Capture authorized OpenSearch
node CPU/JVM/OS memory metrics separately with `GET /_nodes/stats/process,jvm,os`, or existing
monitoring; the report marks cluster resource metrics as not collected. This needs an
existing account with node-monitor permissions; do not grant permissions automatically.
GPU utilization/memory, provisioned CPU/RAM, host type, cache state, duration and cost must
be attached to a production acceptance record. Small samples cannot support reliable tail
latency claims; increase repetitions only within an approved resource budget.

Ranking quality is **not evaluated** without reviewed relevance judgments. Optional
`--qrels judgments.json` requires `reviewed: true`, a `source` description, exact
`queries_sha256`, exact `index_uuid`, and `judgments: {query_id: {content_id: grade}}` with
integer grades 0–3 and at least one relevant record per evaluated query. Output is nDCG@N
and recall of judged-relevant documents; unjudged documents count as nonrelevant and can
bias incomplete judgments. Synthetic judgments in tests are not quality evidence.

## Deployment templates and remaining approvals

`Dockerfile` packages the existing app/models without baked credentials. It has **not**
been built with real model dependencies in this increment; pin a tested base digest and
freeze Python/model dependencies before deployment. `deploy/compose.validation.yml` is a
Linux local/on-demand template with separate benchmark/server profiles, one worker,
explicit model commits, restart disabled, bounded CPU/RAM and loopback web binding. It
uses the development cluster in `compose.opensearch.yml`, not production security/HA.
It persists cache volumes; stopping Compose retains disk usage and costs on a paid host.
Do not start it on a paid machine until the worksheet is approved. Commands are not executed
by repository CI. No Cloud Run revision or traffic is changed by these templates.

Before a production server deployment: provide an approved total-cost plan, chosen host/
cluster, endpoint/CA, secret references, existing runtime identity and network path, tested
image digest, actual-model benchmark and acceptance thresholds. Any new IAM/network change
needs its own authorization. Use a new validation service with no production traffic first;
keep the existing `search-agent` service and sample revision unchanged. Deployment and
real-model/cluster performance validation remain blocked by these missing prerequisites,
not by the draft PR.
