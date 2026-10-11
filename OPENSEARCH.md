# OpenSearch indexing and release runbook

This increment extends the existing media backend. It ingests metadata only; it does
not fetch videos. Qwen/Qwen3-Embedding-4B produces 2560-dimensional normalized vectors;
BAAI/bge-reranker-v2-m3 reranks the fused results. No server, credentials, IAM changes,
paid infrastructure or Cloud Run deployment are included.

## Local development

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
docker compose -f compose.opensearch.yml up -d --wait
export OPENSEARCH_URL=http://127.0.0.1:9200
export OPENSEARCH_ALLOW_HTTP_LOCAL=true
export OPENSEARCH_TEST=1
pytest -q tests/test_opensearch_integration.py
```

The pinned 2.19.3 container is development/CI only: security is disabled, host binding
is loopback, and data lives in a named volume. Reserve at least 2 GB for OpenSearch.
`docker compose -f compose.opensearch.yml stop` preserves data. Do not expose its port
externally or use this Compose security configuration in production.
Integration tests remove only their own uniquely named synthetic test indices.
Tests use fabricated metadata, vectors and reranker scores: they do **not** establish
real Qwen/BGE quality, GPU compatibility or performance. Existing `scripts/smoke_models.py`
is the separate real-model sample smoke. Before rollout, a separately approved real-model
metadata-to-OpenSearch evaluation with labeled Korean queries and latency measurements is required.

## Production prerequisites and configuration

Provisioning remains an operator task after approval. Required resources:

- Private, supported OpenSearch cluster compatible with Faiss cosine/HNSW (2.19+;
  2.19.3 is the tested baseline, not a promise of current security support). At least
  two data nodes for one replica, capacity planning, monitoring, snapshots and a tested restore.
- Verified HTTPS endpoint/CA, network reachability from the application and indexing job,
  and separately supplied least-privilege Basic-auth credentials. This client does not
  implement AWS SigV4, workload identity or client-certificate authentication.
- A read account for search/count/aggregations/mappings/settings on the read alias and
  retained concrete versions (search pins the concrete index). Admin status additionally
  needs limited index-health access. A separate indexing account needs create/mapping,
  settings, bulk and refresh rights on the staging prefix. A serialized release operator
  needs alias and write-block permissions. No data deletion privilege is needed by the app.
- Secret injection through your existing secret manager. Never commit credentials or
  use the dev/CI test key. Admin APIs retain bearer-key authentication, constant-time
  comparison and disabled-by-default behavior when no key is configured. Set an existing
  `SEARCH_ADMIN_API_KEY` with at least 32 printable non-space ASCII characters only when
  admin access is wanted. Serve admin over HTTPS behind restricted ingress; rotate the
  key and audit operator actions. Browser storage behavior is unchanged.
- Qwen and BGE runtime resources and approved model downloads/cache. Pin both model
  revisions to verified 40-character Hugging Face commit hashes. GPU/memory sizing,
  cold starts, concurrency, deadlines and Cloud Run connectivity require separate testing.
- Durable writable `MEDIA_ROOT` for checkpoints and embedding cache. Cloud Run ephemeral
  storage is not a durable job system. Run ingestion in one controlled job/process with
  persistent storage; the local file lock does not serialize workers on different hosts.

Set these values using your deployment configuration; angle-bracket values are placeholders,
not usable endpoints, secrets or generated model revisions:

```dotenv
SEARCH_ENVIRONMENT=production
SEARCH_MODE=real
SEARCH_BACKEND=opensearch
SEARCH_DIMENSIONS=2560
SEARCH_EMBEDDING__REVISION=<verified-Qwen-commit>
SEARCH_RERANKER__REVISION=<verified-BGE-commit>
OPENSEARCH_PRODUCTION=true
OPENSEARCH_URL=https://<private-cluster>:9200
OPENSEARCH_USERNAME=<injected-account>
OPENSEARCH_PASSWORD=<injected-secret>
OPENSEARCH_CA_CERTS=<trusted-CA-file-if-not-system-trusted>
OPENSEARCH_ALLOW_HTTP_LOCAL=false
OPENSEARCH_SHARDS=1
OPENSEARCH_REPLICAS=1
OPENSEARCH_INDEX=kids-media-read
MEDIA_ROOT=<durable-job-directory>
```

Production settings reject mock mode, mutable model revisions, unauthenticated OpenSearch,
HTTP and zero configured replicas. TLS verification cannot be disabled. The sample backend
remains supported in real production mode; model/server errors never select mock or sample.
Both web and CLI force OpenSearch production validation when `SEARCH_ENVIRONMENT=production`,
even if `OPENSEARCH_PRODUCTION=false` was supplied. Conversely, `OPENSEARCH_PRODUCTION=true`
requires `SEARCH_ENVIRONMENT=production`, so default development model settings cannot
silently weaken the production contract. Configuration validation errors omit input values.
`/health` remains process liveness, not proof of cluster/model readiness; use authenticated
admin status and a representative real search for readiness verification.

## Build, resume, validate, publish

1. Use a fresh concrete index name such as `kids-media-v1-20261011a` for the indexing
   process. Keep the search process configured to `kids-media-read`. `media-v1` is the
   strict mapping version; model identity or incompatible schema changes require a new
   index and complete re-embedding. Existing `kids-media-v1` indices remain readable;
   publication requires `<prefix>-v1-<build>`. Do not rename/delete an existing index.
2. Validate the reviewed metadata manifest without network/model access:

   ```sh
   python -m search_agent.media.cli ingest data/metadata_manifest.json
   ```

3. In the indexing job's environment, set `OPENSEARCH_INDEX` to the new concrete version,
   then execute (requires real model packages and the cluster):

   ```sh
   python -m search_agent.media.cli ingest data/metadata_manifest.json --execute --resume
   ```

   Bulk replacement writes use stable content IDs, bounded item/byte limits and
   `refresh=wait_for`, waiting for all shard copies. Defaults: 32 items, 5 MB, three retries
   with exponential jitter. Tune `OPENSEARCH_BULK_SIZE`, `OPENSEARCH_BULK_MAX_BYTES`,
   `OPENSEARCH_BULK_RETRIES`, `OPENSEARCH_RETRY_BASE_SECONDS` and timeout to your workload.
   Oversize batches fail explicitly; reduce batch size. HTTP/item 429/502/503/504 and
   ambiguous transport/malformed-response failures are retried; permanent failures are
   not. Reports contain content ID, status, HTTP code and attempts, never upstream error
   reasons. Exit 1 means per-item failure; exit 2 means setup/command failure.

   Checkpoints bind successful content fingerprints to schema, full embedding identity,
   endpoint, concrete index name and index UUID. `--resume` skips only matching successes;
   changed metadata/model/recreated index triggers new writes. Before any new write, a durable
   pending checkpoint invalidates the previous success, including when a process crashes
   after the remote commit but before recording its acknowledgement. Without `--resume`, all
   entries are replaced. Lost receipts may replay a write safely. Failed checkpoints are
   retried. Do not use resume after external edits/deletions to that same index: checkpoints
   attest acknowledged ingestion, not current remote content. Reconcile with a full replay.
   One job owns each staging index; concurrent writers are unsupported. Full entry metadata
   stays in the index; checkpoints contain hashes and operational provenance, not videos.
4. Require a completed report with the expected unique content count, zero failed entries,
   exact mapping identity, and approved retrieval-quality checks. Count alone does not
   prove catalog completeness or semantic quality. The runtime checks finite, nonzero,
   2560-dimensional vectors and exact index embedding provenance.
5. Review publication with the actual old concrete name (use `NONE` only for the first
   publication). This is a no-network plan until `--execute` is supplied:

   ```sh
   python -m search_agent.media.cli publish kids-media-read \
     --expected-current NONE --expected-count 8
   # Same command plus --execute applies the reviewed change.
   ```

   The publisher checks a single expected alias target, schema/model identity and green
   health, blocks writes on the new version and requires complete shard/target acknowledgement
   after in-flight writes, refreshes and
   verifies exact count, then atomically removes/adds the alias and verifies the result.
   It never deletes an index. Existing filtered/routed aliases are rejected before any mutation;
   publication never strips their access restrictions. Retain the previous version and
   matching model configuration.
   Serialize all publishers externally: OpenSearch alias APIs are atomic but are not a
   general compare-and-swap, particularly when the alias is initially absent. The expected
   target check and `must_exist` removal guard common stale updates, not arbitrary external
   alias writers. Initial publication needs the same single-operator discipline.
6. Run authenticated admin status/browse and representative public/admin searches through
   the alias. Each search resolves one concrete version before keyword/vector retrieval;
   identical hard filters apply to both, equal-weight reciprocal rank fusion selects
   candidates, then BGE scores determine final order. No branch silently degrades to mock.
   Vocabulary above 1,000 terms is explicitly rejected pending a scalable vocabulary adapter.

Published versions are immutable and the read alias is explicitly non-writable. Existing
admin preview/browse/search/status/refresh UI remains available. Admin ingestion/ensure
continues to work when pointed at a writable staging index; it intentionally cannot mutate
a published read alias. Use a separate operator process/job for production indexing instead
of granting the public web process write privileges. Publication is CLI-only in this increment.

## Rollback and failure handling

Configure the release job's `OPENSEARCH_INDEX` to the retained previous version and use
`publish` with `--expected-current` set to the currently active version and the previous
version's expected count. Run the plan, then the same command with `--execute`. The same
checks apply; both versions remain intact. If the embedding identity changed, coordinate
application model configuration/revision rollback as well; mismatches fail closed.

Block ownership is a release invariant: only this workflow may install/remove write blocks,
using the Blocks API. Do not set `index.blocks.write` directly through `_settings`, and do not
allow concurrent external metadata/block mutations. OpenSearch 2.19 returns an exact
`acknowledged=true, shards_acknowledged=false, indices=[]` no-op for an already finalized
block; rollback accepts only that shape plus a fresh concrete-index `write=true` setting.
Every other incomplete acknowledgement fails closed. Temporary unfinished blocks are
reverified by OpenSearch rather than producing this no-op. Direct external settings changes
can bypass that drain guarantee and are unsupported.

A failure after applying the write block leaves the candidate blocked and the old alias
untouched unless an alias request was already sent. Inspect the alias after timeout/lost
acknowledgement before retrying: the switch may have succeeded. Do not blindly undo it.
For a failed count/validation gate, investigate the report and build a fresh version;
manual unblocking of an unpublished staging index requires an operator's deliberate action.
Checkpoints are a resumable synchronous job aid, not a queue or cross-host lock.

Do not delete old indices as part of publication/rollback. Retention, backups, credential
rotation and eventual cleanup need their own reviewed operator procedures. No live Cloud Run
traffic promotion or verification is performed by this work; the reported sample promotion
remains separate from this undeployed OpenSearch path.

API references: [Bulk](https://docs.opensearch.org/2.19/api-reference/document-apis/bulk/),
[atomic alias operations](https://docs.opensearch.org/2.19/api-reference/index-apis/alias/),
[Blocks API](https://docs.opensearch.org/2.19/api-reference/index-apis/blocks/), and
[2.19 block finalization implementation](https://github.com/opensearch-project/OpenSearch/blob/2.19/server/src/main/java/org/opensearch/cluster/metadata/MetadataIndexStateService.java).

## Sizing assumptions and current limits

Catalog size and update frequency are not yet known. The initial configuration uses one
primary shard, configurable replicas and bounded 32-record bulk requests; these are starting
values, not a production capacity estimate. A raw float32 vector alone is 2560 × 4 = 10,240
bytes per document before HNSW, metadata, `_source`, replication and engine overhead. Measure
actual index size and p95/p99 query latency with representative metadata and concurrent load
before selecting node memory/disk, shards or replica count. The development container's
512 MB JVM heap and 2 GB container limit are only for small synthetic fixtures.

The existing manifest limit is 2 MB and the admin batch limit defaults to 100 records
(`SEARCH_ADMIN_MAX_BATCH_SIZE`). Split larger catalogs into reviewed manifests, use the
same new staging index and durable checkpoint root, and publish only after validating the
combined expected unique-ID count. No document is removed merely because it is absent from
an incremental batch; build a fresh version for a full catalog replacement. The local job
lock and synchronous admin execution are intentional limits of this increment, not a
distributed ingestion architecture. Filtered/routed aliases are rejected because resolving
them to a concrete index would otherwise bypass their filter/routing semantics.
