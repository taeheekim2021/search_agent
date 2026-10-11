# Finite Cloud Run validation package (not deployed)

One Cloud Run **Job**, not a service: one task, parallelism one, retries zero, 8 CPU,
32 GiB, 1,800-second task timeout. No interactive UI, public endpoint, VM, scheduler,
always-on cluster, VPC connector or new IAM binding. The existing service is untouched.
This is a small functional/latency smoke test, not production capacity or relevance validation.

## Proposed execution and evidence

- Use at most 20 reviewed public metadata-only records and exactly three queries.
- Required models remain Qwen/Qwen3-Embedding-4B (2560 dimensions) and
  BAAI/bge-reranker-v2-m3, pinned to verified commits already in the existing image.
- CPU bfloat16, batch one, sequence length 128, candidate top-k five, result top-n three.
- Start OpenSearch in at most two minutes; index in at most 12 minutes; search in at most
  ten minutes. The supervisor stops work after 25 minutes, leaving five minutes inside
  the platform timeout for cleanup/evidence. Every child uses its own process group;
  terminate, wait, then kill/reap on success, failure or interruption.
- Search concurrency one, repeats two, warmup one: one first request + one warmup + six
  measured requests = eight calls. Require successful real-inference flags on all eight,
  actual loaded flags, model revisions and code identity. Remove sample percentiles from
  exported reports; individual timings are retained. Quality remains unevaluated.
- Emit a baseline containing the exact reviewed metadata and queries before engine startup,
  then index, search and final evidence. Each artifact has bounded base64 JSON chunks and a
  completion marker with SHA-256. At most four artifacts of 1 MB each; no environment dump,
  credentials, upstream exception bodies, engine logs or model download output. The base64
  envelope is encoding, **not encryption**: only approved public metadata belongs here.
- Export is stdout to the existing Cloud Logging path, not a new bucket. Logging retention,
  exclusions and reader permission must be checked before execution. A printed completion
  marker is not proof of durable delivery: after the task finishes, read logs back and verify
  chunks/checksums before treating evidence preservation as successful. OOM/SIGKILL can
  prevent final export; earlier complete artifacts can survive and are not a success claim.
- `/validation-work` is a 2 GiB memory-backed volume holding index, logs, temporary files,
  embedding cache and reports. It consumes part of the 32 GiB allocation. Do not copy model
  weights into it. Index data is intentionally ephemeral; export metadata/results, not the
  binary index. Volume-full or OOM fails the trial; do not silently enlarge/retry it.

## Image feasibility and unresolved inputs

`Dockerfile` layers the OpenSearch 2.19.3 distribution and this repository's source over the
existing baked-model image. No pip installs or weight downloads are performed by this
recipe. Both source images must use immutable digests. Mandatory build arguments are
`MODEL_IMAGE`, `OPENSEARCH_IMAGE`, and `VALIDATION_INPUT_DIR` (a reviewed directory inside
the build context containing `manifest.json` and `queries.json`). The model image's exact
digest/cache layout/dependencies/architecture are **not available in this workspace**.
Do not build until inspecting them through authorized read-only registry/service tools.

Verify Linux amd64, Python >=3.12, installed application/model dependencies, compatible
libc for the copied JDK/native k-NN libraries, and readable HF snapshots at the pinned
revisions for UID 10001. The supervisor refuses root, dotenv, missing snapshot configs or
safetensors, mutable revisions, task retries and multiple tasks. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` are forced for model children; no remote download fallback.
The complete snapshots must include tokenizer files and shard links; checking config and
weights alone is preliminary, and real loading remains the definitive test.

## Security exception requiring explicit approval before execution

The proposed test engine sets `plugins.security.disabled: true` **only** with HTTP and
transport bound to `127.0.0.1`. The optional performance-analyzer plugin is omitted so it
cannot expose another listener. The runner requires `VALIDATION_LOOPBACK_AUTH_APPROVED=1`
in addition to execution opt-in. The user explicitly approved this narrow exception on
2026-10-11; the template records that approval. Paid execution remains gated separately.
No public listener or service deployment is provided. This must never become a production
configuration. No syscall-filter disabling, privileged mode, host sysctl, ulimit raise,
IAM mutation or network permission expansion is part of this package.

[Cloud Run's runtime contract](https://docs.cloud.google.com/run/docs/container-contract)
documents the 25,000 hard file-descriptor limit, default `vm.max_map_count=65535`,
memory-backed writable filesystem and job termination behavior. Production OpenSearch
recommends higher limits in its [installation guidance](https://docs.opensearch.org/2.19/install-and-configure/install-opensearch/index/).
Here `node.store.allow_mmap=false` follows the existing tested development fixture and
avoids mmap storage. Memory locking stays false. In the
[2.19.3 bootstrap source](https://github.com/opensearch-project/OpenSearch/blob/2.19.3/server/src/main/java/org/opensearch/bootstrap/BootstrapChecks.java),
the mmap check is conditional on mmap storage; ordinary descriptor checks warn for a
loopback single-node development process. Always-enforced checks remain active.
This is a feasibility inference, **not proof the composite image boots on Cloud Run**.
If platform restrictions prevent boot, stop and report; never weaken host security to force it.

## Cost and deployment gate

No build, registry push, job creation or execution has occurred. The template deliberately
contains unresolved placeholders. Before paid work, record actual billing currency, Seoul
region rates, month-to-date gross usage and expected remaining costs of the existing service,
build, image storage/retention, logs, network, taxes and this single execution. Include credit
consumption as cost: do not subtract credits/free allowances to claim the user's cap is met.

The nominal task envelope is 14,400 vCPU-seconds + 57,600 GiB-seconds. Multiply by the
account's applicable [Cloud Run Job rates](https://cloud.google.com/run/pricing), then include
startup/billing overhead and other services. A 30-minute task timeout is not a monetary cap.
Do not guess USD/KRW conversion or treat a budget alert as enforcement. Proceed only if the
verified entire monthly total remains <=10,000 KRW with explicit contingency. Build timeout
and build machine costs require their own bound; no automatic build or task retries.
Stop if existing monthly spend or a rate is unavailable. Prefer reuse of the existing image
layers and run exactly once; another execution needs a fresh remaining-budget check.

Deployment review must verify taskCount=1, parallelism=1, maxRetries=0, timeoutSeconds=1800,
8 CPU/32 GiB, image digest, 2 GiB work volume, existing least-privilege task identity and
no triggers. See [task timeout](https://docs.cloud.google.com/run/docs/configuring/task-timeout)
and [retries](https://docs.cloud.google.com/run/docs/configuring/max-retries). Operator execution
overrides can change these platform controls: do not use overrides or concurrent executions.

## Retrieve before declaring completion

After the approved one-time execution, export only that job execution's structured log
records using authorized read-only Cloud Logging access to a local JSON array. For example,
use `gcloud logging read` with resource.type `cloud_run_job`, exact project/region/job and
`labels."run.googleapis.com/execution_name"` equal to the observed execution name, plus
`jsonPayload.validation_evidence:*`, and `--format=json`. Do not export arbitrary service logs.
The artifact format is independent of entry order and tolerates identical duplicate chunks.

```bash
python -m search_agent.validation_evidence execution-logs.json ./retained-evidence
```

The destination must be new. Inspect the reconstructed `final.json`: completed status,
expected input/model/code hashes, real loaded/inference flags and eight calls are required.
Missing final evidence or a failed task remains a failed/incomplete test even if earlier
artifacts were recovered. Verify the execution is terminal and no task remains running.
The dormant Job definition is not an always-on server; preserve its execution history.
Image/log retention still has cost. Do not delete shared images, data or existing services.

## Local verification scope

Unit tests mock Cloud Run orchestration, model subprocesses and engine readiness. They cover
opt-in/approval/task/revision/input guards, offline environment, timeout process-group cleanup,
partial failure evidence, chunk limits/checksums and reconstruction. They do **not** validate
real model loading, the composite image, Cloud Logging durability, constrained Cloud Run boot,
deployment permissions or costs. Existing engine integration evidence on main is separate.
