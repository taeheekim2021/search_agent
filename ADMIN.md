# 검색 관리자 서비스

기존 FastAPI 서비스의 `/admin`에서 운영 현황, 콘텐츠 조회·메타데이터 적재, 검색 테스트, 인덱스 관리를 제공합니다.
실행 중인 검색 에이전트의 카탈로그·모델·OpenSearch 연결을 그대로 사용합니다.
관리자 API는 `SEARCH_ADMIN_API_KEY`를 설정해야 활성화됩니다.

## 1. 빠른 실행: 샘플·모의 모드

Python 3.12 이상, 저장소 루트의 Bash 셸에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export SEARCH_BACKEND=sample
export SEARCH_MODE=mock
export SEARCH_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# 본인 터미널에서 생성된 키를 확인하고 관리자 화면에 입력합니다.
python -c 'import os; print(os.environ["SEARCH_ADMIN_API_KEY"])'
python -m uvicorn search_agent.app:app --host 127.0.0.1 --port 8000 --workers 1
```

[관리자 화면](http://127.0.0.1:8000/admin)을 열고 생성한 키로 **운영 콘솔 연결**을 누릅니다.
키는 최소 32자의 공백 없는 출력 가능한 ASCII 문자열이어야 합니다. 위 명령은 임의의 키를 생성합니다.
샘플 모드에서는 가상 카탈로그를 조회하고 키워드 검색을 실행할 수 있으며, Qwen/BGE는 실행하지 않습니다.
기존 사용자 화면 `/`와 `/api/search`는 계속 공개되어 있습니다. 관리자 키가 일반 검색까지 보호하지는 않습니다.

## 2. OpenSearch와 실제 모델 연결

실행 중인 권한 있는 OpenSearch 서버를 준비하고 같은 가상환경에 모델 의존성을 설치합니다.
접속 인증서·계정, CPU/GPU·메모리 및 모델 revision 설정은 [MEDIA.md](MEDIA.md)와 [README.md](README.md)를 참고하세요.

```bash
python -m pip install -e '.[models,dev]'
export SEARCH_BACKEND=opensearch
export SEARCH_MODE=real
export SEARCH_DIMENSIONS=2560
export OPENSEARCH_URL=https://your-authorized-cluster:9200
export OPENSEARCH_INDEX=kids-media-v1
export MEDIA_ROOT=.media
# 필요한 OPENSEARCH_USERNAME/PASSWORD/CA_CERTS와 기존 관리자 키를 환경에 설정합니다.
python -m uvicorn search_agent.app:app --host 127.0.0.1 --port 8000 --workers 1
```

OpenSearch는 2.19 이상, 미디어 벡터는 2560차원을 사용합니다. 관리자 화면에서 클러스터나 계정을 생성하지 않습니다.
인덱스 준비·메타데이터 적재·검색은 최초 모델 로딩으로 오래 걸릴 수 있습니다.
모델 로딩·추론에 실패하면 오류 또는 항목별 실패를 표시하며 모의 모드로 자동 전환하지 않습니다.

## 3. 화면별 기능

| 기능 | 샘플 백엔드·모의 모드 | OpenSearch 백엔드·실제 모드 |
|---|---|---|
| 운영 현황 | 카탈로그 건수, 설정, 모델 비활성 상태 | 인덱스 존재·상태·건수·매핑 식별자와 모델 상태 |
| 콘텐츠 조회 | ID·제목·설명 등의 부분 문자열 조회 | 메타데이터 검색과 페이지 조회, 출처·라이선스 확인 |
| 검색 테스트 | 기존 키워드 검색; 실제 모델 점수 없음 | 기존 BM25 + Qwen 의미검색 + BGE 재정렬 |
| 메타데이터 내용 검증 | 가능; 외부 요청·색인 쓰기 없음 | 가능; 외부 요청·색인 쓰기 없음 |
| 메타데이터 적재 | 지원하지 않음 | 확인 후 실제 임베딩과 설정된 인덱스 upsert |
| 인덱스 준비·새로고침 | 지원하지 않음 | 확인 후 설정된 인덱스에만 실행 |

`sample`에서 `SEARCH_MODE=real`을 선택하면 샘플 검색에도 실제 모델을 사용합니다. 샘플 카탈로그가 실제 서비스 데이터로 바뀌는 것은 아닙니다.
콘텐츠 상세 조회에는 벡터·서버 로컬 미디어 경로·전체 자막·임의의 원본 메타데이터 객체를 반환하지 않습니다.

### 메타데이터 적재 순서

1. **콘텐츠 관리 → JSON 파일 불러오기**에서 [data/metadata_manifest.json](data/metadata_manifest.json) 같은 기존 `MediaManifest` 파일을 선택합니다.
2. **1. 내용 검증**으로 필수 필드·출처·라이선스·provenance와 적재 예정 항목을 확인합니다. 이 단계에서는 모델·OpenSearch를 호출하거나 파일을 쓰지 않습니다.
3. OpenSearch 환경에서 **2. 인덱스에 적재**를 누르고 대상 인덱스와 건수를 확인해 실행합니다. 같은 콘텐츠 ID는 전체 문서 upsert로 갱신합니다.
4. 응답의 전체 상태와 항목별 결과를 확인한 뒤 콘텐츠 목록을 다시 조회합니다. 메타데이터를 편집하면 다시 내용 검증을 진행합니다.

알려지지 않은 연령을 추정해 채우지 않습니다. 연령을 지정한 검색에서는 연령 미상 콘텐츠가 제외됩니다.
전체 manifest 형식, 식별자 생성 및 이전 문서 갱신 동작은 [MEDIA.md](MEDIA.md)에 설명되어 있습니다.

## 4. 관리자 API 계약

모든 `/api/admin/*` 요청에 `Authorization: Bearer <관리자 키>`를 사용합니다. URL 쿼리나 쿠키로는 인증하지 않습니다.
JSON 요청은 `Content-Type: application/json`을 지정합니다. 인증을 먼저 검사하고 실제 요청 본문 크기도 제한합니다.

| 메서드·경로 | 입력 | 동작 |
|---|---|---|
| `GET /api/admin/status` | 없음 | 상태·지원 기능·제한·기본 Top K/N 반환; 모델 로딩 없음 |
| `GET /api/admin/contents` | `q`, `offset=0`, `limit=20` | `items`, `total`, `has_more`, `window_limited` 반환 |
| `POST /api/admin/search` | `query`, 선택적 `top_k`, `top_n` | 사용자 검색과 같은 파이프라인·응답 계약 |
| `POST /api/admin/ingest` | `manifest`, `dry_run=true`, `confirmed=false` | 기본값은 내용 검증; 실행은 두 플래그를 명시적으로 변경 |
| `POST /api/admin/index/ensure` | `{"confirmed":true}` | 없으면 생성, 있으면 현재 임베딩과 호환성 검증 |
| `POST /api/admin/index/refresh` | `{"confirmed":true}` | 기존 인덱스 refresh; 문서·벡터 재생성 없음 |

`manifest`는 `{"version":1,"entries":[...]}` 구조이며 각 원소는 기존 `MediaEntry`입니다.
실제 적재에는 **`dry_run: false`와 `confirmed: true`가 모두 필요**합니다. 두 플래그는 문자열·숫자가 아닌 JSON 불리언이어야 합니다.
내용 검증은 승인 토큰을 발급하지 않습니다. API 직접 호출 시에도 서버가 manifest를 다시 검증하고 `confirmed`를 확인합니다.
인덱스 작업도 `confirmed: true`가 필요합니다. 요청 본문에서 인덱스 이름, 파일 경로, 다운로드 옵션을 지정할 수 없습니다.

```bash
curl -sS http://127.0.0.1:8000/api/admin/status \
  -H "Authorization: Bearer $SEARCH_ADMIN_API_KEY"
curl -sS http://127.0.0.1:8000/api/admin/search \
  -H "Authorization: Bearer $SEARCH_ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"query":"5살 아이가 볼 공룡 영상","top_k":20,"top_n":5}'
```

검색어는 1~500자, Top K/N은 각각 1~100이며 Top N은 Top K 이하여야 합니다. 생략하면 서버 설정을 적용합니다.
알 수 없는 요청 필드는 거부합니다. OpenSearch 콘텐츠 목록은 콘텐츠 ID로 정렬하고 부분 응답·시간 초과·ID 불일치를 오류로 처리합니다.

### 제한과 오류

| 항목 | 값·동작 |
|---|---|
| 요청 본문 | 최대 2,000,000바이트; JSON 포장과 UTF-8 문자 바이트 포함 |
| 한 번에 적재·검증할 항목 | 기본 100개; `SEARCH_ADMIN_MAX_BATCH_SIZE`로 1~1000 지정 |
| 콘텐츠 페이지 | `limit` 최대 100; `offset + limit` 최대 10,000 |
| 관리자 키 미설정·불일치 | 각각 404 `ADMIN_DISABLED`, 401 `ADMIN_UNAUTHORIZED` |
| 실행 확인 누락·백엔드 미지원·다른 작업 진행 중 | 409 `CONFIRMATION_REQUIRED` / `BACKEND_UNSUPPORTED` / `ADMIN_BUSY` |
| 본문 초과·잘못된 입력 | 413 `REQUEST_TOO_LARGE`; 입력·건수·조회 범위 오류는 422 |
| 모델·OpenSearch·파일 저장소 오류 | 503과 `MODEL_UNAVAILABLE` / `OPENSEARCH_UNAVAILABLE` / `STORAGE_UNAVAILABLE` |

오류는 `detail.code`, `detail.message`를 포함하며 모델 오류에는 `stage`가 추가됩니다.
운영 현황은 OpenSearch에 연결하지 못해도 HTTP 200과 `status: degraded`, `errors`를 반환할 수 있습니다. 확인하지 못한 건수를 0으로 표시하지 않습니다.

## 5. 실행 상태와 재시도

모델의 `loaded`는 현재 프로세스에 가중치가 로딩되었는지를 표시합니다. `inference_readiness: not_checked`는 별도 추론 점검을 하지 않았다는 뜻입니다.
상태 조회는 모델의 `.identity`를 읽지 않으며 실제 추론을 실행하지 않습니다. 주입된 테스트 어댑터는 `adapter_unverified`로 표시합니다.
인덱스 `available`은 존재 확인이며 현재 모델과의 호환성 보장이 아닙니다. **인덱스 준비는 Qwen을 로딩할 수 있지만 임베딩·리랭킹 추론은 실행하지 않습니다.**
모델·인덱스 표시와 검색 응답의 `model_inference_performed`를 함께 확인하세요. 이들 표시는 검색 품질 평가를 대신하지 않습니다.

적재는 요청 안에서 순차 실행하는 동기 작업입니다. 서버를 `--workers 1`로 실행하면 기존 검색과 관리자 검색·쓰기 작업이 같은 모델 잠금을 공유합니다.
다른 검색·쓰기 작업이 진행 중이면 새 관리자 검색·쓰기 요청은 기다리지 않고 409 `ADMIN_BUSY`를 반환합니다. 상태·목록 조회와 내용 검증은 별도로 사용할 수 있습니다.
기존 CLI와 관리자 적재는 같은 `MEDIA_ROOT` 파일 잠금도 사용합니다. 별도 작업 프로세스의 진행률이나 작업 이력을 제공하는 서비스는 아닙니다.

실행 응답의 `status`는 `completed` 또는 `partial_failure`이며 항목별 `status`, `stage`, `error_type`을 확인해야 합니다. HTTP 200만으로 전체 성공을 판단하지 마세요.
`MEDIA_ROOT`의 `<content_id>.metadata-receipt.json`에 처리 기록을 남기고 임베딩 캐시를 사용합니다. 일부 실패 시 이미 반영한 문서를 롤백하지 않습니다.
업서트 뒤 처리 기록 저장에 실패할 수도 있습니다. 항목별 기록과 현재 목록을 확인한 후 같은 manifest를 재실행하면 동일 ID에 다시 upsert합니다.
브라우저 연결 해제·요청 시간 초과는 서버 작업 취소가 아닙니다. 결과를 받지 못했다면 진행 중 여부·처리 기록을 확인한 뒤 재시도하세요.

## 6. 접속과 제공 범위

브라우저의 관리자 키는 현재 페이지의 메모리에만 보관하며 localStorage·sessionStorage·쿠키에 저장하지 않습니다. 새로고침·연결 해제 후에는 다시 입력합니다.
키를 교체하려면 `SEARCH_ADMIN_API_KEY`를 변경하고 서버를 재시작합니다. 키를 저장소나 공유 문서에 넣지 마세요.
원격 접속은 HTTPS 역방향 프록시를 통해 제공하고 키를 암호화된 연결로 전달하세요. 이 구현은 배포나 프록시 구성을 자동으로 수행하지 않습니다.

관리자 적재는 메타데이터 전용이며 영상 원본 다운로드를 제공하지 않습니다. 모델 다운로드와 OpenSearch 접속은 발생할 수 있습니다.
콘텐츠·인덱스 삭제, 임의 인덱스 선택·alias 교체, 내구성 있는 작업 큐, 사용자 계정·역할별 권한(RBAC)은 포함하지 않습니다.
기존 다운로드 CLI 기능과 상세 데이터 정책은 [MEDIA.md](MEDIA.md)를 따릅니다.
