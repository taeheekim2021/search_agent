# 실제 미디어 인제스트와 OpenSearch

원본 영상/음성은 `MEDIA_ROOT`에 보관하고, OpenSearch에는 검색할 **텍스트 메타데이터 + Qwen 2560차원 벡터**를 보관합니다. Qwen3-Embedding-4B는 텍스트 모델이므로 영상 프레임이나 음성 바이트를 직접 처리하지 않습니다. 자동 ASR/자막 생성/영상 분석은 구현하지 않았습니다. 제공되고 재사용 가능한 자막만 명시적 출처·라이선스와 함께 넣을 수 있습니다.

## 실행

기존 Python 가상환경과 패키지를 설치한 뒤 `.env.example`을 참고합니다. 인제스트 CLI는 Linux/POSIX 파일 잠금을 사용합니다.

```bash
python -m pip install -e '.[dev]'
# 제공된 출처 기록 → 검증된 인제스트 스키마 변환 (결과 파일도 포함되어 있음)
python -m search_agent.media.cli adapt-commons data/source_manifest.json data/media_manifest.json
# 아래 환경 변수는 허용된 다운로드 대상만 명시하며 시스템 네트워크 권한을 변경하지 않음
export MEDIA_ALLOWED_DOMAINS='["upload.wikimedia.org"]'
# 기본은 dry-run. 파일 다운로드/모델/OpenSearch 호출 및 파일 쓰기 없음
python -m search_agent.media.cli ingest data/media_manifest.json
# 먼저 1.7MB 영상 하나만 다운로드. 실제 Qwen/OpenSearch 호출 없음
python -m search_agent.media.cli ingest data/media_manifest.small.json --download-only
# 실제 Qwen과 OpenSearch로 다운로드 및 색인
python -m pip install -e '.[models]'
python -m search_agent.media.cli ingest data/media_manifest.small.json --execute
```

`--execute`는 mock 모드를 사용하지 않고 지정된 실제 Qwen 어댑터를 실행합니다. BGE는 색인이 아닌 검색 시에 호출됩니다. `--limit N`으로 manifest 앞 N개만 처리할 수 있습니다. 전체 자료 약 116.5MiB, 파일별 40MB/배치 160MB 제한을 권장합니다. 기본 `MediaSettings` 파일 한도는 50MiB이며 예제 환경에서는 40,000,000 bytes로 제한합니다.

다운로드에 실패하면 같은 명령을 다시 실행합니다. 이미 다운로드한 파일은 보관된 checksum receipt 또는 출처의 expected SHA-256이 일치할 때만 재사용합니다. 임베딩 캐시는 모델·해결된 revision·차원·본문 메타데이터 해시를 구분합니다. OpenSearch는 안정적인 `content_id = media- + SHA256(canonical_url)`를 `_id`로 전체 문서 PUT하므로 반복 실행해도 중복 문서가 생기지 않습니다. 같은 URL의 내용이 변경됐는지 네트워크로 자동 재검증하지 않으므로 새 원본을 의도하면 source checksum을 갱신하거나 해당 다운로드 파일/receipt를 제거하고 재수집하세요. 서로 다른 canonical URL은 동일 바이트여도 라이선스/출처가 다를 수 있어 합치지 않습니다.

### 이미 적법하게 확보한 로컬 파일

```bash
python -m search_agent.media.cli import-local data/media_manifest.small.json \
  /absolute/path/chinstrap_penguin_jumps.webm --probe
# 이어 --execute 하면 검증한 로컬 파일을 재사용해 색인
```

단일 항목 manifest와 사전에 검증된 `expected_sha256`이 필수입니다. 복사 중 크기/파일 시그니처/체크섬을 검사하며 `--probe`는 설치된 ffprobe로 형식/스트림/길이를 확인합니다. 소스의 원래 메타데이터는 임의 변경하지 않고 측정 결과를 receipt에 기록합니다. 로컬 import는 `url_download_performed: false`로 원본 URL 다운로드와 구분합니다. ffprobe는 선택적이며 다운로드 기본 경로는 MIME+파일 시그니처+체크섬 검증까지입니다. ASR이나 재인코딩은 없습니다.

## OpenSearch 연결과 검색

**OpenSearch 2.19 이상**의 Faiss/HNSW `cosinesimil` + `knn_vector dimension:2560`을 사용합니다. 최신 공식 [벡터 필드](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/), [엔진](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/), [k-NN query](https://docs.opensearch.org/latest/query-dsl/specialized/k-nn/index/), [효율적 필터](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/) 문서를 확인했으며 실제 2.19.3 서버에서 테스트했습니다. 색인 시 `_meta`에 실제 임베딩 revision과 설정을 넣고 검색 시 동일성을 검사합니다. 설정이나 차원이 다르면 자동 덮어쓰지 않고 새 인덱스 생성/재색인을 요구합니다.

```dotenv
SEARCH_BACKEND=opensearch
SEARCH_MODE=real
SEARCH_DIMENSIONS=2560
OPENSEARCH_URL=https://your-authorized-cluster:9200
OPENSEARCH_INDEX=kids-media-v1
OPENSEARCH_CA_CERTS=/absolute/path/to/trusted-ca.pem
# 기존 권한 있는 계정의 인증정보를 환경변수로 제공. Git에 넣지 마세요.
OPENSEARCH_USERNAME=your-existing-user
OPENSEARCH_PASSWORD=your-existing-secret
```

```bash
python -m uvicorn search_agent.app:app --host 127.0.0.1 --port 8000 --workers 1
```

HTTPS와 인증서 검증이 기본이며 `verify=false` 설정은 제공하지 않습니다. URL 안의 자격증명은 거부합니다. 인증은 기존 계정의 basic auth 또는 HTTPS 연결 환경에 맡기며 새로운 클러스터/계정을 자동 생성하지 않습니다. 명시적 로컬 테스트용 `OPENSEARCH_ALLOW_HTTP_LOCAL=true`는 루프백 HTTP만 허용하지만 이번 검증에서는 사용하지 않았습니다.

`POST /api/search`는 기존 query/TopK/TopN 계약을 유지합니다. backend가 OpenSearch이면 BM25 multi_match와 k-NN 내부 filter를 각각 실행하고, 앱에서 RRF→TopK→BGE 원시 점수→TopN 순으로 처리합니다. 두 분기 모두 동일한 나이/주제/캐릭터 필터입니다. 점수 동률은 안정적인 content ID로 정렬합니다. 한국어 BM25는 기본 analyzer이며 형태소 분석 플러그인은 필요하지 않지만 실제 데이터 품질 평가는 별도입니다.

응답의 `backend: opensearch`, `content_id`, 콘텐츠 원본 언어·형식·기간·출처·저작자·라이선스·attribution·checksum·상대 local_path·metadata_provenance를 확인할 수 있습니다. 모델 점수는 확률이 아닙니다. 원본 바이너리/로컬 파일 경로를 스트리밍하는 공개 HTTP 엔드포인트는 제공하지 않습니다. UI는 원문 출처 링크를 표시합니다.

### 미상 연령 정책

공식 연령 정보는 양 끝과 `age_rating_source`가 모두 있어야 합니다. 추정 연령이나 가짜 자막을 채우지 않습니다. **나이를 명시한 검색에서 연령 미상 문서는 항상 제외**합니다. 따라서 현재 Commons 8개만 색인했다면 `5살 펭귄 영상`은 빈 결과, `펭귄 영상`은 미상 표시와 함께 검색됩니다. 이것은 자료가 어린이에게 적합하다는 판정이 아닙니다. 실제 언어도 미상이면 `und`, 언어 내용이 없다고 확인된 경우 `zxx`입니다.

## 스키마·안전·복구

- `MediaManifest(version=1, entries=[...])`는 extra 필드를 거부합니다. 설명/주제/언어/기간 provenance가 필수이며 제목/설명의 assistant 번역과 편집 태그를 원문과 구분합니다. `rights_verified`는 제공된 출처 페이지에서 라이선스 표시를 확인했다는 입력자 확인이며, 모든 용도/전 세계 권리/아동 적합성 보장이 아닙니다. NASA 등 추가 조건은 `license_notes`와 `source_metadata`에 남깁니다.
- 원본 manifest 및 파생 manifest와 개별 미디어의 라이선스는 코드와 별도입니다. [ATTRIBUTIONS.md](data/ATTRIBUTIONS.md)를 함께 보존합니다. AI Hub 승인 자료는 목록에 없습니다.
- 다운로드는 HTTPS/443, 정확한 hostname allowlist만 허용합니다. 사용자정보·fragment·커스텀 포트·IP literal을 거부합니다. DNS의 모든 반환 주소가 공인 주소인지 검사하고 검증한 IP에 직접 연결하며 TLS는 원래 hostname을 검증합니다. 모든 리다이렉트를 재검증하고 최대 횟수를 제한합니다. 다운로드·DNS·연결/헤더/본문 전체에 deadline을 적용합니다.
- 관리형 프록시가 설정된 환경에서는 직접 소켓이 egress 정책을 우회하지 않도록 **fail closed**합니다. 프록시를 해제하거나 403을 우회하지 않습니다. 승인된 환경 또는 정책을 보장하는 별도 `Transport` 구현이 필요합니다.
- 파일/전체 배치 크기 제한, bounded retry(4xx 접근 거부/429는 재시도 안 함), MIME/시그니처 검증, expected checksum 검증, atomic temp→rename, 경로 생성 및 symlink 차단을 적용합니다. 파일 이름은 content ID+검증된 확장자로만 만듭니다. archive 추출이나 원격 스크립트 실행은 하지 않습니다.
- `MEDIA_ROOT`는 신뢰하는 서버 전용 디렉터리여야 합니다. 인제스트/로컬 import는 같은 lock을 사용합니다. 오류 receipt에는 signed URL query나 인증정보를 포함하지 않습니다. 원본 URL 자체는 출처 보존 목적의 정상 다운로드 receipt와 인덱스에 저장하므로 장기 비밀 토큰 URL을 manifest에 넣지 마세요.
- `downloaded`와 `indexed`를 구분하고 실패 단계(`download`, `embedding`, `index_setup`, `index_upsert`)와 오류 유형을 기록합니다. 다운로드가 완료된 뒤 색인 실패해도 receipt/파일을 남겨 재개합니다. CLI 종료 코드는 성공/계획 0, 항목별 부분실패 1, 구성/manifest 오류 2입니다.
- 로컬 파일 저장은 `MediaStorage`, 색인 쓰기는 `IndexWriter`, 검색은 `OpenSearchStore` 경계로 분리했습니다. 객체 저장소로 교체할 수 있습니다. 소규모 구현이며 병렬 배치/샤딩 설계/청크별 자막 검색/클러스터 운영 설정은 포함하지 않습니다. 기본 인덱스는 shard 1/replica 0으로 개발용입니다.

## 검증

```bash
pytest -q
ruff check .
ruff format --check .
mypy search_agent scripts
# 운영 데이터와 별개인 무작위 테스트 인덱스를 생성·삭제합니다.
# 권한 있는 테스트용 서버 접속 변수를 먼저 지정하세요.
OPENSEARCH_TEST=1 pytest -q tests/test_opensearch_integration.py
```

통합 테스트는 **실제 OpenSearch + 명시적 synthetic 벡터/메타데이터**입니다. Qwen/BGE의 실행이나 미디어 다운로드 성공으로 표현하지 않습니다. TLS root CA 및 임시 인증을 사용한 2.19.3 서버에서 실행했으며 보안 플러그인을 끄지 않았습니다. 미디어/Hugging Face 외부 접속과 Library 영상 전송은 현재 실행환경에서 차단되어 실제 원본 취득→Qwen→색인→BGE 전체 경로는 미검증입니다. 상세 기록은 [VALIDATION.md](VALIDATION.md)를 참고하세요.
