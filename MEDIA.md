# 미디어 메타데이터 색인과 OpenSearch

기본 경로는 **영상 원본을 다운로드하지 않는 metadata-only 인제스트**입니다. 제목·설명·태그와 허가된 선택적 자막 텍스트를 Qwen/Qwen3-Embedding-4B로 임베딩하고, 메타데이터와 2560차원 벡터를 OpenSearch에 upsert합니다. 검색 시 BAAI/bge-reranker-v2-m3를 유지합니다. 미디어 URL을 열거나 DNS를 조회하지 않으며 영상 파일·로컬 경로·미디어 체크섬이 없어도 실행됩니다. 모델 다운로드와 OpenSearch 접속은 별개로 필요할 수 있습니다.

자동 ASR/자막 생성/영상 분석은 구현하지 않았습니다. 없는 연령·언어·길이·자막을 만들지 않습니다. 이전에 저장한 영상 파일이나 다운로드 receipt도 삭제하지 않습니다.

## 기본 실행: 메타데이터만 색인

```bash
python -m pip install -e '.[models,dev]'
# 계획 확인: 다운로드·모델·OpenSearch 호출과 파일 쓰기 없음
python -m search_agent.media.cli ingest data/metadata_manifest.json
# 실제 Qwen 텍스트 임베딩 + OpenSearch upsert. 영상 다운로드 없음
python -m search_agent.media.cli ingest data/metadata_manifest.json --execute
```

먼저 아래의 OpenSearch 접속 설정을 지정합니다. `MEDIA_ALLOWED_DOMAINS`는 메타데이터 경로에서 필요하지 않습니다. `--limit N`으로 N개 항목만 처리할 수 있습니다. `MEDIA_ROOT`에는 메타데이터 처리 receipt, 임베딩 캐시와 잠금 파일만 쓰며, 이 경로에서도 원본 미디어 바이트를 읽거나 쓰지 않습니다. 실제 Qwen 로딩/추론이 실패하면 색인 성공으로 처리하지 않습니다.

`source_id`, 제목·설명·태그, `canonical_url`(출처 페이지), `original_url`(원본/재생 링크), 저자·라이선스·attribution·provenance를 보존합니다. 구 manifest의 `source_id`가 없으면 기존 `source_metadata.source_id`, 그마저 없으면 canonical URL을 식별자로 사용합니다. `content_id`는 이전과 동일한 canonical URL의 SHA-256이므로 반복 upsert 시 중복 문서가 생기지 않습니다.

형식·언어·길이·연령·자막은 선택 정보입니다. 원본 언어를 모르면 `und`, 미디어 형식·길이·연령은 `null`입니다. `MediaRecord.local_path`, `checksum_sha256`, `size_bytes`는 선택 항목이고 기본 색인 문서에서 제외합니다. `retrieved_at`은 이 경로에서 메타데이터 처리 시각이며 영상 다운로드 시각을 의미하지 않습니다. 파일 체크섬과 임베딩 캐시용 메타데이터 해시는 구분합니다.

기존 media-v1 인덱스에 `source_id` 필드가 없으면 동일 임베딩 식별자를 확인한 뒤 keyword 필드만 추가합니다. 문서를 전체 PUT하므로 이전 색인 문서의 파일 경로/크기 필드는 metadata-only 실행 후 생략될 수 있지만 **로컬에 보관한 실제 파일과 다운로드 receipt는 그대로 남습니다**. 기존 메타데이터/차원과 호환되지 않는 매핑은 자동 재생성하지 않습니다.

`.metadata-receipt.json`은 다운로드 receipt와 분리되며 실패 후 같은 명령을 재실행하면 동일 ID에 upsert합니다. 필수 제목·설명·출처/재생 URL·저자·라이선스·attribution과 알려진 메타데이터 provenance는 검증합니다. 출처 manifest와 파생 메타데이터 라이선스는 [ATTRIBUTIONS.md](data/ATTRIBUTIONS.md)를 따릅니다.

### 기존 선택적 파일 취득 기능

과거 다운로드 기능은 호환성을 위해 남아 있으나 기본 CLI/API 경로에서는 호출하지 않습니다. `--execute --download-media` 또는 `--download-only`를 **명시**한 경우에만 사용됩니다. `import-local`도 별도 명령으로 유지합니다. 현재 요청에서는 어느 경로도 실행하지 않았고 파일을 삭제하지 않았습니다. 파일 취득을 선택하는 경우에만 HTTPS allowlist·SSRF·크기·시간·checksum 검사를 적용하며 형식을 명시해야 합니다.

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

- `MediaManifest(version=1, entries=[...])`는 extra 필드를 거부합니다. 제목/설명/주제와 값이 알려진 언어/기간의 provenance가 필수이며 제목/설명의 assistant 번역과 편집 태그를 원문과 구분합니다. `rights_verified`는 제공된 출처 페이지에서 라이선스 표시를 확인했다는 입력자 확인이며, 모든 용도/전 세계 권리/아동 적합성 보장이 아닙니다. NASA 등 추가 조건은 `license_notes`와 `source_metadata`에 남깁니다.
- 원본 manifest 및 파생 manifest와 개별 미디어의 라이선스는 코드와 별도입니다. [ATTRIBUTIONS.md](data/ATTRIBUTIONS.md)를 함께 보존합니다. AI Hub 승인 자료는 목록에 없습니다.
- 다운로드는 HTTPS/443, 정확한 hostname allowlist만 허용합니다. 사용자정보·fragment·커스텀 포트·IP literal을 거부합니다. DNS의 모든 반환 주소가 공인 주소인지 검사하고 검증한 IP에 직접 연결하며 TLS는 원래 hostname을 검증합니다. 모든 리다이렉트를 재검증하고 최대 횟수를 제한합니다. 다운로드·DNS·연결/헤더/본문 전체에 deadline을 적용합니다.
- 관리형 프록시가 설정된 환경에서는 직접 소켓이 egress 정책을 우회하지 않도록 **fail closed**합니다. 프록시를 해제하거나 403을 우회하지 않습니다. 승인된 환경 또는 정책을 보장하는 별도 `Transport` 구현이 필요합니다.
- 파일/전체 배치 크기 제한, bounded retry(4xx 접근 거부/429는 재시도 안 함), MIME/시그니처 검증, expected checksum 검증, atomic temp→rename, 경로 생성 및 symlink 차단을 적용합니다. 파일 이름은 content ID+검증된 확장자로만 만듭니다. archive 추출이나 원격 스크립트 실행은 하지 않습니다.
- `MEDIA_ROOT`는 신뢰하는 서버 전용 디렉터리여야 합니다. 인제스트/로컬 import는 같은 lock을 사용합니다. 오류 receipt에는 signed URL query나 인증정보를 포함하지 않습니다. 원본 URL 자체는 출처 보존 목적의 정상 다운로드 receipt와 인덱스에 저장하므로 장기 비밀 토큰 URL을 manifest에 넣지 마세요.
- `downloaded`와 `indexed`를 구분하고 실패 단계(`download`, `embedding`, `index_setup`, `index_upsert`)와 오류 유형을 기록합니다. 다운로드가 완료된 뒤 색인 실패해도 receipt/파일을 남겨 재개합니다. CLI 종료 코드는 성공/계획 0, 항목별 부분실패 1, 구성/manifest 오류 2입니다.
- 기본 메타데이터 색인은 미디어 저장소를 생성하거나 호출하지 않습니다. 선택적 로컬 파일 저장은 `MediaStorage`, 색인 쓰기는 `IndexWriter`, 검색은 `OpenSearchStore` 경계로 분리했습니다. 객체 저장소로 교체할 수 있습니다. 소규모 구현이며 병렬 배치/샤딩 설계/청크별 자막 검색/클러스터 운영 설정은 포함하지 않습니다. 기본 인덱스는 shard 1/replica 0으로 개발용입니다.

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

통합 테스트는 **실제 OpenSearch + 명시적 synthetic 벡터/메타데이터**입니다. Qwen/BGE의 실행이나 미디어 다운로드 성공으로 표현하지 않습니다. TLS root CA 및 임시 인증을 사용한 2.19.3 서버에서 실행했으며 보안 플러그인을 끄지 않았습니다. 실제 Qwen/BGE 추론은 Hugging Face 접속 차단 이력 때문에 미검증입니다. 기본 경로는 미디어 다운로드가 필요하지 않아 Wikimedia/Library 영상 전송 차단의 영향을 받지 않습니다. 이번 검증에서도 실제 모델 성공을 주장하지 않습니다. 상세 기록은 [VALIDATION.md](VALIDATION.md)를 참고하세요.
