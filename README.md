# 아이들나라 콘텐츠 SearchAgent — 가상 샘플 프로토타입

한국어 자연어 질의를 나이·주제·캐릭터 조건으로 해석하고, 키워드 + **Qwen/Qwen3-Embedding-4B** 의미검색 후보를 합친 뒤 **BAAI/bge-reranker-v2-m3**로 재정렬합니다. FastAPI와 정적 HTML UI만 사용합니다. 외부 API 요금이나 별도 검색 인프라가 필요하지 않습니다.

**모든 콘텐츠 12개, 캐릭터, 설명은 이 프로젝트에서 직접 만든 가상 예시입니다. 실제 아이들나라 DB, 저작권 있는 실제 줄거리, 고객정보는 포함하지 않습니다. 실제 모델 추론은 이 환경에서 아직 검증되지 않았습니다.**

## 빠른 실행: 명시적 모의 UI

Python 3.12 이상, 저장소 루트에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
SEARCH_MODE=mock python -m uvicorn search_agent.app:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. 모의 모드는 키워드만 사용하고 UI에 큰 라벨을 표시합니다. 의미검색/리랭커를 실행하지 않으며 `fusion_score`, `reranker_score`는 모두 `null`입니다. 모의 결과를 실제 AI 성능으로 해석하면 안 됩니다. 모드를 생략하면 기본값은 **real**입니다.

이 클라우드 환경의 `uv` 사용 시 쓰기 가능한 캐시를 지정해야 합니다.

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -e '.[dev]'
```

## 실제 모델 실행

```bash
source .venv/bin/activate
python -m pip install -e '.[models,dev]'
cp .env.example .env
# CPU 기본값은 float32. 충분한 메모리를 먼저 확보하세요.
python -m uvicorn search_agent.app:app --host 127.0.0.1 --port 8000 --workers 1
```

CPU 전용 PyTorch wheel이 필요하면 먼저 공식 PyTorch 설치 안내에 맞는 CPU wheel을 설치합니다. GPU 환경은 해당 CUDA 버전과 맞는 PyTorch를 먼저 설치하고 아래 설정을 사용합니다. GPU를 자동 생성하거나 유료 서비스를 호출하지 않습니다.

```dotenv
SEARCH_MODE=real
SEARCH_EMBEDDING__DEVICE=cuda
SEARCH_EMBEDDING__DTYPE=float16
SEARCH_RERANKER__DEVICE=cuda
SEARCH_RERANKER__DTYPE=float16
```

Qwen 4B 가중치만 16bit 약 8GB, float32 약 16GB입니다. BGE, 토크나이저, activation, 로딩 공간이 추가로 필요합니다. 기본 CPU float32는 이 작업 환경의 16GiB 한도에서 실행하기 어렵습니다. 메모리 여유가 있는 머신에서 실행하거나, CPU의 16bit 지원·성능을 먼저 확인한 뒤 `bfloat16`/`float16`을 명시 설정하세요. CUDA에서도 가중치 크기 이상의 VRAM이 필요합니다. 프로세스 하나와 batch 1부터 시작하세요. 서버는 최초 검색에서 모델을 로딩하며 병렬 검색은 메모리 폭증을 막기 위해 직렬 처리합니다.

모델 ID는 코드 상수로 고정되어 임의 모델로 변경되지 않습니다. 두 모델의 revision/device/dtype/batch/max_length는 별도 설정입니다. 재현성을 위해 `SEARCH_EMBEDDING__REVISION`과 `SEARCH_RERANKER__REVISION`을 검토한 Hugging Face commit SHA로 고정할 수 있습니다. 기본 `main`도 로딩 후 실제 commit SHA를 확인해 캐시 키에 포함합니다.

모델 다운로드·로딩·추론에 실패하면 API는 **503 MODEL_UNAVAILABLE**와 실패 단계(`embedding`, `reranker`)를 반환합니다. 실제 원인은 서버 로그에 남습니다. 해시 임베딩, 다른 모델, 키워드 리랭킹, 모의 모드로 자동 대체하지 않습니다. `/health`는 HTTP 서버 상태만 확인하며 `model_readiness: not_checked`로 모델 준비 여부를 구분합니다.

## 검색 규칙과 한계

1. 숫자 나이(`5살`, `만 5세`)와 일부 한글 나이(`다섯 살`)를 규칙으로 추출합니다. 나이는 0~18 정수 하나만 허용합니다. 주제·캐릭터는 카탈로그 사전과 소수의 주제 별칭을 사용합니다. 범용 LLM 조건 해석기는 아닙니다.
2. **나이는 하드 필터**이며 양 끝을 포함한 권장 연령 범위에 들어야 합니다. 인식한 주제와 캐릭터도 AND 조건으로 먼저 필터링합니다. 인식된 모든 조건을 만족해야 합니다. 명시적으로 `없는친구 캐릭터`라고 요청하면 빈 결과입니다. 사전에 없는 일반 단어나 복잡한 부정/범위/OR 표현은 완전히 해석하지 못하므로 UI의 추출 조건을 확인하세요.
3. 필터 후 키워드·의미검색 각각 TopK를 가져옵니다. 한국어 키워드는 형태소 분석 없이 메타데이터 부분 문자열을 사용합니다. 각 소스 내 ID 중복 제거 → 동일 가중치 RRF(`1/(60+rank)`) → 통합 TopK입니다. 동점은 안정적인 ID 오름차순입니다.
4. Qwen은 **쿼리에만** 영어 검색 instruction을 붙입니다. 문서는 메타데이터 본문 그대로입니다. left padding + last-token pooling, 기본 2560차원이며 32~2560 차원을 설정할 수 있습니다. 잘라낸 뒤 L2 정규화하고 내적(코사인)을 계산합니다.
5. BGE는 instruction 없이 원래 query-passage 쌍을 입력하는 cross-encoder입니다. 원시 logits 내림차순, 동점 ID 오름차순, 결과 TopN을 반환합니다. 점수는 확률·신뢰도·모델 속마음이 아닙니다. 근거는 연령·주제·캐릭터·길이 등 관찰 가능한 메타데이터만 표시합니다.
6. 필터 결과가 없으면 모델을 호출하지 않고 빈 목록과 `model_inference_performed: false`를 반환합니다. 일반 의미검색에는 미보정 임계값을 적용하지 않으므로 무관한 질의도 가까운 후보를 반환할 수 있습니다. 관련성 없음 판정은 향후 실제 데이터 평가가 필요합니다.

공식 구현 참고: [Qwen 모델 카드](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [BGE 모델 카드](https://huggingface.co/BAAI/bge-reranker-v2-m3). Qwen은 Transformers 4.51 이상을 사용합니다.

## 캐시와 교체 지점

- `.cache/huggingface`: Hugging Face 모델/토크나이저 다운로드 캐시.
- `.cache/embeddings`: 콘텐츠 벡터 `.npy`. 모델 ID·요청 revision·해결된 commit SHA·차원·dtype·device·max_length·pooling schema·전체 콘텐츠 데이터의 SHA-256으로 구분합니다. 콘텐츠가 바뀌면 새 캐시를 생성합니다. 손상 파일은 같은 실제 모델로 재계산합니다. 오래된 캐시는 서버를 중지한 뒤 삭제해도 됩니다.
- `CatalogRepository`: 일관된 카탈로그 스냅샷을 반환하는 DB 어댑터로 교체. 샘플 `Content`는 `is_sample=true`를 강제하므로 실제 DB 연결 시 데이터 타입과 UI의 샘플 표시도 함께 변경해야 합니다.
- `KeywordAdapter`: 검색엔진 순위 어댑터로 교체.
- `EmbeddingAdapter`, `RerankerAdapter`: 서로 독립된 모델 계약. `SemanticSearch`는 작은 샘플용 NumPy 전수 벡터 검색입니다. 대규모 데이터에는 필터를 지원하는 인덱스 어댑터가 필요합니다.

## API 계약

`GET /` UI, `GET /health` 서버 모드와 기본 TopK/TopN, `POST /api/search` 검색, `GET /docs` OpenAPI 문서입니다.

```bash
curl -s http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"5살 아이가 볼 공룡 영상","top_k":20,"top_n":5}'
```

입력: 공백 제거 후 1~500자 `query`; 정수 `top_k`/`top_n`은 각각 1~100, 생략 시 설정값 20/5. 유효한 기본값까지 적용한 뒤 `top_n <= top_k`여야 합니다. 알 수 없는 필드는 거부합니다.

응답: `mode`, `sample_notice`, `model_inference_performed`, `conditions`(age/topics/characters/extraction_method), `candidate_count`, `elapsed_ms`, `notice`, `results`. 각 결과는 `content`, `evidence`, `retrieval_sources`, `fusion_score`, `reranker_score`를 포함합니다. 메타데이터 설명은 검색/캐시에도 사용됩니다. 빈 결과도 HTTP 200입니다. 잘못된 입력/해석 불가능한 나이는 422, 모델 실패는 503입니다. 점수는 서로 다른 종류이므로 RRF와 BGE 수치를 직접 비교하지 마세요.

## 검증과 스모크 테스트

```bash
python -m pytest -q
ruff check .
ruff format --check .
mypy search_agent scripts
# 다운로드/메모리 조건을 먼저 확인한 후 명시적으로 실제 모델 실행:
python scripts/smoke_models.py --run-real
```

단위/계약 테스트는 작은 고정 fixture와 NumPy 기반 HF test double을 사용합니다. 실제 모델 추론이나 품질 검증이 아닙니다. 스모크 스크립트만 실제 모델을 실행하며 한국어 관련/무관 문장의 Qwen 코사인, BGE 원시 점수, 차원·정규화, 로딩 포함 지연을 출력하고 관련 문장의 우선 순위를 검증합니다. 실패 시 비정상 종료하며 가짜 결과를 출력하지 않습니다.

브라우저 검증은 모의 서버 실행 후 `pip install -e '.[ui-test]'`와 `python scripts/smoke_ui.py`로 재현합니다. 이 스크립트는 `/usr/bin/chromium`이 필요합니다. 검증된 경량 패키지 버전은 `requirements-tested.txt`에 있으며, 모델 의존성은 이 환경에서 설치/검증하지 않았습니다.

실제 테스트 현황과 남은 제한은 [VALIDATION.md](VALIDATION.md)에 기록합니다. 이 프로토타입은 로컬 검토용이며 인증/운영용 배포를 포함하지 않습니다. 서비스 배포는 포함하지 않습니다.
