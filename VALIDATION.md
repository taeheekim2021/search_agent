# 구현 및 검증 기록

날짜: 2026-10-03. 작업 위치: `/workspace/search_agent`.

- 체크아웃: origin `https://github.com/taeheekim2021/search_agent.git`, 초기 브랜치 `work`, 기존 커밋 없음. 메타데이터 기본 브랜치는 전달받은 `main`; 실제 원격 브랜치 생성/수정 없음.
- `/workspace` 및 저장소에 `AGENTS.md`, `.agents/skills` 없음. 기존 구현 파일 없음.
- 환경: Python 3.12.14, CPU quota 4코어, cgroup RAM 17,179,869,184 bytes (16GiB), swap 없음, 시작 시 디스크 여유 약 30GiB. `nvidia-smi` 없음, 사용 가능한 GPU 미확인. GPU/유료 API 리소스 생성 없음.

## 성공한 검증

- pytest: **41 passed**. 조건 추출, 나이 하드 필터, 빈 결과, 입력/TopK/TopN 경계, 단일 후보, ID 중복 제거, RRF 통합, 점수 내림차순 및 동점 ID 정렬, 두 모델 ID, 쿼리 전용 instruction, last-token pooling, 2560 기본 차원 및 축소 후 정규화, BGE 쌍 입력/원시 logits, 독립적 로딩/추론 오류, 캐시 재사용/모델·revision·차원·콘텐츠 변경 무효화/손상 재계산을 검증.
- Ruff lint 및 format 검사 통과.
- mypy `search_agent scripts`: 11개 소스 파일 통과.
- FastAPI TestClient로 모의/실제 계약의 200/422/503와 UI 라우트 확인.
- 실제 로컬 Uvicorn 서버 + Chromium + Playwright 모의 UI 검증 통과:
  - `5살 아이가 볼 공룡 영상`: 가상 콘텐츠 3개, 나이 범위와 주제 충족.
  - `토리 캐릭터 영상`: 가상 콘텐츠 4개.
  - `18살 공룡 영상`: 0개.
  - 첫 검색 반복: 동일한 ID/제목/순서.
  - TopK 1 / TopN 5: 입력 오류 메시지.
  - 390px 모바일 너비 가로 overflow 없음, JavaScript 오류 없음.
- 테스트 시 upstream Starlette의 httpx TestClient 지원 예정 변경에 관한 DeprecationWarning 1개가 발생했으며 테스트는 통과함.

## 실제 모델 미검증 및 네트워크 차단

공식 모델 카드는 브라우징 도구로 읽었습니다. 그러나 선택된 실행환경에서 다음 메타데이터 요청이 모두 실패했습니다.

- hostname: `huggingface.co`
- 경로: `/api/models/Qwen/Qwen3-Embedding-4B`
- 경로: `/api/models/BAAI/bge-reranker-v2-m3`
- 오류: `URLError <urlopen error Tunnel connection failed: 403 Forbidden>`

후속 상태: 사용자가 `huggingface.co` 허용을 승인했으나 공식 Settings → Codex Cloud → search_agent → Edit 흐름이 `환경 편집을 시작하지 못했습니다` 오류로 실패했다고 전달받았습니다. 허용 설정은 변경되지 않았고 새 환경 버전도 없습니다.

차단을 우회하거나 허용 도메인을 임의 변경하지 않았습니다. 리다이렉트 이전에 차단되어 가중치 CDN hostname은 확인되지 않았습니다. 정식 허용 변경 이후 해당 실행환경에 적용됐는지 위 hostname 요청부터 다시 확인하고, 실제로 관측한 추가 다운로드 hostname만 검토해야 합니다.

PyTorch/Transformers 모델 의존성과 수 GB 가중치를 다운로드하지 않았습니다. 기본 CPU float32 모델은 RAM 한도에도 여유가 없습니다. **실제 Qwen→BGE 추론, 관련성 품질, 실제 모델 지연은 미검증**입니다. 단위테스트의 고정 fixture/HF test double 및 모의 UI 성공은 실제 모델 실행 성공이 아닙니다. `scripts/smoke_models.py --run-real`을 네트워크와 메모리가 충분한 조건에서 실행하면 실제 점수/지연과 두 한국어 문장의 순위를 확인할 수 있습니다.

## 범위와 남은 작업

가상 샘플 12개에 대한 로컬 검토용 구현입니다. 규칙 기반 추출은 모든 한국어 표현/부정/다중 연령/OR 조건을 지원하지 않습니다. 실제 DB, 형태소 분석, 관련성 없음 임계값, 실제 콘텐츠 평가, 운영 보안/동시처리/대용량 인덱스는 후속 작업입니다.

이 문서는 최초 원격 게시 이전의 검증 기록입니다. PR·서비스 배포는 검증 범위에 포함하지 않습니다. 전달 ZIP에는 소스·샘플 데이터·테스트·문서·비밀값 없는 `.env.example`만 포함하고 `.git`, `.env`, 모델 가중치, 캐시, 가상환경은 제외합니다.

# OpenSearch·실제 미디어 확장 검증 (2026-10-03 KST)

이 절은 위 초기 샘플 구현 이후의 확장 결과입니다. 원격 기존 `main`은 `8eb63895bf16e663e6e0c2e75e350bfdab1454bc`; 로컬 `work`의 기반 커밋 `349de7ed45e667b03d8462ad1e2ee2cc4525bbd2`와 원격 파일 내용이 같음을 확인한 뒤 확장했습니다. 이 절은 확장본 게시 전 완료한 검증 기록입니다.

## 구현

`search_agent/media/`에 strict manifest/출처·번역 provenance, 원본 미디어 저장소, HTTPS allowlist·DNS pinning·TLS·SSRF 차단, 용량/시간/재시도/리다이렉트 제한, 원자적 파일 저장, 체크섬·파일 시그니처, CLI dry-run/다운로드/체크섬 기반 로컬 import, 부분 실패 receipt와 재개, 실제 OpenSearch PUT 및 BM25/k-NN 검색을 추가했습니다. 기존 sample 백엔드는 유지했습니다. OpenSearch에서는 나이 미상 자료를 나이를 지정한 검색에서 제외합니다.

## 실제 서버 및 테스트

- Docker client/daemon 28.4.0이 동작했고 기존 이미지/컨테이너 목록은 비어 있었습니다.
- 공식 `opensearchproject/opensearch:2.19.3` 이미지를 가져왔습니다. digest: `sha256:e96cc6ae1500a073d973c0906f30f7cf4d9c461f32f855f9242a2da933660cdd`.
- 독립 임시 컨테이너에 메모리 4GiB, CPU 2, JVM heap 512MiB를 지정하고 호스트 포트를 `127.0.0.1:19200`으로만 열었습니다. security plugin·TLS·인증서 hostname 검증을 유지하고 번들 root CA를 신뢰했습니다. 테스트용 임시 비밀번호만 사용했으며 저장소/로그/전달물에는 넣지 않았습니다. 호스트 sysctl·방화벽·네트워크 보안 설정을 바꾸지 않았습니다.
- HTTPS로 서버 버전 2.19.3 / HTTP 200 확인.
- 무작위 이름의 테스트 인덱스에서 실제 **2560차원 Faiss HNSW/cosinesimil 매핑**, `_meta` 임베딩 식별자 검사, 반복 PUT upsert 시 4문서 유지, BM25 및 k-NN 내 동일 연령/태그 필터, 미상 연령 제외, 빈 결과를 검증했습니다. 테스트 후 해당 인덱스, 임시 컨테이너, 임시 인증정보를 삭제했습니다.
- **83 passed** (전체 단위/계약 + live OpenSearch 통합), 9.34초. 실제 서버가 없으면 기본 pytest는 **82 passed, 1 skipped**로 통합 테스트를 명시적으로 생략합니다. upstream Starlette httpx deprecation warning 1개는 남아 있습니다.
- 추가 테스트는 manifest·라이선스/provenance 보존, URL/사설 IP/루프백/metadata IP 차단, DNS timeout, 검증된 IP에 연결하고 원래 TLS hostname 사용, managed proxy fail-closed, redirect/403 비재시도, 다운로드 checksum·시그니처·크기·deadline·symlink, atomic cleanup, 실패 후 파일 재사용/재색인, batch budget, 로컬 import와 URL 취득 구분, secure OpenSearch 설정/API 503을 포함합니다.
- Ruff lint/format, mypy `search_agent scripts`(20 소스 파일) 통과.
- 기존 모의 UI의 Chromium 회귀 테스트: 공룡 3개, 토리 4개, 빈 결과, 반복 순서 동일, TopK/TopN 오류, 모바일 overflow 없음, JavaScript 오류 없음. 기존 8000 포트는 사용 중이라 건드리지 않고 새 루프백 8011에서 실행했습니다.

**서버 통합 테스트의 벡터/메타데이터는 synthetic fixture입니다. 실제 Qwen/BGE 추론·순위 품질·실제 원본 수집 성공을 의미하지 않습니다.**

## 실제 소스 입력과 취득 한계

- 전달받은 Library `source_manifest.json` 29,391바이트 전체를 공식 read로 가져와 원본 출처 기록을 보존했습니다. signed transfer materialize는 차단되어 읽기 경로를 사용했습니다.
- Commons 8개 출처 기록, 파생 `media_manifest.json`, 단일 소형 펭귄 영상용 `media_manifest.small.json`, `data/ATTRIBUTIONS.md`를 포함했습니다. 미디어별 라이선스와 CC BY-SA 4.0 메타데이터 라이선스를 분리합니다. 공식 연령/자막은 모두 미상/없음이고 한국어 제목을 원본 언어로 오인하지 않습니다.
- 승인된 가장 작은 영상의 공식 원본 hostname `upload.wikimedia.org`에 HEAD 요청을 했으나 `<urlopen error Tunnel connection failed: 403 Forbidden>`으로 실패했습니다. 반복 우회하지 않았습니다.
- 같은 펭귄 영상의 공식 Library materialize도 `oaisdmntprsouthcentralus.blob.core.windows.net`에서 `library file transfer failed: download failed`로 실패했습니다. 바이너리는 이 실행환경에 도착하지 않았습니다. 제공된 출처 조사에서의 ffprobe 확인 사실과 현재 실행환경의 검증을 구분합니다.
- 실제 CLI 단일항목 dry-run 성공(네트워크/쓰기/모델 없음). `--download-only`는 관리형 프록시를 발견해 `UnsafeDownload`/`stage:download`, `indexed:0`, `bytes_accounted:0`으로 안전하게 중단됐습니다. 다운로드 구현은 직접 소켓으로 관리형 egress를 우회하지 않습니다.
- Hugging Face 403 및 환경 Edit 실패 상태는 유지됩니다. 가중치/Git LFS는 변경하지 않았습니다. 실제 원본 취득 → Qwen 임베딩 → OpenSearch → BGE 전체 파이프라인과 실제 모델 지연은 **미검증**입니다.
