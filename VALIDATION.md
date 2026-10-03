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
