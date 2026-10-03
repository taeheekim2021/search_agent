# 미디어·메타데이터 출처와 라이선스

`source_manifest.json`은 2026-10-03에 확인한 Wikimedia Commons 8개 파일의 출처 기록이며 **CC BY-SA 4.0**입니다. Commons 설명문에서 파생한 한국어 제목/설명과 주제 태그는 assistant 번역·요약/편집 메타데이터로, 공식 현지화·자막·대본·연령등급이 아닙니다. `metadata_manifest.json`, `media_manifest.json`, `media_manifest.small.json`은 그 기록을 인제스트 스키마로 변환한 파생 메타데이터로, 출처·변경 사항·동일 라이선스를 유지합니다.

원본 manifest의 개별 저자, 원문 설명, `license_evidence_url`, attribution과 특기사항을 그대로 보존합니다. Commons 페이지의 설명문 라이선스와 구조화된 데이터(CC0), 미디어 파일 라이선스를 혼동하지 않습니다. 저장소 코드의 라이선스가 아래 자료의 라이선스를 덮어쓰지 않습니다. 미디어 바이너리는 Git에 포함하지 않습니다.

| 자료/출처 | 저작자 | 미디어 라이선스 |
|---|---|---|
| [Solarsystem.webm](https://commons.wikimedia.org/wiki/File:Solarsystem.webm) | Simpleshow Japan | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Elephant.webm](https://commons.wikimedia.org/wiki/File:Elephant.webm) | OJjnr | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [Penguin was sleeping.webm](https://commons.wikimedia.org/wiki/File:Penguin_was_sleeping.webm) | Chicomint | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [Chinstrap penguin jumps.webm](https://commons.wikimedia.org/wiki/File:Chinstrap_penguin_jumps.webm) | BrokenSegue | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Monarch butterfly 1.webm](https://commons.wikimedia.org/wiki/File:Monarch_butterfly_1.webm) | Truem | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [Butterfly flying.webm](https://commons.wikimedia.org/wiki/File:Butterfly_flying.webm) | Subhashish Panigrahi | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| [Small Copper Butterfly](https://commons.wikimedia.org/wiki/File:Small_Copper_Butterfly_(Lycaena_phlaeas).webm) | The Nature Box | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Solar System Resource Page](https://commons.wikimedia.org/wiki/File:Solar_System_Resource_Page_(SVS20391).webm) | NASA's Scientific Visualization Studio; Jenny McElligott; David Ladd; Michael Lentz; Walt Feimer | [PD-USGov-NASA / NASA 사용 조건](https://www.nasa.gov/nasa-brand-center/images-and-media/) |

NASA 자료의 필수 크레딧: **NASA's Goddard Space Flight Center Conceptual Image Lab**. 출처 페이지는 미국 내 public domain으로 표시합니다. 이를 전 세계의 모든 권리 허가로 확장하지 않으며 NASA의 보증을 암시하지 않습니다. 인물·상표·제3자 자료 등의 별도 조건은 원본 manifest의 `license_notes`에 남깁니다.

현재 코퍼스는 자연·우주에 편중되어 있으며 한국어 어린이 방송 콘텐츠의 실제 카탈로그가 아닙니다. 8개 모두 공식 연령등급은 미상, 자막/대본 없음입니다. 태양계 설명 영상의 실제 언어는 일본어(`ja`); 침묵이 확인된 자료는 `zxx`, 그 외 언어 미상은 `und`입니다. 한국어 제목만 보고 원본 언어를 한국어로 지정하지 않습니다.

소스 조사 단계에서 작은 턱끈펭귄 영상 1개만 다운로드/ffprobe로 확인됐습니다. 그 사실은 제공된 source manifest의 기록이며 **현재 코드 실행환경의 외부 다운로드 성공을 의미하지 않습니다**. 나머지 7개는 원본 링크 확인까지입니다.

AI Hub [교육 영상 데이터 71808](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71808&topMenu=100)은 신청·승인이 필요하므로 자동 다운로드 목록에서 제외했습니다. DataHub는 이번 조사에서 적합한 재생형 미디어 코퍼스를 확인하지 못해 사용하지 않았습니다.
