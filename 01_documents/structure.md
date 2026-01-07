# 데이터 구조

## 폴더 구성
```
data/
├── sj_1/          1,701개 PNG
├── sj_1.csv       1,702행
├── sj_2/          4,516개 PNG
├── sj_2.csv       4,517행
├── tk/            4,216개 PNG
└── tk.csv         4,217행
```

## CSV 구조

### 시간 정보
- Frame: 프레임 번호
- Elapsed_Time_In_Milliseconds: 경과 시간
- Time: 시각 (HH:MM:SS:MMM)

### 손가락 각도 (10개)
- Thumb: CMC_Flex, PIP_Flex
- Index: MCP_Flex, PIP_Flex
- Middle: MCP_Flex, PIP_Flex
- Ring: MCP_Flex, PIP_Flex
- Pinky: MCP_Flex, PIP_Flex

#### 측정된 각도 범위
전체 데이터셋(10,433개 프레임) 분석 결과:

**관절 타입별**
- **CMC** (Carpometacarpal - 엄지 손가락 기저부): -2° ~ 64° (평균: 48.3°)
- **MCP** (Metacarpophalangeal - 손가락 첫 번째 관절): -36° ~ 116° (평균: 48.6°)
- **PIP** (Proximal Interphalangeal - 손가락 두 번째 관절): -5° ~ 110° (평균: 32.3°)

**손가락별 상세**
| 손가락 | 관절 | 최소값 | 최대값 | 평균 |
|--------|------|--------|--------|------|
| Thumb  | CMC  | -2°    | 64°    | 48.3° |
| Thumb  | PIP  | 1°     | 59°    | 22.5° |
| Index  | MCP  | -30°   | 100°   | 43.0° |
| Index  | PIP  | 0°     | 110°   | 33.2° |
| Middle | MCP  | -36°   | 100°   | 47.2° |
| Middle | PIP  | 0°     | 110°   | 37.3° |
| Ring   | MCP  | -26°   | 102°   | 50.4° |
| Ring   | PIP  | -1°    | 110°   | 38.9° |
| Pinky  | MCP  | -26°   | 116°   | 53.6° |
| Pinky  | PIP  | -5°    | 110°   | 29.5° |

*참고: 음수 값은 센서 오차 또는 손가락이 과도하게 펴진 상태를 나타냄*

## 이미지 파일
- 형식: `us_HHMMSS_MMM.png`
- 예: `us_140244_236.png` = 14:02:44.236
- 프레임 간격: 약 130-140ms
- 이미지 크기: 508x380 (width x height)

## 총계
- 총 이미지: 10,433개
- 총 데이터: 10,436행
- 세션: 3개 (sj_1, sj_2, tk)
