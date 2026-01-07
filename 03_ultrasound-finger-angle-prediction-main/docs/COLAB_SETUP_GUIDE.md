# Google Colab 사용 가이드 🚀

CNN-TCN 모델을 Google Colab에서 무료 GPU로 학습하는 방법입니다.

---

## 📋 준비물

1. ✅ Google 계정
2. ✅ 데이터 파일들 (압축하여 업로드)
3. ✅ Colab 노트북 파일

---

## 1️⃣ 데이터 압축하기

### 방법 1: 필요한 파일만 압축 (권장)

```bash
cd /home/kimsj/delto_m_ros2/model

# data.zip 생성 (이미지 + CSV 파일들)
zip -r data.zip \
  data/251002_wrist \
  data/train.csv \
  data/val.csv \
  data/test.csv
```

압축 후 크기 확인:
```bash
ls -lh data.zip
# 예상 크기: 약 100-300MB
```

### 방법 2: Google Drive 업로드 (큰 데이터셋용)

```bash
# data 폴더 전체를 Google Drive에 업로드
# 브라우저에서 Google Drive 접속 → 폴더 업로드
```

---

## 2️⃣ Colab에서 노트북 열기

### 방법 A: 로컬에서 업로드

1. **Colab 접속**: https://colab.research.google.com/
2. **파일 → 노트북 업로드**
3. `Ultrasound_Angle_Prediction_Colab.ipynb` 선택

### 방법 B: GitHub에서 직접 열기 (추천)

1. GitHub에 노트북 업로드
2. Colab에서 **GitHub 탭 → Repository URL 입력**
3. 노트북 선택

---

## 3️⃣ GPU 활성화

**중요!** GPU를 반드시 활성화해야 빠른 학습이 가능합니다.

```
런타임(Runtime) → 런타임 유형 변경(Change runtime type)
→ 하드웨어 가속기(Hardware accelerator): GPU (T4)
→ 저장(Save)
```

확인:
```python
!nvidia-smi
```

출력에서 **Tesla T4** 또는 다른 GPU 이름이 보이면 성공!

---

## 4️⃣ 데이터 업로드

### 옵션 A: 직접 업로드 (간단, 느림)

노트북의 **Section 3** 실행:
```python
from google.colab import files
uploaded = files.upload()  # data.zip 선택
```

### 옵션 B: Google Drive 마운트 (빠름, 권장)

1. **Google Drive에 데이터 업로드** (미리 완료)

2. **Section 2** 실행:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. 경로 설정 (Section 6):
```python
DATA_DIR = '/content/drive/MyDrive/ultrasound_data'
IMAGE_DIR = '/content/drive/MyDrive/ultrasound_data/251002_wrist'
```

---

## 5️⃣ 학습 시작

### 순서대로 실행:

1. ✅ **Section 1**: GPU 확인
2. ✅ **Section 2**: (선택) Drive 마운트
3. ✅ **Section 3**: 데이터 업로드/설정
4. ✅ **Section 4-6**: 모델, 데이터셋, 설정
5. ✅ **Section 7**: 모델 초기화
6. ✅ **Section 8**: 🚀 **학습 시작!**

### 예상 학습 시간 (GPU T4):

- **Per Epoch**: 1-3분
- **50 Epochs**: 약 **1-2.5시간**

---

## 6️⃣ 학습 모니터링

### 실시간 진행 상황:

```
Epoch 1/50
----------------------------------------------------------------------
Training: 100%|██████████| 178/178 [01:23<00:00]  loss: 5.23
Validation: 100%|██████████| 23/23 [00:12<00:00]  loss: 4.89
Train Loss: 5.2341
Val Loss: 4.8912, Val MAE: 4.89°
MCP MAE: 4.23°, PIP MAE: 5.55°
✓ Best model saved! (Val Loss: 4.8912)
```

### 학습 곡선 (Section 9):

- Loss 그래프 자동 생성
- MCP/PIP MAE 그래프
- `training_history.png` 저장

---

## 7️⃣ 학습 완료 후

### 모델 다운로드 (Section 11):

```python
from google.colab import files
files.download('best_model.pth')  # 학습된 모델
files.download('training_history.png')  # 학습 그래프
```

### Test Set 평가 (Section 10):

```
TEST SET RESULTS
======================================================================
Test Loss: 4.5234
Overall MAE: 4.52°
MCP MAE: 4.01°
PIP MAE: 5.03°
======================================================================
```

---

## 8️⃣ 주의사항 ⚠️

### Colab 세션 제한:

- **무료 사용자**: 최대 12시간 연속 사용
- **세션 끊김 방지**:
  - 주기적으로 브라우저 확인
  - 중간 체크포인트 저장 (10 epoch마다)

### 메모리 부족 시:

```python
# CONFIG 수정 (Section 6)
CONFIG = {
    'batch_size': 8,  # 16 → 8로 감소
    'window_size': 5,  # 7 → 5로 감소
}
```

### 재시작 시:

- **Section 1-7** 다시 실행
- 체크포인트에서 재개 가능

---

## 9️⃣ 결과 해석

### 좋은 성능 기준:

- ✅ **Overall MAE < 5°**: 우수
- ✅ **MCP MAE < 5°**: 좋음
- ✅ **PIP MAE < 5°**: 좋음
- ⚠️ **MAE > 10°**: 개선 필요

### 개선 방법:

1. **더 많은 Epoch** (50 → 100)
2. **Learning Rate 조정** (1e-4 → 5e-5)
3. **Window Size 실험** (5, 7, 10)
4. **데이터 증강 강화**

---

## 🔟 트러블슈팅

### 문제 1: GPU가 할당되지 않음
```
해결: 런타임 → 런타임 유형 변경 → GPU 선택
```

### 문제 2: 메모리 부족 (OOM)
```
해결: batch_size를 8 또는 4로 감소
```

### 문제 3: 데이터 로딩 오류
```
해결:
1. DATA_DIR, IMAGE_DIR 경로 확인
2. ls 명령어로 파일 존재 확인
3. CSV 파일 헤더 확인
```

### 문제 4: 세션 끊김
```
해결:
1. 체크포인트 저장 활성화
2. Google Drive에 자동 저장 설정
3. Colab Pro 고려 (유료)
```

---

## 📊 예상 성능

### 학습 후 기대 결과:

| Metric | Expected Value |
|--------|---------------|
| Training MAE | 3-4° |
| Validation MAE | 4-5° |
| Test MAE | 4-6° |
| MCP MAE | 3-5° |
| PIP MAE | 4-6° |

### 학습 속도:

- **GPU (T4)**: ~2분/epoch → **총 1.5-2시간**
- **CPU (로컬)**: ~20분/epoch → **총 16-20시간**

---

## 📁 파일 구조

업로드할 `data.zip` 내용:
```
data/
├── 251002_wrist/
│   ├── ultrasound_frame_dsc_20251002_153224_869.png
│   ├── ultrasound_frame_dsc_20251002_153225_033.png
│   └── ... (3,555 images)
├── train.csv        # 2,844 samples
├── val.csv          # 355 samples
└── test.csv         # 356 samples
```

---

## 🎯 Quick Start Checklist

- [ ] 1. 데이터 압축 완료 (`data.zip`)
- [ ] 2. Colab 노트북 열기
- [ ] 3. GPU 활성화 확인
- [ ] 4. 데이터 업로드 (Drive 또는 직접)
- [ ] 5. Section 1-7 순서대로 실행
- [ ] 6. Section 8 학습 시작
- [ ] 7. 1-2시간 대기
- [ ] 8. Section 9-10 결과 확인
- [ ] 9. Section 11 모델 다운로드
- [ ] 10. 로컬에 저장 및 평가

---

## 💡 추가 팁

1. **중간 저장**: 10 epoch마다 자동 저장됨
2. **Early Stopping**: Validation loss가 5 epoch 동안 개선 안 되면 LR 감소
3. **TensorBoard**: Colab에서 지원 (선택사항)
4. **병렬 실험**: 여러 설정을 각각 다른 노트북에서 실행

---

## 📞 도움말

문제 발생 시:
1. 에러 메시지 확인
2. 경로 및 파일 존재 여부 확인
3. GPU 메모리 확인 (`!nvidia-smi`)
4. 배치 사이즈 줄이기

---

**행운을 빕니다! 🚀**

학습 완료 후 `best_model.pth` 파일을 로컬로 다운로드하여 추론에 사용하세요!
