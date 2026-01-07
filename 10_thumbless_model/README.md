# Thumbless Finger Angle Prediction Model

CNN + LSTM 기반 손가락 각도 예측 모델 (Thumb 제외 8개 각도)

## 프로젝트 구조

```
10_thumbless_model/
├── models/
│   └── cnn_lstm.py          # CNN+LSTM 모델 아키텍처
├── data/
│   └── dataset.py           # 데이터셋 및 데이터로더
├── configs/
│   ├── config_sj.yaml       # SJ 학습 설정
│   └── config_tk.yaml       # TK 학습 설정
├── checkpoints/             # 학습된 모델 저장
├── logs/                    # TensorBoard 로그
├── train.py                 # 학습 스크립트
├── evaluate.py              # 평가 스크립트
├── run_training.py          # 학습 + TensorBoard 통합 실행
├── tensorboard_monitor.py   # TensorBoard 단독 실행
└── README.md
```

## 모델 아키텍처

```
Input: Image Sequence (batch, seq_len=5, 3, 224, 224)
  ↓
ResNet18 (pretrained, shared weights)
  ↓
Features: (batch, seq_len, 512)
  ↓
LSTM (2 layers, hidden=256)
  ↓
FC Layers (256 → 128 → 8)
  ↓
Output: 8 Finger Angles
```

## 데이터 분리

### SJ 모델
- **Train**: sj_2 전체 (4,516 프레임)
- **Val**: sj_1 앞 50% (850 프레임)
- **Test**: sj_1 뒤 50% (851 프레임)

### TK 모델
- **Train**: 처음 70% (2,951 프레임)
- **Val**: 다음 15% (632 프레임)
- **Test**: 마지막 15% (633 프레임)

## 필수 패키지

```bash
pip install torch torchvision
pip install numpy pandas pillow
pip install matplotlib tensorboard tqdm pyyaml
```

## 사용법

### 1. 학습

#### 방법 1: 학습 + TensorBoard 통합 실행 (추천) ⭐
학습과 실시간 TensorBoard 모니터링을 한 번에 시작합니다.

```bash
cd 10_thumbless_model

# SJ 모델
python run_training.py --config configs/config_sj.yaml

# TK 모델
python run_training.py --config configs/config_tk.yaml
```

이 명령어는:
1. TensorBoard 서버를 백그라운드에서 시작
2. 자동으로 브라우저를 열어 http://localhost:6006 접속
3. 학습 시작
4. 학습 진행 상황을 실시간으로 모니터링

#### 방법 2: 학습만 실행
```bash
# SJ 모델 학습
python train.py --config configs/config_sj.yaml

# TK 모델 학습
python train.py --config configs/config_tk.yaml
```

### 2. TensorBoard 실시간 모니터링

#### 학습 중에 TensorBoard만 따로 실행
```bash
# 특정 person 모니터링
python tensorboard_monitor.py --person sj
python tensorboard_monitor.py --person tk

# 모든 로그 모니터링
python tensorboard_monitor.py --person all

# 포트 변경
python tensorboard_monitor.py --person sj --port 6007

# 브라우저 자동 열기 비활성화
python tensorboard_monitor.py --person sj --no-browser
```

#### 기본 TensorBoard 명령어
```bash
tensorboard --logdir logs
```

### 3. 평가

평가 결과는 **자동으로 타임스탬프가 붙은 폴더**에 저장되어 여러 실험 결과를 비교할 수 있습니다.

#### 기본 평가 (타임스탬프 자동 생성)
```bash
# SJ 모델 평가 - results/sj/sj_20250124_153045/ 형식으로 저장
python evaluate.py --config configs/config_sj.yaml \
                   --checkpoint checkpoints/sj/best_model.pth

# TK 모델 평가
python evaluate.py --config configs/config_tk.yaml \
                   --checkpoint checkpoints/tk/best_model.pth
```

#### 커스텀 실험 이름 사용
```bash
# 실험 이름 직접 지정
python evaluate.py --config configs/config_sj.yaml \
                   --checkpoint checkpoints/sj/best_model.pth \
                   --exp_name "seq5_lr0001_mae"
```

#### 여러 실험 결과 비교
```bash
# 모든 실험 비교
python compare_results.py --person all

# 특정 person만 비교
python compare_results.py --person sj
```

### 3. TensorBoard 모니터링

```bash
tensorboard --logdir logs
```

## TensorBoard에서 볼 수 있는 내용

실시간으로 다음 정보를 모니터링할 수 있습니다:

- **Loss/train**: 학습 손실 (매 epoch)
- **Loss/val**: 검증 손실 (매 epoch)
- **Learning_Rate**: 학습률 변화
- **MAE/{angle_name}**: 각도별 MAE (8개 각도)
- **RMSE/{angle_name}**: 각도별 RMSE (8개 각도)

## 출력 결과

### 학습 결과
- `checkpoints/{person}/best_model.pth`: 가장 좋은 성능의 모델
- `checkpoints/{person}/last_model.pth`: 마지막 epoch 모델
- `logs/{person}/`: TensorBoard 로그

### 평가 결과 (실험별 폴더 구조)

각 실험은 `results/{person}/{exp_name}/` 폴더에 저장됩니다:

```
results/
├── sj/
│   ├── sj_20250124_153045/
│   │   ├── experiment_info.json      # 실험 조건 및 결과 요약
│   │   ├── experiment_config.yaml    # 사용한 전체 설정
│   │   ├── model.pth                 # 학습된 모델 (재현용) ⭐
│   │   ├── metrics.txt               # 평가 지표 (MAE, RMSE, MSE)
│   │   ├── scatter_plots.png         # 예측 vs 실제 산점도
│   │   ├── time_series.png           # 시간에 따른 예측 추이
│   │   ├── error_distribution.png    # 오차 분포
│   │   └── predictions.npz           # 예측값 및 실제값
│   └── sj_20250124_180230/
│       └── ...
└── tk/
    └── ...
```

**주요 파일 설명:**
- `model.pth`: 학습된 모델 가중치 - 이 파일로 결과 재현 가능
- `experiment_info.json`: 실험 설정 및 성능 요약
- `experiment_config.yaml`: 전체 하이퍼파라미터 설정

**experiment_info.json 예시:**
```json
{
  "experiment_name": "sj_20250124_153045",
  "timestamp": "2025-01-24 15:30:45",
  "person": "sj",
  "model_config": {
    "sequence_length": 5,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "dropout": 0.3
  },
  "training_config": {
    "learning_rate": 0.0001,
    "optimizer": "Adam",
    "loss_type": "MAE"
  },
  "results": {
    "overall_mae": 3.1256,
    "overall_rmse": 4.0123,
    "overall_mse": 16.0986
  }
}
```

## 하이퍼파라미터

주요 하이퍼파라미터는 `configs/*.yaml` 파일에서 수정 가능:

- `sequence_length`: 시퀀스 길이 (기본: 5)
- `batch_size`: 배치 크기 (기본: 16)
- `learning_rate`: 학습률 (기본: 0.0001)
- `lstm_hidden_size`: LSTM 은닉층 크기 (기본: 256)
- `lstm_num_layers`: LSTM 레이어 수 (기본: 2)
- `num_epochs`: 학습 에포크 (기본: 100)
- `early_stopping_patience`: Early stopping patience (기본: 20)

## 예측 각도

모델은 다음 8개 각도를 예측합니다:

1. Index_MCP_Flex
2. Index_PIP_Flex
3. Middle_MCP_Flex
4. Middle_PIP_Flex
5. Ring_MCP_Flex
6. Ring_PIP_Flex
7. Pinky_MCP_Flex
8. Pinky_PIP_Flex

## 참고사항

- GPU 사용 권장 (CUDA 사용 가능시 자동으로 GPU 사용)
- 슬라이딩 윈도우 방식으로 시간 정보 활용
- ResNet18 pretrained weights 사용으로 빠른 수렴
- Early stopping으로 overfitting 방지
