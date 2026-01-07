# 3D CNN (R3D-18) for Finger Angle Prediction

초음파 이미지 시퀀스로부터 손가락 각도를 예측하는 3D CNN 모델입니다.

## 모델 구조

### R3D-18 (3D ResNet)
- **Backbone**: ResNet-18의 3D 버전
- **Pretrained**: Kinetics-400 데이터셋 (비디오 동작 인식)
- **특징**: 시공간(spatiotemporal) 특징을 동시에 학습

### 왜 3D CNN인가?
1. **시공간 동시 학습**: 2D CNN + LSTM은 공간과 시간을 따로 처리하지만, 3D CNN은 한 번에 처리
2. **모션 패턴 포착**: 컨볼루션 필터가 프레임 간 움직임을 직접 학습
3. **Pretrained 활용**: Kinetics-400으로 학습된 가중치가 일반적인 움직임 패턴을 이미 알고 있음

## 폴더 구조

```
11_3dcnn/
├── configs/
│   ├── config_tk.yaml
│   └── config_sj.yaml
├── data/
│   └── dataset.py
├── models/
│   └── r3d.py
├── train.py
├── evaluate.py
├── checkpoints/
├── logs/
├── results/
└── README.md
```

## 사용법

### 1. 학습

```bash
# TK 데이터로 학습
python train.py --config configs/config_tk.yaml

# SJ 데이터로 학습
python train.py --config configs/config_sj.yaml

# 실험 이름 지정
python train.py --config configs/config_tk.yaml --exp_name my_experiment

# 체크포인트에서 재개
python train.py --config configs/config_tk.yaml --resume checkpoints/tk/best_model.pth
```

### 2. 평가

```bash
# 체크포인트로 평가
python evaluate.py --checkpoint results/tk/tk_YYYYMMDD_HHMMSS/model.pth

# 결과 저장 경로 지정
python evaluate.py --checkpoint checkpoints/tk/best_model.pth --output_dir ./my_results
```

### 3. TensorBoard로 학습 모니터링

```bash
tensorboard --logdir logs/
```

## 설정 파라미터

### Data
- `sequence_length`: 입력 시퀀스 길이 (기본: 16)
- `stride`: 프레임 샘플링 간격 (기본: 1)
- `batch_size`: 배치 크기 (3D CNN은 메모리 많이 사용, 기본: 8)
- `augment`: 데이터 증강 여부
- `image_size`: 입력 이미지 크기 (기본: 224, 디테일 보존 위해 112 대신 224 사용)

### Model
- `model_type`: "r3d" 또는 "r3d_multiscale"
- `dropout`: 드롭아웃 비율
- `pretrained`: Kinetics-400 사전학습 가중치 사용

### Training
- `freeze_backbone_epochs`: 초기 N 에폭 동안 backbone freeze (전이학습 전략)
- `learning_rate`: 학습률 (pretrained 모델은 낮게, 기본: 0.0001)
- `early_stopping_patience`: 조기 종료 patience

## ⚠️ 학습 속도 변화 안내

학습 중 **특정 에폭 이후 속도가 급격히 느려지는 것은 정상**입니다.

### 원인: `freeze_backbone_epochs` 설정

| 구간 | 상태 | 학습 파라미터 | 속도 |
|------|------|--------------|------|
| Epoch 1~5 | Backbone frozen | ~26만개 (regressor만) | 빠름 (~8 it/s) |
| Epoch 6~ | Backbone unfrozen | ~3,300만개 (전체) | 느림 (~2.6 it/s) |

### 왜 이렇게 하나요?

**전이학습 전략**입니다:
1. 처음에는 pretrained backbone을 고정하고 regression head만 학습
2. Head가 어느 정도 수렴하면 backbone도 함께 fine-tuning
3. 이렇게 하면 pretrained 가중치가 망가지지 않고 안정적으로 학습됨

### 속도가 너무 느리다면?

```yaml
# config에서 조절 가능
training:
  freeze_backbone_epochs: 0  # 처음부터 전체 학습 (느리지만 일정)
  # 또는
  freeze_backbone_epochs: 10  # 더 오래 빠르게 학습 후 fine-tuning
```

## 출력

학습 완료 후 `results/{person}/{exp_name}/` 폴더에:
- `experiment_config.yaml`: 실험 설정
- `experiment_info.json`: 실험 결과 요약
- `metrics.txt`: 각도별 MAE, RMSE, MSE
- `predictions.npz`: 예측값과 실제값
- `scatter_plots.png`: 예측 vs 실제 산점도
- `error_distribution.png`: 오차 분포 히스토그램
- `time_series.png`: 시계열 비교

## 기존 CNN+LSTM 대비 차이점

| 항목 | CNN+LSTM | 3D CNN |
|------|----------|--------|
| 입력 크기 | 224x224 | 112x112 |
| 시퀀스 처리 | 각 프레임 독립 → LSTM | 3D 컨볼루션으로 동시 처리 |
| Pretrained | ImageNet (이미지) | Kinetics-400 (비디오) |
| 메모리 | 상대적으로 적음 | 많음 (batch 줄여야 함) |
| 모션 학습 | LSTM이 암시적으로 | 3D conv가 명시적으로 |

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
pandas
matplotlib
pyyaml
tqdm
tensorboard
```
