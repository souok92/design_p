# U-Net + GRU for Finger Angle Prediction

초음파 이미지 시퀀스로부터 손가락 각도를 예측하는 U-Net Encoder + GRU 모델입니다.

## 모델 구조

```
[이미지 시퀀스] (B, T, 3, 224, 224)
       ↓
  U-Net Encoder (각 프레임 독립 처리)
       ↓
  Global Average Pooling
       ↓
  Feature Projection (1024 → 512)
       ↓
  GRU Layer 1 (hidden=256)
       ↓
  GRU Layer 2 (hidden=256)
       ↓
  FC Regressor → 8개 각도
```

### 참고 논문
**"Improved fascicle length estimates from ultrasound using a U-net-LSTM framework"**
- Harvard University, 2024
- U-Net + LSTM x 2 구조로 MAE 0.2mm 달성 (기존 대비 10배 개선)

### 왜 이 구조인가?

1. **U-Net Encoder**: Skip connection 없이 encoder만 사용하여 공간 특징 추출
2. **GRU x 2**: LSTM과 비슷한 성능, 더 적은 파라미터
3. **연속 프레임 (stride=1)**: 프레임 간 연속성이 중요

## 모델 종류

| 모델 | base_features | encoder_depth | gru_hidden | 파라미터 |
|------|---------------|---------------|------------|----------|
| unet_gru | 64 | 4 | 256 | ~11M |
| unet_gru_light | 32 | 3 | 128 | ~1.5M |

## 핵심 설정

### stride=1 (중요!)

논문에서 효과를 본 핵심 설정입니다. 연속된 프레임을 사용해야 시간적 연속성을 학습할 수 있습니다.

| stride | 프레임 간격 | 특징 |
|--------|-----------|------|
| **1** | ~130ms | 연속 프레임, 미세한 변화 포착 (권장) |
| 2 | ~260ms | 변화가 뚜렷하지만 연속성 손실 |

### sequence_length=10

10프레임 × 130ms = 약 1.3초 커버

## 폴더 구조

```
13_unet_gru/
├── configs/
│   ├── config_tk.yaml
│   └── config_sj.yaml
├── data/
│   └── dataset.py
├── models/
│   └── unet_gru.py
├── train.py
├── evaluate.py
├── checkpoints/
├── logs/
└── results/
```

## 사용법

### 1. 학습

```bash
cd C:\Users\suogi\Desktop\design_project\13_unet_gru

# TK 데이터로 학습
python train.py --config configs/config_tk.yaml

# SJ 데이터로 학습
python train.py --config configs/config_sj.yaml
```

### 2. 평가

```bash
python evaluate.py --checkpoint checkpoints/tk/{exp_name}/best_model.pth
```

### 3. TensorBoard

```bash
tensorboard --logdir logs/
```

## 기존 모델과 비교

| 모델 | 구조 | 특징 |
|------|------|------|
| CNN+LSTM | ResNet18 + LSTM | Pretrained encoder, 시간 분리 |
| 3D CNN | R3D-18 | 시공간 동시 학습 |
| TimeSformer | ViT + Temporal Attention | 전역 attention |
| **U-Net+GRU** | **U-Net Encoder + GRU x 2** | **공간/시간 특화, 논문 검증** |

## 설정 파라미터

### Model
- `base_features`: U-Net 기본 채널 (64)
- `encoder_depth`: U-Net encoder 깊이 (4)
- `gru_hidden_size`: GRU hidden size (256)
- `gru_num_layers`: GRU 레이어 수 (2)

### Data
- `sequence_length`: 프레임 수 (10)
- `stride`: 프레임 간격 (1, 연속)
- `image_size`: 이미지 크기 (224)

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
