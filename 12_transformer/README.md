# TimeSformer for Finger Angle Prediction

초음파 이미지 시퀀스로부터 손가락 각도를 예측하는 TimeSformer 모델입니다.

## 모델 구조

### TimeSformer (Divided Space-Time Attention)
- **Temporal Attention**: 같은 공간 위치의 프레임들 간 attention
- **Spatial Attention**: 같은 시간의 패치들 간 attention
- **장점**: 시간/공간 분리로 효율적, 중소규모 데이터에서도 잘 동작

### 모델 크기
| 모델 | dim | depth | heads | 파라미터 |
|------|-----|-------|-------|----------|
| timesformer | 768 | 12 | 12 | ~121M |
| timesformer_small | 384 | 6 | 6 | ~22M |
| timesformer_tiny | 192 | 4 | 3 | ~5M |

## 핵심 설정: stride

프레임 간격을 조절하여 시간 해상도와 커버 범위를 조절합니다.

| stride | 프레임 간격 | 8프레임 커버 시간 |
|--------|-----------|------------------|
| 1 | ~130ms | ~1초 |
| **2** | **~260ms** | **~2초** (기본값) |
| 3 | ~390ms | ~3초 |

**stride=2 선택 이유:**
- 프레임 간 변화가 더 뚜렷해짐 (비슷한 이미지 연속 문제 해결)
- 1초 정도의 적당한 딜레이
- 손가락 움직임 패턴 포착에 적합

## 폴더 구조

```
12_transformer/
├── configs/
│   ├── config_tk.yaml
│   └── config_sj.yaml
├── data/
│   └── dataset.py       # 강화된 augmentation 포함
├── models/
│   └── timesformer.py
├── train.py
├── evaluate.py
├── checkpoints/
├── logs/
└── results/
```

## 사용법

### 1. 학습

```bash
cd C:\Users\suogi\Desktop\design_project\12_transformer

# TK 데이터로 학습
python train.py --config configs/config_tk.yaml

# SJ 데이터로 학습
python train.py --config configs/config_sj.yaml

# 실험 이름 지정
python train.py --config configs/config_tk.yaml --exp_name my_experiment
```

### 2. 평가

```bash
python evaluate.py --checkpoint results/tk/{실험명}/best_model.pth
```

### 3. TensorBoard

```bash
tensorboard --logdir logs/
```

## 설정 파라미터

### Data
- `sequence_length`: 프레임 수 (기본: 8)
- `stride`: 프레임 샘플링 간격 (기본: 2)
- `augment`: 데이터 증강 여부
- `image_size`: 이미지 크기 (기본: 224)

### Model
- `model_type`: "timesformer", "timesformer_small", "timesformer_tiny"
- `patch_size`: 패치 크기 (기본: 16)
- `dropout`: 드롭아웃 비율

### Training
- `warmup_epochs`: Learning rate warmup 기간
- `scheduler`: CosineAnnealingWarmRestarts 사용 (주기적 재시작)

## Data Augmentation

강화된 augmentation 적용:
- RandomCrop
- GaussianBlur (p=0.2)
- ColorJitter (brightness, contrast, saturation, hue)
- GaussianNoise
- RandomErasing (p=0.1)

시간축 augmentation은 제외 (프레임 순서가 중요하므로)

## 3D CNN vs TimeSformer

| 항목 | 3D CNN (R3D) | TimeSformer |
|------|--------------|-------------|
| 시간 모델링 | 3D Conv | Self-attention |
| 장거리 의존성 | 제한적 | 전역적 |
| Pretrained | Kinetics-400 | 없음 (scratch) |
| 메모리 | 중간 | 높음 |
| 해석성 | 낮음 | Attention 시각화 가능 |

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
einops>=0.4.0  # 필수!
numpy
pandas
matplotlib
pyyaml
tqdm
tensorboard
```

### einops 설치

```bash
pip install einops
```
