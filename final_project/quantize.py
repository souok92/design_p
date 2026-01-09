import random
import torch.multiprocessing
import pickle
import os
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch

from model_resnet50_channelstack import CNN_ResNet50_ChannelStack_DPU # CNN_TCN_Grayscale_DPU factory

from torch.utils.data import DataLoader
from pytorch_nndct.apis import Inspector

print("Device Set....")
device = torch.device('cpu')

window_size = 7
model = CNN_ResNet50_ChannelStack_DPU(window_size=window_size) #, device=device, pretrained=False
checkpoint = torch.load("./checkpoints_dpu/best_model.pth", map_location=device)#, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
#model.load_state_dict(torch.load("./checkpoints_dpu/best_model.pth", map_location=device), strict=False)
model.eval()
print("[OK] Model loaded")

quant_mode = 'test' # 'calib' or 'test'
quant_dir = './build/quantized'

print("Quantizer Initialize...")

from pytorch_nndct.apis import torch_quantizer

# ============================ #
# 여기부터 dummy input 대신
# ./data/251001_image 안의 실제 이미지 사용
# ============================ #
from PIL import Image                 # <<< 추가
import torchvision.transforms as T    # <<< 추가
import glob                           # <<< 추가

img_dir = "./data/251002_wrist"       # <<< 추가

# 이미지 파일 리스트 가져오기 (확장자 여러 개 대응)  <<< 추가
img_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
    img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
img_paths = sorted(img_paths)

if len(img_paths) == 0:
    raise RuntimeError(f"No image files found in {img_dir}")

# 학습 때 썼던 전처리랑 최대한 비슷하게 맞추는 게 중요함  <<< 추가
transform = T.Compose([
    T.Resize((380, 508)),   # 380x508 -> 380x380 (CPU에서 처리)
    T.ToTensor(),           # (1, H, W), [0,1]
    # 학습 때 Normalize 썼으면 여기 추가:
    T.Normalize(mean=[0.5], std=[0.5])
])

# 7프레임(window_size) 쌓기: 일단 간단하게 앞에서부터 7장 사용  <<< 추가
frames = []
for i in range(window_size):
    img_path = img_paths[i % len(img_paths)]   # 이미지가 7장 미만이어도 돌려쓰기
    img = Image.open(img_path).convert('L')    # 그레이스케일
    img_t = transform(img)                     # (1, 380, 380)
    frames.append(img_t)

# (7, 380, 380)으로 쌓고, 배치 차원 추가해서 (1, 7, 380, 380) 만들기  <<< 추가
stack = torch.cat(frames, dim=0)               # (7, 380, 380)
input = stack.unsqueeze(0).to(device)          # (1, 7, 380, 380)

# 기존 dummy input (주석 처리해둠)
# input = torch.randn([1, 7, 380, 380], dtype=torch.float32).to(device)
# ============================ #
# 여기까지가 "이미지 넣는 부분" 수정 끝
# ============================ #

print("Quantize model...")

quantizer = torch_quantizer(quant_mode, model, (input), output_dir = quant_dir, device=device)
quant_model = quantizer.quant_model.to(device)

_ = quant_model(input)

if quant_mode == 'calib':
     quantizer.export_quant_config()
if quant_mode == 'test':
     quantizer.export_xmodel(deploy_check=False, output_dir=quant_dir)

