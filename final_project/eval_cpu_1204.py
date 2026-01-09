#!/usr/bin/env python3
"""
CPU Baseline Evaluation
- best_model.pth 기반 동일 조건 inference 수행
- train.csv 사용하여 CPU avg latency per window / throughput 측정
"""

import os
import csv
import time
import numpy as np
import torch
import cv2

from model_resnet50_channelstack import CNN_ResNet50_ChannelStack_DPU

# ==============================
# 경로 설정
# ==============================
PROJECT_DIR = "/home/mics1234/Vitis-AI/my_project_12031710"  # 필요 시 수정
CSV_PATH    = os.path.join(PROJECT_DIR, "data/test.csv")
IMG_DIR     = os.path.join(PROJECT_DIR, "data/251002_wrist")
CKPT_PATH   = os.path.join(PROJECT_DIR, "checkpoints_dpu/best_model.pth")


WINDOW_SIZE = 7               # 채널 스택 크기
IMG_H, IMG_W = 380, 508       # 학습 시 resize 사용 크기
MEAN, STD = 0.5, 0.5          # Normalize 동일 유지

# ==============================
# 이미지 전처리
# ==============================
def preprocess_frame_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img  # (H,W)

# ==============================
# Main
# ==============================
def main():

    # -----------------------------
    # 1) 모델 로드 (CPU 강제)
    # -----------------------------
    device = torch.device("cpu")
    model = CNN_ResNet50_ChannelStack_DPU(window_size=WINDOW_SIZE)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("\n=====================================")
    print("CPU BASELINE EVALUATION START")
    print("Model:", CKPT_PATH)
    print("CSV:", CSV_PATH)
    print("=====================================\n")

    # -----------------------------
    # 2) CSV 로드
    # -----------------------------
    with open(CSV_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    preds = []
    gts   = []

    total_infer_time = 0.0
    total_samples    = 0

    # -----------------------------
    # 3) 7프레임 슬라이딩 윈도우 inference
    # -----------------------------
    for idx in range(WINDOW_SIZE - 1, len(rows)):
        start = idx - WINDOW_SIZE + 1

        frames = []
        for i in range(start, idx + 1):
            img_name = rows[i]["image_filename"]
            img_path = os.path.join(IMG_DIR, img_name)
            frame = preprocess_frame_gray(img_path)
            frames.append(frame)

        volume = np.stack(frames, axis=0)              # (7,H,W)
        volume = torch.from_numpy(volume).unsqueeze(0) # (1,7,H,W)
        volume = volume.to(device)

        # -----------------------------
        # 추론시간 측정
        # -----------------------------
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(volume)
        t1 = time.perf_counter()

        infer_time = t1 - t0
        total_infer_time += infer_time
        total_samples += 1

        pred = output.cpu().numpy().reshape(-1)
        preds.append(pred)

        gts.append([
            float(rows[idx]["MCP_Flex"]),
            float(rows[idx]["PIP_Flex"])
        ])


    # -----------------------------
    # 4) MAE 계산
    # -----------------------------
    preds = np.array(preds)
    gts   = np.array(gts)

    abs_err = np.abs(preds - gts)
    mae_mcp = abs_err[:,0].mean()
    mae_pip = abs_err[:,1].mean()
    mae_all = abs_err.mean()

    # -----------------------------
    # 5) 성능 결과
    # -----------------------------
    avg_latency_ms = (total_infer_time / total_samples) * 1000
    throughput     = total_samples / total_infer_time

    print("\n========== CPU BASELINE RESULT ==========")
    print(f"CPU MAE (MCP): {mae_mcp:.4f} deg")
    print(f"CPU MAE (PIP): {mae_pip:.4f} deg")
    print(f"CPU MAE (ALL): {mae_all:.4f} deg")
    print("------------------------------------------")
    print(f"Avg latency per window: {avg_latency_ms:.3f} ms")
    print(f"Throughput            : {throughput:.2f} window/s")
    print("==========================================\n")

if __name__ == "__main__":
    main()
