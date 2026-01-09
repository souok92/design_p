#!/usr/bin/env python3
"""
eval_dpu_mae.py

ResNet50 Channel-Stack DPU xmodel 평가 스크립트
- 입력: 7장 그레이스케일 프레임을 채널 방향으로 쌓은 텐서 (1, 7, H, W)
- 출력: (1, 2) = [MCP_angle, PIP_angle]
- 지표: MCP/PIP 각각 MAE, 전체 평균 MAE
"""

import os
import csv
import numpy as np
import cv2
import xir
import vart
import time

# ==============================
# 경로 설정 (필요시 수정)
# ==============================
PROJECT_DIR = "/home/petalinux/my_project"

# xmodel 파일 이름은 네가 실제 사용 중인 이름으로 수정하면 됨
# 예: "u96v2_CNN_ResNet50_ChannelStack_DPU_int.xmodel"
XMODEL_NAME = "kr260_CNN_ResNet50_ChannelStack_DPU_int.xmodel.xmodel"
XMODEL_PATH = os.path.join(PROJECT_DIR, XMODEL_NAME)

CSV_PATH = os.path.join(PROJECT_DIR, "data", "test.csv")
IMG_DIR  = os.path.join(PROJECT_DIR, "data", "251002_wrist")

# ==============================
# 모델 / 데이터 설정
# ==============================
WINDOW_SIZE = 7

# 학습에 사용한 이미지 크기로 맞춰야 함
# 여기서는 예시로 380x508 사용 (네가 실제 학습/quant 때 쓴 사이즈로 수정)
IMG_H = 380
IMG_W = 508

# 그레이스케일 정규화
# 학습 때 [0,1] -> [-1,1] 썼다면 아래처럼 유지
MEAN = 0.5
STD  = 0.5


# ==============================
# 유틸 함수
# ==============================
def preprocess_frame_gray(path):
    """
    1) 그레이스케일 로드
    2) (IMG_W, IMG_H)로 리사이즈
    3) [0,255] -> [0,1]
    4) 정규화: (x - MEAN) / STD
    5) (H, W) float32 배열 반환
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    img = cv2.resize(img, (IMG_W, IMG_H))  # (W, H)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img  # (H, W)


def load_graph(path):
    return xir.Graph.deserialize(path)


def get_dpu_runner(graph):
    """
    Graph에서 DPU 서브그래프를 찾아서 Runner 생성
    """
    root = graph.get_root_subgraph()
    # 루트가 곧 DPU인 경우
    if root.has_attr("device") and root.get_attr("device").upper() == "DPU":
        dpu_subgraph = root
    else:
        dpu_subgraph = None
        for sg in root.get_children():
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                dpu_subgraph = sg
                break

    if dpu_subgraph is None:
        raise RuntimeError("No DPU subgraph found in the xmodel graph.")

    return vart.Runner.create_runner(dpu_subgraph, "run")


# ==============================
# 메인 평가 루프
# ==============================
def main():
    # ----- CSV 로드 -----
    with open(CSV_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    print(f"[INFO] Loaded {len(rows)} entries from CSV")

    # ----- xmodel 로드 -----
    print(f"[INFO] Loading xmodel from: {XMODEL_PATH}")
    graph = load_graph(XMODEL_PATH)
    runner = get_dpu_runner(graph)

    input_tensor  = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]

    input_shape  = tuple(input_tensor.dims)
    output_shape = tuple(output_tensor.dims)

    print(f"[INFO] Model input shape : {input_shape}")
    print(f"[INFO] Model output shape: {output_shape}")

    ################################# 김상진 수정
    print(f"[INFO] Model output shape: {output_shape}")

    # Input Tensor Quantization 정보 확인
    print("\n========== Input Tensor Info ==========")
    print(f"Input tensor name: {input_tensor.name}")
    print(f"Input tensor dtype: {input_tensor.dtype}")

    if input_tensor.has_attr("fix_point"):
        input_fix_point = input_tensor.get_attr("fix_point")
        input_scale = 1.0 / (2 ** input_fix_point)
        print(f"Input fix_point: {input_fix_point}")
        print(f"Input scale: {input_scale}")
    else:
        print("No input fix_point found")

    # Output Tensor Quantization 정보 확인
    print("\n========== Output Tensor Info ==========")
    print(f"Output tensor name: {output_tensor.name}")
    print(f"Output tensor dtype: {output_tensor.dtype}")

    if output_tensor.has_attr("fix_point"):
        output_fix_point = output_tensor.get_attr("fix_point")
        output_scale = 1.0 / (2 ** output_fix_point)
        print(f"Output fix_point: {output_fix_point}")
        print(f"Output scale: {output_scale}")
    else:
        print("No output fix_point found")

    print("=======================================\n")

    ##################################



    if len(input_shape) != 4:
        raise RuntimeError(f"Expected 4D input (NHWC), got {len(input_shape)}D: {input_shape}")

    batch, in_h, in_w, in_channels = input_shape  # NHWC

    if batch != 1:
        print(f"[WARN] batch dimension is {batch}, script assumes batch=1.")
    if in_channels != WINDOW_SIZE:
        print(f"[WARN] channels (last dim) = {in_channels}, WINDOW_SIZE = {WINDOW_SIZE}")

    # xmodel이 기대하는 H,W를 그대로 사용
    IMG_H = in_h
    IMG_W = in_w
    print(f"[INFO] Using IMG_H={IMG_H}, IMG_W={IMG_W} based on xmodel dims")

    if in_h != IMG_H or in_w != IMG_W:
        print(f"[WARN] xmodel expects H,W=({in_h},{in_w}) but script uses ({IMG_H},{IMG_W}).")
        print("      → IMG_H, IMG_W를 xmodel에 맞게 수정해야 MAE가 정확해짐.")

    preds = []
    gts   = []
    
    total_infer_time = 0.0  # 초 단위
    num_runs = 0

    # ----- 슬라이딩 윈도우 (7프레임씩) -----

    for idx in range(WINDOW_SIZE - 1, len(rows)):
        start = idx - WINDOW_SIZE + 1

        # 1) 프레임 7장 로드 & 전처리 → (H,W) 7개
        frames = []
        for i in range(start, idx + 1):
            img_name = rows[i]["image_filename"]
            img_path = os.path.join(IMG_DIR, img_name)
            frame = preprocess_frame_gray(img_path)  # (H, W)
            if frame is None:
                print("NOONENENENE")
            frames.append(frame)

        # 2) (H, W, 7) 로 스택 (채널 last)
        volume = np.stack(frames, axis=-1).astype(np.float32)  # (H, W, WINDOW_SIZE)

        # 3) DPU 입력 버퍼 준비: (N, H, W, C) = (1, IMG_H, IMG_W, 7)
        input_data  = np.zeros(input_shape, dtype=np.float32)
        output_data = np.empty(output_shape, dtype=np.float32)

        # batch=1 가정, C=WINDOW_SIZE
        input_data[0, :, :, :WINDOW_SIZE] = volume  # (H,W,7)

        # 4) DPU 실행
        t0 = time.perf_counter()
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)
        t1 = time.perf_counter()

        infer_time = t1 - t0  # 초 단위
        total_infer_time += infer_time
        num_runs += 1
        print("[DEBUG] output_data dtype:", output_data.dtype)
        #print("[DEBUG] output_data min/max:", output_data.min(), output_data.max())
        #print("[DEBUG] first 10 values:", output_data.reshape(-1)[:10])


        # 출력: (1, 2) 가정 → (2,)로 reshape
        pred = output_data.reshape(-1)
        preds.append(pred)

        print(pred)

        # GT
        gts.append([
            float(rows[idx]["MCP_Flex"]),
            float(rows[idx]["PIP_Flex"])
        ])




    preds = np.array(preds)  # (N, 2)
    gts   = np.array(gts)    # (N, 2)

    # ----- MAE 계산 -----
    abs_err = np.abs(preds - gts)  # (N, 2)
    mae_mcp = abs_err[:, 0].mean()
    mae_pip = abs_err[:, 1].mean()
    mae_all = abs_err.mean()

    print("\n========== RESULT ==========")
    print(f"DPU MAE (MCP): {mae_mcp:.4f} deg")
    print(f"DPU MAE (PIP): {mae_pip:.4f} deg")
    print(f"DPU MAE (ALL): {mae_all:.4f} deg")
    print("============================\n")

    # 성능 요약
    if num_runs > 0:
        avg_latency_ms = (total_infer_time / num_runs) * 1000.0
        fps = num_runs / total_infer_time
        print(f"\n[DPU PERFORMANCE]")
        print(f"Avg latency per window: {avg_latency_ms:.3f} ms")
        print(f"Throughput            : {fps:.2f} FPS")
    print("============================\n")

if __name__ == "__main__":
    main()

