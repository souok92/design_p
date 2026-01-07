import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. 설정 (학습 때와 동일)
# ==========================================
CONFIG = {
    'data_root': '../00_data',
    'seq_len': 8,
    'img_size': (224, 224),
    'batch_size': 1,  # 평가할 때는 하나씩 정밀하게 보거나, 시각화를 위해 1 권장
    'num_classes': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

TARGET_COLS = [
    'Index_MCP', 'Index_PIP',
    'Middle_MCP', 'Middle_PIP',
    'Ring_MCP', 'Ring_PIP',
    'Pinky_MCP', 'Pinky_PIP'
]

# ==========================================
# 2. 모델 & 데이터셋 클래스 (Train과 동일해야 함)
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f); o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ResNetConvLSTM(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetConvLSTM, self).__init__()
        resnet = models.resnet18(weights=None) # 평가 때는 가중치를 로드하므로 None
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.conv_lstm = ConvLSTMCell(input_dim=512, hidden_dim=128, kernel_size=3, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        c_in = x.view(b * seq_len, c, h, w)
        features = self.backbone(c_in)
        _, c_out, h_out, w_out = features.size()
        features = features.view(b, seq_len, c_out, h_out, w_out)
        hidden_state = self.conv_lstm.init_hidden(b, (h_out, w_out))
        for t in range(seq_len):
            hidden_state = self.conv_lstm(features[:, t, :, :, :], hidden_state)
        h_last, _ = hidden_state
        out = self.avgpool(h_last).flatten(1)
        return self.fc(out)

class UltrasoundDataset(Dataset):
    def __init__(self, data_root, session_name, seq_len=5, transform=None, mode='test'):
        self.seq_len = seq_len
        self.transform = transform
        
        img_dir = os.path.join(data_root, session_name)
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        csv_path = os.path.join(data_root, f"{session_name}.csv")
        df = pd.read_csv(csv_path)
        
        # 컬럼 이름 매핑 (실제 CSV 헤더 확인 필요, 여기서는 args에 맞춤)
        # 만약 CSV 헤더가 'Index_MCP_Flex' 처럼 되어있다면 아래 리스트 수정 필요
        # 여기서는 편의상 iloc으로 가져오거나, 기존 TARGET_COLS 사용
        # 실제 CSV 헤더에 맞춰 TARGET_COLS_CSV를 정의
        TARGET_COLS_CSV = [
            'Index_MCP_Flex', 'Index_PIP_Flex',
            'Middle_MCP_Flex', 'Middle_PIP_Flex',
            'Ring_MCP_Flex', 'Ring_PIP_Flex',
            'Pinky_MCP_Flex', 'Pinky_PIP_Flex'
        ]
        
        min_len = min(len(self.img_files), len(df))
        self.img_files = self.img_files[:min_len]
        self.labels = df.iloc[:min_len][TARGET_COLS_CSV].values.astype('float32')
        
        # TK 모델 분할 (Test Set 추출)
        if session_name == 'tk':
            total = len(self.img_files)
            val_idx = int(total * 0.85)
            # Test 셋은 뒷부분 15%
            self.img_files = self.img_files[val_idx:]
            self.labels = self.labels[val_idx:]
        else:
            # SJ 모델: sj_1 전체를 Test로 간주
            pass 
            
        print(f"[{session_name} - {mode}] Loaded {len(self.img_files)} frames for evaluation.")

    def __len__(self):
        return len(self.img_files) - self.seq_len + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.seq_len):
            img_path = self.img_files[idx + i]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            imgs.append(image)
        img_seq = torch.stack(imgs)
        label = self.labels[idx + self.seq_len - 1]
        return img_seq, torch.tensor(label)

# ==========================================
# 3. 평가 함수
# ==========================================
def evaluate_model(session_name, model_path):
    print(f"=== Evaluating Session: {session_name} ===")
    print(f"Loading weights from: {model_path}")

    # 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # SJ 모델인 경우 평가 데이터는 sj_1, TK는 tk의 뒷부분
    target_data_folder = 'sj_1' if session_name == 'sj' else 'tk'
    test_dataset = UltrasoundDataset(CONFIG['data_root'], target_data_folder, CONFIG['seq_len'], transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 모델 로드
    model = ResNetConvLSTM(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    checkpoint = torch.load(model_path, map_location=CONFIG['device'])
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    all_preds = []
    all_targets = []

    # 예측 루프
    print("Starting Inference...")
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(CONFIG['device'])
            outputs = model(imgs)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # 배열 합치기
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # ==========================================
    # 4. 결과 분석 및 출력
    # ==========================================
    print("\n" + "="*40)
    print("       EVALUATION RESULTS       ")
    print("="*40)

    # 1. 전체 MAE & R2
    total_mae = mean_absolute_error(all_targets, all_preds)
    total_r2 = r2_score(all_targets, all_preds)
    print(f"Total MAE (Mean Absolute Error): {total_mae:.4f} degrees")
    print(f"Total R2 Score                 : {total_r2:.4f}")
    print("-" * 40)

    # 2. 관절별 상세 분석
    print(f"{'Joint Name':<15} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 40)
    for i, col_name in enumerate(TARGET_COLS):
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        print(f"{col_name:<15} | {mae:.4f}     | {r2:.4f}")

    # ==========================================
    # 5. 시각화 (그래프 저장)
    # ==========================================
    # 시계열 그래프 (처음 200 프레임만)
    plt.figure(figsize=(15, 10))
    limit = 300 # 표시할 프레임 수
    
    # 8개 관절을 4x2 그리드로 표현
    for i in range(8):
        plt.subplot(4, 2, i+1)
        plt.plot(all_targets[:limit, i], label='Actual', color='black', alpha=0.7)
        plt.plot(all_preds[:limit, i], label='Predicted', color='red', alpha=0.7, linestyle='--')
        plt.title(f"{TARGET_COLS[i]} (MAE: {mean_absolute_error(all_targets[:,i], all_preds[:,i]):.2f})")
        plt.legend(loc='upper right')
        if i >= 6: plt.xlabel('Frames')
        
    plt.tight_layout()
    save_img_path = f'evaluation_result_{session_name}.png'
    plt.savefig(save_img_path)
    print(f"\n[Graph Saved] Time-series comparison plot saved to: {save_img_path}")
    
    # 산점도 (Scatter Plot)
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets.flatten(), all_preds.flatten(), alpha=0.1, s=1)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('Actual Angle')
    plt.ylabel('Predicted Angle')
    plt.title(f'Actual vs Predicted (R2: {total_r2:.3f})')
    plt.grid(True)
    plt.savefig(f'scatter_plot_{session_name}.png')
    print(f"[Graph Saved] Scatter plot saved to: scatter_plot_{session_name}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_name', type=str, default='sj', help='Session name (sj or tk)')
    # 학습된 모델 경로 지정 (기본값은 best model)
    parser.add_argument('--model_path', type=str, default=None, help='Path to .pth file')
    
    args = parser.parse_args()
    
    # 모델 경로가 없으면 자동 지정
    if args.model_path is None:
        args.model_path = f"./saved_models/best_model_{args.session_name}.pth"
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
    else:
        evaluate_model(args.session_name, args.model_path)