import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datetime import datetime

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    'data_root': '../00_data',      # 데이터 루트 폴더
    'seq_len': 8,               # 시퀀스 길이 (LSTM 입력 프레임 수)
    'img_size': (224, 224),     # 모델 입력 이미지 크기 (ResNet 기본)
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 100,
    'patience': 10,             # <--- Early Stopping 기다릴 횟수 추가
    'num_classes': 8,           # 예측할 각도 개수 (Thumb 제외)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# CSV에서 사용할 컬럼명 (실제 CSV 헤더에 맞춰 수정 필요할 수 있음)
# 프롬프트 설명에 따른 추정 헤더명입니다. 실제 CSV 파일의 헤더를 확인해주세요.
TARGET_COLS = [
    'Index_MCP_Flex', 'Index_PIP_Flex',
    'Middle_MCP_Flex', 'Middle_PIP_Flex',
    'Ring_MCP_Flex', 'Ring_PIP_Flex',
    'Pinky_MCP_Flex', 'Pinky_PIP_Flex'
]

# ==========================================
# 2. 데이터셋 클래스 (Custom Dataset)
# ==========================================
class UltrasoundDataset(Dataset):
    def __init__(self, data_root, session_name, seq_len=5, transform=None, mode='train'):
        """
        Args:
            data_root: 데이터 루트 경로
            session_name: 'sj_1', 'sj_2', 'tk' 중 하나
            seq_len: 시퀀스 길이 (Default: 5)
        """
        self.seq_len = seq_len
        self.transform = transform
        
        # 1. 이미지 경로 로드
        img_dir = os.path.join(data_root, session_name)
        # 파일명 기준 정렬 (시간순)
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        # 2. CSV 로드
        csv_path = os.path.join(data_root, f"{session_name}.csv")
        df = pd.read_csv(csv_path)
        
        # 3. 데이터 정합성 체크 (이미지와 CSV 행 개수 맞추기)
        # 보통 이미지 수와 CSV 행 수가 약간 다를 수 있으므로 교집합(min) 사용
        min_len = min(len(self.img_files), len(df))
        self.img_files = self.img_files[:min_len]
        self.labels = df.iloc[:min_len][TARGET_COLS].values.astype('float32')
        
        # TK 모델인 경우 Train/Val/Test 분할 로직 (70/15/15)
        if session_name == 'tk':
            total = len(self.img_files)
            train_idx = int(total * 0.7)
            val_idx = int(total * 0.85)
            
            if mode == 'train':
                self.img_files = self.img_files[:train_idx]
                self.labels = self.labels[:train_idx]
            elif mode == 'val':
                self.img_files = self.img_files[train_idx:val_idx]
                self.labels = self.labels[train_idx:val_idx]
            else: # test
                self.img_files = self.img_files[val_idx:]
                self.labels = self.labels[val_idx:]
        
        # SJ 모델의 경우 (sj_2: Train, sj_1: Val) - 외부에서 호출 시 구분함
        
        print(f"[{session_name} - {mode}] Loaded {len(self.img_files)} frames.")

    def __len__(self):
        # 시퀀스 데이터이므로 전체 길이 - (시퀀스 길이 - 1)
        return len(self.img_files) - self.seq_len + 1

    def __getitem__(self, idx):
        # idx부터 idx + seq_len 까지의 이미지 로드
        imgs = []
        for i in range(self.seq_len):
            img_path = self.img_files[idx + i]
            # 흑백 이미지지만 ResNet 전이학습을 위해 RGB로 변환
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            imgs.append(image)
            
        # (Seq, C, H, W) 형태로 스택
        img_seq = torch.stack(imgs)
        
        # 정답 라벨은 시퀀스의 '마지막 프레임' 기준으로 설정
        label = self.labels[idx + self.seq_len - 1]
        
        return img_seq, torch.tensor(label)

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중단"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
            verbose (bool): True일 경우 각 validation loss 개선 사항 메세지 출력
            delta (float): 개선되었다고 인정하기 위한 최소 변화량
            path (str): checkpoint저장 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==========================================
# 3. 모델 정의 (ResNet + ConvLSTM)
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
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # (Batch, 512, 7, 7) output
        
        self.conv_lstm = ConvLSTMCell(input_dim=512, hidden_dim=128, kernel_size=3, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        c_in = x.view(b * seq_len, c, h, w)
        
        # CNN Backbone
        features = self.backbone(c_in)
        _, c_out, h_out, w_out = features.size()
        features = features.view(b, seq_len, c_out, h_out, w_out)
        
        # ConvLSTM
        hidden_state = self.conv_lstm.init_hidden(b, (h_out, w_out))
        for t in range(seq_len):
            hidden_state = self.conv_lstm(features[:, t, :, :, :], hidden_state)
        
        # Final Prediction
        h_last, _ = hidden_state
        out = self.avgpool(h_last).flatten(1)
        return self.fc(out)

# ==========================================
# 4. 학습 루프 (Training Loop)
# ==========================================
def train_model():
    # Transform 정의
    transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드 (SJ 모델 기준: sj_2 학습, sj_1 검증)
    # *참고: TK 모델을 학습하려면 session_name='tk', mode='train'/'val'로 변경
    train_dataset = UltrasoundDataset(CONFIG['data_root'], 'sj_2', CONFIG['seq_len'], transform, mode='train')
    val_dataset = UltrasoundDataset(CONFIG['data_root'], 'sj_1', CONFIG['seq_len'], transform, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 모델, 손실함수, 최적화
    model = ResNetConvLSTM(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    criterion = nn.MSELoss() # 또는 nn.L1Loss() (MAE)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print(f"Start training on {CONFIG['device']}...")
    
    for epoch in range(CONFIG['epochs']):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        mae_score = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                mae_score += torch.abs(outputs - targets).mean().item() * imgs.size(0)

        # 결과 출력
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_mae = mae_score / len(val_dataset)

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val MAE: {epoch_mae:.4f}")
        
        # 모델 저장 (매 epoch 혹은 best 저장 로직 추가 가능)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"./saved_models/model_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    # 저장 폴더 생성
    os.makedirs('./saved_models', exist_ok=True)
    train_model()