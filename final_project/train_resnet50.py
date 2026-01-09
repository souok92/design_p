#!/usr/bin/env python3
"""
Training Script for ResNet50 Channel-Stack Model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from model_resnet50_channelstack import CNN_ResNet50_ChannelStack_DPU


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ChannelStackedUSDataset(Dataset):
    """Dataset that returns grayscale frames stacked in channel dimension"""
    def __init__(self, csv_file, img_dir, window_size=7, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.window_size = window_size
        self.transform = transform

        # Sort by filename
        self.data = self.data.sort_values('image_filename').reset_index(drop=True)
        self.valid_indices = list(range(len(self.data) - window_size + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Load window_size consecutive frames as grayscale
        frame_list = []
        for i in range(self.window_size):
            img_name = self.data.iloc[start_idx + i]['image_filename']
            img_path = os.path.join(self.img_dir, img_name)

            # Load as grayscale
            image = Image.open(img_path).convert('L')

            if self.transform:
                image = self.transform(image)

            frame_list.append(image)

        # Stack frames in CHANNEL dimension
        stacked_frames = torch.cat(frame_list, dim=0)

        # Get target angles
        target_row = self.data.iloc[start_idx + self.window_size - 1]
        mcp_angle = target_row['MCP_Flex']
        pip_angle = target_row['PIP_Flex']
        target = torch.tensor([mcp_angle, pip_angle], dtype=torch.float32)

        return stacked_frames, target


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    mcp_errors = []
    pip_errors = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for frames, targets in pbar:
        frames = frames.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            mcp_mae = torch.abs(outputs[:, 0] - targets[:, 0]).mean().item()
            pip_mae = torch.abs(outputs[:, 1] - targets[:, 1]).mean().item()
            mcp_errors.append(mcp_mae)
            pip_errors.append(pip_mae)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'MCP': f'{mcp_mae:.2f}°',
            'PIP': f'{pip_mae:.2f}°'
        })

    return running_loss / len(dataloader), np.mean(mcp_errors), np.mean(pip_errors)


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    mcp_errors = []
    pip_errors = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")
    with torch.no_grad():
        for frames, targets in pbar:
            frames = frames.to(device)
            targets = targets.to(device)

            outputs = model(frames)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            mcp_mae = torch.abs(outputs[:, 0] - targets[:, 0]).mean().item()
            pip_mae = torch.abs(outputs[:, 1] - targets[:, 1]).mean().item()
            mcp_errors.append(mcp_mae)
            pip_errors.append(pip_mae)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'MCP': f'{mcp_mae:.2f}°',
                'PIP': f'{pip_mae:.2f}°'
            })

    return running_loss / len(dataloader), np.mean(mcp_errors), np.mean(pip_errors)


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_mcp_mae'], label='Train')
    axes[0, 1].plot(history['val_mcp_mae'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (degrees)')
    axes[0, 1].set_title('MCP Angle MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_pip_mae'], label='Train')
    axes[1, 0].plot(history['val_pip_mae'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (degrees)')
    axes[1, 0].set_title('PIP Angle MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    train_combined = [(m + p) / 2 for m, p in zip(history['train_mcp_mae'], history['train_pip_mae'])]
    val_combined = [(m + p) / 2 for m, p in zip(history['val_mcp_mae'], history['val_pip_mae'])]
    axes[1, 1].plot(train_combined, label='Train')
    axes[1, 1].plot(val_combined, label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE (degrees)')
    axes[1, 1].set_title('Average MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def main():
    # Configuration
    WINDOW_SIZE = 7
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    IMG_SIZE = (380, 508)  # Original image size (H, W)

    DATA_DIR = '../data/251002_wrist'
    TRAIN_CSV = '../data/train.csv'
    VAL_CSV = '../data/val.csv'

    CHECKPOINT_DIR = './checkpoints_resnet50_1204'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # TensorBoard
    LOG_DIR = os.path.join('../runs', 'resnet50_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("ResNet50 Channel-Stack Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"Backbone: ResNet50 (pretrained)")
    print(f"Window Size: {WINDOW_SIZE} frames")
    print(f"Input Size: {IMG_SIZE} (H, W)")
    print(f"TensorBoard Log: {LOG_DIR}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        #transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Datasets
    print("\nLoading datasets...")
    train_dataset = ChannelStackedUSDataset(TRAIN_CSV, DATA_DIR, WINDOW_SIZE, train_transform)
    val_dataset = ChannelStackedUSDataset(VAL_CSV, DATA_DIR, WINDOW_SIZE, val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print("\nInitializing model...")
    model = CNN_ResNet50_ChannelStack_DPU(window_size=WINDOW_SIZE).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001, verbose=True)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mcp_mae': [], 'val_mcp_mae': [],
        'train_pip_mae': [], 'val_pip_mae': []
    }

    best_val_loss = float('inf')

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mcp_mae, train_pip_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_mcp_mae, val_pip_mae = validate(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mcp_mae'].append(train_mcp_mae)
        history['val_mcp_mae'].append(val_mcp_mae)
        history['train_pip_mae'].append(train_pip_mae)
        history['val_pip_mae'].append(val_pip_mae)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
        print(f"  Train - Loss: {train_loss:.4f} | MCP: {train_mcp_mae:.2f}° | PIP: {train_pip_mae:.2f}°")
        print(f"  Val   - Loss: {val_loss:.4f} | MCP: {val_mcp_mae:.2f}° | PIP: {val_pip_mae:.2f}°")

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train_mcp', train_mcp_mae, epoch)
        writer.add_scalar('MAE/train_pip', train_pip_mae, epoch)
        writer.add_scalar('MAE/val_mcp', val_mcp_mae, epoch)
        writer.add_scalar('MAE/val_pip', val_pip_mae, epoch)
        writer.add_scalar('MAE/train_avg', (train_mcp_mae + train_pip_mae) / 2, epoch)
        writer.add_scalar('MAE/val_avg', (val_mcp_mae + val_pip_mae) / 2, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mcp_mae': val_mcp_mae,
                'val_pip_mae': val_pip_mae,
            }, checkpoint_path)
            print(f"  ✓ Best model saved!")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, checkpoint_path)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"{'='*70}")
            break

        print("-" * 70)

    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Close TensorBoard writer
    writer.close()
    print(f"\nTensorBoard logs saved to: {LOG_DIR}")
    print(f"To view, run: tensorboard --logdir={LOG_DIR}")

    plot_training_history(history, os.path.join(CHECKPOINT_DIR, 'training_history.png'))

    final_path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == '__main__':
    main()
