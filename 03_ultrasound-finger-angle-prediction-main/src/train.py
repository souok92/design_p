#!/usr/bin/env python3
"""
Training script for CNN-TCN Ultrasound-to-Angle Prediction Model
Uses MAE (Mean Absolute Error) as primary loss function
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from model import get_model
from dataset import get_dataloaders


class Trainer:
    """
    Trainer class for CNN-TCN model
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = get_model(
            window_size=config['window_size'],
            device=self.device
        )

        # Loss function: MAE (L1 Loss)
        self.criterion = nn.L1Loss()  # Mean Absolute Error

        # Optimizer: Adam
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            data_dir=config['data_dir'],
            image_dir=config['image_dir'],
            window_size=config['window_size'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        # TensorBoard writer
        log_dir = os.path.join(config['log_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.best_val_loss = float('inf')
        self.epoch = 0

        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Window size: {config['window_size']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Loss function: MAE (L1 Loss)")
        print(f"{'='*70}\n")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_mae = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)  # (batch, window, C, H, W)
            targets = targets.to(self.device)  # (batch, 2)

            # Forward pass
            predictions = self.model(images)  # (batch, 2)

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_mae += loss.item()  # MAE is the same as L1 loss

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Average metrics
        avg_loss = running_loss / len(self.train_loader)
        avg_mae = running_mae / len(self.train_loader)

        return avg_loss, avg_mae

    def validate(self, loader, mode='Val'):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {self.epoch+1} [{mode}]")
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images)
                loss = self.criterion(predictions, targets)

                running_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Aggregate predictions
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        # Compute metrics
        avg_loss = running_loss / len(loader)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # Per-angle MAE
        mcp_mae = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))
        pip_mae = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))

        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'mcp_mae': mcp_mae,
            'pip_mae': pip_mae
        }

        return metrics

    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}\n")

        for epoch in range(self.config['epochs']):
            self.epoch = epoch

            # Train
            train_loss, train_mae = self.train_epoch()

            # Validate
            val_metrics = self.validate(self.val_loader, mode='Val')

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Logging
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}°")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}°")
            print(f"  MCP MAE: {val_metrics['mcp_mae']:.4f}°, PIP MAE: {val_metrics['pip_mae']:.4f}°")

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('MAE/train', train_mae, epoch)
            self.writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
            self.writer.add_scalar('MAE/MCP', val_metrics['mcp_mae'], epoch)
            self.writer.add_scalar('MAE/PIP', val_metrics['pip_mae'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best model saved! (Val Loss: {val_metrics['loss']:.4f})")

            # Save regular checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(is_best=False)

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}\n")

        # Final evaluation on test set
        print("Evaluating on test set...")
        test_metrics = self.validate(self.test_loader, mode='Test')
        print(f"\nTest Results:")
        print(f"  Test MAE: {test_metrics['mae']:.4f}°")
        print(f"  MCP MAE: {test_metrics['mcp_mae']:.4f}°")
        print(f"  PIP MAE: {test_metrics['pip_mae']:.4f}°")

        self.writer.close()

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
        else:
            path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{self.epoch+1}.pth')

        torch.save(checkpoint, path)


def main():
    # Training configuration
    config = {
        # Data
        'data_dir': '/home/kimsj/delto_m_ros2/model/data',
        'image_dir': '/home/kimsj/delto_m_ros2/model/data/251002_wrist',

        # Model
        'window_size': 7,  # 7 frames ≈ 210ms temporal context

        # Training
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,

        # Data loading
        'num_workers': 4,

        # Logging and saving
        'log_dir': '/home/kimsj/delto_m_ros2/model/runs',
        'checkpoint_dir': '/home/kimsj/delto_m_ros2/model/checkpoints',
        'save_freq': 10  # Save checkpoint every 10 epochs
    }

    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Save config
    config_path = os.path.join(config['checkpoint_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
