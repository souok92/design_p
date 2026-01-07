import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from datetime import datetime
import json
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.r3d import create_model
from data.dataset import create_dataloaders


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model, config):
    """Create optimizer from config"""
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config"""
    scheduler_name = config['training']['scheduler']
    
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler_factor'],
            patience=config['training']['scheduler_patience']
        )
    elif scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('scheduler_step_size', 30),
            gamma=config['training']['scheduler_factor']
        )
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif scheduler_name is None or scheduler_name == 'None':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def create_loss_function(config):
    """Create loss function from config"""
    loss_type = config['loss']['type']
    
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'MAE' or loss_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_type == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return criterion


def freeze_backbone(model):
    """Freeze R3D backbone parameters"""
    for name, param in model.named_parameters():
        if 'regressor' not in name:  # regressor만 학습
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen. Trainable parameters: {trainable:,}")


def unfreeze_backbone(model):
    """Unfreeze all parameters"""
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone unfrozen. Trainable parameters: {trainable:,}")


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Free memory
        del outputs, loss
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    
    # Calculate per-angle metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mae_per_angle = np.mean(np.abs(all_predictions - all_targets), axis=0)
    rmse_per_angle = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))
    
    return avg_loss, mae_per_angle, rmse_per_angle


def save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_dir, filename):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D CNN (R3D-18) for finger angle prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Create experiment name with timestamp
    person = config['data']['person']
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{person}_{timestamp}"
    
    # Create experiment directories
    exp_dir = f"./results/{person}/{exp_name}"
    checkpoint_dir = f"./checkpoints/{person}/{exp_name}"
    log_dir = f"./logs/{person}/{exp_name}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config to experiment directory
    shutil.copy(args.config, os.path.join(exp_dir, 'experiment_config.yaml'))
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Results: {exp_dir}")
    print(f"{'='*70}")
    print(yaml.dump(config, default_flow_style=False))
    print(f"{'='*70}\n")
    
    # Set seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        person=config['data']['person'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        stride=config['data'].get('stride', 1),
        augment=config['data'].get('augment', False),
        image_size=config['data'].get('image_size', 224)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config['model'])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer, scheduler, and loss function
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_loss_function(config)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {checkpoint['epoch']}, val_loss={best_val_loss:.4f}")
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    freeze_epochs = config['training'].get('freeze_backbone_epochs', 0)
    
    # Freeze backbone for initial epochs
    if freeze_epochs > 0 and start_epoch <= freeze_epochs:
        freeze_backbone(model)
    
    print("\nStarting training...")
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        # Unfreeze backbone after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            # Reset optimizer with lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['training']['learning_rate'] * 0.1
            print(f"Learning rate reduced to {param_group['lr']} for fine-tuning")
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch)
        
        # Validate
        val_loss, mae_per_angle, rmse_per_angle = validate(model, val_loader, criterion, device, epoch)
        
        # Calculate overall metrics
        overall_mae = np.mean(mae_per_angle)
        overall_rmse = np.mean(rmse_per_angle)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {overall_mae:.2f}°")
        print(f"  Val RMSE: {overall_rmse:.2f}°")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to tensorboard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Metrics/Overall_MAE', overall_mae, epoch)
        writer.add_scalar('Metrics/Overall_RMSE', overall_rmse, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP',
                       'Ring_MCP', 'Ring_PIP', 'Pinky_MCP', 'Pinky_PIP']
        
        for i, name in enumerate(angle_names):
            writer.add_scalar(f'MAE/{name}', mae_per_angle[i], epoch)
            writer.add_scalar(f'RMSE/{name}', rmse_per_angle[i], epoch)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_dir, 'best_model.pth')
            print(f"  *** New best model! ***")
        else:
            patience_counter += 1
        
        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_dir, 'last_model.pth')
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save experiment info
    exp_info = {
        'exp_name': exp_name,
        'best_val_loss': float(best_val_loss),
        'final_epoch': epoch,
        'config': config
    }
    
    with open(os.path.join(exp_dir, 'experiment_info.json'), 'w') as f:
        json.dump(exp_info, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*70}")
    
    writer.close()


if __name__ == '__main__':
    main()
