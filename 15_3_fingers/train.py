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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_lstm import create_model
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
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return criterion


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

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

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

            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches

    # Calculate per-angle metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae_per_angle = np.mean(np.abs(all_predictions - all_targets), axis=0)
    rmse_per_angle = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))

    return avg_loss, mae_per_angle, rmse_per_angle


def save_checkpoint(model, optimizer, epoch, val_loss, config, filename):
    """Save model checkpoint"""
    checkpoint_dir = config['logging']['checkpoint_dir']
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
    parser = argparse.ArgumentParser(description='Train CNN+LSTM model for finger angle prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (auto-generated if not provided)')
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

    # Update config paths with experiment name
    config['logging']['checkpoint_dir'] = f"./checkpoints/{person}/{exp_name}"
    config['logging']['log_dir'] = f"./logs/{person}/{exp_name}"

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")
    print(yaml.dump(config, default_flow_style=False))
    print(f"{'='*70}\n")

    # Set seed
    set_seed(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        person=config['data']['person'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
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

    # Create tensorboard writer
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']

    print("\nStarting training...")

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch)

        # Validate
        val_loss, mae_per_angle, rmse_per_angle = validate(model, val_loader, criterion, device, epoch)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")

        # Log to tensorboard
        # Use add_scalars to plot train and val loss on the same graph
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        num_outputs = config['model']['num_outputs']
        if num_outputs == 6:
            angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP', 'Ring_MCP', 'Ring_PIP']
        elif num_outputs == 8:
            angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP', 'Ring_MCP', 'Ring_PIP', 'Pinky_MCP', 'Pinky_PIP']
        else:
            angle_names = [f'Angle_{i}' for i in range(num_outputs)]

        for i, name in enumerate(angle_names):
            if i < len(mae_per_angle):
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
            if config['logging']['save_best_only']:
                save_checkpoint(model, optimizer, epoch, val_loss, config, 'best_model.pth')
        else:
            patience_counter += 1

        # Save last model
        if config['logging']['save_last']:
            save_checkpoint(model, optimizer, epoch, val_loss, config, 'last_model.pth')

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\nTraining completed!")
    writer.close()


if __name__ == '__main__':
    main()
