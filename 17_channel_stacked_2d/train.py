import os
import sys
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

from models.channel_stacked_2d import create_model
from data.dataset import create_dataloaders


# ============================================
# CONFIGURATION PARAMETERS
# ============================================

# Data settings
DATA_DIR = "../00_data"
PERSON = "tk"  # 'sj' or 'tk'
SEQUENCE_LENGTH = 16  # 16 frames = 16 channels
STRIDE = 1
BATCH_SIZE = 16  # Higher than 3D CNN (2D is faster, can handle larger batches)
NUM_WORKERS = 4
AUGMENT = True
IMAGE_SIZE = 384  # Higher resolution for better feature extraction

# Model settings
MODEL_TYPE = "channel_stacked_2d"
NUM_FRAMES = 16  # Number of temporal frames (treated as channels)
NUM_OUTPUTS = 8
DROPOUT = 0.3
PRETRAINED = True  # ImageNet ResNet18 weights

# Training settings
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
OPTIMIZER = "AdamW"  # 'Adam', 'AdamW', or 'SGD'
SCHEDULER = "ReduceLROnPlateau"  # 'ReduceLROnPlateau', 'CosineAnnealingLR', 'StepLR', or None
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 20  # More patience for 2D approach
GRADIENT_CLIP = 1.0

# Loss settings
LOSS_TYPE = "MAE"  # 'MSE', 'MAE', or 'SmoothL1'

# Logging settings
LOG_INTERVAL = 10

# Reproducibility
SEED = 42

# Experiment name (optional, leave None for auto timestamp)
EXP_NAME = None  # e.g., "channel_2d_experiment_1" or None


# ============================================
# HELPER FUNCTIONS
# ============================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer_fn(model):
    """Create optimizer"""
    if OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    return optimizer


def create_scheduler_fn(optimizer):
    """Create learning rate scheduler"""
    if SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )
    elif SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=SCHEDULER_FACTOR
        )
    elif SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS
        )
    elif SCHEDULER is None or SCHEDULER == 'None':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {SCHEDULER}")

    return scheduler


def create_loss_function():
    """Create loss function"""
    if LOSS_TYPE == 'MSE':
        criterion = nn.MSELoss()
    elif LOSS_TYPE == 'MAE' or LOSS_TYPE == 'L1':
        criterion = nn.L1Loss()
    elif LOSS_TYPE == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {LOSS_TYPE}")

    return criterion


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
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
        if GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

        optimizer.step()

        # Update metrics
        total_loss += loss.item()

        # Free memory
        del outputs, loss

        # Update progress bar
        pbar.set_postfix({'loss': f"{total_loss / (batch_idx + 1):.4f}"})

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


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, filename):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': {
            'data_dir': DATA_DIR,
            'person': PERSON,
            'sequence_length': SEQUENCE_LENGTH,
            'batch_size': BATCH_SIZE,
            'model_type': MODEL_TYPE,
            'num_frames': NUM_FRAMES,
            'learning_rate': LEARNING_RATE
        }
    }

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


# ============================================
# MAIN TRAINING LOOP
# ============================================

def main():
    # Create experiment name with timestamp
    if EXP_NAME:
        exp_name = EXP_NAME
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{PERSON}_channel2d_{timestamp}"

    # Create experiment directories
    exp_dir = f"./results/{PERSON}/{exp_name}"
    checkpoint_dir = f"./checkpoints/{PERSON}/{exp_name}"
    log_dir = f"./logs/{PERSON}/{exp_name}"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Print configuration
    print(f"\n{'='*70}")
    print(f"17_channel_stacked_2d: Channel-Stacked 2D CNN Training")
    print(f"{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Results: {exp_dir}")
    print(f"\nConfiguration:")
    print(f"  Person: {PERSON}")
    print(f"  Sequence Length: {SEQUENCE_LENGTH} frames (treated as {NUM_FRAMES} channels)")
    print(f"  Batch Size: {BATCH_SIZE} (higher than 3D CNN)")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Scheduler: {SCHEDULER}")
    print(f"  Loss: {LOSS_TYPE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"\nKey Insight: Time is encoded as CHANNELS (not explicit 3D conv)")
    print(f"{'='*70}\n")

    # Set seed
    set_seed(SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        person=PERSON,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        stride=STRIDE,
        augment=AUGMENT,
        image_size=IMAGE_SIZE
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    config = {
        'model_type': MODEL_TYPE,
        'num_frames': NUM_FRAMES,
        'num_outputs': NUM_OUTPUTS,
        'dropout': DROPOUT,
        'pretrained': PRETRAINED
    }
    model = create_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer, scheduler, and loss function
    optimizer = create_optimizer_fn(model)
    scheduler = create_scheduler_fn(optimizer)
    criterion = create_loss_function()

    # Create tensorboard writer
    writer = SummaryWriter(log_dir)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

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
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, 'best_model.pth')
            print(f"  *** New best model! ***")
        else:
            patience_counter += 1

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, 'last_model.pth')

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Save experiment info
    exp_info = {
        'exp_name': exp_name,
        'best_val_loss': float(best_val_loss),
        'final_epoch': epoch,
        'person': PERSON,
        'model_type': MODEL_TYPE,
        'batch_size': BATCH_SIZE,
        'num_frames': NUM_FRAMES
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
