import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.channel_stacked_2d import create_model
from data.dataset import create_dataloaders


# ============================================
# CONFIGURATION PARAMETERS
# ============================================

# Path to checkpoint to evaluate
CHECKPOINT_PATH = "./checkpoints/tk/tk_channel2d_YYYYMMDD_HHMMSS/best_model.pth"  # CHANGE THIS!

# Data settings (should match training)
DATA_DIR = "../00_data"
PERSON = "tk"  # 'sj' or 'tk'
SEQUENCE_LENGTH = 16
STRIDE = 1
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 224

# Output directory (will be created in same folder as checkpoint)
OUTPUT_DIR = None  # None = auto (same as checkpoint dir)


# ============================================
# EVALUATION FUNCTIONS
# ============================================

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    config = {
        'model_type': 'channel_stacked_2d',
        'num_frames': 16,
        'num_outputs': 8,
        'dropout': 0.3,
        'pretrained': False  # Don't need pretrained when loading checkpoint
    }

    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    return model


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()

    all_predictions = []
    all_targets = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print(f"Evaluated {len(predictions)} samples")

    return predictions, targets


def compute_metrics(predictions, targets, angle_names):
    """Compute evaluation metrics"""
    errors = predictions - targets
    abs_errors = np.abs(errors)

    metrics = {}

    for i, name in enumerate(angle_names):
        metrics[name] = {
            'MAE': np.mean(abs_errors[:, i]),
            'RMSE': np.sqrt(np.mean(errors[:, i] ** 2)),
            'MSE': np.mean(errors[:, i] ** 2),
            'Std': np.std(errors[:, i]),
            'Max': np.max(abs_errors[:, i]),
            'Min': np.min(abs_errors[:, i])
        }

    # Overall metrics
    metrics['Overall'] = {
        'MAE': np.mean(abs_errors),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MSE': np.mean(errors ** 2),
        'Std': np.std(errors),
        'Max': np.max(abs_errors),
        'Min': np.min(abs_errors)
    }

    return metrics


def save_metrics(metrics, output_path):
    """Save metrics to text file"""
    angle_names = [k for k in metrics.keys() if k != 'Overall']

    with open(output_path, 'w') as f:
        f.write("PER-ANGLE METRICS (17_channel_stacked_2d)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Angle':<20} {'MAE':>10} {'RMSE':>10} {'MSE':>14}\n")
        f.write("-" * 60 + "\n")

        for name in angle_names:
            m = metrics[name]
            f.write(f"{name:<20} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MSE']:>14.4f}\n")

        f.write("-" * 60 + "\n")
        m = metrics['Overall']
        f.write(f"{'Overall':<20} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MSE']:>14.4f}\n")

    print(f"Metrics saved to {output_path}")


def save_predictions(predictions, targets, angle_names, output_path):
    """Save predictions and targets to numpy file"""
    np.savez(
        output_path,
        predictions=predictions,
        targets=targets,
        angle_names=angle_names
    )
    print(f"Predictions saved to {output_path}")


def create_visualizations(predictions, targets, angle_names, output_dir):
    """Create and save visualization plots"""
    # Scatter plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10)
        ax.plot([targets[:, i].min(), targets[:, i].max()],
                [targets[:, i].min(), targets[:, i].max()],
                'r--', lw=2)
        ax.set_xlabel('Ground Truth (degrees)')
        ax.set_ylabel('Prediction (degrees)')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'scatter_plots.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plots saved to {scatter_path}")

    # Error distribution
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, name in enumerate(angle_names):
        ax = axes[i]
        errors = predictions[:, i] - targets[:, i]
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Error (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} - MAE: {np.mean(np.abs(errors)):.2f}°')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    error_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error distribution saved to {error_path}")

    # Time series (first 500 samples)
    n_samples = min(500, len(predictions))
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()

    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.plot(targets[:n_samples, i], label='Ground Truth', alpha=0.7)
        ax.plot(predictions[:n_samples, i], label='Prediction', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    timeseries_path = os.path.join(output_dir, 'time_series.png')
    plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved to {timeseries_path}")


# ============================================
# MAIN EVALUATION
# ============================================

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine output directory
    if OUTPUT_DIR is None:
        output_dir = os.path.dirname(CHECKPOINT_PATH)
    else:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model(CHECKPOINT_PATH, device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        person=PERSON,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        stride=STRIDE,
        augment=False,
        image_size=IMAGE_SIZE
    )

    print(f"Test batches: {len(test_loader)}")

    # Evaluate
    predictions, targets = evaluate(model, test_loader, device)

    # Compute metrics
    angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP',
                   'Ring_MCP', 'Ring_PIP', 'Pinky_MCP', 'Pinky_PIP']

    metrics = compute_metrics(predictions, targets, angle_names)

    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS (17_channel_stacked_2d)")
    print("="*60)
    print(f"{'Angle':<20} {'MAE':>10} {'RMSE':>10}")
    print("-"*60)

    for name in angle_names:
        m = metrics[name]
        print(f"{name:<20} {m['MAE']:>10.2f}° {m['RMSE']:>10.2f}°")

    print("-"*60)
    m = metrics['Overall']
    print(f"{'Overall':<20} {m['MAE']:>10.2f}° {m['RMSE']:>10.2f}°")
    print("="*60)

    # Save results
    save_metrics(metrics, os.path.join(output_dir, 'metrics.txt'))
    save_predictions(predictions, targets, angle_names, os.path.join(output_dir, 'predictions.npz'))
    create_visualizations(predictions, targets, angle_names, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
