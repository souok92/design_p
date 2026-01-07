import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.r3d import create_model
from data.dataset import create_dataloaders


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
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
        f.write("PER-ANGLE METRICS\n")
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


def plot_results(predictions, targets, angle_names, output_dir):
    """Plot evaluation results"""
    
    # 1. Scatter plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.3, s=5)
        
        # Perfect prediction line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel('Ground Truth (°)')
        ax.set_ylabel('Prediction (°)')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plots.png'), dpi=150)
    plt.close()
    print(f"Scatter plots saved")
    
    # 2. Error distribution
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    errors = predictions - targets
    
    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.hist(errors[:, i], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Error (°)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nMAE={np.mean(np.abs(errors[:, i])):.2f}°')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150)
    plt.close()
    print(f"Error distribution saved")
    
    # 3. Time series (first 500 samples)
    n_samples = min(500, len(predictions))
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.plot(targets[:n_samples, i], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(predictions[:n_samples, i], 'r-', label='Prediction', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Angle (°)')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series.png'), dpi=150)
    plt.close()
    print(f"Time series plot saved")


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D CNN model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as checkpoint
        output_dir = os.path.dirname(args.checkpoint)
        if not output_dir:
            output_dir = '.'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    _, _, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        person=config['data']['person'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        stride=config['data'].get('stride', 1),
        augment=False,
        image_size=config['data'].get('image_size', 224)
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    predictions, targets = evaluate(model, test_loader, device)
    
    # Compute metrics
    angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP',
                   'Ring_MCP', 'Ring_PIP', 'Pinky_MCP', 'Pinky_PIP']
    
    metrics = compute_metrics(predictions, targets, angle_names)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Angle':<20} {'MAE':>10} {'RMSE':>10}")
    print("-" * 60)
    
    for name in angle_names:
        m = metrics[name]
        print(f"{name:<20} {m['MAE']:>10.2f}° {m['RMSE']:>10.2f}°")
    
    print("-" * 60)
    m = metrics['Overall']
    print(f"{'Overall':<20} {m['MAE']:>10.2f}° {m['RMSE']:>10.2f}°")
    print("=" * 60)
    
    # Save results
    save_metrics(metrics, os.path.join(output_dir, 'metrics.txt'))
    
    # Save predictions
    np.savez(os.path.join(output_dir, 'predictions.npz'),
             predictions=predictions, targets=targets, angle_names=angle_names)
    print(f"Predictions saved")
    
    # Plot results
    plot_results(predictions, targets, angle_names, output_dir)
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
