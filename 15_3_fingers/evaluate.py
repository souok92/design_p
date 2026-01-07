import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_lstm import create_model
from data.dataset import create_dataloaders


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.6f}")
    return model


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()

    all_predictions = []
    all_targets = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_predictions, all_targets


def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(predictions - targets), axis=0)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    mse = np.mean((predictions - targets) ** 2, axis=0)

    # Overall metrics
    overall_mae = np.mean(mae)
    overall_rmse = np.mean(rmse)
    overall_mse = np.mean(mse)

    return {
        'mae_per_angle': mae,
        'rmse_per_angle': rmse,
        'mse_per_angle': mse,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_mse': overall_mse
    }


def print_metrics(metrics, angle_names):
    """Print evaluation metrics"""

    print("\n" + "="*60)
    print("PER-ANGLE METRICS")
    print("="*60)
    print(f"{'Angle':<15} {'MAE':>12} {'RMSE':>12} {'MSE':>12}")
    print("-"*60)

    for i, name in enumerate(angle_names):
        print(f"{name:<15} {metrics['mae_per_angle'][i]:>12.4f} "
              f"{metrics['rmse_per_angle'][i]:>12.4f} "
              f"{metrics['mse_per_angle'][i]:>12.4f}")

    print("-"*60)
    print(f"{'Overall':<15} {metrics['overall_mae']:>12.4f} "
          f"{metrics['overall_rmse']:>12.4f} "
          f"{metrics['overall_mse']:>12.4f}")
    print("="*60)


def plot_predictions(predictions, targets, angle_names, save_dir):
    """Plot prediction vs ground truth"""
    os.makedirs(save_dir, exist_ok=True)
    num_angles = len(angle_names)

    # Determine subplot layout
    if num_angles <= 4:
        scatter_layout = (1, num_angles)
        ts_layout = (num_angles, 1)
    elif num_angles == 6:
        scatter_layout = (2, 3)
        ts_layout = (3, 2)
    else: # Default for 8 or other numbers
        scatter_layout = (2, 4)
        ts_layout = (4, 2)

    # Plot 1: Scatter plots for each angle
    fig, axes = plt.subplots(scatter_layout[0], scatter_layout[1], figsize=(5 * scatter_layout[1], 5 * scatter_layout[0]))
    axes = axes.flatten()

    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10)
        ax.plot([targets[:, i].min(), targets[:, i].max()],
                [targets[:, i].min(), targets[:, i].max()],
                'r--', linewidth=2)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
    
    for i in range(num_angles, len(axes)): axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_plots.png'), dpi=150)
    plt.close()

    # Plot 2: Time series for each angle (first 500 samples)
    n_samples = min(500, len(predictions))
    fig, axes = plt.subplots(ts_layout[0], ts_layout[1], figsize=(6 * ts_layout[1], 3 * ts_layout[0]))
    axes = axes.flatten()

    for i, name in enumerate(angle_names):
        ax = axes[i]
        x = np.arange(n_samples)
        ax.plot(x, targets[:n_samples, i], label='Ground Truth', linewidth=1.5, alpha=0.8)
        ax.plot(x, predictions[:n_samples, i], label='Prediction', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for i in range(num_angles, len(axes)): axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series.png'), dpi=150)
    plt.close()

    # Plot 3: Error distribution
    errors = predictions - targets
    fig, axes = plt.subplots(scatter_layout[0], scatter_layout[1], figsize=(5 * scatter_layout[1], 5 * scatter_layout[0]))
    axes = axes.flatten()

    for i, name in enumerate(angle_names):
        ax = axes[i]
        ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} - Error Distribution')
        ax.grid(True, alpha=0.3)

    for i in range(num_angles, len(axes)): axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN+LSTM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for plots')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (auto-generated if not provided)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")

    # Create experiment directory with timestamp
    person = config['data']['person']
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{person}_{timestamp}"

    output_dir = os.path.join(args.output_dir, person, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    _, _, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        person=config['data']['person'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config['model'])
    model = model.to(device)

    # Load checkpoint
    model = load_checkpoint(args.checkpoint, model, device)

    # Evaluate
    predictions, targets = evaluate(model, test_loader, device)

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)

    # Dynamically create angle names
    num_outputs = config['model'].get('num_outputs', predictions.shape[1])
    if num_outputs == 6:
        angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP', 'Ring_MCP', 'Ring_PIP']
    elif num_outputs == 8:
        angle_names = ['Index_MCP', 'Index_PIP', 'Middle_MCP', 'Middle_PIP', 'Ring_MCP', 'Ring_PIP', 'Pinky_MCP', 'Pinky_PIP']
    else:
        angle_names = [f'Angle_{i}' for i in range(num_outputs)]
    
    print_metrics(metrics, angle_names)

    # Save experiment configuration
    config_save_path = os.path.join(output_dir, 'experiment_config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\nExperiment config saved to: {config_save_path}")

    # Save experiment info
    checkpoint = torch.load(args.checkpoint, map_location=device)
    exp_info = {
        'experiment_name': exp_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'person': person,
        'config_file': args.config,
        'checkpoint_file': args.checkpoint,
        'checkpoint_epoch': checkpoint.get('epoch', 'N/A'),
        'checkpoint_val_loss': checkpoint.get('val_loss', 'N/A'),
        'device': str(device),
        'test_samples': len(test_loader.dataset),
        'model_config': {
            'sequence_length': config['model']['sequence_length'],
            'lstm_hidden_size': config['model']['lstm_hidden_size'],
            'lstm_num_layers': config['model']['lstm_num_layers'],
            'dropout': config['model']['dropout'],
        },
        'training_config': {
            'learning_rate': config['training']['learning_rate'],
            'optimizer': config['training']['optimizer'],
            'loss_type': config['loss']['type'],
        },
        'results': {
            'overall_mae': float(metrics['overall_mae']),
            'overall_rmse': float(metrics['overall_rmse']),
            'overall_mse': float(metrics['overall_mse']),
        }
    }

    exp_info_path = os.path.join(output_dir, 'experiment_info.json')
    with open(exp_info_path, 'w', encoding='utf-8') as f:
        json.dump(exp_info, f, indent=2, ensure_ascii=False)
    print(f"Experiment info saved to: {exp_info_path}")

    # Create plots
    plot_predictions(predictions, targets, angle_names, output_dir)

    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("PER-ANGLE METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"{'Angle':<15} {'MAE':>12} {'RMSE':>12} {'MSE':>12}\n")
        f.write("-"*60 + "\n")

        for i, name in enumerate(angle_names):
            f.write(f"{name:<15} {metrics['mae_per_angle'][i]:>12.4f} "
                    f"{metrics['rmse_per_angle'][i]:>12.4f} "
                    f"{metrics['mse_per_angle'][i]:>12.4f}\n")

        f.write("-"*60 + "\n")
        f.write(f"{'Overall':<15} {metrics['overall_mae']:>12.4f} "
                f"{metrics['overall_rmse']:>12.4f} "
                f"{metrics['overall_mse']:>12.4f}\n")

    print(f"\nMetrics saved to {metrics_file}")

    # Save predictions
    pred_file = os.path.join(output_dir, 'predictions.npz')
    np.savez(pred_file, predictions=predictions, targets=targets)
    print(f"Predictions saved to {pred_file}")

    # Copy checkpoint to results folder for reproducibility
    import shutil
    checkpoint_copy_path = os.path.join(output_dir, 'model.pth')
    shutil.copy2(args.checkpoint, checkpoint_copy_path)
    print(f"Model checkpoint copied to: {checkpoint_copy_path}")

    print("\nEvaluation completed!")
    print(f"\n{'='*70}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
