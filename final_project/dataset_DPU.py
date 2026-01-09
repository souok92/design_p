#!/usr/bin/env python3
"""
Dataset for Grayscale Ultrasound-to-Angle Prediction (DPU-Friendly)

Features:
- Grayscale (1-channel) ultrasound images
- Temporal sequences with sliding window
- MCP angle filtering: excludes 1/3 of data in 60-70° range (training only)
- Compatible with CNN_TCN_Grayscale_DPU model
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AddGaussianNoise:
    """
    Add Gaussian noise to simulate ultrasound speckle noise
    """
    def __init__(self, mean=0., std=0.03):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor: Input tensor in range [0, 1]

        Returns:
            Noisy tensor clipped to [0, 1]
        """
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class UltrasoundAngleDataset_DPU(Dataset):
    """
    Temporal Dataset for Grayscale Ultrasound-to-Angle Prediction

    Features:
    - Loads grayscale (1-channel) ultrasound images
    - Creates temporal sequences using sliding window
    - Optional MCP angle filtering for training data
    """
    def __init__(self, csv_file, image_dir, window_size=7, transform=None,
                 filter_mcp_range=None, filter_ratio=0.33, random_seed=42):
        """
        Args:
            csv_file: Path to CSV file (train.csv, val.csv, or test.csv)
            image_dir: Directory containing ultrasound images
            window_size: Number of consecutive frames for temporal modeling (default: 7)
            transform: Optional transform to be applied on images
            filter_mcp_range: Tuple (min, max) for MCP angle filtering (e.g., (60, 70))
                              If None, no filtering is applied
            filter_ratio: Ratio of data to exclude in filter range (default: 0.33 = 1/3)
            random_seed: Random seed for reproducible filtering (default: 42)
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.window_size = window_size
        self.transform = transform

        print(f"\nDataset initialization:")
        print(f"  CSV file: {csv_file}")
        print(f"  Total samples in CSV: {len(self.data)}")
        print(f"  Window size: {window_size}")

        # Apply MCP angle filtering if specified
        if filter_mcp_range is not None:
            self._apply_mcp_filtering(filter_mcp_range, filter_ratio, random_seed)
        else:
            print(f"  No MCP filtering applied")

        # For temporal sequences, we can only use samples where we have enough history
        # Valid indices start from window_size - 1
        self.valid_indices = list(range(window_size - 1, len(self.data)))

        print(f"  Valid temporal sequences: {len(self.valid_indices)}")

    def _apply_mcp_filtering(self, filter_range, filter_ratio, random_seed):
        """
        Filter out a portion of data where MCP angle is in the specified range

        Args:
            filter_range: Tuple (min, max) for MCP angle range
            filter_ratio: Ratio of data to exclude (0.33 = 1/3)
            random_seed: Random seed for reproducibility
        """
        min_angle, max_angle = filter_range

        # Find indices where MCP is in the filter range
        in_range_mask = (self.data['MCP_Flex'] >= min_angle) & (self.data['MCP_Flex'] <= max_angle)
        in_range_indices = self.data[in_range_mask].index.tolist()

        num_in_range = len(in_range_indices)
        num_to_exclude = int(num_in_range * filter_ratio)

        print(f"\n  MCP Filtering:")
        print(f"    Range: {min_angle}° - {max_angle}°")
        print(f"    Samples in range: {num_in_range}")
        print(f"    Exclude ratio: {filter_ratio:.2%}")
        print(f"    Samples to exclude: {num_to_exclude}")

        if num_to_exclude > 0:
            # Randomly select indices to exclude
            np.random.seed(random_seed)
            indices_to_exclude = np.random.choice(
                in_range_indices,
                size=num_to_exclude,
                replace=False
            )

            # Remove selected indices from dataset
            original_len = len(self.data)
            self.data = self.data.drop(indices_to_exclude).reset_index(drop=True)
            new_len = len(self.data)

            print(f"    Excluded samples: {original_len - new_len}")
            print(f"    Remaining samples: {new_len}")
        else:
            print(f"    No samples to exclude")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns a temporal sequence of grayscale images and corresponding target angles

        Returns:
            images: Tensor of shape (window_size, 1, H, W) - Grayscale
            target: Tensor of shape (2,) containing [MCP_angle, PIP_angle]
        """
        # Get the actual index in the dataframe
        actual_idx = self.valid_indices[idx]

        # Get window of frames [actual_idx - window_size + 1, ..., actual_idx]
        start_idx = actual_idx - self.window_size + 1
        end_idx = actual_idx + 1

        # Load sequence of images
        images = []
        for i in range(start_idx, end_idx):
            img_filename = self.data.iloc[i]['image_filename']
            img_path = os.path.join(self.image_dir, img_filename)

            # Load image as grayscale
            image = Image.open(img_path).convert('L')  # 'L' = grayscale

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            images.append(image)

        # Stack images along new dimension: (window_size, 1, H, W)
        images = torch.stack(images, dim=0)

        # Get target angles (from the last frame in the window)
        mcp_angle = float(self.data.iloc[actual_idx]['MCP_Flex'])
        pip_angle = float(self.data.iloc[actual_idx]['PIP_Flex'])
        target = torch.tensor([mcp_angle, pip_angle], dtype=torch.float32)

        return images, target


def get_transforms_DPU(train=True, img_size=224):
    """
    Get image transforms for grayscale ultrasound images

    Args:
        train: If True, apply data augmentation
        img_size: Target image size (default: 224)

    Returns:
        transforms: Composed transforms for grayscale images
    """
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            # 1. Resize
            transforms.Resize((img_size, img_size)),

            # 2. Ensure grayscale (1 channel)
            transforms.Grayscale(num_output_channels=1),

            # 3. Geometric augmentation (NO Flip)
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,              # Rotation handled above
                translate=(0.1, 0.1),   # ±10% translation
                scale=(0.9, 1.1),       # 90-110% scale
                fill=0                  # Fill with black
            ),

            # 4. Intensity augmentation (mild strength)
            transforms.ColorJitter(
                brightness=0.1,  # ±10% brightness
                contrast=0.1     # ±10% contrast
            ),

            # 5. Convert to tensor [0, 1]
            transforms.ToTensor(),

            # 6. Add Gaussian noise (ultrasound speckle simulation)
            AddGaussianNoise(mean=0., std=0.03),

            # 7. Normalization for 1-channel
            transforms.Normalize(mean=[0.5], std=[0.5])  # [0,1] -> [-1,1]
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    return transform


def get_dataloaders_DPU(data_dir, image_dir, window_size=7, batch_size=16,
                        num_workers=4, filter_train_mcp=True, filter_ratio=0.33):
    """
    Create train, validation, and test dataloaders for DPU model

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        image_dir: Directory containing ultrasound images
        window_size: Temporal window size (default: 7)
        batch_size: Batch size for training (default: 16)
        num_workers: Number of worker processes for data loading (default: 4)
        filter_train_mcp: If True, filter 1/3 of MCP 60-70° data in training (default: True)
        filter_ratio: Ratio to exclude from filter range (default: 0.33 = 1/3)

    Returns:
        train_loader, val_loader, test_loader
    """
    # File paths
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    # Get transforms
    train_transform = get_transforms_DPU(train=True, img_size=224)
    test_transform = get_transforms_DPU(train=False, img_size=224)

    # Create datasets
    print("\n" + "=" * 70)
    print("CREATING DATASETS (DPU-Friendly)")
    print("=" * 70)

    print("\nTrain Dataset:")
    train_dataset = UltrasoundAngleDataset_DPU(
        csv_file=train_csv,
        image_dir=image_dir,
        window_size=window_size,
        transform=train_transform,
        filter_mcp_range=(60, 70) if filter_train_mcp else None,
        filter_ratio=filter_ratio,
        random_seed=42
    )

    print("\nValidation Dataset:")
    val_dataset = UltrasoundAngleDataset_DPU(
        csv_file=val_csv,
        image_dir=image_dir,
        window_size=window_size,
        transform=test_transform,
        filter_mcp_range=None,  # No filtering for validation
        filter_ratio=0.0
    )

    print("\nTest Dataset:")
    test_dataset = UltrasoundAngleDataset_DPU(
        csv_file=test_csv,
        image_dir=image_dir,
        window_size=window_size,
        transform=test_transform,
        filter_mcp_range=None,  # No filtering for test
        filter_ratio=0.0
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\n" + "=" * 70)
    print("DATALOADER SUMMARY")
    print("=" * 70)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")
    print(f"Window size: {window_size}")
    print(f"Image channels: 1 (grayscale)")
    print(f"MCP filtering (train): {'Enabled' if filter_train_mcp else 'Disabled'}")
    if filter_train_mcp:
        print(f"  - Range: 60-70°")
        print(f"  - Exclude ratio: {filter_ratio:.2%}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("=" * 70)
    print("TESTING DATASET_DPU")
    print("=" * 70)

    data_dir = '/home/kimsj/delto_m_ros2/model_DPU/data'
    image_dir = '/home/kimsj/delto_m_ros2/model_DPU/data/251002_wrist'

    # Test with MCP filtering enabled
    train_loader, val_loader, test_loader = get_dataloaders_DPU(
        data_dir=data_dir,
        image_dir=image_dir,
        window_size=7,
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        filter_train_mcp=True,  # Enable MCP filtering
        filter_ratio=0.33       # Exclude 1/3 of 60-70° data
    )

    # Test loading one batch
    print("\n" + "=" * 70)
    print("TESTING BATCH LOADING")
    print("=" * 70)

    for images, targets in train_loader:
        print(f"\nBatch loaded successfully!")
        print(f"  Images shape: {images.shape}")  # (batch, window, 1, H, W)
        print(f"  Expected: (4, 7, 1, 224, 224)")
        print(f"  Targets shape: {targets.shape}")  # (batch, 2)
        print(f"  Sample target (MCP, PIP): {targets[0]}")
        print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
        break

    # Check MCP distribution
    print("\n" + "=" * 70)
    print("MCP ANGLE DISTRIBUTION")
    print("=" * 70)

    all_mcp_angles = []
    for _, targets in train_loader:
        all_mcp_angles.extend(targets[:, 0].tolist())

    all_mcp_angles = np.array(all_mcp_angles)
    in_range = ((all_mcp_angles >= 60) & (all_mcp_angles <= 70)).sum()
    total = len(all_mcp_angles)

    print(f"\nTrain set MCP statistics:")
    print(f"  Total samples: {total}")
    print(f"  MCP in 60-70° range: {in_range} ({in_range/total*100:.1f}%)")
    print(f"  MCP min: {all_mcp_angles.min():.1f}°")
    print(f"  MCP max: {all_mcp_angles.max():.1f}°")
    print(f"  MCP mean: {all_mcp_angles.mean():.1f}°")
    print(f"  MCP std: {all_mcp_angles.std():.1f}°")

    print("\n" + "=" * 70)
    print("✓ Dataset test passed!")
    print("=" * 70)
