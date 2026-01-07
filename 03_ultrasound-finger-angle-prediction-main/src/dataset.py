#!/usr/bin/env python3
"""
Dataset and DataLoader for Temporal Ultrasound-to-Angle Prediction
Implements sliding window approach for temporal sequences
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UltrasoundAngleDataset(Dataset):
    """
    Temporal Dataset for Ultrasound-to-Angle Prediction

    Uses sliding window to create sequences of frames for TCN input
    """
    def __init__(self, csv_file, image_dir, window_size=7, transform=None):
        """
        Args:
            csv_file: Path to CSV file (train.csv, val.csv, or test.csv)
            image_dir: Directory containing ultrasound images
            window_size: Number of consecutive frames for temporal modeling
            transform: Optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.window_size = window_size
        self.transform = transform

        # For temporal sequences, we can only use samples where we have enough history
        # Valid indices start from window_size - 1
        self.valid_indices = list(range(window_size - 1, len(self.data)))

        print(f"Dataset initialized:")
        print(f"  Total samples in CSV: {len(self.data)}")
        print(f"  Window size: {window_size}")
        print(f"  Valid temporal sequences: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns a temporal sequence of images and corresponding target angles

        Returns:
            images: Tensor of shape (window_size, 3, H, W)
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

            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            images.append(image)

        # Stack images along new dimension: (window_size, C, H, W)
        images = torch.stack(images, dim=0)

        # Get target angles (from the last frame in the window)
        mcp_angle = float(self.data.iloc[actual_idx]['MCP_Flex'])
        pip_angle = float(self.data.iloc[actual_idx]['PIP_Flex'])
        target = torch.tensor([mcp_angle, pip_angle], dtype=torch.float32)

        return images, target


def get_transforms(train=True, img_size=224):
    """
    Get image transforms for training or validation/test

    Args:
        train: If True, apply data augmentation
        img_size: Target image size

    Returns:
        transforms: Composed transforms
    """
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),  # Mild augmentation
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def get_dataloaders(data_dir, image_dir, window_size=7, batch_size=16, num_workers=4):
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        image_dir: Directory containing ultrasound images
        window_size: Temporal window size
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    # File paths
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    # Get transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # Create datasets
    print("\n" + "=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    print("\nTrain Dataset:")
    train_dataset = UltrasoundAngleDataset(
        train_csv, image_dir, window_size=window_size, transform=train_transform
    )

    print("\nValidation Dataset:")
    val_dataset = UltrasoundAngleDataset(
        val_csv, image_dir, window_size=window_size, transform=test_transform
    )

    print("\nTest Dataset:")
    test_dataset = UltrasoundAngleDataset(
        test_csv, image_dir, window_size=window_size, transform=test_transform
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

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("=" * 70)
    print("TESTING DATASET")
    print("=" * 70)

    data_dir = '/home/kimsj/delto_m_ros2/model/data'
    image_dir = '/home/kimsj/delto_m_ros2/model/data/251002_wrist'

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        image_dir=image_dir,
        window_size=7,
        batch_size=4,
        num_workers=0  # Use 0 for testing
    )

    # Test loading one batch
    print("\n" + "=" * 70)
    print("TESTING BATCH LOADING")
    print("=" * 70)

    for images, targets in train_loader:
        print(f"\nBatch loaded successfully!")
        print(f"  Images shape: {images.shape}")  # (batch, window, C, H, W)
        print(f"  Targets shape: {targets.shape}")  # (batch, 2)
        print(f"  Sample target (MCP, PIP): {targets[0]}")
        break

    print("\n" + "=" * 70)
    print("✓ Dataset test passed!")
    print("=" * 70)
