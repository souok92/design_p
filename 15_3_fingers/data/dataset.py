import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FingerAngleDataset(Dataset):
    """
    Dataset for finger angle prediction with temporal information.
    Uses sliding window to create sequences of images.
    """

    def __init__(self, data_dir, csv_files, image_folders, sequence_length=5,
                 transform=None, start_idx=None, end_idx=None):
        """
        Args:
            data_dir: Base data directory (e.g., '00_data')
            csv_files: List of CSV filenames (e.g., ['sj_1.csv', 'sj_2.csv'])
            image_folders: List of image folder names (e.g., ['sj_1', 'sj_2'])
            sequence_length: Number of frames in each sequence
            transform: Image transformations
            start_idx: Starting index for each session (dict: {session_name: idx})
            end_idx: Ending index for each session (dict: {session_name: idx})
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform or self._default_transform()

        # Load and process data
        self.samples = []

        for csv_file, img_folder in zip(csv_files, image_folders):
            csv_path = os.path.join(data_dir, csv_file)
            img_dir = os.path.join(data_dir, img_folder)

            # Load CSV
            df = pd.read_csv(csv_path, encoding='utf-8')

            # Apply index range if specified
            if start_idx and img_folder in start_idx:
                df = df[start_idx[img_folder]:]
            if end_idx and img_folder in end_idx:
                df = df[:end_idx[img_folder]]

            # MODIFIED: Extract target angles for 3 fingers (Index, Middle, Ring)
            angle_columns = [
                'Index_MCP_Flex', 'Index_PIP_Flex',
                'Middle_MCP_Flex', 'Middle_PIP_Flex',
                'Ring_MCP_Flex', 'Ring_PIP_Flex'
            ]

            # Create sequences using sliding window
            for i in range(sequence_length - 1, len(df)):
                # Get sequence of image paths
                image_paths = []
                valid_sequence = True

                for j in range(i - sequence_length + 1, i + 1):
                    time_str = df.iloc[j]['Time']  # Format: HH:MM:SS:MMM
                    # Convert to filename format: us_HHMMSS_MMM.png
                    time_parts = time_str.replace(':', '_').split('_')
                    if len(time_parts) == 4:
                        filename = f"us_{time_parts[0]}{time_parts[1]}{time_parts[2]}_{time_parts[3]}.png"
                        img_path = os.path.join(img_dir, filename)

                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                        else:
                            valid_sequence = False
                            break
                    else:
                        valid_sequence = False
                        break

                if valid_sequence and len(image_paths) == sequence_length:
                    # Get target angles for the last frame
                    angles = df.iloc[i][angle_columns].values.astype(np.float32)

                    self.samples.append({
                        'image_paths': image_paths,
                        'angles': angles,
                        'session': img_folder,
                        'frame_idx': i
                    })

        print(f"Dataset initialized with {len(self.samples)} samples")

    def _default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet standard input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and transform images
        images = []
        for img_path in sample['image_paths']:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack images: (sequence_length, C, H, W)
        images = torch.stack(images)

        # Target angles
        angles = torch.from_numpy(sample['angles'])

        return images, angles


def create_dataloaders(data_dir, person='sj', sequence_length=5, batch_size=32, num_workers=4):
    """
    Create train, val, test dataloaders for a specific person.

    Args:
        data_dir: Base data directory
        person: 'sj' or 'tk'
        sequence_length: Number of frames in sequence
        batch_size: Batch size
        num_workers: Number of workers for dataloader

    Returns:
        train_loader, val_loader, test_loader
    """

    if person == 'sj':
        # Train: sj_2 전체
        train_dataset = FingerAngleDataset(
            data_dir=data_dir,
            csv_files=['sj_2.csv'],
            image_folders=['sj_2'],
            sequence_length=sequence_length
        )

        # Val: sj_1 앞 50%
        val_dataset = FingerAngleDataset(
            data_dir=data_dir,
            csv_files=['sj_1.csv'],
            image_folders=['sj_1'],
            sequence_length=sequence_length,
            start_idx={'sj_1': 0},
            end_idx={'sj_1': 850}
        )

        # Test: sj_1 뒤 50%
        test_dataset = FingerAngleDataset(
            data_dir=data_dir,
            csv_files=['sj_1.csv'],
            image_folders=['sj_1'],
            sequence_length=sequence_length,
            start_idx={'sj_1': 850},
            end_idx={'sj_1': None}
        )

    elif person == 'tk':
        # Load full dataset first to get indices
        full_df = pd.read_csv(os.path.join(data_dir, 'tk.csv'), encoding='utf-8')
        total_len = len(full_df)

        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)

        # Train: 70%
        train_dataset = FingerAngleDataset(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=sequence_length,
            start_idx={'tk': 0},
            end_idx={'tk': train_end}
        )

        # Val: 15%
        val_dataset = FingerAngleDataset(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=sequence_length,
            start_idx={'tk': train_end},
            end_idx={'tk': val_end}
        )

        # Test: 15%
        test_dataset = FingerAngleDataset(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=sequence_length,
            start_idx={'tk': val_end},
            end_idx={'tk': None}
        )
    else:
        raise ValueError(f"Unknown person: {person}. Must be 'sj' or 'tk'")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
