import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class GaussianNoise:
    """Add Gaussian noise to image"""
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


class RandomGaussianBlur:
    """Random Gaussian blur"""
    def __init__(self, p=0.3, radius_range=(0.5, 1.5)):
        self.p = p
        self.radius_range = radius_range
    
    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class FingerAngleDatasetTransformer(Dataset):
    """
    Dataset for Transformer-based finger angle prediction.
    
    특징:
    - stride 지원으로 프레임 간격 조절
    - 강화된 data augmentation
    - 8프레임 + stride=2 기본값 (약 1초 커버, 260ms 간격)
    """
    
    def __init__(self, data_dir, csv_files, image_folders, sequence_length=8,
                 transform=None, start_idx=None, end_idx=None, 
                 stride=2, augment=False, image_size=224):
        """
        Args:
            data_dir: Base data directory
            csv_files: List of CSV filenames
            image_folders: List of image folder names
            sequence_length: Number of frames in each sequence (default: 8)
            transform: Image transformations
            start_idx: Starting index for each session
            end_idx: Ending index for each session
            stride: Frame sampling stride (default: 2, 약 260ms 간격)
            augment: Apply data augmentation
            image_size: Output image size (default: 224)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.augment = augment
        self.image_size = image_size
        self.transform = transform or self._default_transform()
        
        if augment:
            self.augment_transform = self._augment_transform()
        
        # Load and process data
        self.samples = []
        
        for csv_file, img_folder in zip(csv_files, image_folders):
            csv_path = os.path.join(data_dir, csv_file)
            img_dir = os.path.join(data_dir, img_folder)
            
            # Load CSV
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Apply index range if specified
            s_idx = start_idx.get(img_folder, 0) if start_idx else 0
            e_idx = end_idx.get(img_folder, len(df)) if end_idx else len(df)
            df = df.iloc[s_idx:e_idx].reset_index(drop=True)
            
            # Extract target angles (exclude Thumb)
            angle_columns = [
                'Index_MCP_Flex', 'Index_PIP_Flex',
                'Middle_MCP_Flex', 'Middle_PIP_Flex',
                'Ring_MCP_Flex', 'Ring_PIP_Flex',
                'Pinky_MCP_Flex', 'Pinky_PIP_Flex'
            ]
            
            # Calculate actual sequence span considering stride
            seq_span = (sequence_length - 1) * stride + 1
            
            # Create sequences using sliding window
            for i in range(seq_span - 1, len(df)):
                image_paths = []
                valid_sequence = True
                
                frame_indices = range(i - seq_span + 1, i + 1, stride)
                
                for j in frame_indices:
                    time_str = df.iloc[j]['Time']
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
                    angles = df.iloc[i][angle_columns].values.astype(np.float32)
                    
                    self.samples.append({
                        'image_paths': image_paths,
                        'angles': angles,
                        'session': img_folder,
                        'frame_idx': i
                    })
        
        interval_ms = stride * 130  # 약 130ms per frame
        total_ms = (sequence_length - 1) * stride * 130
        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"  - Sequence: {sequence_length} frames, stride={stride}")
        print(f"  - Frame interval: ~{interval_ms}ms, total coverage: ~{total_ms}ms ({total_ms/1000:.1f}s)")
    
    def _default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.CenterCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def _augment_transform(self):
        """강화된 data augmentation"""
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.RandomCrop((self.image_size, self.image_size)),
            RandomGaussianBlur(p=0.2),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Select transform (same transform for all frames in sequence)
        if self.augment and random.random() > 0.5:
            transform = self.augment_transform
        else:
            transform = self.transform
        
        # Load and transform images
        images = []
        for img_path in sample['image_paths']:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
        
        # Stack images: (sequence_length, C, H, W)
        images = torch.stack(images)
        
        # Target angles
        angles = torch.from_numpy(sample['angles'])
        
        return images, angles


def create_dataloaders(data_dir, person='sj', sequence_length=8, 
                       batch_size=8, num_workers=4, stride=2, 
                       augment=False, image_size=224):
    """
    Create train, val, test dataloaders.
    
    Args:
        data_dir: Base data directory
        person: 'sj' or 'tk'
        sequence_length: Number of frames (default: 8)
        batch_size: Batch size
        num_workers: Number of workers
        stride: Frame sampling stride (default: 2)
        augment: Apply augmentation to training data
        image_size: Image size (default: 224)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    if person == 'sj':
        train_dataset = FingerAngleDatasetTransformer(
            data_dir=data_dir,
            csv_files=['sj_2.csv'],
            image_folders=['sj_2'],
            sequence_length=sequence_length,
            stride=stride,
            augment=augment,
            image_size=image_size
        )
        
        val_dataset = FingerAngleDatasetTransformer(
            data_dir=data_dir,
            csv_files=['sj_1.csv'],
            image_folders=['sj_1'],
            sequence_length=sequence_length,
            start_idx={'sj_1': 0},
            end_idx={'sj_1': 850},
            stride=stride,
            augment=False,
            image_size=image_size
        )
        
        test_dataset = FingerAngleDatasetTransformer(
            data_dir=data_dir,
            csv_files=['sj_1.csv'],
            image_folders=['sj_1'],
            sequence_length=sequence_length,
            start_idx={'sj_1': 850},
            end_idx={'sj_1': None},
            stride=stride,
            augment=False,
            image_size=image_size
        )
    
    elif person == 'tk':
        full_df = pd.read_csv(os.path.join(data_dir, 'tk.csv'), encoding='utf-8')
        total_len = len(full_df)
        
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)
        
        train_dataset = FingerAngleDatasetTransformer(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=sequence_length,
            start_idx={'tk': 0},
            end_idx={'tk': train_end},
            stride=stride,
            augment=augment,
            image_size=image_size
        )
        
        val_dataset = FingerAngleDatasetTransformer(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=sequence_length,
            start_idx={'tk': train_end},
            end_idx={'tk': val_end},
            stride=stride,
            augment=False,
            image_size=image_size
        )
        
        test_dataset = FingerAngleDatasetTransformer(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=sequence_length,
            start_idx={'tk': val_end},
            end_idx={'tk': None},
            stride=stride,
            augment=False,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown person: {person}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
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
