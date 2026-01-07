import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class FingerAngleDatasetUNet(Dataset):
    """
    Dataset for U-Net + GRU finger angle prediction.
    
    논문에서 효과를 본 설정:
    - 연속 프레임 (stride=1)
    - 시퀀스 길이 10~15
    """
    
    def __init__(self, data_dir, csv_files, image_folders, sequence_length=10,
                 transform=None, start_idx=None, end_idx=None, 
                 stride=1, augment=False, image_size=224):
        """
        Args:
            data_dir: Base data directory
            csv_files: List of CSV filenames
            image_folders: List of image folder names
            sequence_length: Number of frames (default: 10)
            stride: Frame sampling stride (default: 1, 연속 프레임)
            augment: Apply data augmentation
            image_size: Output image size
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.augment = augment
        self.image_size = image_size
        self.transform = transform or self._default_transform()
        
        if augment:
            self.augment_transform = self._augment_transform()
        
        self.samples = []
        
        for csv_file, img_folder in zip(csv_files, image_folders):
            csv_path = os.path.join(data_dir, csv_file)
            img_dir = os.path.join(data_dir, img_folder)
            
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            s_idx = start_idx.get(img_folder, 0) if start_idx else 0
            e_idx = end_idx.get(img_folder, len(df)) if end_idx else len(df)
            df = df.iloc[s_idx:e_idx].reset_index(drop=True)
            
            angle_columns = [
                'Index_MCP_Flex', 'Index_PIP_Flex',
                'Middle_MCP_Flex', 'Middle_PIP_Flex',
                'Ring_MCP_Flex', 'Ring_PIP_Flex',
                'Pinky_MCP_Flex', 'Pinky_PIP_Flex'
            ]
            
            seq_span = (sequence_length - 1) * stride + 1
            
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
        
        interval_ms = stride * 130
        total_ms = (sequence_length - 1) * stride * 130
        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"  - Sequence: {sequence_length} frames, stride={stride}")
        print(f"  - Frame interval: ~{interval_ms}ms, total coverage: ~{total_ms}ms ({total_ms/1000:.1f}s)")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def _augment_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size + 20, self.image_size + 20)),
            transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.augment and random.random() > 0.5:
            transform = self.augment_transform
        else:
            transform = self.transform
        
        images = []
        for img_path in sample['image_paths']:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
        
        images = torch.stack(images)
        angles = torch.from_numpy(sample['angles'])
        
        return images, angles


def create_dataloaders(data_dir, person='sj', sequence_length=10, 
                       batch_size=8, num_workers=4, stride=1, 
                       augment=False, image_size=224):
    """Create train, val, test dataloaders"""
    
    if person == 'sj':
        train_dataset = FingerAngleDatasetUNet(
            data_dir=data_dir,
            csv_files=['sj_2.csv'],
            image_folders=['sj_2'],
            sequence_length=sequence_length,
            stride=stride,
            augment=augment,
            image_size=image_size
        )
        
        val_dataset = FingerAngleDatasetUNet(
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
        
        test_dataset = FingerAngleDatasetUNet(
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
        
        train_dataset = FingerAngleDatasetUNet(
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
        
        val_dataset = FingerAngleDatasetUNet(
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
        
        test_dataset = FingerAngleDatasetUNet(
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
