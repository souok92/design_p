import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FingerAngleDataset3D(Dataset):
    """
    Dataset for 3D CNN finger angle prediction.
    Uses sliding window to create sequences of images.
    """
    
    def __init__(self, data_dir, csv_files, image_folders, sequence_length=16,
                 transform=None, start_idx=None, end_idx=None, 
                 stride=1, augment=False, image_size=224):
        """
        Args:
            data_dir: Base data directory (e.g., '../00_data')
            csv_files: List of CSV filenames
            image_folders: List of image folder names
            sequence_length: Number of frames in each sequence
            transform: Image transformations
            start_idx: Starting index for each session
            end_idx: Ending index for each session
            stride: Frame sampling stride
            augment: Apply data augmentation
            image_size: Output image size (224 recommended for detail)
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
                # Get sequence of image paths with stride
                image_paths = []
                valid_sequence = True
                
                frame_indices = range(i - seq_span + 1, i + 1, stride)
                
                for j in frame_indices:
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
        
        print(f"Dataset initialized with {len(self.samples)} samples "
              f"(seq_len={sequence_length}, stride={stride}, size={image_size}x{image_size})")
    
    def _default_transform(self):
        """Default image transformations"""
        # 원본 비율 유지하면서 resize 후 center crop
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),  # 약간 크게
            transforms.CenterCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],  # Kinetics mean
                                 std=[0.22803, 0.22145, 0.216989])   # Kinetics std
        ])
    
    def _augment_transform(self):
        """Data augmentation transforms"""
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.3),  # 초음파는 flip 조심
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Select transform
        if self.augment and np.random.random() > 0.5:
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


def create_dataloaders(data_dir, person='sj', sequence_length=16, 
                       batch_size=8, num_workers=4, stride=1, 
                       augment=False, image_size=224):
    """
    Create train, val, test dataloaders for a specific person.
    
    Args:
        data_dir: Base data directory
        person: 'sj' or 'tk'
        sequence_length: Number of frames in sequence
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        stride: Frame sampling stride
        augment: Apply augmentation to training data
        image_size: Image size (224 recommended)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    if person == 'sj':
        # Train: sj_2 전체
        train_dataset = FingerAngleDataset3D(
            data_dir=data_dir,
            csv_files=['sj_2.csv'],
            image_folders=['sj_2'],
            sequence_length=sequence_length,
            stride=stride,
            augment=augment,
            image_size=image_size
        )
        
        # Val: sj_1 앞 50%
        val_dataset = FingerAngleDataset3D(
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
        
        # Test: sj_1 뒤 50%
        test_dataset = FingerAngleDataset3D(
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
        # Load full dataset first to get indices
        full_df = pd.read_csv(os.path.join(data_dir, 'tk.csv'), encoding='utf-8')
        total_len = len(full_df)
        
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)
        
        # Train: 70%
        train_dataset = FingerAngleDataset3D(
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
        
        # Val: 15%
        val_dataset = FingerAngleDataset3D(
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
        
        # Test: 15%
        test_dataset = FingerAngleDataset3D(
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
        raise ValueError(f"Unknown person: {person}. Must be 'sj' or 'tk'")
    
    # Create dataloaders
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


if __name__ == '__main__':
    # Test dataset
    print("Testing dataset with 224x224...")
    
    data_dir = '../00_data'
    
    if os.path.exists(data_dir):
        dataset = FingerAngleDataset3D(
            data_dir=data_dir,
            csv_files=['tk.csv'],
            image_folders=['tk'],
            sequence_length=16,
            stride=1,
            image_size=224,
            start_idx={'tk': 0},
            end_idx={'tk': 100}
        )
        
        if len(dataset) > 0:
            images, angles = dataset[0]
            print(f"Image tensor shape: {images.shape}")  # Expected: (16, 3, 224, 224)
            print(f"Angles shape: {angles.shape}")
            print(f"Angles: {angles}")
            
            # Memory estimate
            # batch_size=8, seq=16, 3x224x224 = 8 * 16 * 3 * 224 * 224 * 4 bytes
            mem_per_batch = 8 * 16 * 3 * 224 * 224 * 4 / (1024**3)
            print(f"\nEstimated memory per batch (input only): {mem_per_batch:.2f} GB")
    else:
        print(f"Data directory not found: {data_dir}")
