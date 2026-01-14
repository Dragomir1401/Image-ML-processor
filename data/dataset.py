import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.target_size = target_size
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_path = self.root_dir / cls
            for img_path in cls_path.glob('*.png'):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
            for img_path in cls_path.glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.target_size:
            image = image.resize(self.target_size, Image.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label

def get_data_loaders(dataset_path, batch_size=32, num_workers=2, target_size=None, augment=False):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageDataset(dataset_path, 'train', train_transform, target_size)
    test_dataset = ImageDataset(dataset_path, 'test', test_transform, target_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    
    val_loader = None
    val_dir = Path(dataset_path) / 'val'
    if val_dir.exists():
        val_dataset = ImageDataset(dataset_path, 'val', test_transform, target_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False)
    
    return train_loader, test_loader, val_loader, train_dataset.classes
