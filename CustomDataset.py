import torch
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, NormalizeIntensity, EnsureType, RandRotate, RandFlip, RandZoom
import os


class GeomCnnDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        super(GeomCnnDataset, self).__init__()
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms      
        self.loader = LoadImage(image_only=True)
        self.sideness = 1 # 0 is left, 1 is right

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
    
        try:
            image_path = self.image_files[idx][self.sideness]
            image = self.loader(image_path)
            image = self.transforms(image)
            label = self.labels[idx]
            return image, label  

        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            

        


def check_batch_sizes(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx} - Image batch size: {images.shape}")
        if images.shape[0] != batch_size:
            print(f"Warning: Batch size mismatch in batch {batch_idx}. Expected {batch_size}, but got {images.shape[0]}")
        
        for i, img in enumerate(images):
            print(f"Image {i} shape in batch {batch_idx}: {img.shape}")
            if img.shape != images[0].shape:
                print(f"Warning: Image size mismatch in batch {batch_idx}, image {i}. Expected {images[0].shape}, but got {img.shape}")


if __name__ == "__main__":
    
    image_files = ["path/to/image1.nii", "path/to/image2.nii"] 
    labels = [0, 1]  
    transforms = None  

    dataset = GeomCnnDataset(image_files, labels, transforms)
    check_batch_sizes(dataset, batch_size=20)
