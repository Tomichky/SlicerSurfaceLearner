from monai.transforms import LoadImage
import torch
import numpy as np


class GeomCnnDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        super(GeomCnnDataset, self).__init__()
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.loader = LoadImage(image_only=True)
          


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path=self.image_files[idx]
        print(image_path)
        image_loaded=self.loader(image_path)
        image_transformed=self.transforms(image_loaded)
        input=np.concatenate(image_transformed,axis=0)
        return input, self.labels[idx]