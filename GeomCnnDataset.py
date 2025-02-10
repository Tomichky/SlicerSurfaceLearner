import logging
from typing import Optional, List, Tuple
from operator import itemgetter

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, RandRotate, RandFlip, RandZoom, EnsureType
import torch

from DeepLearnerLib.data_utils.utils import get_image_files_single_scalar
from DeepLearnerLib.data_utils.CustomDataset import GeomCnnDataset


class GeomCnnDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for handling geometric CNN datasets."""

    def __init__(
        self,
        batch_size: int = -1,
        val_frac: float = 0.2,
        num_workers: int = 4,
        data_tuple: Optional[Tuple[List[str], List[int], List[str], List[int]]] = None,
        file_paths: Optional[dict] = None,
        side: str = 'both'
    ):
        """
        Args:
            batch_size (int): Batch size for DataLoader. If -1, uses full dataset.
            val_frac (float): Fraction of data to use for validation.
            num_workers (int): Number of workers for DataLoader.
            data_tuple (Optional[Tuple]): Pre-split data for training and validation.
            file_paths (Optional[dict]): Dictionary containing file paths and configurations.
            side (str): Specifies which side to use ('left', 'right', or 'both').
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.file_paths = file_paths
        self.side = side
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_tuple = data_tuple

        # Define data transformations
        self.train_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),
            NormalizeIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ])

        self.val_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),
            NormalizeIntensity(),
            EnsureType(),
        ])

        self.test_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),
            NormalizeIntensity(),
            EnsureType(),
        ])

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        print("Setting up data loaders...")
        print(f"Stage: {stage}")

        if stage in (None, "fit"):
            self._setup_fit_stage()

        if stage in (None, "test"):
            self._setup_test_stage()

        print("Finished loading data!")

    def _setup_fit_stage(self):
        """Set up datasets for training and validation."""
        print("Loading training and validation data...")
        train_files, train_labels = get_image_files_single_scalar(
            FILE_PATHS=self.file_paths, data_dir="TRAIN_DATA_DIR"
        )
        print(f"Total training files: {len(train_files)}")

        if len(train_files) == 0:
            raise ValueError("No training files found in the specified directory.")

        if self.batch_size == -1:
            self.batch_size = len(train_files)
            print(f"Batch size set to: {self.batch_size}")

        if self.data_tuple is None:
            print("Splitting data into training and validation sets...")
            train_x, val_x, train_y, val_y = train_test_split(
                train_files, train_labels,
                test_size=self.val_frac,
                shuffle=True,
                stratify=train_labels,
                random_state=42
            )
            print(f"Training samples: {len(train_x)}, Validation samples: {len(val_x)}")
            self.train_ds = GeomCnnDataset(train_x, train_y, self.train_transforms, self.side)
            self.val_ds = GeomCnnDataset(val_x, val_y, self.val_transforms, self.side)
        else:
            print("Using provided data tuple for training and validation...")
            self.train_ds = GeomCnnDataset(*self.data_tuple[:2], self.train_transforms, self.side)
            self.val_ds = GeomCnnDataset(*self.data_tuple[2:], self.val_transforms, self.side)
            print(f"Training samples: {len(self.data_tuple[0])}, Validation samples: {len(self.data_tuple[2])}")

    def _setup_test_stage(self):
        
        print("Loading test data...")
        test_files, test_labels = get_image_files_single_scalar(
            FILE_PATHS=self.file_paths, data_dir="TEST_DATA_DIR"
        )
        print(f"Total test files: {len(test_files)}")

        if len(test_files) == 0:
            raise ValueError("No test files found in the specified directory.")

        print("Sample test file:", test_files[0] if test_files else "None")
        print("Sample test label:", test_labels[0] if test_labels else "None")
        self.test_ds = GeomCnnDataset(test_files, test_labels, self.test_transforms, self.side)

    def train_dataloader(self) -> DataLoader:
        
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
       
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class GeomCnnDataModuleKFold:
    

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        n_splits: int = 2,
        file_paths: Optional[dict] = None,
        side: str = 'both'
    ):
        """
        Args:
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            n_splits (int): Number of splits for K-Fold cross-validation.
            file_paths (Optional[dict]): Dictionary containing file paths and configurations.
            side (str): Specifies which side to use ('left', 'right', or 'both').
        """
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.num_workers = num_workers
        self.file_paths = file_paths
        self.side = side
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.datamodules = self._split_data()

    def _split_data(self) -> List[GeomCnnDataModule]:
       
        print("Splitting data into K-Folds...")
        train_files, train_labels = get_image_files_single_scalar(
            FILE_PATHS=self.file_paths, data_dir="TRAIN_DATA_DIR"
        )
        skf = StratifiedKFold(n_splits=self.n_splits)
        datamodule_list = []

        for train_index, val_index in skf.split(train_files, train_labels):
            train_x = list(itemgetter(*train_index.tolist())(train_files))
            train_y = list(itemgetter(*train_index.tolist())(train_labels))
            valid_x = list(itemgetter(*val_index.tolist())(train_files))
            valid_y = list(itemgetter(*val_index.tolist())(train_labels))

            datamodule_list.append(GeomCnnDataModule(
                batch_size=self.batch_size,
                data_tuple=(train_x, train_y, valid_x, valid_y),
                num_workers=self.num_workers,
                file_paths=self.file_paths,
                side=self.side
            ))

        return datamodule_list

    def get_folds(self) -> List[GeomCnnDataModule]:
        """Return the list of DataModules for each fold."""
        return self.datamodules


# Logger for TensorBoard
logger = TensorBoardLogger(save_dir="logs/", name="geom_cnn_training")