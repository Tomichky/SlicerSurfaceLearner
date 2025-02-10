import logging
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn
from monai.networks.nets import EfficientNetBN, DenseNet, SEResNet50
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBarBase

from DeepLearnerLib.Asynchrony import Asynchrony
from DeepLearnerLib.models.cnn_model import SimpleCNN
from DeepLearnerLib.pl_modules.classifier_modules import ImageClassifier
from DeepLearnerLib.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold


def weight_reset(m):
    """Reset the weights of the model if it has a `reset_parameters` method."""
    if isinstance(m, torch.nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def set_progress_bar(qt_progress_bar_object, value):
    """Set the value of the Qt progress bar."""
    qt_progress_bar_object.setValue(value)


class LitProgressBarBase(ProgressBarBase):
    """Custom progress bar for PyTorch Lightning that updates a Qt progress bar."""
    def __init__(self, qt_progress_bar_object):
        super().__init__()
        self.qt_progress_bar_object = qt_progress_bar_object

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        percent = (pl_module.current_epoch + 1) * 100.0 / trainer.max_epochs

        if not hasattr(Asynchrony._ThreadLocalStorage, 'mainQueue'):
            Asynchrony._ThreadLocalStorage.mainQueue = []

        Asynchrony.RunOnMainThread(lambda: set_progress_bar(self.qt_progress_bar_object, percent))


def save_args_to_file(args, write_dir):
    """Save the arguments to a file for future reference."""
    os.makedirs(write_dir, exist_ok=True)
    args_file_path = os.path.join(write_dir, "args.txt")
    with open(args_file_path, 'w') as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
    print(f"Arguments saved to {args_file_path}")


def cli_main(args):
    """Main function to handle the training process."""
    save_args_to_file(args, args["write_dir"])

    # Data Module Initialization
    if args["n_folds"] == 1:
        data_modules = [GeomCnnDataModule(
            batch_size=args["batch_size"],
            num_workers=args["data_workers"],
            file_paths=args["file_paths"]
        )]
    else:
        data_module_generator = GeomCnnDataModuleKFold(
            batch_size=args["batch_size"],
            num_workers=args["data_workers"],
            n_splits=args["n_folds"],
            file_paths=args["file_paths"],
        )
        data_modules = data_module_generator.get_folds()
        print(f"Number of folds: {len(data_modules)}")

    # Model Initialization
    if args["model"] == "eff_bn":
        backbone = EfficientNetBN(
            model_name="efficientnet-b0",
            in_channels=args["in_channels"],
            pretrained=True,
            num_classes=2
        )
    elif args["model"] == "densenet":
        backbone = DenseNet(
            spatial_dims=2,
            in_channels=args["in_channels"],
            out_channels=2
        )
    elif args["model"] == "resnet":
        backbone = SEResNet50(
            spatial_dims=2,
            in_channels=args["in_channels"],
            num_classes=2,
            pretrained=True
        )
    else:
        backbone = SimpleCNN(
            in_channels=args["in_channels"],
            w=args["w"]
        )
    print(f"Model: {args['model']}")
    print(f"Backbone architecture: {backbone}")

    device = "cuda" if torch.cuda.is_available() and args["use_gpu"] else "cpu"
    print(f"Using device: {device}")

    model = ImageClassifier(
        backbone,
        learning_rate=args["learning_rate"],
        criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, args["pos_weight"]])),
        custom_device=device,
        metrics=["acc", "precision", "recall"]
    )
    print(f"Model initialized successfully.")

    # Training Loop
    for i in range(args["n_folds"]):
        print(f"==============================Fold {i}: Starting Training==============================")
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i)),
            name=args["exp_name"]
        )
        print(f"TensorBoard logger initialized for fold {i}.")

        # Callbacks
        es = EarlyStopping(monitor='validation/valid_loss', patience=30)
        progress_bar = LitProgressBarBase(args["qtProgressBarObject"])
        checkpointer = ModelCheckpoint(
            monitor=args["monitor"],
            save_top_k=args["maxCp"],
            verbose=True,
            save_last=False,
            every_n_epochs=args["cp_n_epoch"],
            dirpath=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "checkpoints")
        )
        print(f"Callbacks initialized for fold {i}.")

        # Trainer
        trainer = pl.Trainer(
            max_epochs=args["max_epochs"],
            accelerator=device,
            log_every_n_steps=5,
            num_sanity_val_steps=1,
            logger=logger,
            callbacks=[checkpointer, es, progress_bar]
        )
        print(f"Trainer initialized for fold {i}.")

        # Training
        print(f"Starting training for fold {i}...")
        try:
            trainer.fit(model, datamodule=data_modules[i])
        except Exception as e:
            print(f"Error during training for fold {i}: {str(e)}")
        print(f"Training completed for fold {i}.")

        # Save Model
        saved_name = os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "model.pt")
        logging.info(f"Saving model: {saved_name}")
        torch.save(model.backbone, saved_name)
        print(f"Model saved for fold {i}.")

        print(f"==============================Fold {i}: Training Finished==============================")
        model.apply(weight_reset)
        print(f"Model weights reset for fold {i}.")

        # Reset Progress Bar
        if args["qtProgressBarObject"] is not None:
            if not hasattr(Asynchrony._ThreadLocalStorage, 'mainQueue'):
                Asynchrony._ThreadLocalStorage.mainQueue = []
            Asynchrony.RunOnMainThread(lambda: set_progress_bar(args["qtProgressBarObject"], 0.0))
            print(f"Progress bar reset for fold {i}.")
    print(f"Finished Training!")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--in_channels', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--data_workers', type=int, default=32)
    parser.add_argument('--model', type=str, default="densenet")
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--n_folds', type=int, default=13)
    parser.add_argument('--write_dir', type=str, default="/work/bigo/data/test_code_V06_0702_both_ct")
    parser.add_argument('--exp_name', type=str, default="Test")
    parser.add_argument('--cp_n_epoch', type=int, default=1)
    parser.add_argument('--maxCp', type=int, default=2)
    parser.add_argument('--monitor', type=str, default="validation/valid_loss")
    parser.add_argument('--pos_weight', type=float, default=5.0)
    parser.add_argument('--side', type=str, choices=['left', 'right', 'both'], default='both')

    args = vars(parser.parse_args())

    # Default File Paths
    DEFAULT_FILE_PATHS = {
        "TRAIN_DATA_DIR": "/NIRAL/work/bigo/data/Non_normalized",
        "FEATURE_DIRS": ["ct"],
        "TIME_POINTS": ["V06"],
        "FILE_SUFFIX": ["_flat", "_flat"],
        "FILE_EXT": ".png",
        "CSV_path": '/work/bigo/data/csv_martin_training_2.csv',
        "group": 'group',
        "ID": 'CombinedID'
    }
    args["file_paths"] = DEFAULT_FILE_PATHS
    args["w"] = 512
    args["side"] = "both"

    # Adjust in_channels based on side
    if args["side"] == "left" or args["side"] == "right":
        args["in_channels"] = len(DEFAULT_FILE_PATHS["TIME_POINTS"]) * len(DEFAULT_FILE_PATHS["FEATURE_DIRS"])
    else:
        args["in_channels"] = len(DEFAULT_FILE_PATHS["TIME_POINTS"]) * len(DEFAULT_FILE_PATHS["FEATURE_DIRS"]) * 2

    args["qtProgressBarObject"] = None
    print(args)
    cli_main(args)