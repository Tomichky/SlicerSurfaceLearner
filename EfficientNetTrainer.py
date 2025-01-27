import logging
import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn
from monai.networks.nets import EfficientNetBN
from monai.networks.nets import DenseNet
from monai.networks.nets import SEResNet50
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar

from DeepLearnerLib.Asynchrony import Asynchrony
from DeepLearnerLib.models.cnn_model import SimpleCNN
from DeepLearnerLib.pl_modules.classifier_modules import ImageClassifier
from DeepLearnerLib.data_utils.GeomCnnDataset import GeomCnnDataModule
from DeepLearnerLib.data_utils.GeomCnnDataset import GeomCnnDataModuleKFold


def weight_reset(m):
    if isinstance(m, torch.nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def setProgressBar(qtProgressBarObject, value):
    qtProgressBarObject.setValue(value)


class LitProgressBar(ProgressBar):
    def __init__(self, qtProgressBarObject):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.qtProgressBarObject = qtProgressBarObject

    def disable(self):
        self.enable = False

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        super().on_train_epoch_end(trainer, pl_module)  # don't forget this :)
        percent = (pl_module.current_epoch + 1) * 100.0 / trainer.max_epochs
        Asynchrony.RunOnMainThread(lambda: setProgressBar(self.qtProgressBarObject, percent))


def cli_main(args):
    # -----------
    # Data
    # -----------
    print("ARGSSSSSSSSSSSSSSSSSS",args)
    if args["n_folds"] == 1:
        data_modules = [
            GeomCnnDataModule(
                batch_size=args["batch_size"],
                num_workers=args["data_workers"],
                file_paths=args["file_paths"]
            )
        ]
    else:
        data_module_generator = GeomCnnDataModuleKFold(
            batch_size=args["batch_size"],
            num_workers=args["data_workers"],
            n_splits=args["n_folds"],
            file_paths=args["file_paths"]
        )
        data_modules = data_module_generator.get_folds()

        # ------------
        # model
        # ------------
    print("Vérification du modèle choisi...")
    if args["model"] == "eff_bn":
        print("EfficientNet choisi.")
        backbone = EfficientNetBN(
            model_name="efficientnet-b0",
            in_channels=args["in_channels"],
            pretrained=True,
            num_classes=2
        )
    elif args["model"] == "densenet":
        print("DenseNet choisi.")
        backbone = DenseNet(
            spatial_dims=2,
            in_channels=args["in_channels"],
            out_channels=2
        )
    elif args["model"] == "resnet":
        print("SEResNet choisi.")
        backbone = SEResNet50(
            spatial_dims=2,
            in_channels=args["in_channels"],
            num_classes=2,
            pretrained=True
        )
    else:
        print("SimpleCNN choisi.")
        backbone = SimpleCNN(
            in_channels=args["in_channels"],
            w=args["w"]
        )

# Création du modèle
    try:
        print("Création du modèle en cours...")
        device = "cuda" if torch.cuda.is_available() and args["gpus"] else "cpu"
        model = ImageClassifier(
            backbone,
            learning_rate=args["learning_rate"],
            criterion=torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor([1.0, args["pos_weight"]]).to(device)),
            device=device,
            metrics=["acc", "precision", "recall"]
        )
        model=model.to(device)
        print("Modèle créé :", model)
    except Exception as e:
        print(f"Erreur lors de la création du modèle : {e}")


    for i in range(args["n_folds"]):
        # logger
        print("PASSAGE AU LOGGER")
        print(args["write_dir"])
        try:
            os.makedirs(os.path.join(args["write_dir"], "logs", args["model"]), exist_ok=True)
            print("Directory created successfully")
        except Exception as e:
            print("Error creating directory:", e)

        logger = TensorBoardLogger(
            save_dir=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i)),
            name=args["exp_name"]
        )
        print("Après création du logger")
        # early stopping
        print("Avant creation du early stopping")
        es = EarlyStopping(
            monitor='validation/valid_loss',
            patience=30
        )
        print("apres creation de l'early stopping")
        progressBar = LitProgressBar(args["qtProgressBarObject"])
        print("apres creation de la progress bar")
        checkpointer = ModelCheckpoint(
            monitor=args["monitor"],
            save_top_k=args["maxCp"], verbose=True, save_last=False,
            every_n_epochs=args["cp_n_epoch"],
            dirpath=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "checkpoints")
        )
        print("Apres la creation du checkpoint ")
        # ------------
        # training
        # ------------

        try:
            trainer = pl.Trainer(
                max_epochs=args["max_epochs"],
                accelerator=device,
                log_every_n_steps=5,
                num_sanity_val_steps=1,
                logger=logger,
                callbacks=[progressBar, checkpointer, es]
            )
            print("Le trainer est bien passe")
        except Exception as e:
            print(f"Erreur lors de l'initialisation du Trainer: {e}")
            raise
        trainer.fit(model, datamodule=data_modules[i])
        saved_name = os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "model.pt")
        logging.info(f"Saving model: {saved_name}")
        torch.save(model.backbone, saved_name)
        print(f"==============================Fold {i}: Training Finished==============================")
        model.apply(weight_reset)
        if args["qtProgressBarObject"] is not None:
            Asynchrony.RunOnMainThread(lambda: setProgressBar(args["qtProgressBarObject"], 0.0))
    print(f"Finished Training!")

#
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument('--batch_size', default=2, type=int)
#     parser.add_argument('--learning_rate', type=float, default=0.0001)
#     parser.add_argument('--in_channels', type=int, default=2)
#     parser.add_argument('--num_classes', type=int, default=2)
#     parser.add_argument('--max_epoch', type=int, default=1)
#     parser.add_argument('--data_workers', type=int, default=4)
#     parser.add_argument('--model', type=str, default="simple_cnn")
#     parser.add_argument('--use_gpu', type=bool, default=False)
#     parser.add_argument('--n_folds', type=int, default=1)
#     parser.add_argument('--write_dir', type=str, default="default_write_dir")
#     parser.add_argument('--exp_name', type=str, default="default_name")
#     parser.add_argument('--cp_n_epoch', type=int, default=1)
#     parser.add_argument('--maxCp', type=int, default=2)
#     parser.add_argument('--monitor', type=str, default="validation/valid_loss")
#     parser.add_argument('--pos_weight', type=float, default=1.0)
#
#     args = vars(parser.parse_args())
#     DEFAULT_FILE_PATHS = {}
#     DEFAULT_FILE_PATHS["TRAIN_DATA_DIR"] = "/Users/mturja/SurfaceDataSimulatedSphere"
#     DEFAULT_FILE_PATHS["FEATURE_DIRS"] = ["eacsf"]
#     DEFAULT_FILE_PATHS["TIME_POINTS"] = ["V06"]
#     DEFAULT_FILE_PATHS["FILE_SUFFIX"] = ["_flat", "_flat"]
#     DEFAULT_FILE_PATHS["FILE_EXT"] = ".png"
#     args["file_paths"] = DEFAULT_FILE_PATHS
#     args["w"] = 512
#     args["max_epochs"] = 10
#
#     args["qtProgressBarObject"] = None
#     cli_main(args)
