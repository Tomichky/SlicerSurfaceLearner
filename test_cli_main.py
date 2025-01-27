import logging
import os
import sys
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from monai.networks.nets import EfficientNetBN, DenseNet, SEResNet50
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
parent_of_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_of_parent_dir)
sys.path.append('/work/bigo/SlicerSALT/SlicerSALT-5.0.0-linux-amd64/bin/Python/qt')

from DeepLearnerLib.models.cnn_model import SimpleCNN
from DeepLearnerLib.pl_modules.classifier_modules import ImageClassifier
from DeepLearnerLib.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold

def cli_main(args):
    print("Début de cli_main")

    print("Préparation des données")
    data_modules = prepare_data_modules(args)
    
    results_file_path = os.path.join(args["write_dir"], "results.txt")
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    def prepare_logger(args, fold_idx):
        log_dir = os.path.join(args["write_dir"], "logs", args["model"], f"fold_{fold_idx}")
        os.makedirs(log_dir, exist_ok=True)
        return TensorBoardLogger(save_dir=log_dir, name=f"logs_fold_{fold_idx}")

    
    
    with open(results_file_path, 'a') as results_file:

        def prepare_logger(args, fold_idx):
            log_dir = os.path.join(args["write_dir"], "logs", args["model"], f"fold_{fold_idx}")
            os.makedirs(log_dir, exist_ok=True)
            return TensorBoardLogger(save_dir=log_dir, name=f"logs_fold_{fold_idx}")
        
        for i, data_module in enumerate(data_modules):
            logger = prepare_logger(args, i)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            checkpoint = prepare_checkpoint(args, i)

            print("Initialisation du modèle...")
            backbone = select_model(args)
            
            print("Création du modèle...")
            model = create_model(backbone, args)
            
            trainer = pl.Trainer(
                max_epochs=args["max_epochs"],
                accelerator="gpu" if args["gpus"] else "cpu",
                log_every_n_steps=5,
                logger=logger,
                callbacks=[early_stopping, checkpoint]
            )

            print(f"Setup pour le fold {i}")
          
            trainer.fit(model, datamodule=data_module)
            
            for batch in data_module.test_dataloader():
                print(batch)
                break
            print(f"Évaluation sur le fold {i}")
            test_results = trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=True)
            for batch in data_module.test_dataloader():
                inputs, labels = batch
                predictions = torch.argmax(model(inputs), dim=1)  
                labels_np = labels.cpu().numpy()
                predictions_np = predictions.cpu().numpy()
 
            results_file.write(f"Fold {i} Test Results:\n")
            for result in test_results:
                results_file.write(f"{result}\n")

            save_model(model, logger, i, args)
            print(f"Fold {i}: Entraînement terminé.")

def prepare_data_modules(args):
    if args["n_folds"] == 1:
        print(f"Chargement des données à partir de {args['file_paths']['TRAIN_DATA_DIR']}")
        print(f"Répertoire des fichiers de formation: {args['file_paths']['TRAIN_DATA_DIR']}")
        print(f"Répertoire des fichiers de test: {args['file_paths']['TEST_DATA_DIR']}")
    else:
        print(f"Folds pour {args['n_folds']}...")
    
    return GeomCnnDataModuleKFold(
        batch_size=args["batch_size"],
        num_workers=args["data_workers"],
        n_splits=args["n_folds"],
        file_paths=args["file_paths"]
    ).get_folds()

def select_model(args):
    if args["model"] == "eff_bn":
        return EfficientNetBN(model_name="efficientnet-b0", in_channels=args["in_channels"], pretrained=True, num_classes=2)
    elif args["model"] == "densenet":
        return DenseNet(spatial_dims=2, in_channels=args["in_channels"], out_channels=2)
    elif args["model"] == "resnet":
 
        return SEResNet50(spatial_dims=2, in_channels=args["in_channels"], num_classes=2, pretrained=True)
    else:

        return SimpleCNN(in_channels=args["in_channels"], w=args["w"])

def create_model(backbone, args):
    device = "cuda" if torch.cuda.is_available() and args["gpus"] else "cpu"
   
    return ImageClassifier(
        backbone,
        learning_rate=args["learning_rate"],
        criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, args["pos_weight"]]).to(device)),
        device=device
    )

def prepare_checkpoint(args, fold_idx):
    return ModelCheckpoint(
        monitor=args["monitor"],
        mode="min",
        save_top_k=args["maxCp"],
        verbose=True,
        save_last=False,
        every_n_epochs=args["cp_n_epoch"],
        dirpath=os.path.join(args["write_dir"], "logs", args["model"], f"fold_{fold_idx}", "checkpoints")
    )

def save_model(model, logger, fold_idx, args):
    saved_name = os.path.join(args["write_dir"], "logs", args["model"], f"fold_{fold_idx}", "model.pt")
    os.makedirs(os.path.dirname(saved_name), exist_ok=True)
    torch.save(model.backbone.state_dict(), saved_name)

    logging.info(f"Model saved: {saved_name}")

if __name__ == "__main__":
    args = {
        'batch_size': 50,
        'learning_rate': 0.0002,
        'in_channels': 1,
        'num_classes': 2,
        'max_epochs':200,
        'gpus': False,
        'model': 'fffff',
        'n_folds': 13,
        'data_workers': 32,
        'write_dir': '/work/bigo/data/training_test_test_3',
        'exp_name': 'default',
        'cp_n_epoch': 1,
        'maxCp': 2,
        'monitor': 'val_loss',
        'pos_weight': 5.0,
        'qtProgressBarObject': None,
        'file_paths': {
            'BASE_DIR': '/home/bigo/SlicerSurfaceLearner',
            'TRAIN_DATA_DIR': '/NIRAL/work/bigo/data/Non_normalized',
            'TEST_DATA_DIR': '/home/bigo/surface_data_test',
            'FEATURE_DIRS': ['sa',"ct","eacsf"],
            'FILE_SUFFIX': ['_flat', '_flat'],
            'TIME_POINTS': ['V06',"V12"],
            'FILE_EXT': '.png',
            'CSV_path': '/work/bigo/data/csv_martin_training_2.csv',
            'group_name': 'group',
            'id_name':'ID'
        },
        'w': 512,
    }
    
    cli_main(args)