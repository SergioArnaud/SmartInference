import os
import argparse
from datetime import datetime
from pprint import pprint
import math

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from MDCNN_trainer_pl import Lit_MDCNN
from preprocess_mdcnn import preprocess_mdcnn


class EAN13Dataset(Dataset):
    def __init__(
        self, basedir, list_of_image_paths, list_of_labels_paths, folder="train"
    ):
        self.basedir = basedir
        self.image_dir = os.path.join(basedir, folder, "images")
        self.labels_dir = os.path.join(basedir, folder, "labels")
        self.imgs = list_of_image_paths
        self.img_labels = list_of_labels_paths

        self.transforms = preprocess_mdcnn

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, str(self.imgs[idx]))
        image = Image.open(img_path)
        image = self.transforms(image)

        label_path = os.path.join(self.labels_dir, str(self.img_labels[idx]))
        barcode_str = next(open(label_path, "r"))
        label = torch.tensor([int(i) for i in barcode_str])
        return image, label


def build_datasets(basedir, split=0.8, random_state=None, test=False):
    train_dir = os.path.join(basedir, "train")
    image_dir = os.path.join(train_dir, "images")
    labels_dir = os.path.join(train_dir, "labels")
    imgs = sorted(os.listdir(image_dir))
    img_labels = sorted(os.listdir(labels_dir))
    if test:
        imgs = imgs[:160]
        img_labels = img_labels[:160]
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        imgs, img_labels, train_size=split, random_state=random_state
    )
    train_dataset = EAN13Dataset(basedir, train_imgs, train_labels)
    if len(val_imgs) != 0:
        test_dataset = EAN13Dataset(basedir, val_imgs, val_labels)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def initialize_wandb(args):
    wandb.login()
    config = dict(
        learning_rate=args.learning_rate,
        backbone=args.backbone,
        pretrained_backbone=args.pretrained,
        architecture="MDCNN",
        dataset_id=args.path.split('/')[-1],
        infra="AWS",
    )

    wandb.init(
        project="decode_barcodes",
        notes="Arquitecture Test",
        tags=["baseline", "paper1"],
        config=config,
    )


def get_checkpoint_callback(args):
    saving_path = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        os.mkdir(saving_path)
    return ModelCheckpoint(dirpath=saving_path, save_top_k=1, monitor = 'val/val_loss')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MDCNN model")
    parser.add_argument(
        "--path",
        type=str,
        default="la-comer/src/data/barcode-decoding-dataset",
        help="path to decoding dataset base",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="Pretrained Backbone (type) bool",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone network to use",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/ec2-user/SageMaker/la-comer/src/models/smart_inference",
        help="path where the model's state dict get saved",
    )
    parser.add_argument(
        "--name", type=str, default=str(datetime.now()), help="name of saving dir"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for the training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-2, help="Maximum Learning rate"
    )
    parser.add_argument(
        "--T_0_scheduler",
        type=int,
        default=10,
        help="Number of iterations for the first restart.",
    )
    parser.add_argument(
        "--T_mult_scheduler",
        type=int,
        default=2,
        help="A factor increases T_i after a restart.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum Number of epochs",
    ) 
    parser.add_argument(
        "--with_scheduler",
        type=bool,
        default=False,
        help="Should use CosineAnnealingWarmRestarts as lr scheduler",
    )

    args = parser.parse_args()
    print('-'*30)
    print('   P A R A M E T E R S')
    pprint(vars(args))
    print('-'*30)
    
    initialize_wandb(args)
    checkpoint_callback = get_checkpoint_callback(args)
    
    trainer = pl.Trainer(
        gpus=-1,
        auto_select_gpus=True,
        accelerator="dp",
        logger=WandbLogger(),
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs
    )
    
    model = Lit_MDCNN(
        backbone=args.backbone,
        pretrained_backbone=args.pretrained,
        lr=args.learning_rate,
        T_0_scheduler=args.T_0_scheduler,
        T_mult_scheduler=args.T_mult_scheduler,
        scheduler=args.with_scheduler
    )

    training_data, test_data = build_datasets(args.path, split=0.95)
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, num_workers = os.cpu_count()
    )
    val_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers = os.cpu_count())

    trainer.fit(model, train_dataloader, val_dataloader)
