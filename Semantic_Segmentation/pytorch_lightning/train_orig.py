import os
import argparse

import wandb
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from utils import label_accuracy_score, add_hist
from transform import  make_transform
from models import SmpModel

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_json_path', type=str, default='../data/train.json')
parser.add_argument('--val_json_path', type=str, default='../data/val.json')

# parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--archi', type=str, default='Unet')
parser.add_argument('--backbone', type=str, default='efficientnet-b0')
parser.add_argument('--pretrained_weights', type=str, default='imagenet')
parser.add_argument('--fp16', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--auto_batch_size', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--scheduler', type=str, default='cosineanneal')
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--RandomScale', type=bool, default=False)
parser.add_argument('--RandomBrightnessContrast', type=bool, default=False)
parser.add_argument('--HueSaturationValue', type=bool, default=False)
parser.add_argument('--RGBShift', type=bool, default=False)
parser.add_argument('--RandomGamma', type=bool, default=False)
parser.add_argument('--HorizontalFlip', type=bool, default=False)
parser.add_argument('--VerticalFlip', type=bool, default=False)
parser.add_argument('--ImageCompression', type=bool, default=False)
parser.add_argument('--ShiftScaleRotate', type=bool, default=False)
parser.add_argument('--ShiftScaleRotateMode', type=int, default=4) # Constant, Replicate, Reflect, Wrap, Reflect101
parser.add_argument('--Downscale', type=bool, default=False)
parser.add_argument('--GridDistortion', type=bool, default=False)
parser.add_argument('--MotionBlur', type=bool, default=False)
parser.add_argument('--RandomResizedCrop', type=bool, default=False)
parser.add_argument('--CLAHE', type=bool, default=False)

args = parser.parse_args()


if __name__ == '__main__':
    # model

    # SWA = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=5, annealing_strategy='cos')
    pl.seed_everything(args.seed)
    wandb_logger = WandbLogger(project='PL_Seg')
    # wandb_logger.log_hyperparams(args)
    # split_num = args.train_json_path.split('_')[0][-1]

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor="val/mIoU",
        dirpath="saved",
        filename=f"{wandb_logger.name}"+"-{epoch:03d}-{val/mIoU:.4f}",
        save_top_k=1,
        mode="max",
    )
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=False, mode="min")

    train_transform, val_transform = make_transform(args)

    model = SmpModel(args, train_transform, val_transform)

    trainer = pl.Trainer(gpus=args.gpus,
                         precision=16 if args.fp16 else 32,
                         max_epochs=args.epochs,
                         # auto_scale_batch_size=args.auto_batch_size,
                         # log_every_n_steps=1,
                         accelerator='ddp',
                         # limit_train_batches=5,
                         # limit_val_batches=2,
                         logger=wandb_logger,
                         callbacks=[lr_monitor, checkpoint_callback])  # , callbacks=[SWA])

    if args.auto_batch_size:
        new_batch_size = trainer.tune(model)
        model.hparams.batch_size = new_batch_size

    trainer.fit(model)

