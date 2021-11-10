import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduelr

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset import CustomDataLoader

class BaseModel(pl.LightningModule):
    def __init__(self, args, train_transform, val_transform):
        super().__init__()
        # self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, num_classes=11)
        # self.criterion = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.train_json_path = args.train_json_path
        self.val_json_path = args.val_json_path
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.args = args

        if args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = lr_scheduelr.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        self.log('train_loss', loss)
        return loss

    def prepare_data(self):
        self.train_data = CustomDataLoader(self.train_json_path, mode='train', transform=self.train_transform)
        self.val_data = CustomDataLoader(self.val_json_path, mode='val', transform=self.val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.args.num_workers)


if __name__ == '__main__':
    model = BaseModel()
    print(model)