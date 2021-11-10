import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduelr
from adamp import AdamP
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.functional import iou, accuracy
import albumentations as A

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset import CustomDataLoader

from torch.utils.data import Sampler,RandomSampler,SequentialSampler
import numpy as np

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None,img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 416
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class SmpModel(pl.LightningModule):
    def __init__(self, args=None, train_transform=None, val_transform=None):
        super().__init__()
        arhci_name_list = sorted([name for name in smp.__dict__ if not (name.islower() or name.startswith('__'))])

        assert (args.archi in arhci_name_list), \
            (f"[!] Architecture Name is wrong, check Archi config, expected: {arhci_name_list} received: {args.archi}")

        self.model = getattr(smp, args.archi)(
            encoder_name=args.backbone,
            encoder_weights=args.pretrained_weights,
            in_channels=3,
            classes=11,
        )
        if train_transform and val_transform:
            self.batch_size = args.batch_size
            self.train_json_path = args.train_json_path
            self.val_json_path = args.val_json_path
            self.train_transform = train_transform
            self.val_transform = val_transform

            self.args = args

            if args.loss == 'ce':
                self.criterion = nn.CrossEntropyLoss()

            self.train_data = CustomDataLoader(self.train_json_path, mode='train', transform=self.train_transform)
            self.val_data = CustomDataLoader(self.val_json_path, mode='val', transform=self.val_transform)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduelr.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduelr.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min= 1e-5,
                                                    last_epoch=-1, verbose=True)
        # return [optimizer], [scheduler]
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/mIoU"}
        # return [optimizer]

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        iou_value = iou(outputs.argmax(dim=1), mask)
        acc_value = accuracy(outputs, mask)

        # self.log('train/loss', loss)
        return {"loss": loss, "IoU": iou_value, "acc": acc_value}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        iou_value = iou(outputs.argmax(dim=1), mask)
        acc_value = accuracy(outputs, mask)

        # self.log('val/loss', loss)
        return {"loss": loss, "IoU": iou_value, "acc": acc_value}

    def training_epoch_end(self, outputs):
        total_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0

        iter_count = len(outputs)

        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()
            total_iou += outputs[idx]['IoU'].item()
            total_acc += outputs[idx]['acc'].item()

        self.log('train/loss', total_loss / iter_count)
        self.log('train/acc', total_acc / iter_count)
        self.log('train/mIoU', total_iou / iter_count)

        # self.log('train', {"loss": total_loss / iter_count,
        #                    "acc": total_acc / iter_count,
        #                    "iou": total_iou / iter_count})

    def validation_epoch_end(self, outputs):
        total_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0

        iter_count = len(outputs)

        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()
            total_iou += outputs[idx]['IoU'].item()
            total_acc += outputs[idx]['acc'].item()

        self.log('val/loss', total_loss / iter_count)
        self.log('val/acc', total_acc / iter_count)
        self.log('val/mIoU', total_iou / iter_count)

        # self.log('val', {"loss": total_loss / iter_count,
        #                  "acc": total_acc / iter_count,
        #                  "iou": total_iou / iter_count})

    def train_dataloader(self):
        if not self.args.RandomScale:
            return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.args.num_workers)
        else:
            return torch.utils.data.DataLoader(self.train_data,
                                               batch_sampler=BatchSampler(RandomSampler(self.train_data),
                                                                          batch_size=self.args.batch_size,
                                                                          drop_last=True,
                                                                          multiscale_step=1,
                                                                          img_sizes=list(range(256, 1024, 32))),
                                               num_workers=self.args.num_workers)

    def val_dataloader(self):
        if not self.args.RandomScale:
            return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.args.num_workers)
        else:
            return torch.utils.data.DataLoader(self.val_data,
                                               batch_sampler=BatchSampler(RandomSampler(self.val_data),
                                                                          batch_size=self.args.batch_size,
                                                                          drop_last=True,
                                                                          multiscale_step=1,
                                                                          img_sizes=list(range(256, 1024, 32))),
                                               num_workers=self.args.num_workers)


if __name__ == '__main__':
    pass

    # from torchinfo import summary
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--archi', type=str, default='unet')
    # parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    # parser.add_argument('--pretrained_weights', type=str, default='imagenet')
    # args = parser.parse_args()
    #
    # model = SmpModel(args)
    # inputs = torch.randn(1, 3, 512, 512)
    # summary(model, (inputs.shape))
    # print()