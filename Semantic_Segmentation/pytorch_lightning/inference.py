import argparse

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
from models import SmpModel
from transform import make_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='test')
parser.add_argument('--archi', type=str, default='Unet')
parser.add_argument('--backbone', type=str, default='efficientnet-b0')
parser.add_argument('--pretrained_weights', type=str, default='imagenet')
parser.add_argument('--fp16', type=bool, default=False)

args = parser.parse_args()
# train_transform, val_transform = make_transform(args)

model = SmpModel.load_from_checkpoint("./saved/sample-mnist-epoch=09-val/mIoU=0.57.ckpt",
                                      args=args,
                                      train_transform=None,
                                      val_transform=None)
model.eval()
model = model.cuda()
test_path = '../data/test.json'

test_transform = A.Compose([
    A.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2471, 0.2435, 0.2616],
    ),
    ToTensorV2()
])

def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16,
                                          num_workers=4,
                                          collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for imgs, image_infos in tqdm(data_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array

# sample_submisson.csv 열기
submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

file_names, preds = test(model, test_loader, device)
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())},
                                   ignore_index=True)

submission.to_csv(f"./submission/{args.backbone}_{args.archi}_best_model_{args.name}.csv", index=False)