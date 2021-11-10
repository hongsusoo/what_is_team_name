import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset_path = '../data/'
category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam',
                  'Plastic bag', 'Battery', 'Clothing']

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __len__(self) -> int:
        return len(self.coco.getImgIds())

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.uint8)
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.uint8)

            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            # return images, masks, image_infos
            return images, masks.long()

        # 5. test mode면 return (image)
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
            # return images


if __name__ == "__main__":
    dataset_path = '../data/'
    # train_path = dataset_path + 'train_all.json'
    # test_path = dataset_path + 'test.json'
    # loader = CustomDataLoader(data_dir=train_path, mode='train')
    #
    # # data_dir = ['train_01.json', 'train_02.json', 'train_03.json']
    #
    # file_list = os.listdir('../data/')
    # train_file_list = [file for file in file_list if file.endswith('_all.json')]
    # # for i in range(len(train_file_list)):
    # # file_real_path = file_list + train_file_list
    # with open('../data/train0_all.json','w') as f:
    #     d_update = json.load(f, indent=4)
    # with open('../data/train1_all.json','w') as f:
    #     json.dump(d_update, f, indent=4)
    # with open('../data/train2_all.json','w') as f:
    #     json.dump(d_update, f, indent=4)
    # with open('../data/train3_all.json','w') as f:
    #     json.dump(d_update, f, indent=4)
    # with open('../data/train4_all.json', 'w') as f:
    #     json.dump(d_update, f, indent=4)                          #  기막힌 방법이 없나
    #
    # js = json.load(open('../data/train4_all.json'))
    #
    # # train_path = dataset_path + '/train0_all.json'
    # # data_dir = train_path, mode = 'train', transform = train_transform
    # print(train_file_list)
    # print(json_data)
    # print('train/test dataset run')