import os
import cv2
import json
import torch
import random
import warnings 
import webcolors
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns; sns.set()
import albumentations as A

from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from collections import defaultdict
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import label_accuracy_score, add_hist
from matplotlib.patches import Patch
from imantics import Polygons, Mask


random_seed = 2021
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


class SegCutmix():

    def __init__(self, root, train_all_path, object_info_json):
        self.root = root
        self.train_all_path  = train_all_path
        self.object_info_json = object_info_json
        self.make_obj_json(self.train_all_path,self.object_info_json)
        self.object_json = self.read_obj_json(object_info_json)
        self.coco = COCO(self.train_all_path)
        
    def extract_obj_img(self, train_all_path):
        '''
        class 별로 구분한 object_info_json file 생성
        '''
        clsfy_by_object = [[] for i in range(11)]    
        
        with open(train_all_path, 'r') as readjson:
            train_all = json.load(readjson)
        
        train_annot = train_all['annotations']
        
        for img_info in train_annot:
            id = img_info['id']
            img_id = img_info['image_id']
            img_cls = img_info['category_id']
            clsfy_by_object[img_cls].append({'img_id':img_id,'ann_id':id})
        return clsfy_by_object

    def get_image(self,index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        images = cv2.imread(os.path.join(self.root, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        return images
    
    def get_ann_info(self, index, img_all = False):
        seg_info = []
        if img_all: 
            image_infos = self.coco.loadImgs(index)[0]
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
        else : ann_ids = [index]
        anns = self.coco.loadAnns(ann_ids)
        return anns

    def make_obj_json(self, train_all_path, object_info_json):
        clsfy_by_object = self.extract_obj_img(train_all_path)
        with open(object_info_json, 'w') as outfile:
            json.dump(clsfy_by_object, outfile,indent = 4)
    
    def read_obj_json(self, object_info_json):
        with open(object_info_json, 'r') as readjson:
            object_json = json.load(readjson)        
        return object_json

    def mul_image_mask(self, image,mask):
        masks = np.stack([mask,mask,mask], axis=2)
        return np.where(masks !=0,image,0)

    def extract_image(self, image,mask,bbox):
        x,y,w,h = bbox
        return self.mul_image_mask(image,mask)[y:y+h,x:x+w,:]

    def extract_seg(self, mask,bbox):
        x,y,w,h = bbox
        return mask[y:y+h,x:x+w]

    def extract_bbox(self, index, img_all):
        '''extract_bbox
        '''
        anns = self.get_ann_info(index, img_all=img_all)
        bbox = []
        for ann in anns:
            x,y,w,h = map(int,ann['bbox'])
            bbox += [[x,y,w,h]]
        return bbox

    def get_seg_by_class(self, add_class):    
        add_list = self.object_json[add_class] # battery
        len_add_list = len(add_list)
        list_idx = random.choice(range(0, len_add_list))
        ann_idx = add_list[list_idx]['ann_id']
        img_idx = add_list[list_idx]['img_id']
        anns = self.coco.loadAnns(ann_idx)
        
        image = self.get_image(img_idx)
        mask = self.coco.annToMask(anns[0])
        mask = np.where(mask!=0, add_class,0)
        bbox = self.extract_bbox(ann_idx, img_all=False)
        return image,mask,bbox[0]

    def transform_image_masks(self, image,mask,bbox,transform = None):
        images = self.extract_image(image,mask,bbox)
        masks = self.extract_seg(mask,bbox)
        if transform !=None:
            transformed = transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
        h,w = images.shape[:2]
        return images,masks,[0,0,w,h]
    
    def select_insert_point(self, insert_area, seg_img):
        '''
        삽입할 위치 찾고, img 사이즈 조정 box 크기에 랜덤하게 여분추가(5% ~ 50%, 5%단위로)  
        '''
        x,y,w,h = insert_area
        i_x,i_y = seg_img.shape[1],seg_img.shape[0]
        ratio = random.choice([i*1/20 for i in range(1,11)])
        r = max(i_x/w,i_y/h)
        i_x,i_y = int(seg_img.shape[1]/(r+ratio)),int(seg_img.shape[0]/(r+ratio))
        w-= i_x
        h-= i_y
        insert_x = random.choice(range(x, x+w+1))
        insert_y = random.choice(range(y, y+h+1))
        return insert_x,insert_y,(i_x,i_y) #insert좌표, resize 크기

    def make_insert_mask(self, x_insert,y_insert,seg_image,seg_mask,seg_bbox):
        x,y,w,h = x_insert, y_insert, seg_bbox[2], seg_bbox[3]
        insert_image = np.zeros((512, 512, 3))
        insert_mask = np.zeros((512, 512))
        insert_image[y:y+h,x:x+w,:] = seg_image
        insert_mask[y:y+h,x:x+w] = seg_mask
        return insert_image, insert_mask

    def cutmix_image(self, origin_img, origin_mask, add_cls = 9):
        insert_area = [0,0,512,512]
        image,mask,bbox = self.get_seg_by_class(add_cls)
        seg_image = self.extract_image(image,mask,bbox)
        x_insert,y_insert,resize = self.select_insert_point(insert_area,seg_image)
        
        transform = A.Compose([A.Resize(resize[1],resize[0]), 
                               A.HorizontalFlip(always_apply=False, p=0.25), 
                               A.VerticalFlip(always_apply=False, p=0.25)])
        
        seg_image,seg_mask,seg_bbox = self.transform_image_masks(image,mask,bbox, transform)
        seg_mask = np.where(seg_mask!=0 ,add_cls,0)
        
        seg_image, seg_mask = self.make_insert_mask(x_insert,y_insert,seg_image,seg_mask,seg_bbox)
        
        seg_mask3d = np.dstack([seg_mask]*3)
        
        cutmix_image = np.where(seg_mask3d==add_cls,seg_image,origin_img)
        cutmix_mask = np.where(seg_mask==add_cls,seg_mask,origin_mask)
        
        return cutmix_image, cutmix_mask

    def shift(self, X, dx, dy):
        X = np.roll(X, dy, axis=0)
        X = np.roll(X, dx, axis=1)
        # if dy>0:
        #     X[:dy, :] = 0
        # elif dy<0:
        #     X[dy:, :] = 0
        # if dx>0:
        #     X[:, :dx] = 0
        # elif dx<0:
        #     X[:, dx:] = 0
        return X
    

    def duplicate_image(self, origin_img, origin_mask):
        
        origin_mask3d = np.dstack([origin_mask]*3)
        seg_image = np.where(origin_mask3d!=0, origin_img, 0)
        insert_area = [0,0,512,512]
        
        dx = random.randrange(-250,260,10)
        dy = random.randrange(-250,260,10)
        
        seg_image = self.shift(seg_image,dx,dy)
        seg_mask = self.shift(origin_mask,dx,dy)

        duplicate_image = np.where(seg_image!=0,seg_image,origin_img)
        duplicate_mask = np.where(seg_mask!=0,seg_mask,origin_mask)
        
        return duplicate_image, duplicate_mask