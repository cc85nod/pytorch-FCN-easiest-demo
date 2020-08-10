import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms # 提供 data 的操作, 如 tensor, PIL, numpy 的轉換, normalize 等
from multiprocessing import set_start_method

from config import *

import cv2
from onehot import onehot

class ImageDataset(Dataset):
    def __init__(self):
        # 組合各個 transform
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 將 (h,w,c) 轉成 (c,h,w)
            # 正規化 tensor
            # normalized_img = (img-mean)/std
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_len = len(os.listdir('data/images'))
        self.label_len = len(os.listdir('data/labels'))
        
    def __len__(self):
        return self.image_len

    def __getitem__(self, idx):
        img_name = os.listdir('data/images')[idx]
        img = cv2.imread('data/images/'+img_name)
        img = cv2.resize(img, IMAGE_SIZE)

        label = cv2.imread('data/labels/'+img_name, 0)
        label = cv2.resize(label, IMAGE_SIZE)

        label = label / 255.0
        label = label.astype('uint8') # transform type
        label = onehot(label, 2)
        # In pytorch, images are represented as [channels, height, width]
        # 256,256,2 -> 2,256,256
        label = label.transpose(2, 0, 1)
        label = torch.cuda.FloatTensor(label)
        
        if self.transform:
            img = self.transform(img)

        return img, label

dataset = ImageDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Randomly split dataset by given size
train_dataset, test_dataset = random_split(
    dataset,[train_size, test_size]
)

# num_workers: allocate batch to workers
# larger num -> faster but need more GPU resource
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKER_NUM,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKER_NUM,
)

if __name__ =='__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    for train_batch in train_dataloader:
        print(train_batch)
    for test_batch in test_dataloader:
        print(test_batch)
