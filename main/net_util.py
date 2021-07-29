import torch
import os
import numpy as np
from torch import nn
import torchvision
from config import config as cfg
import torch.utils.data
from torchvision import datasets, transforms, models
import cv2
from data_pre import json_to_numpy


# box_3D的数据仓库
class Dataset(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = os.listdir(os.path.join(dataset_path, 'imgs'))

    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img = cv2.imread(os.path.join(self.dataset_path, 'imgs', self.img_name_list[index]))
        img = cv2.resize(img, (512, 352))
        img = transforms.ToTensor()(img)

        # 读入标签
        mask = json_to_numpy(self.dataset_path)
        # mask = np.load(os.path.join(self.dataset_path, 'masks', self.img_name_list[index].split('.')[0] + '.npy'))
        mask = torch.tensor(mask, dtype=torch.float32)

        return img, mask

    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)
