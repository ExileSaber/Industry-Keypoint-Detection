import os
import json
import numpy as np
import matplotlib.pyplot as plt
from config import config as cfg
import cv2


# json变成加入高斯的np
def json_to_numpy(dataset_path):
    # test of train_flag
    # way = 'test'
    # date = '07_23'

    # 保存的路径
    imgs_path = os.path.join(dataset_path, 'imgs')
    labels_path = os.path.join(dataset_path, 'labels')
    # save_path = os.path.join('..', 'data', date, way, 'masks')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    # 开始处理
    for name in os.listdir(imgs_path):
        # 读入label
        with open(os.path.join(os.path.join(labels_path),
                               name.split('.')[0] + '.json'), 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            points = json_data['shapes']

        # print(points)
        landmarks = []
        for point in points:
            for p in point['points'][0]:
                landmarks.append(p)

        # print(landmarks)
        landmarks = np.array(landmarks)

        # 保存为np
        # np.save(os.path.join(save_path, name.split('.')[0] + '.npy'), landmarks)

        return landmarks


if __name__ == '__main__':
    json_to_numpy()
