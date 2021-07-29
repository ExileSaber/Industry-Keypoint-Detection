import os
import torch

config = {
    # 网络训练部分
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'batch_size': 1,
    'epochs': 1000,
    'save_epoch': 100,


    # 网络评估部分
    'test_batch_size': 1,
    'test_threshold': 0.5,

    # 设置路径部分
    'train_date': '07_23_2',
    'train_way': 'train',
    'test_date': '07_23_2',
    'test_way': 'test',

}
