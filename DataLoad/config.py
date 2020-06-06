'''
@Author: your name
@Date: 2020-05-19 08:31:15
@LastEditTime: 2020-06-01 08:37:34
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gitClone/Build-Your-Own-Face-Model/detection/config.py
'''
import torch
from torchvision import transforms as T


class Config:
    # preprocess
    data_enhance = {}
    # data_enhance['PAD_SIZE'] = [1000,600]   # 将原始图像填充到指定尺寸，若原始图像尺寸大于目标尺寸，则填充到目标尺寸的比例
    # data_enhance['CROP'] = [0.7, 1.]        # 随机剪裁
    # data_enhance['PAD_MULTI'] = [32,32]     # 将原始图像填充到某个尺寸的整数倍
    # data_enhance['FLIP'] = True             # 将原始图像随机翻转
    # data_enhance['MASK'] = [0.2, 0.3]       # 对原始图像中的人脸进行mask
    data_enhance['RESIZE'] = [640,384]      # 将原始图像resize到指定尺寸

    channels = 3
    train_transforms = T.Compose([
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    # dataset
    data_root = '/root/yw/AI_train_machine/DataSets/FaceDetectLandmark'

    # checkpoints
    checkpoints = 'checkpoints'
    restore = False
    restore_model = '80.pth'

    # training
    epoch = 90
    lr = 5e-4
    batch_size = 16
    pin_memory = True
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # log
    logName = '0929'

    # inference
    threshold = 0.5
