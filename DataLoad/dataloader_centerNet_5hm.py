# -*- coding: utf-8 -*-
from .__enhance__ import *

import os
import yaml
import cv2
import copy
import torch
import numpy as np

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm



# 1. dataSet 重载
# 2. dataLoader 重载

class dataLoader():
    def __init__(self, batch_size=16, shuffle=True, num_workers=0, pin_memory=True):
        # super(DataLoader, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def loader(self, dataset):
        return torch.utils.data.DataLoader(dataset = dataset,
                                            batch_size = self.batch_size,
                                            shuffle = self.shuffle,
                                            num_workers = self.num_workers,
                                            pin_memory = self.pin_memory)

class dataSet(Dataset):
    def __init__(self, root, config_enhance, datasetType, datatransforms=None):
        super(dataSet, self).__init__()
        self.loader = DataPreProcess(   dataroot=root,
                                        config_enhance=config_enhance,
                                        datatype=datasetType,
                                        datatransforms=datatransforms
                                        )

    def __getitem__(self, item):
        # img, labels, fn = self.loader.get_item(item)
        # return img, json.dumps(labels), fn
        img, hm, fn = self.loader.get_item(item)
        return img, hm, fn

    def __len__(self):
        return self.loader.get_len()

class DataPreProcess(object):

    def __init__(self, dataroot, config_enhance, datatype='train', datatransforms=None):

        self.datatype = datatype
        self.root = dataroot
        self.config_enhance = config_enhance
        self.transforms = datatransforms
        self.images, self.labels = self.__count_items__()

        self.funcs = {}
        if datatype == 'train':
            self.funcs['RESIZE'] = resize
            self.funcs['PAD_SIZE'] = pad_size
            self.funcs['PAD_MULTI'] = pad_multi
            self.funcs['FLIP'] = flip
            self.funcs['MASK'] = mask
            self.funcs['CROP'] = crop

    def __count_items__(self):
        source_root = os.path.abspath(self.root)
        print('DataSet Root: {}'.format(source_root))

        source_root = os.path.join(source_root, self.datatype)
        source_label = os.path.join(source_root, 'label')
        source_image = os.path.join(source_root, 'image')
        labels = []
        images = []
        for file in os.listdir(source_label):
            if not 'txt' in file:
                continue
            src_image = os.path.join(source_image, file[:-3]+'jpg')
            if not os.path.exists(src_image):
                continue
            with open( os.path.join(source_label, file) ) as f:
                label = json.load(f)
            labels.append(label)
            images.append(src_image)
        return images, labels

    def get_item(self, index):
        """根据所选index返回一个数据
            Args:
            index: 数据的index
            Returns:
            无
            Raises:
            若index溢出，报'image index out of range !'
        """       
        assert index<len(self.images), 'image index out of range !'

        img = cv2.imread(self.images[index])
        label = copy.deepcopy(self.labels[index])

        fn = self.images[index]
        fn = os.path.split(fn)[-1]

        for key in self.config_enhance.keys():
            if key in self.funcs.keys():
                img, label = self.funcs[key](img, label, self.config_enhance[key])
                
        if self.transforms is not None:
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img = self.transforms(img)

        hm = self._get_label(img, label)

        return img, hm, fn[:-4]

    def _get_label(self, img, label):
        scale = 4

        num_objs = len(label)
        input_cls, input_h, input_w = img.shape
        output_cls, output_h, output_w = input_cls, input_h // 4, input_w // 4

        # hm = np.zeros((output_h, output_w), dtype=np.float32)
        grid_x = np.tile(np.arange(output_w), reps=(output_h, 1))
        grid_y = np.tile(np.arange(output_h), reps=(output_w, 1)).transpose()
        res = np.zeros([5, output_h, output_w], dtype=np.float32)
        
        for k in range(num_objs):
            rect = []
            rect = label[k]['rect']
            x, y = rect[0] / scale, rect[1] / scale
            w, h = abs(rect[2] - rect[0]) / scale, abs(rect[3] - rect[1]) / scale

            # 计算 centerPoint
            ct_p = np.array([x + w / 2, y + h / 2], dtype=np.float32)
            ct_p_int = ct_p.astype(np.int32)

            # 准备 heartMap
            grid_dist = (grid_x - ct_p_int[0]) ** 2 + (grid_y - ct_p_int[1]) ** 2
            heatmap = np.exp(-0.5 * grid_dist / 2.65 ** 2)
            res[0] = np.maximum(heatmap, res[0])
            # 偏移值的获取
            res[1][int(y), int(x)] = ct_p[0] - ct_p_int[0]
            res[2][int(y), int(x)] = ct_p[1] - ct_p_int[1]
            # 在160*96尺寸下的目标宽高
            res[3][int(y), int(x)] = np.log(w + 1e-4)
            res[4][int(y), int(x)] = np.log(h + 1e-4)
            
        return res

    def get_len(self):
        return len(self.images)


if __name__ == '__main__':
    from config import Config as cfg
    from __enhance__ import *

    DATA_Root = cfg.data_root
    DATA_Enhance = cfg.data_enhance
    DATA_TRANSFORMS = cfg.train_transforms

    trainset = dataSet(DATA_Root, DATA_Enhance, 'train', DATA_TRANSFORMS)

    dataloader = dataLoader()
    trainData = dataloader.loader(trainset)

    for data in tqdm(trainData, desc=f"loading data", total=len(trainData)):
        img, labels, fn = data
        print(img.shape)
        print(labels.shape)
        print(fn)

        hm = labels[:, 0]
        off = labels[:, [1,2]]
        wh =  labels[:, [3,4]]
        print(hm.shape)
        print(off.shape)
        print(wh.shape)

        break
        # temp = np.array(labels[0][0])
        # # print(fn)
        # cv2.imwrite('heatmap.jpg', temp*255)
        # break

