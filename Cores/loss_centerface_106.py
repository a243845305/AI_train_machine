# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import numpy as np
import time
import os

from Cores.utils import make_hm_center, make_hm_landmark
from Cores.sinkhorn import geometryDistance, sinkhornDistance


def _hm_loss1_(hm1, hm2):
    return None


def _clip_sinkhorn_loss_(hm_gt, hm_pred, labels, expand=4, device=torch.device('cpu')):
    height, width = hm_gt.shape[1], hm_gt.shape[2]
    ch = hm_gt.shape[0]
    rects = [label['rect'] for label in labels]
    rects = np.array(rects, dtype=np.float32)
    rects /= 4.
    loss = 0
    for rect in rects:
        x1, y1 = rect[0]-expand, rect[1]-expand
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = rect[2]+expand, rect[3]+expand
        x2, y2 = min(int(x2), width-1), min(int(y2), height-1)
        dist = geometryDistance(y2-y1, x2-x1)
        dist = dist.to(device)
        for i in range(ch):  # 106层heat map
            # print(i)
            gt_face = hm_gt[i, y1:y2, x1:x2]
            pred_face = hm_pred[i, y1:y2, x1:x2]
            loss += sinkhornDistance(gt_face, pred_face, dist)
    return loss


def _clip_other_loss_(hm_gt, hm_pred, labels, expand=4, device=torch.device('cpu'), loss_fun=None):
    height, width = hm_gt.shape[1], hm_gt.shape[2]
    ch = hm_gt.shape[0]
    rects = [label['rect'] for label in labels]
    rects = np.array(rects, dtype=np.float32)
    rects /= 4.
    loss = 0
    for rect in rects:
        x1, y1 = rect[0]-expand, rect[1]-expand
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = rect[2]+expand, rect[3]+expand
        x2, y2 = min(int(x2), width-1), min(int(y2), height-1)
        for i in range(ch):  # 106层heat map
            gt_face = hm_gt[i, y1:y2, x1:x2]
            pred_face = hm_pred[i, y1:y2, x1:x2]
            loss += loss_fun(gt_face, pred_face, device)
    return loss


class LossLayer(nn.Module):
    def __init__(self):
        super(LossLayer, self).__init__()

    def forward(self, net_out_list, annotations, device=torch.device('cpu')):

        hm_cls_pred, offset_pred, scale_pred, hm_landmark_pred = net_out_list
        # hm_landmark_pred = hm_landmark_pred.abs()
        step = 4  # 从原始图到featuremap的缩放比例
        batch = hm_cls_pred.shape[0]
        height, width = hm_cls_pred.shape[2]*step, hm_cls_pred.shape[3]*step

        label_list = [json.loads(annotation) for annotation in annotations]

        hm_cls_gt = []
        offset_gt = []
        scale_gt = []
        hm_landmark_gt = []
        for labels in label_list:

            hm_landmark_gt.append(torch.tensor(
                make_hm_landmark(height, width, labels, step), dtype=torch.float32
            ).to(device))
            # todo: offset
            # todo: sacle
            # todo: class

        tic = time.time()
        landmark_loss = [_clip_sinkhorn_loss_(hm_gt, hm_pred, labels, 4, device)
                         for hm_gt, hm_pred, labels in zip(hm_landmark_gt, hm_landmark_pred, label_list)]
        print(time.time()-tic)

        return sum(landmark_loss)



if __name__ == '__main__':
    from Models.pytorch.header_centerface_106 import CernterFaceHeader, _conv_1x1

    header = CernterFaceHeader(24, hm_landmark_conv=_conv_1x1).to(torch.device('cuda:0'))

    loss_layer = LossLayer()
    annotations = []
    for file in os.listdir('labels'):
        with open(os.path.join('labels', file), 'r') as f:
            annotations.append(f.readline())

    x = torch.rand(len(annotations), 24, 96, 160, dtype=torch.float32).to(torch.device('cuda:0'))
    out = header(x)

    loss = loss_layer(out, annotations, torch.device('cuda:0'))

