'''
@Author: your name
@Date: 2020-05-19 08:31:14
@LastEditTime: 2020-06-03 06:12:29
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gitClone/Build-Your-Own-Face-Model/detection/api.py
'''
import sys
sys.path.append('../..')
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np


# local imports
from config import Config as cfg
from Models.backbone_centerTemp import centerface
from DataLoad.__enhance__ import *
import cv2


def load_model(device):
    net = centerface().to(device)
    path = osp.join(cfg.checkpoints, cfg.restore_model)
    weights = torch.load(path, map_location=device)
    net.load_state_dict(weights)
    net.eval()
    return net

def preprocess(im):
    funcs = {}
    funcs['RESIZE'] = resize
    funcs['PAD_SIZE'] = pad_size
    funcs['PAD_MULTI'] = pad_multi
    funcs['FLIP'] = flip
    funcs['MASK'] = mask
    funcs['CROP'] = crop

    for key in cfg.data_enhance.keys():
            if key in funcs.keys():
                img, label = funcs[key](im, None, cfg.data_enhance[key])

    return img

def detect(im, device):
    data = cfg.test_transforms(im)
    data = data[None, ...]
    data = data.to(device)
    net = load_model(device)
    with torch.no_grad():
        out_hm, out_wh, out_off = net(data)
        return out_hm, out_wh, out_off

def decode(heatmap, scale, offset, size, threshold=0.75):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    boxes = []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])

        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes[:, :4], boxes[:, 4], 0.25)
        boxes = boxes[keep, :]

    return boxes

def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep


def visualize(im, bboxes):
    img = im.copy()
    for box in bboxes:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
    return img


if __name__ == '__main__':
    impath = '/root/yw/AI_train_machine/DataSets/FaceDetectLandmark/test/image/jucan_8.jpg'
    im = cv2.imread(impath)
    device = torch.device("cuda:1" if cfg.use_cuda else "cpu")

    img = preprocess(im)
    out_hm, out_wh, out_off = detect(img, device)
    # res = np.array(out_hm)
    # res = res.transpose(1, 2 ,0)
    # cv2.imwrite('hm.jpg',res*255)
    bboxes = decode(out_hm, out_wh, out_off,[384,640])
    bboxes = bboxes[:, : 4]

    img = visualize(img, bboxes)
    cv2.imwrite('res_02_.jpg', img)