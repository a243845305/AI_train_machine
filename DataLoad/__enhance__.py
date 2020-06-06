# -*- coding: utf-8 -*-

import cv2
import json
import math
import numpy as np



def resize( im, labels, size ):
    """将原始图像resize到指定尺寸

        Args:
            im: 输入的图像
            labels: 输入的label
            size: 需要调整到的尺寸的大小

        Returns:
            im: 处理后的图像
            labels: 处理后的label

        Raises:
            无
    """
    w, h = size[0], size[1]
    scale = np.array([w, h], dtype=np.double)
    scale = scale/im.shape[1::-1]
    scale = scale.reshape(1,-1)
    # print(scale)
    im = cv2.resize(im, (w,h))
    
    if labels is None:
        return im, labels
    for label in labels:
        rect_out = np.array( label['rect'], dtype=np.double )
        landmark_out = np.array( label['landmark'], dtype=np.double )

        rect_out = rect_out.reshape(-1,2)
        landmark_out = landmark_out.reshape(-1,2)

        rect_out = rect_out*scale
        landmark_out = landmark_out*scale

        label['rect'] = rect_out.flatten()
        label['landmark'] = landmark_out.flatten()

        label['landmark'] = label['landmark'].tolist()
        label['rect'] = label['rect'].tolist()



    return im, labels


def pad_size( im, labels, size, color=None ):
    """将原始图像填充到指定尺寸，若原始图像尺寸大于目标尺寸，则填充到目标尺寸的比例

        Args:
            im: 输入的图像
            labels: 输入的label
            size: 需要填充到的尺寸的大小

        Returns:
            im: 处理后的图像
            labels: 处理后的label

        Raises:
            无
    """
    w, h = size[0], size[1]
    w_org, h_org = im.shape[1], im.shape[0]

    if w_org>w or h_org>h:
        if float(w)/float(h) > float(w_org)/float(h_org):
            w = (float(w)/float(h))*float(h_org)
            w = math.ceil(w)
            h = h_org
        else:
            h = (float(h) / float(w)) * float(w_org)
            h = math.ceil(h)
            w = w_org

    # print('org shape: {}'.format(im.shape))
    # print('pad shape: {}'.format((h,w)))

    if color is None:
        color = np.random.randint(0,255,(1,3))
    shape = np.array([h, w, 3])
    delta_shape = shape-im.shape
    start_y = 0
    start_x = 0

    if delta_shape[0]>0:
        start_y = np.random.randint(0,delta_shape[0])
    if delta_shape[1]>0:
        start_x = np.random.randint(0,delta_shape[1])

    im_out = np.empty(shape)
    im_out[:] = color
    im_out[start_y:im.shape[0]+start_y, start_x:im.shape[1]+start_x, :] = im

    for label in labels:
        rect = np.array(label['rect'])
        rect = rect.reshape(-1,2)
        rect += np.array([start_x, start_y])
        label['rect'] = rect.flatten()

        landmark = np.array(label['landmark'])
        landmark = landmark.reshape(-1,2)
        landmark += np.array([start_x, start_y])
        label['landmark'] = landmark.flatten()

        label['landmark'] = label['landmark'].tolist()
        label['rect'] = label['rect'].tolist()

    return im_out, labels


def pad_multi( im, labels, size, color=None ):
    """将原始图像填充到某个尺寸的整数倍

        Args:
            im: 输入的图像
            labels: 输入的label
            size: 填充的最小整数倍

        Returns:
            im: 处理后的图像
            labels: 处理后的label

        Raises:
            无
    """
    dw, dh = size[0], size[1]

    if color is None:
        color = np.random.randint(0,255,(1,3))

    w_adjusted = int( int(im.shape[1]/dw) * dw ) + dw
    h_adjusted = int( int(im.shape[0]/dh) * dh ) + dw

    shape = np.array([h_adjusted, w_adjusted, 3])
    delta_shape = shape - im.shape

    start_y = 0
    start_x = 0

    if delta_shape[0] > 0:
        start_y = np.random.randint(0, delta_shape[0])
    if delta_shape[1] > 0:
        start_x = np.random.randint(0, delta_shape[1])

    im_out = np.empty(shape)
    im_out[:] = color
    im_out[start_y:im.shape[0]+start_y, start_x:im.shape[1]+start_x, :] = im

    for label in labels:
        rect = np.array(label['rect'])
        rect = rect.reshape(-1,2)
        rect += np.array([start_x, start_y])
        label['rect'] = rect.flatten()

        landmark = np.array(label['landmark'])
        landmark = landmark.reshape(-1,2)
        landmark += np.array([start_x, start_y])
        label['landmark'] = landmark.flatten()

        label['landmark'] = label['landmark'].tolist()
        label['rect'] = label['rect'].tolist()

    return im_out, labels


def flip( im, labels, enable=True ):
    """将原始图像随机翻转

        Args:
            im: 输入的图像
            labels: 输入的label

        Returns:
            im: 处理后的图像
            labels: 处理后的label

        Raises:
            无
    """
    if enable is not True:
        return im, labels

    is_flip = np.random.uniform(0,1)
    if is_flip<0.5:
        return im, labels

    w_max = im.shape[1]-1
    im_out = cv2.flip(im, 1)

    left_landmark_mask = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                          33,34,35,36,37,47,48,52,53,54,55,56,57,
                          64,65,66,67,72,73,74,78,80,82,84,85,86,
                          95,94,96,97,103,104]
    right_landmark_mask = [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,
                           42,41,40,39,38,51,50,61,60,59,58,63,62,
                           71,70,69,68,75,76,77,79,81,83,90,89,88,
                           91,92,100,99,101,105]

    for label in labels:
        rect = np.array(label['rect'])
        rect_out = np.array(label['rect'])
        landmark = np.array(label['landmark'])
        landmark_out = np.array(label['landmark'])

        rect[::2] = w_max - rect[::2]
        rect_out[::2] = w_max - rect_out[::2]
        landmark[::2] = w_max - landmark[::2]
        landmark_out[::2] = w_max - landmark_out[::2]

        rect_out[0] = rect[2]
        rect_out[2] = rect[0]
        rect_out = rect_out.flatten()

        landmark = landmark.reshape(-1,2)
        landmark_out = landmark_out.reshape(-1,2)
        landmark_out[left_landmark_mask] = landmark[right_landmark_mask]
        landmark_out[right_landmark_mask] = landmark[left_landmark_mask]
        landmark_out = landmark_out.flatten()

        label['rect'] = rect_out
        label['landmark'] = landmark_out

        label['landmark'] = label['landmark'].tolist()
        label['rect'] = label['rect'].tolist()

    return im_out, labels


def mask( im, labels, scale_limits=(0.2,0.4), color=None ):
    """对原始图像中的人脸进行mask

        Args:
            im: 输入的图像
            labels: 输入的label
            scale_limits: mask相对于人脸框的尺寸

        Returns:
            im: 处理后的图像
            labels: 处理后的label

        Raises:
            无
    """
    if color is None:
        color = np.random.randint(0,255,(1,3))

    for label in labels:
        rect_x1 = label['rect'][0]
        rect_y1 = label['rect'][1]
        rect_x2 = label['rect'][2]
        rect_y2 = label['rect'][3]
        rect_w = label['rect'][2]-label['rect'][0]
        rect_h = label['rect'][3]-label['rect'][1]
        # print(label['rect'])
        mask_w_min = int(rect_w*scale_limits[0])
        mask_w_max = int(rect_w*scale_limits[1])

        mask_h_min = int(rect_h*scale_limits[0])
        mask_h_max = int(rect_h*scale_limits[1])

        if mask_h_min >= mask_h_max or mask_w_min >= mask_w_max:
            continue

        mask_w = np.random.randint(mask_w_min, mask_w_max)
        mask_h = np.random.randint(mask_h_min, mask_h_max)

        if mask_w == 0 or mask_h == 0:
            continue

        img_w = im.shape[1]
        img_h = im.shape[0]

        mask_x1 = np.random.randint(rect_x1, rect_x2-mask_w)
        mask_y1 = np.random.randint(rect_y1, rect_y2-mask_h)
        mask_x2 = mask_x1+mask_w
        mask_y2 = mask_y1+mask_h

        im[mask_y1:mask_y2, mask_x1:mask_x2, :] = color
        # print(im.shape)
        # print(mask_x1, mask_x2, mask_y1, mask_y2)
    return im, labels


def crop( im, labels, scale_limits ):
    """对原始图像进行剪裁，若剪裁边界触碰到人脸，则放弃剪裁，返回原始图像和label

        Args:
            im: 输入的图像
            labels: 输入的label
            scale_limits: mask相对于人脸框的尺寸

        Returns:
            im: 处理后的图像
            labels: 处理后的label

        Raises:
            无
    """

    def compute_cross(rect_all, rect_part):
        left1, top1, right1, down1 = rect_all[0], rect_all[1], rect_all[2], rect_all[3]
        left2, top2, right2, down2 = rect_part[0], rect_part[1], rect_part[2], rect_part[3]

        area1 = (rect1[2]-rect1[0])*(rect1[3]-rect1[1])
        area2 = (rect2[2]-rect2[0])*(rect2[3]-rect2[1])

        left = max(left1, left2)
        top = max(top1, top2)
        right = min(right1, right2)
        down = min(down1, down2)

        if left>right or top > down:
            area = 0
        else:
            area = (right-left)*(down-top)

        return area/area2

    scale_wh = np.random.uniform(scale_limits[0], scale_limits[1])
    scale_x = np.random.uniform(0, 1-scale_wh)
    scale_y = np.random.uniform(0, 1-scale_wh)
    shape = im.shape
    bbox = [ int(scale_x*shape[1]),
             int(scale_y*shape[0]),
             int(scale_x*shape[1]+scale_wh*shape[1]),
             int(scale_y*shape[0]+scale_wh*shape[0]) ]
    # print(bbox)

    rect1 = np.array(bbox)
    w = im.shape[1]
    h = im.shape[0]

    if rect1[2] >= w or rect1[3] >= h:
        return im, labels

    labels_out = []
    face_cut = False
    for label in labels:
        rect2 = np.array(label['rect'])
        cross = compute_cross(rect1, rect2)

        if cross == 0:
            continue
        elif cross == 1.:
            labels_out.append(label)
        else:
            face_cut = True
            break

    if len(labels_out) == 0:
        return im, labels
    if face_cut is True:
        return im, labels

    im_out = np.empty((rect1[3]-rect1[1],rect1[2]-rect1[0],3))
    im_out[:,:,:] = im[rect1[1]:rect1[3], rect1[0]:rect1[2], :]

    for label in labels:
        label['landmark'][::2] = label['landmark'][::2]-rect1[0]
        label['landmark'][1::2] = label['landmark'][1::2]-rect1[1]
        label['rect'][::2] = label['rect'][::2]-rect1[0]
        label['rect'][1::2] = label['rect'][1::2]-rect1[1]

    return im_out, labels


# if __name__ == '__main__':
#     from DataSets.pytorch.FaceDetectLandmark.__init__ import face_detect_landmark_root
#     import os

#     image_dir = os.path.join(face_detect_landmark_root,'dataset_used/train/image')
#     label_dir = os.path.join(face_detect_landmark_root,'dataset_used/train/gt')
#     image_file_list = os.listdir(image_dir)
#     image_file = image_file_list[np.random.randint(0, len(image_file_list))]
#     label_file = image_file[:-3]+'txt'
#     image_path = os.path.join(image_dir, image_file)
#     label_path = os.path.join(label_dir, label_file)

#     print(image_path, label_path)

#     im = cv2.imread(image_path)
#     with open(label_path, 'r') as f:
#         labels = json.load(f)

#     im, labels = crop( im, labels, (0.6,1) )
#     im, labels = resize( im, labels, (740, 555) )
#     im, labels = pad_size( im, labels, (1000, 1000) )
#     im, labels = pad_multi( im, labels, (32, 32) )
#     im, labels = flip( im, labels )
#     im, labels = resize( im, labels, (960, 640) )
#     im, labels = mask( im, labels )

#     im_out_aa = draw_image_from_data( im, labels)

#     cv2.imwrite(os.path.join(face_detect_landmark_root,'test.jpg'), im_out_aa)
