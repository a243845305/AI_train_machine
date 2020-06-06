import torch
import numpy as np


def calculate_iou(a, b):

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


# 使用全图画高斯分布，中心可以是float值
def __gaussian1__(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, np.float32)
    y = np.arange(0, height, 1, np.float32)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return 1 * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def make_hm_landmark(height, width, labels, step=4, sigma=3):
    assert width % step == 0 and height % step == 0
    ch = len(labels[0]['landmark']) // 2
    hm = np.zeros((ch, height//step, width//step))

    for label in labels:
        for i in range(ch):
            x, y = label['landmark'][i*2]/step, label['landmark'][i*2+1]/step
            mask = __gaussian1__(height//step, width//step, sigma, center=[x, y])
            hm[i, :, :] = np.maximum(hm[i, :, :], mask)
    return hm


def __gaussian_radius_fixed__(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)

    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


# 按指定尺寸绘制高斯分布，然后整体copy到heatmap
def __gaussian2__(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def make_hm_center(height, width, labels, step=4):
    heatmap = np.zeros((height//step, width//step), dtype=np.float32)

    for i, label in enumerate(labels):
        ct = [label['rect'][2]+label['rect'][0], label['rect'][3]+label['rect'][1]]
        ct = [s / (2*step) for s in ct]
        wh = [label['rect'][2]-label['rect'][0], label['rect'][3]-label['rect'][1]]
        wh = [s / step for s in wh]

        redius = __gaussian_radius_fixed__(wh)
        radius = max(0, int(2.*redius+1))
        ct = np.array(ct, dtype=np.float32)
        ct_int = ct.astype(np.int32)

        # diameter = 2. * radius + 1  # 例如半径是2时,直径是5. 这里+1的目的就是为了准备一个中心点
        diameter = (radius//2)*2+1
        # 以直径为边长,生成一个2维的 直径*直径 高斯分布图,具体就是在图的中心值为1,离中心越远,值越小
        gaussian = __gaussian2__((diameter, diameter), sigma=diameter/6)
        # 获取真实物体中心点坐标 该坐标已经经过padding处理以及除以4之后的坐标
        x, y = ct_int[0], ct_int[1]
        # 热力图的尺寸,一般为输入图像尺寸的四分之一
        height, width = heatmap.shape
        # 这里进行min操作的目的在于防止真实物体的中心点过于靠近热力图边缘时,防止热半径超出热力图范围,
        # +1的操作是因为切片操作中取左不取右,所以要取到理想中的值,:右边的值需要+1
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # 获取热力图中真实物体周围热半径内(radius)的热力图,如果目标热半径超出热力图的话,则舍弃超出部分
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        # 同理提取出对应的高斯分布,目标不在热力图边缘的正常情况下,masked_gaussian == gaussian
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        # 一般情况下 masked_gaussian.shape == masked_heatmap.shape
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            # 如果某一个类的两个高斯分布重叠，重叠的点取最大值就行,这里的out返回值就是修改后的masked_heatmap
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
        else:
            print('gauss', masked_gaussian.shape)
            print('heatmap', masked_heatmap.shape)
            # exit()

    return heatmap







# def main():
#     import yaml
#     from Models.pytorch.tool_anchor import Anchors
#
#     config_path = "../Projects/face_detect_landmark/face_detect_landmark.yaml"
#     with open(config_path) as f:
#         data = f.read()
#     cfg = yaml.load(data, Loader=yaml.FullLoader)
#     cfg_retinaface_anchor = cfg['CORE']['ANCHOR']
#
#     anchor = Anchors()
#     anchors = anchor(
#         cfg_retinaface_anchor['INPUT_SIZE'],
#         cfg_retinaface_anchor['FEATURE_MAP_LIST'],
#         cfg_retinaface_anchor['SCALE_LIST'],
#         cfg_retinaface_anchor['RATIO_LIST']
#     )
#     print(anchors.size())
#     print(anchors[0:10])
#     print(anchors[-10:])
#
#
# if __name__ == '__main__':
#     main()

