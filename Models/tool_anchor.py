import torch
import torch.nn as nn
import numpy as np


def make_anchor(map_size, step, scale_list, ratio_list):
    w = map_size[0]*step
    h = map_size[1]*step
    center_x_list = np.arange(0, w, step) + step/2.
    center_y_list = np.arange(0, h, step) + step/2.
    center_x_list, center_y_list = np.meshgrid(center_x_list, center_y_list)
    center_x_list = center_x_list.reshape(map_size[0]*map_size[1], 1)
    center_y_list = center_y_list.reshape(map_size[0]*map_size[1], 1)

    anchor_list = np.zeros([map_size[0]*map_size[1], len(scale_list)*len(ratio_list), 4])
    anchor_list[:, :, 0] = center_x_list
    anchor_list[:, :, 1] = center_y_list

    for i,scale in enumerate(scale_list):
        for j,ratio in enumerate(ratio_list):
            anchor_w = step*np.sqrt(scale)*ratio
            anchor_h = step*np.sqrt(scale)/ratio
            anchor_x = -anchor_w/2.
            anchor_y = -anchor_h/2.
            anchor_index = i*len(ratio_list)+j
            anchor_list[:,anchor_index,0] += anchor_x
            anchor_list[:,anchor_index,1] += anchor_y
            anchor_list[:,anchor_index,2] = anchor_list[:,anchor_index,0]+anchor_w
            anchor_list[:,anchor_index,3] = anchor_list[:,anchor_index,1]+anchor_h

    anchor_list = anchor_list.reshape(-1,4)

    return anchor_list


class Anchors(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(Anchors, self).__init__()
        self.device = device

    def forward(self, input_size, feature_maps, scale_lists, ratio_lists):
        # feature_maps = config['FEATURE_MAP_LIST']
        # steps = [config['INPUT_SIZE'][1] / feature_map[0] for feature_map in feature_maps]
        # scale_lists = config['SCALE_LIST']
        # ratio_lists = config['RATIO_LIST']
        steps = [input_size[1] / feature_map[0] for feature_map in feature_maps]

        anchor_list = []
        for feature_map, step, scale_list, ratio_list in zip(feature_maps, steps, scale_lists, ratio_lists):
            anchor_list.append(
                torch.tensor(
                    make_anchor(feature_map, step, scale_list, ratio_list), dtype=torch.float32).to(self.device)
            )
        anchor_list = torch.cat(anchor_list, dim=0)

        return anchor_list


