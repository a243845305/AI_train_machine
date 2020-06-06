# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import json

from Cores.utils import calculate_iou


class LossLayer(nn.Module):
    def __init__(self):
        super(LossLayer, self).__init__()

    def forward(self, classifications, bbox_regressions, ldm_regressions, anchors, annotations,
                device=torch.device('cpu'), scale_xy=0.1, scale_wh=0.2):

        self.device = device
        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

        batch_size = classifications.shape[0]
        classification_losses = []
        bbox_regression_losses = []
        ldm_regression_losses = []

        anchor = anchors
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights


        # temp
        # positive_indices_list = []

        for j in range(batch_size):
            classification = classifications[j, :, :]
            bbox_regression = bbox_regressions[j, :, :]
            ldm_regression = ldm_regressions[j, :, :]

            annotation = json.loads(annotations[j])
            rects = [label['rect'] for label in annotation]
            bbox_annotation = torch.tensor(rects, dtype=torch.float32, requires_grad=False).to(self.device)
            landmarks = [label['landmark'] for label in annotation]
            ldm_annotation = torch.tensor(landmarks, dtype=torch.float32).to(self.device)

            if bbox_annotation.shape[0] == 0:
                bbox_regression_losses.append(
                    torch.tensor(0., requires_grad=True, dtype=torch.float32).to(self.device)
                )
                classification_losses.append(
                    torch.tensor(0., requires_grad=True, dtype=torch.float32).to(self.device)
                )
                ldm_regression_losses.append(
                    torch.tensor(0., requires_grad=True, dtype=torch.float32).to(self.device)
                )

                # positive_indices_list.append([])

                continue

            IoU = calculate_iou(anchor, bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            targets = torch.ones(classification.size(), dtype=torch.float32) * -1
            targets = targets.to(self.device)

            # those whose iou<0.3 have no object
            negative_indices = torch.lt(IoU_max, 0.3)
            targets[negative_indices, :] = 0
            targets[negative_indices, 1] = 1

            # those whose iou>0.5 have object
            positive_indices = torch.ge(IoU_max, 0.5)
            # print('find {} positive'.format(positive_indices.sum()))

            # temp
            # positive_indices_list.append(positive_indices)

            num_positive_anchors = positive_indices.sum()

            # keep positive and negative ratios with 1:3
            keep_negative_anchors = num_positive_anchors * 3

            bbox_assigned_annotations = bbox_annotation[IoU_argmax, :]
            ldm_assigned_annotations = ldm_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, 0] = 1

            ldm_sum = ldm_assigned_annotations.sum(dim=1)
            ge0_mask = ldm_sum > 0
            ldm_positive_indices = ge0_mask & positive_indices

            # OHEM
            classification = F.log_softmax(classification, dim=0)
            negative_losses = classification[negative_indices, 1] * -1
            sorted_losses, _ = torch.sort(negative_losses, descending=True)
            if sorted_losses.numel() > keep_negative_anchors:
                sorted_losses = sorted_losses[:keep_negative_anchors]
            positive_losses = classification[positive_indices, 0] * -1

            if positive_indices.sum() > 0:
                classification_losses.append(positive_losses.mean() + sorted_losses.mean())
            else:
                classification_losses.append(
                    torch.tensor(0., requires_grad=True, dtype=torch.float32).to(self.device)
                )
            # compute bboxes loss
            if positive_indices.sum() > 0:
                # bbox
                bbox_assigned_annotations = bbox_assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = bbox_assigned_annotations[:, 2] - bbox_assigned_annotations[:, 0]
                gt_heights = bbox_assigned_annotations[:, 3] - bbox_assigned_annotations[:, 1]
                gt_ctr_x = bbox_assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = bbox_assigned_annotations[:, 1] + 0.5 * gt_heights

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi + 1e-14)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi + 1e-14)
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                bbox_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                bbox_targets = bbox_targets.t()

                # Rescale
                bbox_scale = torch.tensor(
                        [[self.scale_xy, self.scale_xy, self.scale_wh, self.scale_wh]], dtype=torch.float32
                ).to(self.device)
                bbox_targets = bbox_targets/bbox_scale

                # smooth L1
                # box losses
                bbox_regression_loss = nn.SmoothL1Loss(bbox_targets, bbox_regression[positive_indices, :])
                bbox_regression_losses.append(bbox_regression_loss)
            else:
                bbox_regression_losses.append(
                    torch.tensor(0., requires_grad=True, dtype=torch.float32).to(self.device)
                )

                # compute landmarks loss
            if ldm_positive_indices.sum() > 0:
                ldm_assigned_annotations = ldm_assigned_annotations[ldm_positive_indices, :]

                anchor_widths_l = anchor_widths[ldm_positive_indices]
                anchor_heights_l = anchor_heights[ldm_positive_indices]
                anchor_ctr_x_l = anchor_ctr_x[ldm_positive_indices]
                anchor_ctr_y_l = anchor_ctr_y[ldm_positive_indices]

                anchor_widths_l = torch.unsqueeze(anchor_widths_l, 1)
                anchor_heights_l = torch.unsqueeze(anchor_heights_l, 1)
                anchor_ctr_x_l = torch.unsqueeze(anchor_ctr_x_l, 1)
                anchor_ctr_y_l = torch.unsqueeze(anchor_ctr_y_l, 1)

                landmarks_trans = torch.zeros(ldm_assigned_annotations.size())
                landmarks_trans = landmarks_trans.to(self.device)
                landmarks_trans[:,::2] = (ldm_assigned_annotations[:, ::2] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                landmarks_trans[:, 1::2] = (ldm_assigned_annotations[:, 1::2] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)

                ldm_targets = landmarks_trans
                # ldm_targets = ldm_targets.t()

                # Rescale
                ldm_targets = ldm_targets/self.scale_xy

                ldm_regression_loss = nn.SmoothL1Loss(ldm_targets, ldm_regression[ldm_positive_indices, :])
                ldm_regression_losses.append(ldm_regression_loss)
            else:
                ldm_regression_losses.append(
                    torch.tensor(0., requires_grad=True, dtype=torch.float32).to(self.device)
                )

        return (torch.stack(classification_losses).mean(),
                torch.stack(bbox_regression_losses).mean(),
                torch.stack(ldm_regression_losses).mean())
        # return positive_indices_list, torch.stack(classification_losses), torch.stack(bbox_regression_losses),torch.stack(ldm_regression_losses)


