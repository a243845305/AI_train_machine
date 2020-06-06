'''
Context
包含内容：{
    1. DataSet
    2. Model_Train{
        2.0 model
        2.1 Loss
            2.1.1 Anchor(if use)
        2.2 Optimizer
        2.3 Epoch                  
    }
    3. Output
   }
'''

import sys
sys.path.append('../..')
from DataLoad.dataloader_centerNet_5hm import dataLoader, dataSet
from Models.backbone_centerTemp import centerface
from Models.mnet import MobileNetSeg
from Train.train import train
import torch.nn as nn
import torch
import yaml

from config import Config as cfg

class RegLoss(nn.Module):
    """Regression loss for CenterFace, especially 
    for offset, size and landmarks
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred = pred[mask]
        gt = gt[mask]
        loss = self.loss(pred, gt)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss

class Context():
    def __init__(self, cfg):
        super().__init__()
        
        self.dataSet = {}
        self.outPut = {}

        # 解析 config 
        use_cuda = cfg.use_cuda
        device = torch.device("cuda:1" if use_cuda else "cpu")

        # 1. 数据集处理
        # ===========================================================================
        DATA_Root = cfg.data_root
        DATA_Enhance = cfg.data_enhance
        DATA_TRANSFORMS = cfg.train_transforms

        trainset = dataSet(DATA_Root, DATA_Enhance, 'train', DATA_TRANSFORMS)
        valset = dataSet(DATA_Root, DATA_Enhance, 'val', DATA_TRANSFORMS)
        testset = dataSet(DATA_Root, DATA_Enhance, 'test', None)

        dataloader = dataLoader(batch_size=cfg.batch_size, pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
        trainData = dataloader.loader(trainset)
        valData = dataloader.loader(valset)
        testData = dataloader.loader(testset)

        # 2. 模型处理
        # =========================================================================
        model = centerface().to(device)
        # heads = {'hm':1, 'wh':2, 'off':2}
        # head_conv=24
        # model = MobileNetSeg('mobilenetv2_{}'.format(10), heads,
        #          pretrained=True,
        #          head_conv=head_conv).to(device)
        
        
        # 2.1 Loss 处理
        # =================================================
        loss_fun = self._LossFun

        # 2.3 Optimizer 处理
        # =================================================
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # 2.4 Epoch 处理
        epoch = cfg.epoch

        # 3. OutPut 的解析及处理
        # =========================================================================
        log_name = cfg.log_name
        checkpoints = cfg.checkpoints
        restore = cfg.restore
        restore_model = cfg.restore_model

        # 装填内容
        # =========================================================================
        self.dataSet['trainData'] = trainData
        self.dataSet['valData'] = valData
        self.dataSet['testData'] = testData

        self.device = device
        self.model = model
        
        self.focal_loss = nn.MSELoss(reduction='sum').to(device)
        self.regloss = RegLoss().to(device)

        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.epoch = epoch

        self.outPut['log_name'] = log_name
        self.outPut['checkpoints'] = checkpoints
        self.outPut['restore'] = restore
        self.outPut['restore_model'] = restore_model
    
    def _LossFun(self, data, device):

        imgs, labels, fn = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        out_hm, out_wh, out_off = self.model(imgs)

        # out = self.model(imgs)    
        # out_hm = torch.cat([o['hm'].squeeze() for o in out], dim=0) 
        # out_off = torch.cat([o['off'].squeeze() for o in out], dim=0)
        # out_wh = torch.cat([o['wh'].squeeze() for o in out], dim=0)

        loss_hm = self.focal_loss(out_hm, labels[:, 0])
        loss_off = self.regloss(out_off,  labels[:, [1,2]])
        loss_wh = self.regloss(out_wh,  labels[:, [3,4]])

        losses = {}
        losses['loss_hm'] = loss_hm
        losses['loss_wh'] = loss_wh
        losses['loss_offset'] = loss_off
        
        loss = loss_hm + 0.1 * loss_wh + loss_off

        return losses, loss



if __name__=='__main__':

    context = Context(cfg)
    train(context)