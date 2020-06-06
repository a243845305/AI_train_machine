'''
@Author: your name
@Date: 2020-06-02 02:59:43
@LastEditTime: 2020-06-02 09:56:47
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /AI_train_machine/Train/train2.py
'''
import time
import os
import os.path as osp
import logging
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter

def train(context):
    dataSet = context.dataSet
    outPut = context.outPut

    # Network Setup
    net = context.model

    # Training Setup
    epoch = context.epoch
    device = context.device
    optimizer = context.optimizer
    loss_fun = context.loss_fun

    # Checkpoints Setup
    checkpoints = outPut['checkpoints']
    restore = outPut['restore']
    restore_model = outPut['restore_model']
    os.makedirs(checkpoints, exist_ok=True)

    if restore:
        weights_path = osp.join(checkpoints, restore_model)
        net.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"load weights from checkpoints: {restore_model}")

    # logger Setup
    log_name = outPut['log_name']
    timestr = time.strftime('%m%d_%H',time.localtime(time.time()))
    writer = SummaryWriter(logdir='./log/'+log_name+'_'+timestr+'/', comment=log_name )  

    logging.basicConfig(level = logging.DEBUG,
                        format = '[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%a,%d %b %Y %H:%M:%S',
                        filename='./log/'+log_name+'_'+timestr+'.log',
                        filemode='w')
    logger=logging.getLogger()
    ch=logging.StreamHandler()
    logger.addHandler(ch)

    # Start training
    net.train()

    for e in range(epoch):
        for data in tqdm(dataSet['trainData'], desc=f"Epoch {e}/{epoch}",
                                ascii=True, total=len(dataSet['trainData'])):           
            optimizer.zero_grad()
            # 前向传播 计算loss
            losses, loss = loss_fun(data, device)

            # 反向传播
            loss.backward()
            optimizer.step()

            l_heatmap = losses['loss_hm']
            l_off = losses['loss_offset']
            l_wh = losses['loss_wh']


        logger.info(f"Epoch {e}/{epoch}, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}")
        writer.add_scalar("scalar/{}".format('Loss'), loss, e)
        writer.add_scalar("scalar/{}".format('heatmap'), l_heatmap, e)
        writer.add_scalar("scalar/{}".format('offset'), l_off, e)
        writer.add_scalar("scalar/{}".format('wh'), l_wh, e)
        writer.flush()
        # print(f"Epoch {e}/{cfg.epoch}, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}")

        backbone_path = osp.join(checkpoints, f"{log_name}_{e}.pth")
        torch.save(net.state_dict(), backbone_path)