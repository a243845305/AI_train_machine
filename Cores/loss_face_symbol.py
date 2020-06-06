import os
import sys
import mxnet as mx
from config import config,default

sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import resnet
import mobilefacenet

def get_symbol(args):
  embedding = eval(config.net_name).get_symbol()
  #Dynamic output network fc1 layer value
  all_label = mx.symbol.Variable('softmax_label')
  gt_label = all_label
  is_softmax = True
  if config.loss_name=='softmax': #softmax
    _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
        lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
    if config.fc7_no_bias:
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
    else:
      _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
      fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=config.num_classes, name='fc7')

  elif config.loss_name=='margin_softmax':
    _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
        lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
    s = config.loss_s
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
    if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
      if config.loss_m1==1.0 and config.loss_m2==0.0:
        s_m = s*config.loss_m3
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = s_m, off_value = 0.0)
        fc7 = fc7-gt_one_hot
      else:
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy/s
        t = mx.sym.arccos(cos_t)
        if config.loss_m1!=1.0:
          t = t*config.loss_m1
        if config.loss_m2>0.0:
          t = t+config.loss_m2
        body = mx.sym.cos(t)
        if config.loss_m3>0.0:
          body = body - config.loss_m3
        new_zy = body*s
        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7+body

  elif config.loss_name.find('triplet')>=0:
    is_softmax = False
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
    anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=default.per_batch_size//3)
    positive = mx.symbol.slice_axis(nembedding, axis=0, begin=default.per_batch_size//3, end=2*default.per_batch_size//3)
    negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2*default.per_batch_size//3, end=default.per_batch_size)
    if config.loss_name=='triplet':
      ap = anchor - positive
      an = anchor - negative
      ap = ap*ap
      an = an*an
      ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
      an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
      triplet_loss = mx.symbol.Activation(data = (ap-an+config.triplet_alpha), act_type='relu')
      triplet_loss = mx.symbol.mean(triplet_loss)
    else:
      ap = anchor*positive
      an = anchor*negative
      ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
      an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
      ap = mx.sym.arccos(ap)
      an = mx.sym.arccos(an)
      triplet_loss = mx.symbol.Activation(data = (ap-an+config.triplet_alpha), act_type='relu')
      triplet_loss = mx.symbol.mean(triplet_loss)
    triplet_loss = mx.symbol.MakeLoss(triplet_loss)
  out_list = [mx.symbol.BlockGrad(embedding)]
  
  if is_softmax:
    print("out list is",out_list)
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    if config.ce_loss:
      #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/default.per_batch_size
      body = mx.symbol.SoftmaxActivation(data=fc7)
      body = mx.symbol.log(body)
      _label = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)
      body = body*_label
      ce_loss = mx.symbol.sum(body)/default.per_batch_size
      out_list.append(mx.symbol.BlockGrad(ce_loss))
  else:
    print("out list is....",out_list)
    out_list.append(mx.sym.BlockGrad(gt_label))
    out_list.append(triplet_loss)
  out = mx.symbol.Group(out_list)
  return out