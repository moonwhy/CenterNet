from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


def main(opt):
  torch.manual_seed(opt.seed)  # 使得每次获取的随机数都是一样的
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test  # 结构固定、形状固定时可以加快运行速度
  Dataset = get_dataset(opt.dataset, opt.task)   # (coco ctdet) 得到实例Dataset(实例COCO, 实例CTDetDataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)  # 根据数据集和arch生成检测器头所需的参数
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)  # 当前测试模型结构（DLA、hourglass）、检测头、检测头卷即层设施
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)  # 设置优化器、迭代返回模型参数
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]  #
  trainer = Trainer(opt, model, optimizer)  # 产生一个训练实例
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  writer = SummaryWriter(log_dir=os.path.join(opt.save_dir, 'runs'))
  y=[]

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'   # 模型保存参数

    log_dict_train, _ = trainer.train(epoch, train_loader)  # 训练

    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

    writer.add_scalar(os.path.join(opt.save_dir, 'runs/scalar/train70-2'), log_dict_train['loss'], epoch)
    '''
    x = range(0,epoch)
    y.append(log_dict_train['loss'])
    plt.plot(x, y, '.-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.show()
    '''
  logger.close()
  writer.close()


if __name__ == '__main__':
  minglingstr = 'ctdet --exp_id fod_seg_hg2 --dataset fod --arch hourglass --add_segmentation ' \
                '--num_epochs 70 --batch_size 1 --lr 2.5e-4 --lr_step 50 ' \
                '--load_model ../models/ctdet_coco_hg.pth'
#                '--num_iters 5'
  opt = opts().parse(minglingstr.split())
  main(opt)

  '''
  查看tensorboardX结果，在fod_hg文件夹中打开终端
  运行tensorboard --logdir runs，
  打开链接即可
  '''