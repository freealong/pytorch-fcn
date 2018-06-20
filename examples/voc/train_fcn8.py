import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX


CONFIG = {
  'max_iter': 100000,
  'lr': 0.0001,
  'momentum':  0.9,
  'weight_decay': 0.0005,
  'interval_validate': 4000,

  'batch_size': 8,
  'print_freq': 100
}

def main():
  import fcn

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataroot', type=str, default='datasets')
  parser.add_argument('--run_dir', type=str, default='runs')
  parser.add_argument('--name', type=str, default='voc_fcn8')
  parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
  args = parser.parse_args()

  str_ids = args.gpu_ids.split(',')
  gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      gpu_ids.append(id)

  if not os.path.isdir(args.run_dir):
    os.mkdir(args.run_dir)
  base_path = os.path.join(args.run_dir, args.name)
  if not os.path.isdir(base_path):
    os.mkdir(base_path)
  log_dir = os.path.join(base_path, 'logs')
  ckp_dir = os.path.join(base_path, 'ckps')
  if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
  if not os.path.isdir(ckp_dir):
    os.mkdir(ckp_dir)

  # 1. dataset
  dataset_root = args.dataroot
  train_dataset = fcn.datasets.VOC2012ClassSeg(dataset_root, split='train', transform=True)
  train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
  print('load train data with %d images' % len(train_dataset))

  val_dataset = fcn.datasets.VOC2012ClassSeg(dataset_root, split='val', transform=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
  print('load eval data with %d images' % len(val_dataset))

  num_classes = len(train_dataset.class_names)

  # 2. model
  model = fcn.models.FCN8(num_classes=num_classes)
  print('init model okey')

  # 3. optimizer, criterion, measure
  optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['lr'],
                              momentum=CONFIG['momentum'],
                              weight_decay=CONFIG['weight_decay'])
  criterion = fcn.utils.cross_entropy2d
  measure = fcn.utils.measure

  # 4. writer(optional)
  writer = tensorboardX.SummaryWriter(log_dir=log_dir)
  print('write log to ' + log_dir)

  # 5. trainer
  trainer = fcn.trainer.Trainer(model=model, optimizer=optimizer, criterion=criterion,
                                measure=measure, train_loader=train_loader, val_loader=val_loader,
                                max_iter=CONFIG['max_iter'], print_freq=CONFIG['print_freq'],
                                interval_eval=CONFIG['interval_validate'], writer=writer,
                                ckp_path=ckp_dir, gpu_ids=gpu_ids)
  trainer.train()

if __name__ == '__main__':
  import sys
  from os.path import dirname
  sys.path.append(dirname(dirname(sys.path[0])))
#  print(sys.path)
  main()
