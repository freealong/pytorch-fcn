import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX

import fcn

CONFIG = {
  'max_iter': 900000,
  'lr': 1.e-14,
  'momentum':  0.99,
  'weight_decay': 0.0005,
  'interval_validate': 4000,

  'batch_size': 1,
  'print_freq': 1000
}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_dir', type=str, default='runs')
  parser.add_argument('--name', type=str, default='voc_fcn8')
  args = parser.parse_args()

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
  dataset_root = '/mnt/datasets/voc'
  train_dataset = fcn.datasets.VOC2012ClassSeg(dataset_root, split='train', transform=True)
  train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

  val_dataset = fcn.datasets.VOC2012ClassSeg(dataset_root, split='val', transform=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

  num_classes = len(train_dataset.class_names)

  # 2. model
  model = fcn.models.FCN8(num_classes=num_classes)

  # 3. optimizer, criterion, measure
  optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['lr'],
                              momentum=CONFIG['momentum'],
                              weight_decay=CONFIG['weight_decay'])
  criterion = fcn.utils.cross_entropy2d
  measure = fcn.utils.measure

  # 4. writer(optional)
  writer = tensorboardX.SummaryWriter(log_dir=log_dir)

  # 5. trainer
  trainer = fcn.trainer.Trainer(model=model, optimizer=optimizer, criterion=criterion,
                                measure=measure, train_loader=train_loader, val_loader=val_loader,
                                max_iter=CONFIG['max_iter'], print_freq=CONFIG['print_freq'],
                                interval_eval=CONFIG['interval_validate'], writer=writer,
                                ckp_path=ckp_dir)
  trainer.train()

if __name__ == '__main__':
  main()