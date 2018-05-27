import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX

import fcn

CONFIG = {
  'max_iter': 100000,
  'lr': 1.e-14,
  'momentum':  0.99,
  'weight_decay': 0.0005,
  'interval_validate': 4000,

  'batch_size': 32,
  'print_freq': 1000
}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, required=True)
  parser.add_argument('--ckp_dir', type=str, required=True)
  args = parser.parse_args()

  # 1. dataset
  dataset_root = '/mnt/yongqi/datasets/voc'
  dataset = fcn.datasets.VOC2012ClassSeg(dataset_root, transform=True)
  train_loader = DataLoader(dataset=dataset, batch_size=CONFIG['batch_size'], shuffle=True)
  val_loader = DataLoader(dataset=dataset, batch_size=CONFIG['batch_size'], shuffle=False)
  num_classes = len(dataset.class_names)

  # 2. model
  model = fcn.models.FCN8(num_classes=num_classes)

  # 3. optimizer, criterion, measure
  optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['lr'],
                              momentum=CONFIG['momentum'],
                              weight_decay=CONFIG['weight_decay'])
  criterion = nn.CrossEntropyLoss()
  measure = fcn.utils.measure

  # 4. writer(optional)
  writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

  # 5. trainer
  trainer = fcn.trainer.Trainer(model=model, optimizer=optimizer, criterion=criterion,
                                measure=measure, train_loader=train_loader, val_loader=val_loader,
                                max_iter=CONFIG['max_iter'], print_freq=CONFIG['print_freq'],
                                interval_eval=CONFIG['interval_validate'], ckp_path=args.ckp_dir)
  trainer.train()

if __name__ == '__main__':
  main()