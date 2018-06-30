import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
import tensorboardX

def main():
  import fcn

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True, help='config json file')
  parser.add_argument('--dataroot', type=str, default='datasets', help='data root dir')
  parser.add_argument('--run_dir', type=str, default='runs', help='checkpoints and logs root dir')
  parser.add_argument('--name', type=str, default='voc_fcn8', help='experiment name')
  parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
  parser.add_argument('--resume_ckp_num', type=int, default=0, help='checkpoint id')
  args = parser.parse_args()

  with open(args.config) as f:
    config = json.loads(f.read())

  str_ids = args.gpu_ids.split(',')
  gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      gpu_ids.append(id)
  resume_ckp_num = args.resume_ckp_num

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
  train_loader = DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'], shuffle=True)
  print('load train data with %d images' % len(train_dataset))

  val_dataset = fcn.datasets.VOC2012ClassSeg(dataset_root, split='val', transform=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=config['eval_batch_size'], shuffle=False)
  print('load eval data with %d images' % len(val_dataset))

  num_classes = len(train_dataset.class_names)

  # 2. model
  model = fcn.models.FCN8(num_classes=num_classes)
  print('init model okey')

  # 3. optimizer, criterion, measure
  optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
  criterion = fcn.utils.cross_entropy2d
  measure = fcn.utils.measure

  # 4. writer(optional)
  writer = tensorboardX.SummaryWriter(log_dir=log_dir)
  print('write log to ' + log_dir)

  # 5. trainer
  trainer = fcn.trainer.Trainer(model=model, optimizer=optimizer, criterion=criterion,
                                measure=measure, train_loader=train_loader, val_loader=val_loader,
                                max_iter=config['max_iter'], print_freq=config['print_freq'],
                                save_freq=config['save_freq'], eval_freq=config['eval_freq'],
                                writer=writer, ckp_path=ckp_dir,
                                resume_ckp_num=resume_ckp_num, gpu_ids=gpu_ids)
  trainer.train()

if __name__ == '__main__':
  import sys
  from os.path import dirname
  sys.path.append(dirname(dirname(sys.path[0])))
#  print(sys.path)
  main()
