import math
import os

import torch
import torch.nn as nn


class Trainer(object):
  def __init__(self, model, optimizer, criterion, measure,
               train_loader, val_loader, max_iter,
               print_freq, interval_eval, writer,
               ckp_path, resume_ckp_num=0):
    self.global_step = 0
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.measure = measure
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.max_iter = max_iter
    self.print_freq = print_freq
    if interval_eval is None:
      self.interval_eval = len(self.train_loader)
    else:
      self.interval_eval = interval_eval
    self.writer = writer
    # visualize graph if tensorboardX available
    if self.writer != None:
      dummy_input, _ = iter(self.train_loader).next()
      self.writer.add_graph(self.model, (dummy_input,))
    if os.path.isdir(ckp_path):
      self.ckp_path = ckp_path
    else:
      print('checkpoint path {} does not exist.'.format(ckp_path))
      raise ValueError
    if resume_ckp_num != 0:
      resume_ckp_file = os.path.join(ckp_path, 'checkpoint_%06d.tar' % resume_ckp_num)
      if os.path.isfile(resume_ckp_file):
        print("loading checkpoint from {}".format(resume_ckp_file))
        checkpoint = torch.load(resume_ckp_file)
        self.model.load_state_dict(checkpoint['state_dick'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
      else:
        print("checkpoint file {} not found, training from the beginning.".format(resume_ckp_file))

    # using cuda if available
    if torch.cuda.is_available():
      print('%d gpu devices found' % torch.cuda.device_count())
      if torch.cuda.device_count() > 1:
        self.model = nn.DataParallel(self.model)
      self.model = self.model.cuda()
    else:
      print('no gpu device found, using cpu only')


  def train(self):
    max_epoch = int(math.ceil(self.max_iter / len(self.train_loader)))
    num_class = len(self.train_loader.dataset.class_names)

    for epoch in range(1, max_epoch + 1):
      self.model.train()
      running_loss = 0.0
      val_metrics = {}

      for i, data in enumerate(self.train_loader, 1):
        if self.global_step >= self.max_iter:
          break
        # get data
        inputs, labels = data
        if torch.cuda.is_available():
          inputs, labels = inputs.cuda(), labels.cuda()
        # optimize
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        # calculate loss and metrics
        running_loss += loss.item()
        val_metrics = self._add_metrics(self.measure(labels, outputs, num_class), val_metrics)
        # print and log
        if i % self.print_freq == 0:
          running_loss /= self.print_freq * len(inputs)
          val_metrics = self._mean_metrics(val_metrics, self.print_freq * len(inputs))
          print('epoch[%d/%d], iter[%d/%d], global_step[%d/%d] loss: %.3f' %
                (epoch, max_epoch, i, len(self.train_loader), self.global_step, self.max_iter, running_loss))
          if self.writer != None:
            self.writer.add_scalar('loss', running_loss, epoch * i)
            for name, value in val_metrics.items():
              self.writer.add_scalar(name, value, epoch * i)
          running_loss = 0
          val_metrics = {}


        if self.global_step % self.interval_eval == 0:
          self.eval()
          torch.save({'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'global_step': self.global_step},
                     os.path.join(self.ckp_path, 'checkpoint_%06d.tar' % self.global_step))

    print('Finish Training')

  def eval(self):
    training = self.model.training
    self.model.eval()

    num_class = len(self.val_loader.dataset.class_names)

    val_loss = 0
    val_metrics = []
    for i, data in enumerate(self.val_loader, 1):
      inputs, labels = data
      if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()

      outputs = self.model(inputs)
      loss = self.criterion(outputs, labels)
      val_loss += loss.item() / len(inputs)
      val_metrics = self._add_metrics(self.measure(labels, outputs, num_class), val_metrics)

    val_loss /= len(self.val_loader)
    val_metrics = self._mean_metrics(val_metrics, len(self.val_loader))

    print("eval loss: %.3f" % val_loss)
    print(val_metrics)
    if self.writer != None:
      self.writer.add_scalar('eval_loss', val_loss, self.global_step)
      for name, value in val_metrics.items():
        self.writer.add_scalar('eval_'.join(name), value, self.global_step)

    if training:
      self.model.train()

  def _add_metrics(self, A, B):
    for name, value in A.items():
      A[name] += B[name]
    return A

  def _mean_metrics(self, m, num):
    if len(m) == 0:
      return m
    for name, value in m.items():
      m[name] /= num
    return m
