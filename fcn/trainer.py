import math
import os

import torch
import torch.nn as nn
import torchvision.utils as vutils


class Trainer(object):
  def __init__(self, model, optimizer, criterion, measure,
               train_loader, val_loader, max_iter,
               print_freq, save_freq, eval_freq,
               writer, ckp_path, resume_ckp_num=0,
               gpu_ids=list()):
    self.global_step = 0
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.measure = measure
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.max_iter = max_iter
    self.print_freq = print_freq
    self.save_freq = save_freq
    self.eval_freq = eval_freq
    self.writer = writer
    # visualize graph if tensorboardX available
    # if self.writer != None and resume_ckp_num == 0:
    #   dummy_input, _ = iter(self.val_loader).next()
    #   self.writer.add_graph(self.model, (dummy_input[0],))


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
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
      else:
        print("checkpoint file {} not found, training from the beginning.".format(resume_ckp_file))

    # using cuda if available
    self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    if len(gpu_ids) > 0 and torch.cuda.is_available():
      if len(gpu_ids) > 1:
        self.model = nn.DataParallel(self.model, gpu_ids)
      else:
        self.model.to(self.device)
        # optim load_state_dict has a problem: https://github.com/pytorch/pytorch/issues/2830
        if resume_ckp_num != 0:
          for state in optimizer.state.values():
            for k, v in state.items():
              if isinstance(v, torch.Tensor):
                state[k] = v.to(self.device)
      print('Using gpu: ', gpu_ids)
    else:
      print('Using cpu only')

  def train(self):
    max_epoch = int(math.ceil(self.max_iter / len(self.train_loader)))

    running_loss = 0.0
    for epoch in range(1, max_epoch + 1):
      self.model.train()

      for i, data in enumerate(self.train_loader, 1):
        if self.global_step >= self.max_iter:
          break
        # get data
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # optimize
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        running_loss += loss.item()

        # print and log
        if self.global_step % self.print_freq == 0:
          running_loss /= self.print_freq * len(inputs)
          print('epoch[%d/%d], iter[%d/%d], global_step[%d/%d] loss: %.3f' %
                (epoch, max_epoch, i, len(self.train_loader), self.global_step, self.max_iter, running_loss))
          if self.writer != None:
            self.writer.add_scalar('loss', running_loss, self.global_step)
            input_images = vutils.make_grid(inputs.cpu().detach(), normalize=True, scale_each=True)
            self.writer.add_image('inputs', input_images, self.global_step)
            lables_images = vutils.make_grid(labels.cpu().detach(), normalize=True, scale_each=True)
            self.writer.add_image('labels', lables_images, self.global_step)
            pred_labels_images = vutils.make_grid(outputs.max(1)[1].cpu().detach(), normalize=True, scale_each=True)
            self.writer.add_image('pred labels', pred_labels_images, self.global_step)
          running_loss = 0

        # save checkpoint
        if self.save_freq != 0 and self.global_step % self.save_freq == 0:
          torch.save({'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'global_step': self.global_step},
                     os.path.join(self.ckp_path, 'checkpoint_%06d.tar' % self.global_step))

        # eval
        if self.eval_freq != 0 and self.global_step % self.eval_freq == 0:
          self.eval()

    print('Finish Training')

  def eval(self):
    training = self.model.training
    self.model.eval()

    num_class = len(self.val_loader.dataset.class_names)

    val_loss = 0
    val_metrics = {}
    for i, data in enumerate(self.val_loader, 1):
      inputs, labels = data
      inputs, labels = inputs.to(self.device), labels.to(self.device)

      outputs = self.model(inputs)
      loss = self.criterion(outputs, labels)
      val_loss += loss.item() / len(inputs)
      pred_labels = outputs.max(1)[1].cpu().detach().numpy()[:, :, :]
      true_labels = labels.cpu().detach().numpy()
      val_metrics = self._add_metrics(self.measure(true_labels, pred_labels, num_class), val_metrics)

    val_loss /= len(self.val_loader)
    val_metrics = self._mean_metrics(val_metrics, len(self.val_loader))

    print("global_step[%d/%d] eval loss: %.3f" % (self.global_step, self.max_iter, val_loss))
    print(val_metrics)
    if self.writer != None:
      self.writer.add_scalar('eval_loss', val_loss, self.global_step)
      for name, value in val_metrics.items():
        self.writer.add_scalar('eval_' + name, value, self.global_step)

    if training:
      self.model.train()

  def _add_metrics(self, A, B):
    if len(A) == 0 and len(B) == 0:
      return {}
    elif len(A) == 0 and len(B) != 0:
      return B
    elif len(A) != 0 and len(B) == 0:
      return A
    else:
      for name, value in A.items():
        A[name] += B[name]
      return A

  def _mean_metrics(self, m, num):
    if len(m) == 0:
      return m
    for name, value in m.items():
      m[name] /= num
    return m
