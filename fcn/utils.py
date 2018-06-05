import numpy as np

import torch.nn.functional as F

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def measure(label_trues, label_preds, num_class):
  """Returns accuracy score evaluation result.
    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
  """
  hist = np.zeros((num_class, num_class))
  for lt, lp in zip(label_trues, label_preds):
    hist += _fast_hist(lt.flatten(), lp.flatten(), num_class)
  acc = np.diag(hist).sum() / hist.sum()
  acc_cls = np.diag(hist) / hist.sum(axis=1)
  acc_cls = np.nanmean(acc_cls)
  iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
  mean_iu = np.nanmean(iu)
  freq = hist.sum(axis=1) / hist.sum()
  fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
  return {'acc': acc,
          'acc_cls': acc_cls,
          'mean_iu': mean_iu,
          'fwavacc': fwavacc
          }

def cross_entropy2d(input, target, weight=None, size_average=True):
  # input: (n, c, h, w), target: (n, h, w)
  n, c, h, w = input.size()
  # log_p: (n, c, h, w)
  log_p = F.log_softmax(input, dim=1)
  # log_p: (n*h*w, c)
  log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
  log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
  log_p = log_p.view(-1, c)
  # target: (n*h*w,)
  mask = target >= 0
  target = target[mask]
  loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
  if size_average:
    loss /= mask.sum().float()
  return loss
