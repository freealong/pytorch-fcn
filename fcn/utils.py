import numpy as np

class Metrics(object):
  def __init__(self):
    self.acc = 0

  def __add__(self, other):
    pass

  def __iadd__(self, other):
    self.__add__(other)

  def __truediv__(self, other):
    pass

  def __idiv__(self, other):
    self.__truediv__(other)

  def __str__(self):
    pass

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


