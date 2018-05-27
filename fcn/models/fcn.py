import torch
import torch.nn as nn

import numpy as np

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
  """Make a 2D bilinear kernel suitable for upsampling"""
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  og = np.ogrid[:kernel_size, :kernel_size]
  filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
  weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
  weight[range(in_channels), range(out_channels), :, :] = filt
  return torch.from_numpy(weight).float()

class FCN8(nn.Module):
  def __init__(self, num_classes=21, input_channel=3):
    super(FCN8).__init__()
    # conv1
    self.conv1_1 = nn.Conv2d(input_channel, 64, 3, padding=100)
    self.relu1_1 = nn.ReLU(inplace=True)
    self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
    self.relu1_2 = nn.ReLU(inplace=True)
    self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/2

    # conv2
    self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.relu2_1 = nn.ReLU(inplace=True)
    self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
    self.relu2_2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/4

    # conv3
    self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
    self.relu3_1 = nn.ReLU(inplace=True)
    self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_2 = nn.ReLU(inplace=True)
    self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_3 = nn.ReLU(inplace=True)
    self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/8

    # conv4
    self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
    self.relu4_1 = nn.ReLU(inplace=True)
    self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_2 = nn.ReLU(inplace=True)
    self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_3 = nn.ReLU(inplace=True)
    self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/16

    # conv5
    self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_1 = nn.ReLU(inplace=True)
    self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_2 = nn.ReLU(inplace=True)
    self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_3 = nn.ReLU(inplace=True)
    self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32

    # fc6
    self.fc6 = nn.Conv2d(512, 4096, 7)
    self.relu6 = nn.ReLU(inplace=True)
    self.drop6 = nn.Dropout2d()

    # fc7
    self.fc7 = nn.Conv2d(4096, 4096, 1)
    self.relu7 = nn.ReLU(inplace=True)
    self.drop7 = nn.Dropout2d()

    self.score_fr = nn.Conv2d(4096, num_classes, 1)
    self.score_pool3 = nn.Conv2d(256, num_classes, 1)
    self.score_pool4 = nn.Conv2d(512, num_classes, 1)

    self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
    self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
    self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)


  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.zero_()
        if m.bias is not None:
          m.bias.data.zero_()
      if isinstance(m, nn.ConvTranspose2d):
        assert m.kernel_size[0] == m.kernel_size[1]
        initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)

  def forward(self, x):
    x = self.relu1_1(self.conv1_1(x))
    x = self.relu1_2(self.conv1_2(x))
    x = self.pool1(x)

    x = self.relu2_1(self.conv2_1(x))
    x = self.relu2_2(self.conv2_2(x))
    x = self.pool2(x)

    x = self.relu3_1(self.conv3_1(x))
    x = self.relu3_2(self.conv3_2(x))
    x = self.relu3_3(self.conv3_3(x))
    x = self.pool3(x)
    pool3 = x # 1/8

    x = self.relu4_1(self.conv4_1(x))
    x = self.relu4_2(self.conv4_2(x))
    x = self.relu4_3(self.conv4_3(x))
    x = self.pool4(x)
    pool4 = x # 1/16

    x = self.relu5_1(self.conv5_1(x))
    x = self.relu5_2(self.conv5_2(x))
    x = self.relu5_3(self.conv5_3(x))
    x = self.pool5(x)

    x = self.relu6(self.fc6(x))
    x = self.drop6(x)

    x = self.relu7(self.fc7(x))
    x = self.drop7(x)

    x = self.score_fr(x)
    x = self.upscore2(x)
    upscore2 = x # 1/16

    x = self.score_pool4(pool4)
    x = x[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
    score_pool4c = x # 1/16

    x = score_pool4c + upscore2
    x = self.upscore_pool4(x)
    upscore_pool4 = x # 1/8

    x = self.score_pool3(pool3)
    x = x[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
    score_pool3c = x # 1/8

    x = upscore_pool4 + score_pool3c

    x = self.upscore8(x)

    x = x[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

    return x