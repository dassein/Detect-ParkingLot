import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# out = floor((in + 2 * p - k) / s) + 1
# shrink block, out = in / 2
def block_shrink(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(),)


class Net(nn.Module):
    def __init__(self, in_channels, out=2):
        super().__init__()
        self.conv1 = block_shrink(in_channels, 16)
        self.conv2 = block_shrink(16, 32)
        self.conv3 = block_shrink(32, 64)
        self.conv4 = block_shrink(64, 64)
        self.out = nn.Linear(64 * 4 * 4, out)
        # init weight
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x4.view(x4.size(0), -1)
        out = self.out(x4)
        # out = F.sigmoid(out)
        return out


if __name__ == '__main__':
    net = Net(3, 1).float()
    x = np.ones((4, 3, 64, 64))
    x = torch.Tensor(x)
    x1 = net(x)
    print(x1)

