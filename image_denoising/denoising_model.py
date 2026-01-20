__all__ =['ConvDenoiser']

import torch.nn as nn
import torch.nn.functional as F

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(2,2)
        # 定义批归一化层
        self.t_conv1 = nn.ConvTranspose2d(in_channels=8, out_channels=8 ,kernel_size=3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_out = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        # 前向传播过程
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))






        x = self.conv3(x)
        return x
