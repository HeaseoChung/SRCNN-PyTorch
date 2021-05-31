import torch.nn as nn
from torch.cuda import amp


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5//2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
    
    @amp.autocast()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
