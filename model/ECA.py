import torch
from torch import nn


class ECA(nn.Module):
    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2)
        y = y.unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

if __name__ == "__main__":
    x = torch.randn(16, 256 * 64, 7, 7)
    eca = ECA()
    y = eca(x)
    print(y.shape)
