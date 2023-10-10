import torch
import torch.nn as nn
import torch.nn.functional as F


conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1)
conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1)

max_pool1 = nn.MaxPool2d(kernel_size=2)
max_pool2 = nn.MaxPool2d(kernel_size=2)


def process(x):
    x = conv1(x)
    x = max_pool1(x)
    x = F.sigmoid(x)

    x = conv2(x)
    x = max_pool2(x)
    x = F.sigmoid(x)

    return x


if __name__ == "__main__":
    bs = 10
    x = torch.rand(bs, 1, 28, 28)
    out = process(x)
    print(out.shape)
