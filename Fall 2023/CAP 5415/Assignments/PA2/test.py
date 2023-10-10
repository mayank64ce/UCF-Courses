from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet
import argparse
import numpy as np

from ConvNet import ConvNet

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data/', train=False, download=True,
                                  transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=10,
                             shuffle=True, num_workers=4)

    model = ConvNet(mode=5)

    for data, target in test_loader:
        # print(data.shape, target)

        out = model(data)

        print(out.shape)

        # pred = out.argmax(dim=1, keepdim=True).squeeze()
