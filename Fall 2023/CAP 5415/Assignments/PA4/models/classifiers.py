import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClassifierExtended(Classifier):
    def __init__(self, ):
        super(ClassifierExtended, self).__init__()
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)  # new conv layer is introduced
        # fc1 is redefined to accomodate the new conv layer
        self.fc1 = nn.Linear(64 * 2 * 2, 120)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))  # new conv layer
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = ClassifierExtended()
    # net = Classifier()
    x = torch.rand(12, 3, 32, 32)
    out = net(x)
    print(out.shape)
