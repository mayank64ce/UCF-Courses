import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing

        self.num_classes = 10

        hidden_layers1 = 100
        hidden_layers2 = 1000
        prob = 0.5

        # Flatten layers
        self.flatten = nn.Flatten()

        # activations
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Linear Layers
        self.fc1 = nn.Linear(28*28, hidden_layers1)  # for model 1
        self.fc2 = nn.Linear(40*4*4, hidden_layers1)
        self.fc3 = nn.Linear(hidden_layers1, hidden_layers1)  # for model 4

        self.fc4 = nn.Linear(40*4*4, hidden_layers2)
        self.fc5 = nn.Linear(hidden_layers2, hidden_layers2)

        self.fc_out1 = nn.Linear(hidden_layers1, self.num_classes)
        self.fc_out2 = nn.Linear(
            hidden_layers2, self.num_classes)  # for model 5

        # Convolution Layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=40, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(
            in_channels=40, out_channels=40, kernel_size=5, stride=1)

        # Maxpool Layers
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        # dropout
        self.dropout = nn.Dropout(p=prob)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    # Baseline model. step 1

    def model_1(self, x):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc_out1(x)

        return x

    # Use two convolutional layers.
    def model_2(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.

        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.sigmoid(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.sigmoid(x)

        x = self.flatten(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc_out1(x)

        return x

    # Replace sigmoid with ReLU.
    def model_3(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc_out1(x)

        return x

    # Add one extra fully connected layer.
    def model_4(self, x):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc_out1(x)

        return x

    # Use Dropout now.
    def model_5(self, x):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_out2(x)

        return x
