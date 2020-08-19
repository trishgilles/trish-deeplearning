import pylint, yaml, argparse
import torch
import datetime, time
import matplotlib.pyplot as plt
import numpy as np
import sys, warnings, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets


class MyNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define model
        # This one will have 2 conv layers each followed by a relu activation, followed by 3 fully connected layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4,kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8,kernel_size = 3)
        self.fc1 = nn.Linear(in_features = 64, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 64)
        self.fc3 = nn.Linear(in_features = 64, out_features = 64)

    def num_features(self, x):
        size  = x.size()
        n = 1
        for s in size[1:]:
         n = n*s
        return n
    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(2)
        #flatten x
        x = x.view(-1, )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x




def main():

    #load data (images are 28 * 28 pixels)
    MNIST_train = torchvision.datasets.MNIST(root='data', download=True, train = True)
    MNIST_test = torchvision.datasets.MNIST(root='data', download=True, train = False)
    # #TODO : do I need this?
    #
    # train_data, train_labels = MNIST.train_data, MNIST.train_labels
    # test_data, test_labels = MNIST.test_data, MNIST.test_labels


    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=2, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=2, shuffle=True, num_workers=4)




    #train model
    max_epochs = 50
    model = MyNet()
    optim = torch.optim()




if __name__ == '__main__':
    main()

