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

class VAE(nn.Module):()
    def self __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=2)
        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.encoder  = self._encoder()
        self.decoder = self._decoder()
    def self._encoder(self):







class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # define model
        # 3 fully connected layers

        self.fc1 = nn.Linear(in_features = 28*28, out_features = 512, bias = True)
        self.fc2 = nn.Linear(in_features = 512, out_features = 512, bias = True)
        self.fc3 = nn.Linear(in_features = 512, out_features = 10, bias = True)

    def num_features(self, x):
        size  = x.size()
        n = 1
        for s in size[1:]:
         n = n*s
        return n
    def forward(self, x):
        x = x.view(-1, self.num_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = F.softmax(x)
        return x


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # define model
        # This one will have 2 conv layers each followed by a relu activation, followed by 3 fully connected layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4,kernel_size = 3, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8,kernel_size = 3, padding = 2)
        self.fc1 = nn.Linear(in_features = 512, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 64)
        self.fc3 = nn.Linear(in_features = 64, out_features = 10)

    def num_features(self, x):
        size  = x.size()
        n = 1
        for s in size[1:]:
         n = n*s
        return n

    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        #flatten x
        x = x.view(-1, self.num_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

def acc(pred,target):

    if len(pred) != len(target):
        print('prediction and target must be same length')
        return 0
    else:
        is_true = [pred[i] == target[i] for i in range(len(pred))]
        accuracy  = sum(is_true)/len(pred)
        return accuracy


def test(model, data_loader):
    # test model
    test_pred = []
    test_target = []
    for data, target in data_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        test_pred += (predicted.tolist())
        test_target += (target.tolist())

    return acc(test_pred, test_target)

def main():

    max_epochs = 15

    batch_size = 20

    model_list = ['MLP', 'CNN']
    #model = MLP()
    #model = MyNet()
    MLP_dict = {'name':'MLP', 'model':MLP() }
    CNN_dict = {'name':'CNN', 'model':MyNet() }
    model_list = [MLP_dict, CNN_dict]
    for i in model_list:
        print(i['model'])
    print('done')
    #load data (images are 28 * 28 pixels)
    # convert data to torch.FloatTensor
    transform = torchvision.transforms.ToTensor()
    MNIST_train = torchvision.datasets.MNIST(root='data', download=True, train = True, transform = transform)
    MNIST_test = torchvision.datasets.MNIST(root='data', download=True, train = False, transform = transform)


    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=batch_size, shuffle=True, num_workers=4)

    #train model
    for m in model_list:
        print(m['name'])
        model = m['model']
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_criterion = nn.CrossEntropyLoss()
        model.train()
        #
        train_accuracy_list = []
        test_accuracy_list = []
        loss_list = []

        for epoch in range(max_epochs):
        #train model
            train_loss = 0.0

            for data, target in train_loader:
                # clear gradients
                optimizer.zero_grad()
                #make prediction
                output = model(data)
                # calculate backwards gradient
                loss = loss_criterion(output, target)
                loss.backward()
                # optimize parameters
                optimizer.step()
                train_loss += loss.data.tolist()
            print(epoch, train_loss)
            train_accuracy = test(model, train_loader)
            test_accuracy = test(model, test_loader)
            print('train accuracy is ', train_accuracy)
            print('test accuracy is ', test_accuracy)
            loss_list += [loss.tolist()]
            test_accuracy_list.append(test_accuracy)
            train_accuracy_list.append(train_accuracy)
        x=[i for i in range(max_epochs)]
        #plt.plot(x, loss_list, label = 'Loss')
        s = '-'
        plt.plot(x,  train_accuracy_list, label=s.join((m['name'], 'train accuracy')))
        plt.plot(x,  test_accuracy_list, label=s.join((m['name'], 'test accuracy')))
    plt.legend()
    plt.show()
    plt.savefig('accurancy plot.png')


if __name__ == '__main__':
    main()

