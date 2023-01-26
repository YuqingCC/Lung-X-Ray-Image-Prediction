import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

'''
This file create the architecture for ResNet-34 network
The ResNet model consist of two class: Residual Block which used for layer skipping, and ResNet class to construct the whole network.
Functions:
# build the Res neural network
def model_ResNN(n_class, layer_list): n_class--number of class for the input data
                                  layer_list--list of conv layer for skipping

# training function
def training(model, data_loader, learning_rate, epoch, pre_train=False)

# testing function
def testing(model, data_loader)
'''


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Create the architecture for RNN34 network
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def model_ResNN(n_class, layer_list):
    model = ResNet(block=ResidualBlock, layers=layer_list, num_classes=n_class)
    print("===========Model Structure==============")
    print(model)
    return model


'''
Fit function for Res34 Neural Network
input: ResNet model, training data, learning rate, max_epoch
By default, the model is trained from scratch, but can also use pre-trained weights from ImageNet
'''
def training(model, data_loader, learning_rate, epoch, pre_train=False):
    epoch_saving = ['./output/model/model_res1.pth', './output/model/model_res2.pth', './output/model/model_res3.pth',
                    './output/model/model_res4.pth', './output/model/model_res5.pth', './output/model/model_res6.pth',
                    './output/model/model_res7.pth', './output/model/model_res8.pth', './output/model/model_res9.pth',
                    './output/model/model_res10.pth']
    if pre_train:
        epoch_saving = ['./output/model/model_pre_res1.pth', './output/model/model_pre_res2.pth',
                        './output/model/model_pre_res3.pth',
                        './output/model/model_pre_res4.pth', './output/model/model_pre_res5.pth',
                        './output/model/model_pre_res6.pth',
                        './output/model/model_pre_res7.pth', './output/model/model_pre_res8.pth',
                        './output/model/model_pre_res9.pth',
                        './output/model/model_pre_res10.pth']
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5)
    # acc / loss / lr saver
    acc_train = []
    loss_train = []
    lr_train = []
    epo_train = []
    # dictionary to store the history data
    result = dict.fromkeys(["Epochs", "Accuracy", "Loss", "Learning_rate"])

    for epo in range(epoch):
        model.train()
        print(f'training epoch: {epo}')
        epochs_loss = 0.0
        epochs_acc = 0.0
        step = 0

        for x, y in tqdm(data_loader):
            step += 1
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            _, prediction = torch.max(outputs, 1)

            epochs_acc += (prediction == y.data).sum().item()
            epochs_loss += loss.item() * len(outputs)

            loss.backward()
            optimizer.step()
            scheduler.step()

        data_size = len(data_loader.dataset)
        epochs_loss = epochs_loss / data_size
        epochs_acc = 100. * (epochs_acc / data_size)

        acc_train.append(epochs_acc)
        loss_train.append(epochs_loss)
        lr_train.append(scheduler.get_last_lr()[0])
        epo_train.append(epo)
        torch.save({
            'epoch': epo,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, epoch_saving[epo])
        print(
            f'Epoch {epo + 1}/{epoch} | Loss: {epochs_loss:.4f} | Acc: {epochs_acc:.4f} | learning rate: {scheduler.get_last_lr()[0]}')

    result["Epochs"] = epo_train
    result["Accuracy"] = acc_train
    result["Loss"] = loss_train
    result["Learning_rate"] = lr_train
    return result


def testing(model, data_loader):
    model.eval()
    num_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            image, label = data
            outputs = model(image)
            # calculate the loss
            _, predictions = torch.max(outputs.data, 1)
            num_correct += (predictions == label).sum().item()
    test_acc = 100. * (num_correct / len(data_loader.dataset))
    return test_acc


'''ResNet with pre-trained feature weight'''
def resNet_pretrain():
    model_pre_train = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    print("===========Model Structure==============")
    print(model_pre_train)
    return model_pre_train


if __name__ == "__main__":
    num_classes = 10
    Res_34 = [3, 4, 6, 3]
    print(model_ResNN(num_classes, Res_34))
    # resNet_pretrain()
