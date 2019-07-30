import os
import numpy as np
import torch
import torch.nn as nn
from model import Net
from parklot_dataset import ParklotDataset, ToTensor, Rescale, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# hype params
LR = 0.0001
EPOCH = 20
LOSS_FUNC = nn.CrossEntropyLoss().cuda()


def Train_Val(net, train_loader, val_loader, loss_func=LOSS_FUNC, EPOCH=EPOCH):
    for epoch in range(EPOCH):
        Accuracy_train, Loss_train = Train(net, train_loader, loss_func=LOSS_FUNC)
        Accuracy_val, Loss_val = Val(net, val_loader, loss_func=LOSS_FUNC)
        with open('epoch_loss.txt', 'a') as file:
            print('EPOCH: ', epoch, '| train loss: %.4f' % Loss_train,
                  '| train accuracy: %.4f' % Accuracy_train)
            file.write('EPOCH: ' + str(epoch) + '| train loss: ' + str(Loss_train) +
                       '| train accuracy: ' + str(Accuracy_train) + '\n')
            print('EPOCH: ', epoch, '| val loss:   %.4f' % Loss_val,
                  '| val accuracy:   %.4f' % Accuracy_val)
            file.write('EPOCH: ' + str(epoch) + '| val loss:   ' + str(Loss_val) +
                       '| val accuracy:   ' + str(Accuracy_val) + '\n')


def Train(net, train_loader, loss_func=LOSS_FUNC):
    Loss = 0.0
    Accuracy = 0.0
    Size = 0
    for i, sample_batched in enumerate(train_loader):
        img = sample_batched['img']
        label = sample_batched['label']
        # Size =>count picture numbers
        size = label.size(0)
        Size = Size + size
        # cuda
        img = img.cuda()
        label = torch.squeeze(label, 1)  # (batch_size, 1) => (batch_size)
        label = label.cuda()
        # define optimizer with params of net
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        # forward & back propagation
        output = net(img)  # forward
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate Accuracy
        label_np = label.data.cpu().numpy()
        output_np = output.data.cpu().numpy()[:, 1]
        accuracy = np.logical_xor((output_np > 0.5), (label_np > 0.5))
        accuracy = np.logical_not(accuracy)
        accuracy = np.sum(accuracy, axis=0)
        accuracy = accuracy.item()
        Accuracy = Accuracy + accuracy
        # Loss: sum of loss
        # loss = torch.sum(loss) # loss is sum of a batch
        loss = loss.data.cpu().numpy()
        loss = loss.item()
        Loss = Loss + loss
        # print(accuracy)
        # print(loss)
    Accuracy = Accuracy / Size
    Loss = Loss / (i + 1)
    return Accuracy, Loss


def Val(net, val_loader, loss_func=LOSS_FUNC):
    Loss = 0.0
    Accuracy = 0.0
    Size = 0
    for i, sample_batched in enumerate(val_loader):
        img = sample_batched['img']
        label = sample_batched['label']
        # Size =>count picture numbers
        size = label.size(0)
        Size = Size + size
        # cuda
        img = img.cuda()
        label = torch.squeeze(label, 1)  # (batch_size, 1) => (batch_size)
        label = label.cuda()
        # define optimizer with params of net
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        # forward & back propagation
        output = net(img)  # forward
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate Accuracy
        label_np = label.data.cpu().numpy()
        output_np = output.data.cpu().numpy()[:, 1]
        accuracy = np.logical_xor((output_np > 0.5), (label_np > 0.5))
        accuracy = np.logical_not(accuracy)
        accuracy = np.sum(accuracy, axis=0)
        accuracy = accuracy.item()
        Accuracy = Accuracy + accuracy
        # Loss: sum of loss
        # loss = torch.sum(loss) # loss is sum of a batch
        loss = loss.data.cpu().numpy()
        loss = loss.item()
        Loss = Loss + loss
        # print(accuracy)
        # print(loss)
    Accuracy = Accuracy / Size
    Loss = Loss / (i + 1)
    return Accuracy, Loss


def Save_Param(net, checkpoint_dir='./checkpoint'):
    param_dir = os.path.join(checkpoint_dir, 'Param_Parklot.pth')
    torch.save(net.state_dict(), param_dir)


if __name__ == '__main__':
    net = Net(3).float()
    net.cuda()
    net.load_state_dict(torch.load('./checkpoint/Param_Parklot.pth'))
    net.eval()
    # train
    train_dataset = ParklotDataset(root_dir='./train',
                                   train=True,
                                   transform=transforms.Compose([Rescale((64, 64)), ToTensor()]))
    train_dataset.train = True
    train_dataset.val = False
    train_dataset.root_dir = './train'
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # validate
    val_dataset = ParklotDataset(root_dir='./train',
                                 train=True,
                                 transform=transforms.Compose([Rescale((64, 64)), ToTensor()]))
    val_dataset.train = True
    val_dataset.val = True
    val_dataset.val_dir = './val'
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)
    # Train & Validate and Save Param
    print(len(train_dataset))
    Train_Val(net, train_loader, val_loader)
    Save_Param(net)
