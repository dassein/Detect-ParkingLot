import os
import torch
import torch.nn as nn
from model import Net
from parklot_dataset import ParklotDataset, ToTensor, Rescale, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def Test(net, test_loader,
         checkpoint_dir='./checkpoint'):
    # load trained params of net
    Test_Loader = test_loader
    param_dir = os.path.join(checkpoint_dir, 'Param_Parklot.pth')
    net.load_state_dict(torch.load(param_dir))
    Test_Loader.shuffle = False  # make sure img <=> img's dir
    # Size: record number of pictures
    Size = 0
    for i, sample_batched in enumerate(Test_Loader):
        # prepare img & img's dir
        img = sample_batched['img']
        img = img.cuda()
        size = img.size(0)
        img_dir = test_loader.dataset.image_dir_list[Size:Size + size]
        # classify
        output = net(img)  # forward
        output = (output[:, 1] > output[:, 0])
        # show classifier's result of img
        for j in range(size):
            print(img_dir[j], output[j].cpu().data.numpy())
        Size = Size + size


if __name__ == '__main__':
    net = Net(3).float()
    net.cuda()
    test_dataset = ParklotDataset(root_dir='./test',
                                  train=False,
                                  transform=transforms.Compose([Rescale((32, 32)), ToTensor()]))
    test_dataset.train = False
    test_dataset.val = False
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    Test(net, test_loader)

