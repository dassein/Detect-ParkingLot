import os
# from tqdm import tqdm
import pandas as pd
from skimage import io, transform
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# root_dir = './train'


class ParklotDataset(Dataset):
    def __init__(self, root_dir, val_dir='./val',
                 train=None, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.val = False  # if set self.val = True, use val
        image_dir_list_val = []
        label_list_val = []
        # for val
        if self.train:
            self.val_dir = val_dir
            walk = os.walk(self.val_dir)
            for root, dirs, files in walk:
                root_split = root.split('/')
                # busy => label: 1, free => label: 0
                if root_split[-1] == 'Occupied':
                    label = 1
                    for file in files:
                        if '.jpg' or '.JPG' in file:
                            image_dir_val = os.path.join(root, file)
                            image_dir_list_val.append(image_dir_val)
                            label_list_val.append(label)
                elif root_split[-1] == 'Empty':
                    label = 0
                    for file in files:
                        if '.jpg' or '.JPG' in file:
                            image_dir_val = os.path.join(root, file)
                            image_dir_list_val.append(image_dir_val)
                            label_list_val.append(label)

            self.image_dir_list_val = image_dir_list_val
            self.label_list_val = label_list_val
        # for train or test (test have no label)
        walk = os.walk(self.root_dir)
        image_dir_list = []
        label_list = []
        for root, dirs, files in walk:
            root_split = root.split('/')
            # if train: busy => label: 1, free => label: 0
            if self.train:
                if root_split[-1] == 'Occupied':
                    label = 1
                    for file in files:
                        if '.jpg' or '.JPG' in file:
                            image_dir = os.path.join(root, file)
                            image_dir_list.append(image_dir)
                            label_list.append(label)
                elif root_split[-1] == 'Empty':
                    label = 0
                    for file in files:
                        if '.jpg' or '.JPG' in file:
                            image_dir = os.path.join(root, file)
                            image_dir_list.append(image_dir)
                            label_list.append(label)
        self.image_dir_list = image_dir_list
        if self.train:
            self.label_list = label_list

    def __len__(self):
        # A: 6171 images + 10 img(revise 0~9)
        # B: 6413 images + 10 img(revise 0~9)
        #  => A busy
        #  => A free
        #  => B busy
        #  => B free
        if self.val:
            length = len(self.image_dir_list_val)
        else:
            length = len(self.image_dir_list)
        return length

    def __getitem__(self, item):
        if self.train:
            if self.val:
                image_dir, label = self.image_dir_list_val[item], [self.label_list_val[item]]
                img = Image.open(image_dir).convert("RGB")
                img_np = np.asarray(img).astype(np.float32)
                label_np = np.asarray(label)
                img__label = {'img': img_np, 'label': label_np}
                if self.transform:  # if True, use ToTensor ( transform defined later)
                    img__label = self.transform(img__label)
                return img__label
            else:
                image_dir, label = self.image_dir_list[item], [self.label_list[item]]
                img = Image.open(image_dir).convert("RGB")
                img_np = np.asarray(img).astype(np.float32)
                label_np = np.asarray(label)
                img__label = {'img': img_np, 'label': label_np}
                if self.transform:  # if True, use ToTensor ( transform defined later)
                    img__label = self.transform(img__label)
                return img__label
        else:
            image_dir = self.image_dir_list[item]
            img = Image.open(image_dir).convert("RGB")
            img_np = np.asarray(img).astype(np.float32)
            img__label = {'img': img_np, 'label': np.asarray([-1])}  # if no label => set numpy[-1]
            if self.transform:  # if True, use ToTensor ( transform defined later)
                img__label = self.transform(img__label)
            return img__label



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, img__label):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_np, label_np = img__label['img'], img__label['label']
        img_np = img_np.transpose((2, 0, 1))
        return {'img': torch.FloatTensor(img_np),
                'label': torch.LongTensor(label_np)}


# input: img & label is numpy, Do Rescale + RandomCrop, then ToTensor: numpy => tensor
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h, w = img.shape[:2]  # here img: numpy array (H, W, C) => shape[:2] = (H, W)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w), mode='constant')
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'img': img, 'label': label}


# input: img & label is numpy, Do Rescale + RandomCrop, then ToTensor: numpy => tensor
class RandomCrop(object):
    """Crop randomly the image in a sample. 随机裁剪图像
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h, w = img.shape[:2]  # here img: numpy array (H, W, C) => shape[:2] = (H, W)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = img[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'label': label}


if __name__ == '__main__':
    transformed_dataset = ParklotDataset(root_dir='./train',
                                         train=True,
                                         transform=transforms.Compose([Rescale((150, 150)), ToTensor()]))
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['img'].size(), sample['label'].size())
#        if i == 3:
#            break
    # self.val = False, self.train = True
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=2)
    for i, sample_batched in enumerate(dataloader):
        print(i, sample_batched['img'].size(),
              sample_batched['label'].size())
    # self.val = True, self.train = True
    transformed_dataset.val = True  # self.val = True, self.train = True
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=2)
    for i, sample_batched in enumerate(dataloader):
        print(i, sample_batched['img'].size(),
              sample_batched['label'].size())

    # self.train = False, (any self.val)
    transformed_dataset.train = False
    transformed_dataset.root_dir = './test'
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=2)
    for i, sample_batched in enumerate(dataloader):
        print(i, sample_batched['img'].size(),
              sample_batched['label'].size())