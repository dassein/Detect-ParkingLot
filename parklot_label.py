import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as scio
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def partition(arr, low, high):
    i = (low - 1)  # index of smaller element
    pivot = arr[high]  # pivot
    for j in range(low, high):
        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:
            # increment index of smaller element
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# The main function that implements QuickSort
# arr[] --> Array to be sorted,
# low  --> Starting index,
# high  --> Ending index

# Function to do Quick sort
def quickSort(arr, low, high):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


def RankImg(ImgFolder_dir):
    walk = os.walk(ImgFolder_dir)
    img_list = next(walk)[2]
    quickSort(img_list, 0, len(img_list) - 1)
    return img_list


def ImgRescale(img, output_size):
    h, w = img.shape[:2]  # here img: numpy array (H, W, C) => shape[:2] = (H, W)
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(img, (new_h, new_w), mode='constant')
    # h and w are swapped for landmarks because for images,
    # x and y axes are axis 1 and 0 respectively
    return img


def CropParklot(bbox, img_np, output_size):  # img_dir: full path of 1 img
    img_crop_list = []
    img_crop_list_tensor = []
    for i in range(bbox.shape[0]):
        bbox_used = bbox[i, :]
        img_crop = img_np[bbox_used[1] - 1:bbox_used[1] + bbox_used[3],
                   bbox_used[0] - 1:bbox_used[0] + bbox_used[2], :]
        # plt.imshow(img_crop / 255)
        # plt.show()
        img_crop_tensor = ImgRescale(img_crop, output_size)
        # plt.imshow(img_crop_tensor / 255)
        # plt.show()
        img_crop_tensor = img_crop_tensor.transpose((2, 0, 1))  # H,W,C => C,H,W
        img_crop_tensor = torch.FloatTensor(img_crop_tensor)
        img_crop_list.append(img_crop)
        img_crop_list_tensor.append(img_crop_tensor)
    return img_crop_list, img_crop_list_tensor


class ParklotLabelDataset(Dataset):
    def __init__(self, img_crop_list_tensor):
        self.img_crop_list_tensor = img_crop_list_tensor

    def __len__(self):
        return len(self.img_crop_list_tensor)

    def __getitem__(self, item):
        return self.img_crop_list_tensor[item]


if __name__ == '__main__':
    ImgFolder_dir = './parklot (12-2-2018 7-53-35 AM)'
    img_list = RankImg(ImgFolder_dir)
    for i in range(len(img_list)):
        print(img_list[i])
    bbox_path = 'bbox_parklot.mat'
    bbox_parklot = scio.loadmat(bbox_path)
    bbox_parklot = bbox_parklot.get('bbox_parklot')
    print(bbox_parklot)
    img_dir = img_list[0]
    img_dir = os.path.join(ImgFolder_dir, img_dir)
    img = Image.open(img_dir).convert("RGB")  # Image.open() => PIL object (R, G, B) 0~255
    img_np = np.asarray(img).astype(np.float32)  # H, W, C for C: (R, G, B)  0~255
    print(img_np.shape)
    img_crop_list, img_crop_list_tensor = CropParklot(bbox_parklot, img_np, (32, 32))
    plt.imshow(img_crop_list[1] / 255)
    plt.show()
    # show crop tensor (3, 32, 32), channel = R, G, B
    img_crop_tensor_show = img_crop_list_tensor[1].numpy()
    img_crop_tensor_show = img_crop_tensor_show.transpose((1, 2, 0))  # CHW => HWC
    plt.imshow(img_crop_tensor_show / 255)
    plt.show()
    # test ParklotLabelDataset & DataLoader
    ParklotLabel = ParklotLabelDataset(img_crop_list_tensor)
    print(ParklotLabel[0].shape)
    dataloader = DataLoader(ParklotLabel, batch_size=50,
                            shuffle=False, num_workers=2)  # order not changed => shuffle=False
    for i, sample_batched in enumerate(dataloader):
        print(i, sample_batched.size())
