import os
import torch
import torch.nn as nn
import numpy as np
from model import Net
from parklot_label import ParklotLabelDataset, RankImg, CropParklot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as scio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def TestLabel(net, test_label_loader,
              checkpoint_dir='./checkpoint'):
    param_dir = os.path.join(checkpoint_dir, 'Param_Parklot.pth')
    net.load_state_dict(torch.load(param_dir))
    Test_Label_Loader = test_label_loader
    Test_Label_Loader.shuffle = False  # make sure crop order not changed
    for i, sample_batched in enumerate(Test_Label_Loader):
        img = sample_batched
        img = img.cuda()
        # classify
        output = net(img)  # forward
        output = (output[:, 1] > output[:, 0])
        # print(output.cpu().data.numpy())
    return output


def ImgList(ImgFolder_dir='./warped'):
    img_list = RankImg(ImgFolder_dir)
    return img_list


def Bbox(bbox_path='bbox_parklot.mat'):
    bbox_parklot = scio.loadmat(bbox_path)
    bbox_parklot = bbox_parklot.get('bbox_parklot')
    return bbox_parklot

def ImgCropList(img_list,
                bbox_parklot,
                item,     # order of whole img in img_list
                ImgFolder_dir='./warped'):
    img_dir = img_list[item]
    img_dir = os.path.join(ImgFolder_dir, img_dir)
    img = Image.open(img_dir).convert("RGB")  # Image.open() => PIL object (R, G, B) 0~255
    img_np = np.asarray(img).astype(np.float32)  # H, W, C for C: (R, G, B)  0~255
    img_crop_list, img_crop_list_tensor = CropParklot(bbox_parklot, img_np, (64, 64))
    return img_crop_list, img_crop_list_tensor, img


def MaskImg(img, output, bbox_parklot, img_name, root_dir='.'):
    (W, H) = img.size  # (W, H) = (1920, 1080)
    print(len(output))
    img_mask = np.zeros((H, W, 3))
    for j in range(len(output)):
        bbox_used = bbox_parklot[j, :]  # ( W, H, det W, det H)
        if output[j] == 1:
            img_mask[  # red
            bbox_used[1] - 1:bbox_used[1] + bbox_used[3],
            bbox_used[0] - 1:bbox_used[0] + bbox_used[2], 0] = 255.0
        else:
            img_mask[  # green
            bbox_used[1] - 1:bbox_used[1] + bbox_used[3],
            bbox_used[0] - 1:bbox_used[0] + bbox_used[2], 1] = 255.0
    Img_Mask = Image.fromarray(img_mask.astype('uint8'))
    Img_Masked = Image.blend(img, Img_Mask, 0.2)
    num_busy = sum(output.cpu().numpy())
    num_free = len(output) - num_busy
    d = ImageDraw.Draw(Img_Masked)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMonoBold.ttf', 120)
    d.text((10, 10), "Occupied:" + str(num_busy), font=fnt, fill=(255, 255, 0))
    d.text((10, 210), "Empty:" + str(num_free), font=fnt, fill=(255, 255, 0))
    # save mask & masked
    mask_dir = 'mask_' + img_name
    mask_dir = os.path.join(root_dir, 'warped_mask', mask_dir)
    masked_dir = 'masked_' + img_name
    masked_dir = os.path.join(root_dir, 'warped_masked', masked_dir)
    Img_Mask.save(mask_dir)
    Img_Masked.save(masked_dir)


if __name__ == '__main__':
    net = Net(3).float()
    net.cuda()
    ImgFolder_dir = './warped'
    img_list = ImgList(ImgFolder_dir)
    bbox_parklot = Bbox()
    # print(bbox_parklot)
    for item in range(len(img_list)):
        img_crop_list, img_crop_list_tensor, img = ImgCropList(img_list, bbox_parklot, item, ImgFolder_dir)
        ParklotLabel = ParklotLabelDataset(img_crop_list_tensor)
        test_label_loader = DataLoader(ParklotLabel,
                                       batch_size=len(img_crop_list_tensor),
                                       shuffle=False,
                                       num_workers=2)  # order not changed => shuffle=False
        print(img_list[item])
        output = TestLabel(net, test_label_loader)
        # print(output)
        MaskImg(img, output, bbox_parklot, img_list[item])