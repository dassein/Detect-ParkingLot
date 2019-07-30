import cv2
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from model import Net
from parklot_label import ParklotLabelDataset, RankImg, CropParklot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from test_label import TestLabel

N = 50
def MaskImg_cv2(img, output, bbox_parking_spot):
    img = img.astype(np.float32)
    H, W = img.shape[0], img.shape[1]
    img_mask_np = np.zeros((H, W, 3), np.float32)
    for j in range(len(output)):
        bbox_used = bbox_parking_spot[j, :]  # ( W, H, det W, det H)
        if output[j] == 1:
            img_mask_np[  # red, numpy: (r, g, b), opencv: (b g r)
            bbox_used[1] - 1:bbox_used[1] + bbox_used[3],
            bbox_used[0] - 1:bbox_used[0] + bbox_used[2], 0] = 255.0
        else:
            img_mask_np[  # green
            bbox_used[1] - 1:bbox_used[1] + bbox_used[3],
            bbox_used[0] - 1:bbox_used[0] + bbox_used[2], 1] = 255.0
    r, g, b = cv2.split(img_mask_np)
    img_mask = cv2.merge([b, g, r])
    img_masked = cv2.addWeighted(img, 0.8, img_mask, 0.2, 0)
    num_busy = sum(output.cpu().numpy())
    num_free = len(output) - num_busy
    text_busy = "Occupied:" + str(num_busy)
    text_free = "Empty:" + str(num_free)
    cv2.putText(img_masked, text_busy, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(img_masked, text_free, (100, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), lineType=cv2.LINE_AA)
    return img_masked


if __name__ == "__main__":
    net = Net(3).float()
    net.cuda()
    bbox_parking_spot = scio.loadmat('./bbox_parking_spot/Perspective_PL_Night_45.mat')
    bbox_parking_spot = bbox_parking_spot.get('object')
    video_in = './YouTube/PL_Night_45_1.mp4'
    video_out = './Perspective_PL_Night_45.avi'
    video_reader = cv2.VideoCapture(video_in)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'),
                                   50.0,
                                   (2500, 1600))
    # https://stackoverflow.com/questions/38397964/cant-write-and-save-a-video-file-using-opencv-and-python
    # Python: cv2.VideoWriter.write(image) -> None
    # Parameters:
    # filename – Name of the output video file.
    # fourcc – 4-character code of codec used to compress the frames. For example, CV_FOURCC('P','I','M','1') is a MPEG-1 codec, CV_FOURCC('M','J','P','G') is a motion-jpeg codec etc. List of codes can be obtained at Video Codecs by FOURCC page.
    # fps – Framerate of the created video stream.
    # frameSize – Size of the video frames.
    # isColor – If it is not zero, the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only).
    for i in range(nb_frames):
        ret, img = video_reader.read()
        pts1 = np.float32([[235, 313], [940, 305], [12, 493], [1157, 479]])
        pts2 = np.float32([[500, 300], [2000, 300], [500, 1200], [2000, 1200]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_dst = cv2.warpPerspective(img, M, (2500, 1600))
        b, g, r = cv2.split(img_dst)  # get b,g,r
        img_rgb = cv2.merge([r, g, b])  # switch it to rgb
        img_np = np.asarray(img_rgb)
        img_crop_list, img_crop_list_tensor = CropParklot(bbox_parking_spot, img_np, (64, 64))
        ParklotLabel = ParklotLabelDataset(img_crop_list_tensor)
        test_label_loader = DataLoader(ParklotLabel,
                                       batch_size=len(img_crop_list_tensor),
                                       shuffle=False,
                                       num_workers=2)  # order not changed => shuffle=False
        iteration = TestLabel(net, test_label_loader)
        if i >= (2 * N - 1):
            output_memory = output_memory[0:-1, ...]
        if i == 0:
            output_memory = torch.unsqueeze(iteration, 0)
            output = iteration
        else:
            output_memory = torch.cat((torch.unsqueeze(iteration, 0), output_memory), 0)
            output = torch.sum(output_memory, 0, dtype=torch.uint8)
            output = output // (int(output_memory.shape[0] // 2 + 1))
        img_masked = MaskImg_cv2(img_dst, output, bbox_parking_spot)
        video_writer.write(np.uint8(img_masked))
    video_reader.release()
    video_writer.release()