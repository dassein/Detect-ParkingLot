# -*- coding: utf-8 -*-
'''
透视变换
对于透视变换 ，我们需要一个 3x3 变换矩 。
在变换前后直线 是直线。
构建 个变换矩  你需要在输入图像上找 4 个点， 以及他们在输出图 像上对应的位置。
四个点中的任意三个都不能共线。这个变换矩阵可以用函数 cv2.getPerspectiveTransform() 构建。
然后把这个矩阵传给函数 cv2.warpPerspective。
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

time1 = time.time()
img = cv2.imread('./perspective_label/PL_Obstructed_45.jpg')
# rows, cols, ch = img.shape

## For  PL_Night_45_1.mp4
# pts1 = np.float32([[235, 313], [940, 305], [12, 493], [1157, 479]])
# pts2 = np.float32([[500, 300], [2000, 300], [500, 1200], [2000, 1200]])
# dst = cv2.warpPerspective(img, M, (2500, 1600))

## For  NW Staff Parking 2.mp4
#pts1 = np.float32([[62, 355], [1116, 101], [812, 1007], [1560, 170]])
#pts2 = np.float32([[100, 400], [2100, 70], [100, 1200], [2000, 1250]])
#dst = cv2.warpPerspective(img, M, (2500, 1370))

## For PL_Obstructed_45
pts1 = np.float32([[339, 366], [902, 365], [112, 521], [1108, 553]])
pts2 = np.float32([[500, 300], [2000, 300], [500, 1200], [2000, 1200]])

#x, y
#[339, 366] 	[902, 365]
#[112, 521] 	[1108, 553]


M = cv2.getPerspectiveTransform(pts1, pts2)


dst = cv2.warpPerspective(img, M, (2500, 1600))
cv2.imwrite('./perspective_label/Perspective_PL_Obstructed_45.jpg', dst)
time2 = time.time()
print(time2-time1)
plt.figure(figsize=(8, 7), dpi=98)
p1 = plt.subplot(211)
plt.imshow(img)
# plt.set_title('Input')

p2 = plt.subplot(212)
plt.imshow(dst)
# plt.set_title('Output')

plt.show()

