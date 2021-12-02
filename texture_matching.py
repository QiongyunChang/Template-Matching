import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# Convolution
def image_padding(image, KernelSize):
    p = int(np.floor(KernelSize / 2))
    # print(p)
    (w, h) = np.shape(image)
    # print("====",w,h)
    pad_img = np.zeros((w + 2 * p, h + 2 * p))
    pad_img[int(p):int(-1 * p), int(p):int(-1 * p)] = image
    cv2.imshow('pad_img', pad_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pad_img


def convolution(image, Kernel, stride=1, padding=True):
    # print(Kernel)
    # upside down the kernel to meet the requirement of convolution
    # note that cross cross-correlation does not need rotate the kernal
    Kernel = np.flip(Kernel.T, axis=0)
    KernelSize = np.shape(Kernel)[0]
    KernelSizeC = np.shape(Kernel)[1]
    row, col = image.shape
    # print(KernelSize)
    # print('image',np.shape(image))
    if stride is None:
        stride = KernelSize
    if padding:
        pad_img = image_padding(image, KernelSize)
        resx = np.zeros((row, col))
        print(row,col,"1roooow")
    else:
        pad_img = image
        resx = np.zeros((int((row - KernelSize) / stride) + 1, int((col - KernelSize) / stride) + 1))
    nrow, ncol = pad_img.shape

    xpatch = np.arange(0, nrow - KernelSize + 1, stride)
    ypatch = np.arange(0, ncol - KernelSizeC + 1, stride)
    print(ypatch.shape)
    print(xpatch.shape)

    # print("=", matrix.shape)
    for x_id, x in enumerate(xpatch):
        for y_id, y in enumerate(ypatch):
            matrix = pad_img[x:x + KernelSize, y:y + KernelSizeC]

            # multi_matrix = np.multiply(matrix, Kernel)
            # Normalize sum of squared difference
            s = np.sum((matrix[:, :] - Kernel[:, :]) ** 2)
            # print(s)
            deno = np.square(np.multiply( (np.sum((matrix[:, :])** 2)), (np.sum(Kernel[:, :])  )** 2))
            # print("-",deno)
            # print("s ", s)
            out = s / deno
            # print(out)
    print(matrix.shape)
    print("llll",resx.shape)
            # resx[x_id, y_id] = out


    return resx


def downsample(image):
    img_half = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    return img_half

def upsample(image):
    img_half = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    img_half = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

def Gaussian_Filter(image):
    # create gaussian kernel
    kernel = cv2.imread('./100/100-Template.jpg', 0)
    kernel = downsample(kernel)
    resx = convolution(image, kernel)
    resx = resx.astype(np.uint8)
    return resx


img = cv2.imread('./100/100-1.jpg', 0)
# Downsampling
img = downsample(img)
# img_half = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
cv2.imshow('Half Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(img_org.shape)
# KernelSize = 3
height, width = img.shape

# noise
img = Gaussian_Filter(img)
# print(img)
# img = Gaussian_Filter(KernelSize,img)
plt.imshow(img, cmap='gray')
plt.show()
