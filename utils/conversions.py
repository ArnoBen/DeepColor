import cv2
import numpy as np


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2lab(img):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    result[:, :, 0] *= 100/255 # (0, 255) => (0, 100)
    result[:, :, 1:] -= 128  # (0, 255) => (-127, 127)
    return result.astype(np.int8)


def lab2rgb(img):
    img = img.astype(np.float32)
    img[:, :, 0] *= 255/100  # (0, 100) => (0, 255)
    img[:, :, 1:] += 128  # (-127, 127) => (0, 255)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2RGB)


def bgr2lab(img):
    return rgb2lab(bgr2rgb(img))


def denormalize_lab(img):
    img[:, :, 0] *= 100
    img[:, :, 1:] *= 127
    return img