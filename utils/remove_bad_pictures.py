import os
from glob import glob
import cv2
import numpy as np
from multiprocessing import Pool


def is_grayscale(path):
    img = cv2.imread(path)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.logical_and((r==g).all(), (g==b).all())


def delete_if_grayscale(path):
    if is_grayscale(path):
        print(path)
        os.remove(path)


def delete_if_low_saturation(path):
    if is_low_saturation(path):
        print(path)
        os.remove(path)


def is_low_saturation(path):
    """
    Checks if an image is almost grayscale 
    """
    img = cv2.imread(path)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    overall_saturation = np.mean(hls[..., 2])
    return overall_saturation < 35




if __name__ == "__main__":
    filepaths_set_all = sorted(
                [y for x in os.walk("./images") 
                for y in (glob(os.path.join(x[0], '*.jpg')) + 
                            glob(os.path.join(x[0], '*.png')))])
    
    # with Pool() as p:
    #     p.map(delete_if_grayscale, filepaths_set_all)

    with Pool() as p:
        p.map(delete_if_low_saturation, filepaths_set_all)