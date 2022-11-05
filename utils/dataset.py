import os
from glob import glob
from utils.preprocessing import remove_bw
from utils.conversions import bgr2lab
from tqdm import tqdm
import cv2
import numpy as np


def make_dataset(train_size=40000, val_size=1000, test_size=1000):
    filepaths, set_name2size = get_image_paths()
    desired_sizes = (train_size, val_size, test_size)
    # Getting the min value between the user's desired size and the max set size for each set
    sets_sizes = [(min(size, desired_size)) for size, desired_size in zip(set_name2size.values(), desired_sizes)]
    print("Loading data in dictionary and converting to Lab colorspace.")
    X = {}
    h, w, c = cv2.imread(filepaths['train'][0]).shape  # Assuming all images have same shape
    for set_name, set_size in zip(set_name2size.keys(), sets_sizes):
        X[set_name] = np.zeros((set_size, h, w, 3), dtype=np.int8)
        for i, path in tqdm(enumerate(filepaths[set_name]), total=set_size):
            if i == set_size:
                break
            img = cv2.imread(path)
            # Checking if the image is not black and white (many b/w images in coco) and adds it
            X[set_name][i] = bgr2lab(img)
    print("Done.")
    return X


def get_image_paths():
    print("Getting all images paths and removing greyscale images")
    sets = ('train', 'valid', 'test')
    filepaths = {}
    for set_ in sets:
        dirpath = 'images/' + set_
        filepaths_all = sorted(
            [y for x in os.walk(dirpath) 
               for y in (glob(os.path.join(x[0], '*.jpg')) + 
                         glob(os.path.join(x[0], '*.png')))])
        filepaths[set_] = remove_bw(filepaths_all)
    
    set_name2size = {name: len(filepaths[name]) for name in sets}
    print('\n', set_name2size)

    for set_ in sets:
        assert not [filepath for filepath in filepaths[set_] if not filepath.endswith('.jpg')], f"Non image file exists in {set_}"
    
    return filepaths, set_name2size