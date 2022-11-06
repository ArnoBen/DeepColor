import os
from glob import glob
from .conversions import bgr2lab
from tqdm import tqdm
from multiprocessing import Pool
import cv2
import numpy as np


def _load_and_convert(path):
    img = cv2.imread(path)
    return bgr2lab(img)


def _keep_if_colored(path):
    img = cv2.imread(path)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if not np.logical_and((r==g).all(), (g==b).all()):
        return path
    else:
        return None


def _remove_bw(paths):
    """Only keeps the non black & white pictures within a directory"""
    colored_paths = []

    with Pool() as p:
        colored_paths = list(tqdm(p.imap(_keep_if_colored, paths), total=len(paths)))

    colored_paths = [path for path in colored_paths if path is not None ]

    return colored_paths


def make_dataset(train_size=40000, val_size=1000, test_size=1000):
    filepaths, set_name2size = get_image_paths(train_size, val_size, test_size)
    print("Loading data in dictionary and converting to Lab colorspace.")
    X = {}
    for set_name in filepaths:
        with Pool() as p:
            paths = filepaths[set_name]
            X[set_name] = np.array(list(tqdm(
                p.imap(_load_and_convert, paths),
                total=len(paths))))
    print("Done.")
    return X


def get_image_paths(train_size=40000, val_size=1000, test_size=1000):
    print("Getting all images paths and removing greyscale images")
    sets = {'train': train_size, 'valid': val_size, 'test': test_size}
    filepaths = {}
    for set_name, set_size in sets.items():
        dirpath = 'images/' + set_name
        filepaths_set_all = sorted(
            [y for x in os.walk(dirpath) 
               for y in (glob(os.path.join(x[0], '*.jpg')) + 
                         glob(os.path.join(x[0], '*.png')))])
        filepaths[set_name] = _remove_bw(filepaths_set_all[:set_size])
    
    set_name2size = {name: len(filepaths[name]) for name in sets}
    print("Greyscale images may have been removed, hence the possibly smaller final size")
    print('\n', set_name2size)

    for set_name in sets.keys():
        assert not [filepath for filepath in filepaths[set_name] if not filepath.endswith('.jpg')], f"Non image file exists in {set_name}"
    
    return filepaths, set_name2size


if __name__ == "__main__":
    make_dataset(10000, 1000, 1000)