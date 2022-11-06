import cv2
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool



def square_image_center(img):
    h, w, c = img.shape
    min_length = min((h, w))
    h_border = (h - min_length) // 2
    w_border = (w - min_length) // 2
    return img[h_border:h - h_border, w_border:w - w_border, :]


def downscale(img, size=128):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# def reduce_coco(size=128):
#     paths = [path for path in os.listdir('coco/') if path.endswith(".jpg")]
#     i = 0
#     for path in tqdm(paths):
#         img = cv2.imread('coco/' + path)
#         try:
#             img.shape
#             img_num = f"{i}".zfill(5)
#             write_path = f"images/images_{img_num}.jpg"
#             cv2.imwrite(write_path, downscale(img, size))
#             i += 1
#         except:
#             print(f'{path} invalid')
#         continue
