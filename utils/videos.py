import cv2
import os
from glob import glob
from tqdm import tqdm


def extract_frames(path, output_dir):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    tot_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=tot_frames) as pbar:
        while success:
            write_path = os.path.join(output_dir, f"{count:05d}-frame.jpg")
            cv2.imwrite(write_path, image)
            success,image = vidcap.read()
            count += 1
            pbar.update(1)
    print(f"Extracted {count} frames.")


def create_video_from_frame_files(frames_dir, output_dir, frameSize=(854, 480)):
    write_path = os.path.join(output_dir, "output_video.avi")
    out = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)
    for filename in tqdm(sorted(glob(f"{frames_dir}*.jpg"))):
        img = cv2.imread(filename)
        out.write(img)
    out.release()


def create_video_from_frame_array(frames, output_dir, frameSize=(854, 480)):
    write_path = os.path.join(output_dir, "output_video.avi")
    out = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)
    for frame in tqdm(frames):
        out.write(frame)
    out.release()
