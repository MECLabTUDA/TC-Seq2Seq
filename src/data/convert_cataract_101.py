import os
import shutil

import cv2
import tqdm

root = '/local/scratch/cataract-101/'
target = '/local/scratch/cataract-101-processed/'

SAMPLE_FREQ = 2

assert os.path.isdir(root)

if os.path.exists(target) and os.path.isdir(target):
    shutil.rmtree(target)

os.makedirs(target, exist_ok=False)
shutil.copyfile(root + "annotations.csv", target + "annotations.csv")

for video in tqdm.tqdm(os.listdir(root+'videos/')):
    os.makedirs(target+f'{video.replace(".mp4", "")}/')

    vidcap = cv2.VideoCapture(root+f'videos/{video}')
    success, image = vidcap.read()
    count = 0
    while success:
        # Skip x frames
        if not count % SAMPLE_FREQ:
            cv2.imwrite(target+f'{video.replace(".mp4", "")}/frame%d.jpg' % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


