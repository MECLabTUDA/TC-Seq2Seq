import os
import shutil

import tqdm
import cv2

root = '/local/scratch/Catarakt/videos/micro/'
target = '/local/scratch/CATARACTS-videos-processed/'

SAMPLE_FREQ = 2

assert os.path.isdir(root)

if os.path.exists(target) and os.path.isdir(target):
    shutil.rmtree(target)

os.makedirs(target, exist_ok=False)

for video_file in tqdm.tqdm(os.listdir(root)):

    if not video_file.endswith('.mp4'):
        continue

    os.makedirs(target + f'{video_file.replace(".mp4", "")}/')

    vidcap = cv2.VideoCapture(root + f'{video_file}')
    success, image = vidcap.read()
    count = 0
    while success:
        # Skip x frames
        if not count % SAMPLE_FREQ:
            # Save frame as JPG file
            cv2.imwrite(target + f'{video_file.replace(".mp4", "")}/frame%d.jpg' % count, image)
        success, image = vidcap.read()
        count += 1
