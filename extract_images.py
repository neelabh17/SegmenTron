from imagecorruptions import corrupt
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


f = open('datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt','r')

img_paths = list()

for line in f:
    line = line[:-1]
    # print(line)
    img_path = os.path.join('datasets/VOCdevkit/VOC2012/JPEGImages', line + '.jpg')
    img_paths.append(img_path)

f.close()

for img_path in tqdm(img_paths):
    image = Image.open(img_path)
    image = np.asarray(image)

    corrupted_image = corrupt(image, severity=1, corruption_name='fog')
    # corrupted_image = corrupt(image, severity=1, corruption_name='snow')

    image = Image.fromarray(corrupted_image)
    image.save(os.path.join('datasets/Foggy_VOC/JPEGImages', os.path.basename(img_path)))