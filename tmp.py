import numpy as np
import cv2
from utils import convert_color_space

in_path = 'datasets/official_warped/train/input/0001.png'
out_path = 'test.png'

img = cv2.imread(in_path)
img = img.astype(np.float32) / 255
print(img[0, :10])
img = convert_color_space(img, 'rgb', 'hsv')
print(img[0, :10])
img = convert_color_space(img, 'hsv', 'rgb')
print(img[0, :10])
# cv2.imwrite(out_path, img)
