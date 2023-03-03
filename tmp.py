#%%
import numpy as np
import cv2
from utils import convert_color_space

in_path = 'datasets/official_warped/train/input/0001.png'
out_path = 'test.png'

#%%
img = cv2.imread(in_path)
cv2.imwrite('test2.png', img)
print(img[0, :3])
img = img.astype(np.float32) / 255
img = convert_color_space(img, 'rgb', 'hsv')
print(img[0, :3])
# img[:, :, 0] %= 1
print(img[0, :3])
img = convert_color_space(img, 'hsv', 'rgb')
img *= 255
print(img[0, :3])
cv2.imwrite(out_path, img)

# %%
