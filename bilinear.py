import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def _bilinear(image, height, width):
    img_height, img_width = image.shape[:2]
    resized = np.empty([height, width])
    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
    for i in range(height):
        for j in range(width):
            x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
            x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            pixel = a * (1 - x_weight) * (1 - y_weight)  + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight

            resized[i][j] = pixel
    return resized

    
img1 = cv2.imread('lena64.png', cv2.IMREAD_GRAYSCALE)
img2=_bilinear(img1,128,128)
f, axarr = plt.subplots(1,2)
axarr[0].imshow(img1)
axarr[1].imshow(img2)
plt.savefig('resized.png')
plt.show()