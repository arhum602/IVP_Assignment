import numpy as np
import cv2
import math

def _convolve_previtt(img):

    px_msk = [[-1,0,1],[-1,0,1],[-1,0,1]]
    py_msk = [[-1,-1,-1],[0,0,0],[1,1,1]]

    h,w = img.shape
    out = np.empty([h, w],dtype=np.uint8)
    for i in range(h-2):
        for j in range(w-2):
            px=0
            py=0
            for k in range(3):
                for l in range(3):
                    px+=(img[i+k][j+l]*px_msk[k][l])
                    py+=(img[i+k][j+l]*py_msk[k][l])
            out[i][j]=math.sqrt(px*px+py*py)
    return out


def _convolve_sobel(img):
    sx_msk = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sy_msk = [[-1,-2,-1],[0,0,0],[1,2,1]]

    h,w = img.shape
    out = np.empty([h, w],dtype=np.uint8)
    for i in range(h-2):
        for j in range(w-2):
            px=0
            py=0
            for k in range(3):
                for l in range(3):
                    px+=(img[i+k][j+l]*sx_msk[k][l])
                    py+=(img[i+k][j+l]*sy_msk[k][l])
            out[i][j]=math.sqrt(px*px+py*py)
    return out


img = cv2.imread("cameraman.tif",cv2.IMREAD_GRAYSCALE)
cv2.imshow("original_image",img)
out_prewitt = _convolve_previtt(img)
out_sobel = _convolve_sobel(img)
cv2.imshow("prewitt_convolve",out_prewitt)
cv2.imshow("sobel_convolve",out_sobel)

cv2.imwrite("prewitt_convolve.png",out_prewitt)
cv2.imwrite("sobel_convolve.png",out_sobel)
