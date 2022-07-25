import cv2
import numpy as np
import math


def _rotate_coordinates(x, y, theta, ox, oy):
    s = np.sin(theta)
    c = np.cos(theta)
    x = np.asarray(x) - ox
    y = np.asarray(y) - oy
    
    return x * c - y * s + ox, x * s + y * c + oy

def _linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (y1 - y0)*((x-x0)/(x1-x0))

def nn(bmp, ox, oy):
    return bmp[oy][ox]

def _bilinear_resize(image):
    ox,oy=image.shape[1]//2,image.shape[0]//2
    img_height, img_width = image.shape[:2]
    height,width=img_height,img_width
    resized = image
    for i in range(height):
        for j in range(width):
            if(resized[i][j]==255):
                pixel=None
                try:
                    x0, x1, y0, y1 = int(ox), int(ox)+1, int(oy), int(oy)+1
                    a = _linear_interpolation(ox, x0, x1, image[y0][x0], image[y0][x1])
                    b = _linear_interpolation(ox, x0, x1, image[y1][x0], image[y1][x1])

                    pixel = int(_linear_interpolation(oy, y0, y1, a, b))
                except IndexError:
                    pixel= nn(image, ox, oy)
                resized[i][j] = pixel
    return resized

def rotate_image(src, theta, ox, oy, fill=255):
    theta=(math.pi/180)*theta
    theta = -theta

    sh, sw = src.shape
    cx, cy = _rotate_coordinates([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))
    dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
    sx, sy = _rotate_coordinates(dx + cx.min(), dy + cy.min(), -theta, ox, oy)

    sx, sy = sx.round().astype(int), sy.round().astype(int)
    mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)
    dest = np.empty(shape=(dh, dw), dtype=src.dtype)
    dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
    dest[dy[~mask], dx[~mask]]=fill

    return _bilinear_resize(dest)

img=cv2.imread('cameraman.tif',cv2.IMREAD_GRAYSCALE)
angle=45
ox=img.shape[1]//2
oy=img.shape[0]//2
out=rotate_image(img,angle,ox,oy)
cv2.imshow('Output3',out)
cv2.waitKey(0)
cv2.destroyAllWindows()