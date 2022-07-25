import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def convolve(img,kernel):
    sz=kernel.shape[0]
    h,w=img.shape
    sz=sz-1
    out = np.empty([h-sz, w-sz],dtype=np.float32)
    for i in range(h-sz):
        for j in range(w-sz):
            p=0
            for k in range(sz+1):
                for l in range(sz+1):
                    p+=(img[i+k][j+l]*kernel[k][l])
            out[i][j]=p
    return out


def sobel_grad(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    ix = convolve(img,kx)
    iy = convolve(img,ky)
    G = np.sqrt(np.add(np.multiply(ix,ix),np.multiply(iy,iy)))
    G = G/G.max() * 255
    G = G.astype(np.uint8)
    theta = np.arctan2(ix,iy)
    return (G,theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    Z=Z.astype(np.uint8)
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def edge_dec(img,ker=5):
    g_kernel = gaussian_kernel(ker)
    img=convolve(img,g_kernel)
    img=img.astype(np.uint8)
    img,theta = sobel_grad(img)
    img=non_max_suppression(img,theta)
    img, weak, strong=threshold(img)
    img=hysteresis(img,weak)
    return img

img_orig = cv2.imread("cameramantest2.png",cv2.IMREAD_GRAYSCALE)

img1=edge_dec(img_orig,5)
plt.subplot(121),plt.imshow(img_orig,'gray'),plt.title('original_image')
plt.subplot(122),plt.imshow(img1,'gray'),plt.title('After applying the algorithm')
plt.show()