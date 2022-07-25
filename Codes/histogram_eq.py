import numpy as np
import cv2


def histogram_eq(img):
    frequency = np.zeros((256,),dtype=np.float16)
    mapped = np.zeros((256,),dtype=np.float16)

    h,w=img.shape

    for i in range(w):
        for j in range(h):
            frequency[img[j,i]]+=1

    tmp=1.0/(h*w)

    for i in range(256):
        for j in range(i+1):
            mapped[i]+=frequency[j]*tmp
        mapped[i]=round(mapped[i]*255)
    mapped=mapped.astype(np.uint8)

    for i in range(w):
        for j in range(h):
            img[j,i]=mapped[img[j,i]]


img = cv2.imread("cameraman.tif",cv2.IMREAD_GRAYSCALE)
cv2.imshow("original_image",img)
histogram_eq(img)
cv2.imshow("equalized_image",img)
cv2.imwrite("equalized_image.png",img)
