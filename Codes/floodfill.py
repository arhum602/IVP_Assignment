import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)

v1=0
v2=0

def _floodfill_(img,X,Y,newColor,v1,v2):
    if X < 0 or Y < 0 \
    or X >= img.shape[1] \
    or Y >= img.shape[0] \
    or img[X][Y]<v1 \
    or img[X][Y]>v2 \
    or img[X][Y]!=255:
     return
    img[X][Y]=newColor
    _floodfill_(img,X+1,Y,newColor,v1,v2)
    _floodfill_(img,X-1,Y,newColor,v1,v2)
    _floodfill_(img,X,Y+1,newColor,v1,v2)
    _floodfill_(img,X,Y-1,newColor,v1,v2)
    _floodfill_(img,X+1,Y+1,newColor,v1,v2)
    _floodfill_(img,X-1,Y-1,newColor,v1,v2)
    _floodfill_(img,X-1,Y+1,newColor,v1,v2)
    _floodfill_(img,X+1,Y-1,newColor,v1,v2)

def CCL(img,v1,v2,color):
    b,l = img.shape[0],img.shape[1]
    label = 255
    for Y in range(b):
        for X in range(l):
            if img[X][Y]>=v1 and img[X][Y]<=v2:
                label=255
                _floodfill_(img,X,Y,label,v1,v2)
            elif img[X][Y]!=255 and img[X][Y]<v1 or img[X][Y]>v2:
                img[X][Y]=0

    for X in range(l):
        for Y in range(b):
            if img[X][Y]==255:
                img[X][Y]=color

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h,nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w,ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w))

if __name__=="__main__":
    v1 = int(input("Enter value of v1: "))
    v2 = int(input("Enter value of v2: "))
    image = Image.open('lena512.png')
    img = np.array(image)
    blocks = blockshaped(img,16,16)
    color=0
    for block in blocks:
        color = (color+50)%255
        CCL(block,v1,v2,color)
    newImage=unblockshaped(blocks, 512, 512)
    print ("New Image Saved")
    pilImage = Image.fromarray(newImage)
    pilImage.save('output2.png')

    f, imageArr = plt.subplots(1,2)
    imageArr[0].imshow(image)
    imageArr[1].imshow(pilImage)
    plt.savefig('lena2/plot.png')