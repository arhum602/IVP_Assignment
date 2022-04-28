import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def _dft_matrix(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    t = np.zeros((rows,cols),complex)
    output_img = np.zeros((rows,cols),complex)
    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows,1))
    y = n.reshape((cols,1))
    for row in range(0,rows):
        M1 = 1j*np.sin(-2*np.pi*y*n/cols) + np.cos(-2*np.pi*y*n/cols)
        t[row] = np.dot(M1, input_img[row])
    for col in range(0,cols):
        M2 = 1j*np.sin(-2*np.pi*x*m/cols) + np.cos(-2*np.pi*x*m/cols)
        output_img[:,col] = np.dot(M2, t[:,col])
    return output_img


camtest = cv2.imread("cameraman.png",0)  
dftma= _dft_matrix(camtest)
out_dftma = np.log(np.abs(dftma))

m,n=out_dftma.shape
shifted_dftma=np.empty(shape=[m,n])

for i in range(m):
    for j in range(n):
        shifted_dftma[(m//2)-i][(n//2)-j]=out_dftma[i][j]
plt.subplot(131),plt.imshow(camtest,'gray'),plt.title('original')
plt.subplot(132),plt.imshow(out_dftma,'gray'),plt.title('dftma_output')
plt.subplot(133),plt.imshow(shifted_dftma,'gray'),plt.title('shifted_output')
plt.show()