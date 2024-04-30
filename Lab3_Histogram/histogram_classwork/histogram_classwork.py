import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Input', img)
img_h = img.shape[0]
img_w = img.shape[1]
f1 = plt.figure(1)
plt.hist(img.ravel(),256,[0,256])
histr = cv2.calcHist([img],[0],None,[256],[0,256])
pdf = histr/(img_h*img_w) 
f2 = plt.figure(2)
plt.plot(pdf)
plt.show
cdf = np.zeros(256)
sum = 0
for i in range(256):
    sum += pdf[i]
    cdf[i] = sum
cdf = np.round(cdf*255)
f3 = plt.figure(3)
plt.plot(cdf)
plt.show()
for i in range(0,img_h):
    for j in range(0,img_w):
        x = img[i][j]
        t = cdf[x]
        img[i][j] = t
        
f4 = plt.figure(4)
plt.hist(img.ravel(),256,[0,256])
cv2.imshow('Output', img)
cv2.imwrite("hist.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()