import numpy as np
import matplotlib.pyplot as plt
import cv2
from convolution import *

def gaussian_filter(sigma):
    size = int(2*np.ceil(3*sigma) + 1)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def gaussian_dx(sigma):
    size = int(2*np.ceil(3*sigma) + 1)
    x = np.arange(0, size) - (size - 1) / 2
    return -(x / sigma**2) * gaussian_filter(sigma)

def gaussian_dy(sigma):
    return gaussian_dx(sigma).T  # Transpose of x = y 

sigma = int(input("Enter Value of Sigma: "))
gaussian_kernel = gaussian_filter(sigma)
print("Gaussian Kernel:")
print(gaussian_kernel)
plt.imshow(gaussian_kernel, cmap='gray', interpolation='nearest')
plt.title('Gaussian Kernel')
plt.colorbar()
plt.show()

partial_derivative_x = gaussian_dx(sigma)
partial_derivative_y = gaussian_dy(sigma)

plt.imshow(partial_derivative_x, cmap='gray', interpolation='nearest')
plt.title('Partial Derivative w.r.t. x')
plt.colorbar()
plt.show()

plt.imshow(partial_derivative_y, cmap='gray', interpolation='nearest')
plt.title('Partial Derivative w.r.t. y')
plt.colorbar()
plt.show()

size = int(2*np.ceil(3*sigma) + 1)
c = int(np.floor(size/2))
img = cv2.imread("coins.jpg",0)  
img = cv2.resize(img, (800,500))
cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = convolve(img, partial_derivative_x, (c,c))
cv2.imshow("Convolution X-Derivative Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = convolve(img, partial_derivative_y, (c,c))
cv2.imshow("Convolution Y-Derivative Image", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient_magnitude = np.sqrt(img1 ** 2 + img2 ** 2)
cv2.imshow("Gradient Magnitude", gradient_magnitude.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

average_pixel_value = gradient_magnitude.mean()
threshold_value = average_pixel_value


while True:
    _, thresholded_image = cv2.threshold(gradient_magnitude.astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)
    below_threshold = gradient_magnitude[gradient_magnitude <= threshold_value]
    above_threshold = gradient_magnitude[gradient_magnitude > threshold_value]
    u1 = below_threshold.mean()
    u2 = above_threshold.mean()
    new_threshold_value = (u1 + u2) / 2   
    if np.abs(new_threshold_value - threshold_value) < 0.01:
        break  
    threshold_value = new_threshold_value

cv2.imshow("Final Thresholded Image Edge Detected", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
