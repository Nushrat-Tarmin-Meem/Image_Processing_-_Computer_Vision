import numpy as np
import matplotlib.pyplot as plt
import cv2
from convolution import *



sigma = float(input("Enter Value of Sigma: "))
gaussian_kernel = gaussian_filter(sigma)
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Gaussian Kernel:")
print(gaussian_kernel)
plt.imshow(gaussian_kernel, cmap='gray', interpolation='nearest')
plt.title('Gaussian Kernel')
plt.colorbar()
plt.show()



def non_maximum_suppression(gradient_magnitude, gradient_direction):
    suppressed_image = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            direction = gradient_direction[i, j]
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180) or (-22.5 <= direction < 0) or (-180 <= direction < -157.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i, j + 1]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
            elif (22.5 <= direction < 67.5) or (-157.5 <= direction < -112.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j + 1]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
            elif (67.5 <= direction < 112.5) or (-112.5 <= direction < -67.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
            elif (112.5 <= direction < 157.5) or (-67.5 <= direction < -22.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j + 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j - 1]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
    return suppressed_image



def calculate_threshold(image):
    # Use Otsu's method to calculate optimal threshold
    image = cv2.convertScaleAbs(image)
    _, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5 * thresh_img
    high_threshold = 1.5 * thresh_img
    return low_threshold, high_threshold



def hysteresis_thresholding(image, low_threshold, high_threshold):
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if weak_edges[i, j]:
                if (strong_edges[i - 1: i + 2, j - 1: j + 2]).any():
                    strong_edges[i, j] = True
                else:
                    weak_edges[i, j] = False
    return strong_edges



def canny_edge_detection(image, sigma): 
    gaussian_dx = gaussian_dxx(sigma)
    gaussian_dy = gaussian_dyy(sigma)
    plt.imshow(gaussian_dx, cmap='gray', interpolation='nearest')
    plt.title('Partial Derivative w.r.t. x')
    plt.colorbar()
    plt.show()
    plt.imshow(gaussian_dy, cmap='gray', interpolation='nearest')
    plt.title('Partial Derivative w.r.t. y')
    plt.colorbar()
    plt.show()
    size = int(2*np.ceil(3*sigma) + 1)
    c = int(np.floor(size/2))
    partial_x = convolve(image, gaussian_dx, (c,c))
    partial_y = convolve(image, gaussian_dy, (c,c))
    gradient_magnitude = np.sqrt(partial_x ** 2 + partial_y ** 2)
    cv2.imshow("Gradient Magnitude", gradient_magnitude.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    
    
    gradient_direction = np.arctan2(partial_y, partial_x) * (180 / np.pi)   
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    low_threshold, high_threshold = calculate_threshold(suppressed_image)
    edges = hysteresis_thresholding(suppressed_image, low_threshold, high_threshold)  
    return edges



edges = canny_edge_detection(image, sigma)
cv2.imshow('Canny Edges Detected', np.uint8(edges * 255))
cv2.waitKey(0)
cv2.destroyAllWindows()
