import cv2
import numpy as np
from convolution import *
import matplotlib.pyplot as plt
def create_gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


def create_log_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = (size - 1) // 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = ((x - center)**2 + (y - center)**2 - 2 * sigma**2) * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return kernel - np.mean(kernel)

laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

def create_gaussian_kernel_2(size, sigma_x, sigma_y):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma_x*sigma_y)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma_x*sigma_y)), (size, size))
    return kernel / np.sum(kernel)

def create_mean_kernel(size):
    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= size * size
    return kernel

def custom_mean_filter(image, kernel_size):
    mean_kernel = create_mean_kernel(kernel_size)
    mean_filtered_image = cv2.filter2D(image, -1, mean_kernel)
    return mean_filtered_image

def sobel_filter(image, dx, dy, ksize=3):
    sobel_filtered_image = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)
    sobel_filtered_image = cv2.convertScaleAbs(sobel_filtered_image)
    return sobel_filtered_image


