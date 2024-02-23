import cv2
import numpy as np

def normalize(image):
    return np.round(np.clip(image, 0, 255)).astype(np.uint8)

def convolve(image, kernel, kernel_center):
    cv2.imshow('Input Image', image)
    
    kernel_height, kernel_width = kernel.shape
    pad_top, pad_left = kernel_center
    pad_bottom = kernel_height - pad_top - 1
    pad_right = kernel_width - pad_left - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    cv2.imshow('Bordered Image', padded_image)
    
    output = np.zeros_like(image, dtype='float32')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)
    return normalize(output)
