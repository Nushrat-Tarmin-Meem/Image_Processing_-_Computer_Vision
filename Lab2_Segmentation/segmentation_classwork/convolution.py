import numpy as np
def convolve(image, kernel, kernel_center):
    kernel_height, kernel_width = kernel.shape
    pad_top, pad_left = kernel_center
    pad_bottom = kernel_height - pad_top - 1
    pad_right = kernel_width - pad_left - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    output = np.zeros_like(image, dtype='float32')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)
    return output

def gaussian_filter(sigma):
    size = int(2*np.ceil(3*sigma) + 1)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def gaussian_dxx(sigma):
    size = int(2*np.ceil(3*sigma) + 1)
    x = np.arange(0, size) - (size - 1) / 2
    return -(x / sigma**2) * gaussian_filter(sigma)

def gaussian_dyy(sigma):
    return gaussian_dxx(sigma).T  # Transpose of x = y 