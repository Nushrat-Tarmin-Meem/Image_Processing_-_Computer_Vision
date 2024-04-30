import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_histogram(image):
    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    histogram = histogram / float(np.sum(histogram))   #pdf
    return histogram

def histogram_matching(input_histogram, target_histogram):
    input_cdf = np.cumsum(input_histogram)
    target_cdf = np.cumsum(target_histogram)
    lookup_table = np.zeros(256)
    for i in range(256):
        lookup_table[i] = np.argmin(np.abs(input_cdf[i] - target_cdf))
    return lookup_table


def apply_histogram_matching(image, lookup_table):
    matched_image = np.zeros_like(image)
    for i in range(256):
        matched_image[image == i] = lookup_table[i]
    return matched_image


img = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
input_histogram = generate_histogram(img)


mu1 = int(input("Enter mean1: ")) #30
sigma1 = int(input("Enter sd1: ")) #8
mu2 = int(input("Enter mean2: ")) #165
sigma2 = int(input("Enter sd2: ")) #20


x = np.linspace(0, 255, 256)
gaussian1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
gaussian2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
target_histogram = gaussian1 + gaussian2
target_histogram /= np.sum(target_histogram)   #pdf


lookup_table = histogram_matching(input_histogram, target_histogram)

matched_image = apply_histogram_matching(img, lookup_table)


f1 = plt.figure(1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.show()

f2 = plt.figure(2)
plt.plot(input_histogram)
plt.title('Histogram of the Input Image')
plt.show()

f3 = plt.figure(3)
plt.plot(target_histogram)
plt.title('Target Histogram')
plt.show()

f4 = plt.figure(4)
plt.imshow(matched_image, cmap='gray')
plt.title('Output Image')
plt.show()

output_histogram = generate_histogram(matched_image)
f5 = plt.figure(5)
plt.plot(output_histogram)
plt.title('Histogram of the Output Image')
plt.show()

cv2.imshow("Output Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()