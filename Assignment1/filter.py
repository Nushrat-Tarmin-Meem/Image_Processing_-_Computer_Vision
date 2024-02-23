import cv2
import numpy as np
from kernel_creation import *
def filtering():
    def custom_gaussian_filter(image, sigma_x, sigma_y,ksize_x):
        gaussian_kernel = create_gaussian_kernel_2(ksize_x, sigma_x, sigma_y)
        print(gaussian_kernel)
        gaussian_blur_image = cv2.filter2D(image, -1, gaussian_kernel)
        return gaussian_blur_image
    
    input_image = cv2.imread("lena.jpg")
    sigma_x = float(input("Enter sigma_x value: "))
    sigma_y = float(input("Enter sigma_y value: "))
    cv2.imshow("Original Image", input_image)
    ksize_x = int(6 * sigma_x) + 1
    ksize_y = int(6 * sigma_y) + 1
    if ksize_x % 2 == 0:
        ksize_x += 1
    if ksize_y % 2 == 0:
        ksize_y += 1
    
    gaussian_blur_image_normal = cv2.GaussianBlur(input_image, (ksize_x, ksize_y), sigmaX=sigma_x)
    cv2.imshow("Normal Gaussian Filtered Image", gaussian_blur_image_normal)
    cv2.waitKey(0)
       
    filtered_image = custom_gaussian_filter(input_image, sigma_x, sigma_y, ksize_x)
    cv2.imshow("Custom Funtioned Gaussian Filtered Image", filtered_image)
    cv2.waitKey(0)
    
    mean_filtered_image = custom_mean_filter(input_image, ksize_x)
    cv2.imshow("Mean Filtered Image", mean_filtered_image)
    cv2.waitKey(0)
    
    laplacian_filtered_image = cv2.Laplacian(input_image, cv2.CV_64F)
    laplacian_filtered_image = np.uint8(np.absolute(laplacian_filtered_image))
    cv2.imshow("Laplacian Filtered Image", laplacian_filtered_image)
    cv2.waitKey(0)
       
    log_filtered_image = cv2.Laplacian(gaussian_blur_image_normal, cv2.CV_64F)
    log_filtered_image = np.uint8(np.absolute(log_filtered_image))
    cv2.imshow("Log Filtered Image", log_filtered_image)
    cv2.waitKey(0)
    
    input_image = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
    sobel_x_filtered_image = sobel_filter(input_image, 1, 0)
    sobel_y_filtered_image = sobel_filter(input_image, 0, 1)
    cv2.imshow("Sobel X Filtered Image", sobel_x_filtered_image)
    cv2.imshow("Sobel Y Filtered Image", sobel_y_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

