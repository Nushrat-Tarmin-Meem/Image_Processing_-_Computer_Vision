import cv2
import numpy as np
from convolution import *
from kernel_creation import *
from filter import *
from hsv import *
import matplotlib.pyplot as plt

def display_menu():
    print("Choose Menu:")
    print("1. Grayscale Image")
    print("2. Color Image")
    print("3. HSV Image")
    print("4. Gaussian Subtraction of RGB & HSV")
    print("5. Different Filters")
    print("6. Exit")
    
global g_rgb
global g_hsv
    
def option1(size,sigma,kernel_center):
    print("Grayscale Image Operation")
    img = cv2.imread('lena.jpg',0)            
    
    gaussian_kernel = create_gaussian_kernel(size, sigma)
    plt.imshow(gaussian_kernel, cmap='gray', interpolation='nearest')
    plt.title('Gaussian Kernel')
    plt.colorbar()
    plt.show()
    print(gaussian_kernel)
    result = convolve(img, gaussian_kernel, kernel_center)
    cv2.imshow("Output Image After Gaussian Convolution", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    log_kernel = create_log_kernel(size, sigma)
    plt.imshow(log_kernel, cmap='gray', interpolation='nearest')
    plt.title('LOG Kernel')
    plt.colorbar()
    plt.show()
    print(log_kernel)
    result2 = convolve(img, log_kernel, kernel_center)
    cv2.imshow("Output Image After LOG Convolution", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    plt.imshow(laplacian_kernel, cmap='gray', interpolation='nearest')
    plt.title('Laplacian Kernel')
    plt.colorbar()
    plt.show()
    print(laplacian_kernel)
    result3 = convolve(img, laplacian_kernel, kernel_center)
    cv2.imshow("Output Image After Laplacian Convolution", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def option2(size,sigma,kernel_center):
    print("RGB Operation")
    img = cv2.imread('lena.jpg')
    b1,g1,r1 = cv2.split(img)


    cv2.imshow("Input Image",img)
    cv2.waitKey(0)
    log_kernel = create_log_kernel(size, sigma)
    plt.imshow(log_kernel, cmap='gray', interpolation='nearest')
    plt.title('Log Kernel')
    plt.colorbar()
    plt.show()
    print(log_kernel)  
    cv2.imshow("Green Channel",g1)
    result1 = convolve(g1, log_kernel, kernel_center)
    cv2.imshow("Green Channel After Log Convolution", result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Red Channel",r1)
    result2 = convolve(r1, log_kernel, kernel_center)
    cv2.imshow("Red Channel After Log Convolution", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Blue Channel",b1)
    result3 = convolve(b1, log_kernel, kernel_center)
    cv2.imshow("Blue Channel After Log Convolution", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    merged = cv2.merge((result3,result1,result2))
    cv2.imshow("Merged Log Convolution RGB",merged )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow("Input Image",img)
    cv2.waitKey(0)
    plt.imshow(laplacian_kernel, cmap='gray', interpolation='nearest')
    plt.title('Laplacian Kernel')
    plt.colorbar()
    plt.show()
    print(laplacian_kernel)  
    cv2.imshow("Green Channel",g1)
    result1 = convolve(g1, laplacian_kernel, kernel_center)
    cv2.imshow("Green Channel After Laplacian Convolution", result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Red Channel",r1)
    result2 = convolve(r1, laplacian_kernel, kernel_center)
    cv2.imshow("Red Channel After Laplacian Convolution", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Blue Channel",b1)
    result3 = convolve(b1, laplacian_kernel, kernel_center)
    cv2.imshow("Blue Channel After Laplacian Convolution", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    merged = cv2.merge((result3,result1,result2))
    cv2.imshow("Merged Laplacian Convolution RGB",merged )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow("Input Image",img)
    cv2.waitKey(0)
    gaussian_kernel = create_gaussian_kernel(size, sigma)
    plt.imshow(gaussian_kernel, cmap='gray', interpolation='nearest')
    plt.title('Gaussian Kernel')
    plt.colorbar()
    plt.show()
    print(gaussian_kernel)  
    cv2.imshow("Green Channel",g1)
    result1 = convolve(g1, gaussian_kernel, kernel_center)
    cv2.imshow("Green Channel After Gaussian Convolution", result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Red Channel",r1)
    result2 = convolve(r1, gaussian_kernel, kernel_center)
    cv2.imshow("Red Channel After Gaussian Convolution", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Blue Channel",b1)
    result3 = convolve(b1, gaussian_kernel, kernel_center)
    cv2.imshow("Blue Channel After Gaussian Convolution", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    merged = cv2.merge((result3,result1,result2))
    cv2.imshow("Merged Gausssian Convolution RGB",merged )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return merged

def option3(size,sigma,kernel_center):
    print("HSV Operation")
    gaussian_hsv = hsv(size,sigma,kernel_center)
    return gaussian_hsv

def main():
        while True:
            display_menu()
            choice = input("Enter your choice: ")
        
            if choice == "1":
                size = int(input("Enter Size of kernel: "))
                sigma = float(input("Enter Sigma(SD) for kernel: "))
                kernel_center_x = int(input("Enter kernel center x-coordinate: "))
                kernel_center_y = int(input("Enter kernel center y-coordinate: "))
                kernel_center = (kernel_center_x, kernel_center_y)
                option1(size,sigma,kernel_center)
            elif choice == "2":
                size = int(input("Enter Size of kernel: "))
                sigma = float(input("Enter Sigma(SD) for kernel: "))
                kernel_center_x = int(input("Enter kernel center x-coordinate: "))
                kernel_center_y = int(input("Enter kernel center y-coordinate: "))
                kernel_center = (kernel_center_x, kernel_center_y)
                g_rgb = option2(size,sigma,kernel_center)
            elif choice == "3":
                size = int(input("Enter Size of kernel: "))
                sigma = float(input("Enter Sigma(SD) for kernel: "))
                kernel_center_x = int(input("Enter kernel center x-coordinate: "))
                kernel_center_y = int(input("Enter kernel center y-coordinate: "))
                kernel_center = (kernel_center_x, kernel_center_y)
                g_hsv = option3(size,sigma,kernel_center)
            elif choice == "4":
                
                cv2.destroyAllWindows()
                img1 = cv2.imread('lena.jpg')
                img2 = cv2.cvtColor(img1,cv2.COLOR_RGB2HSV)
                difference_image = cv2.absdiff(img1, img2)
                cv2.imshow("Subtraction of Normal RGB & HSV", difference_image)
                cv2.imwrite("normal_subtraction.jpg", difference_image)             
                cv2.imshow("Gaussian RGB", g_rgb)
                cv2.imwrite("grgb.jpg", g_rgb)
                cv2.waitKey(0)
                cv2.imshow("Gaussian HSV", g_hsv)
                cv2.imwrite("ghsv.jpg", g_hsv)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                difference_image = cv2.absdiff(g_rgb, g_hsv)
                cv2.imshow("Subtraction of Gaussian RGB & HSV", difference_image)
                cv2.imwrite("gaussian_subtraction.jpg", difference_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows() 
            elif choice == "5":
                filtering()
            elif choice == "6":
                print("Exiting...")
                break
            else:
                print("Invalid choice! Please try again!!")

if __name__ == "__main__":
    main()







