from convolution import  convolution2d, fft_convolve2d
import cv2
import numpy as np
import time

if __name__ == "__main__":
    im = cv2.imread("./resources/cropped.jpg", cv2.IMREAD_GRAYSCALE)

    egde_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gausian_blur = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])

    startTime = time.time()
    convolution2d(im, egde_detection_kernel)
    endTime = time.time()
    print("Convolution with edge detection kernel: " + str(endTime - startTime))
    
    startTime = time.time()
    fft_convolve2d(im, egde_detection_kernel)
    endTime = time.time()
    print("FFT convolution with edge detection kernel: " + str(endTime - startTime))

    startTime = time.time()
    convolution2d(im, sharpen_kernel)
    endTime = time.time()
    print("Convolution with sharpen kernel: " + str(endTime - startTime))

    startTime = time.time()
    fft_convolve2d(im, sharpen_kernel)
    endTime = time.time()
    print("FFT convolution with sharpen kernel: " + str(endTime - startTime))

    startTime = time.time()
    convolution2d(im, gausian_blur)
    endTime = time.time()
    print("Convolution with gausian blur kernel: " + str(endTime - startTime))

    startTime = time.time()
    fft_convolve2d(im, gausian_blur)
    endTime = time.time()
    print("FFT convolution with gausian blur kernel: " + str(endTime - startTime))