import numpy as np
import scipy.signal
import copy

def rotate_180(array, M, N):
    out = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            out[i, N-1-j] = array[M-1-i, j]
    return out

def convolution2d(image_org, kernel):
    image = copy.deepcopy(image_org)
    kernel_rotated = rotate_180(kernel, kernel.shape[0], kernel.shape[1])

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel_rotated)
    
    return output

def fft_convolve2d(image_org, kernel):
    image = copy.deepcopy(image_org)
    image_diff = (kernel.shape[0] - 1, kernel.shape[1] - 1)
    kernel_diff = (image.shape[0] - 1, image.shape[1] - 1) 
    
    padded_image = np.pad(image, ((0, image_diff[0]), (0, image_diff[1])), mode='constant')
    padded_kernel = np.pad(kernel, ((0, kernel_diff[0]), (0, kernel_diff[1])), mode='constant')
    
    fft_image = np.fft.fft2(padded_image)
    fft_kernel = np.fft.fft2(padded_kernel)
    
    fft_result = fft_image * fft_kernel
    
    result = np.fft.ifft2(fft_result)
    
    result = result[image_diff[0] // 2:-(image_diff[0] // 2)]
    result = [row[image_diff[1] // 2:-(image_diff[1] // 2)] for row in result]

    return np.real(result)

if __name__ == "__main__":
    image = np.array([[1, 2, 3, 3, 3],
                      [4, 5, 6, 5, 6],
                      [7, 8, 9, 5, 6],
                      [7, 8, 9, 5, 6],
                      [7, 8, 9, 5, 6]])

    kernel = np.array([[1, 0, -1, 0, 1],
                       [1, 0, -1, 0, 1],
                       [1, 0, -1, 0, 1],
                       [1, 0, -1, 0, 1],
                       [1, 0, -1, 0, 1]])
    print("Result of custom 2d fft convolution:")
    print(fft_convolve2d(image, kernel).astype(int))
    print("Result of scipy 2D fft convolution:")
    print(scipy.signal.fftconvolve(image, kernel).astype(int))