import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image

import matplotlib
matplotlib.use('GTK3Agg')

# Q1 Add additive noise to the image----------------------- one photo ---------------------------


def add_gaussian_noise():
    img = cv2.imread("./input/input.jpeg", 0)
    row, col = img.shape
    gauss_noise = np.zeros((row, col), dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise*0.5).astype(np.uint8)
    noisy = cv2.add(img, gauss_noise)
    cv2.imwrite("./output/output.jpeg", noisy)


# Q2 Filter the noisy image using the Gaussian low pass filter-----------------one photo----------------

def filter_gaussian_noise():
    img = cv2.imread("./input/input.jpeg")
    filtered_img = cv2.fastNlMeansDenoising(img, None, 10, 10)
    cv2.imwrite("./output/output.jpeg", filtered_img)


# Convolution function for Q3

def convolve2D(image_data, kernel, padding=0, strides=1):
    # in our case 3x3 it will be 1px in all sides
    padding_width = kernel.shape[0] // 2

    # to add the padding to our image
    x = image_data.shape[0] + padding_width * 2
    y = image_data.shape[1] + padding_width * 2

    # array = np.array([padding] * (x * y))
    # new_arr = array.reshape(x, y)
    padding_img = np.zeros((x, y))

    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    padding_img[padding_width:-padding_width,
                padding_width:-padding_width] = image_data

    # To simplify things
    k = kernel.shape[0]

    # 2D array of zeros
    convolved_img = np.zeros(shape=(image_data.shape[0], image_data.shape[1]))

    # Iterate over the rows
    for i in range(image_data.shape[0]):
        # Iterate over the columns
        for j in range(image_data.shape[0]):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = padding_img[i:i+k, j:j+k]  # => img[i:i+k][j:j+k]

            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img


# Q3

""""
Sobel Function for edge detection:
    Edge detection is an application for the convolution so we are using sobel filter to edge dectect.
    Sobel filter consists of two kernels and the kernels are 3x3 Matrices.
    We convolve each kernel with the image.
    Then we will combined the results
    Gx => the result of convolving kernel1 with image.
    Gy => the result of convolving kernel2 with image.
    by using square root of Gx ** 2 + Gy ** 2, you will get the magnitude of the edge and combine.

"""


def sobel():
    # image = cv2.imread("./samples/dog.jpeg", 0)
    image = cv2.imread("./input/input.jpeg", 0)
    image_data = cv2.resize(image, (400, 400))
    horizontal = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

    vertical = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    # Gx = sc.signal.convolve2d(horizontal, image_data)
    G_x = convolve2D(image_data, horizontal, padding=0, strides=1)
    G_y = convolve2D(image_data, vertical, padding=0, strides=1)
    # Gy = sc.signal.convolve2d(vertical, image_data)
    G_combined = np.sqrt(np.square(G_x) + np.square(G_y))
    # G_combined *= 255.0 / G_combined.max()

    cv2.imwrite("./output/output.jpeg", G_combined)


# Q4 & Q5 those functions i will call them for Draw histogram & distribution curve & Equalize the image
def make_histogram(img):
    histogram = np.zeros(256, dtype=int)
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram


def make_cumsum(histogram):
    cumsum = np.zeros(256, dtype=int)
    cumsum[0] = histogram[0]
    for i in range(1, histogram.size):
        cumsum[i] = cumsum[i-1] + histogram[i]
    return cumsum


def make_mapping(cumsum, img_h, img_w):
    mapping = np.zeros(256, dtype=int)
    grey_levels = 256
    for i in range(grey_levels):
        mapping[i] = max(0, round((grey_levels*cumsum[i])/(img_h*img_w))-1)
    return mapping


def apply_mapping(img, mapping):
    new_image = np.zeros(img.size, dtype=int)
    for i in range(img.size):
        new_image[i] = mapping[img[i]]
    return new_image
# --------------------------------------------


# Q4 --1-- call the function and shoe histogram the image---------one photo

def draw_histogram():

    path = "./input/input.jpeg"

    my_img = Image.open(os.path.join(path))
    image2 = Image.open(os.path.join(path)).convert('L')
    img_arr = np.asarray(image2)
    img_float = img_arr.astype('float32')
    img_h = img_arr.shape[0]
    img_w = img_arr.shape[1]
    flat = img_arr.flatten()
    hist = make_histogram(flat)
    plt.plot(hist)
    plt.savefig("./output/output.jpeg")
    plt.clf()

    return flat, img_h, img_w, hist

# Q4 --2-- call the function and shoe distribution curve for the image---------one photo


def draw_distribution_curve():
    _, _, _, hist = draw_histogram()
    cs = make_cumsum(hist)
    plt.plot(cs)
    plt.savefig("./output/output.jpeg")
    plt.clf()
    
    return cs


# Q5 --- call the functions and shoe equalize the image---------one photo

def draw_histogram_equalization():
    cs = draw_distribution_curve()
    flat, img_h, img_w, _ = draw_histogram()

    new_intensity = make_mapping(cs, img_h, img_w)
    new_img = apply_mapping(flat, new_intensity)
    hist_equ = make_histogram(new_img)
    plt.plot(hist_equ)
    plt.savefig("./output/output.jpeg")
    plt.clf()


# Q6 normalize the image


def normalize():
    img = cv2.imread("./input/input.jpeg")
    min = float(img.min())
    max = float(img.max())
    normalized_img = np.floor((img-min)/(max-min)*255.0)
    cv2.imwrite("./output/output.jpeg", normalized_img)


# Q7 global threshold (1)

def global_threshold(thresh=100):
    image = cv2.imread('./input/input.jpeg', 0)
    modified_image = ((image > thresh) * 255).astype("uint8")
    cv2.imwrite("./output/output.jpeg", modified_image)

# Q7 Local threshold (2)


def Local_threshold():

    path = "./input/input.jpeg"
    image = Image.open(os.path.join(path)).convert('L')
    pixels = asarray(image)
    image_array = np.array(pixels)

    size = 10
    ratio = 0.5
    new_array = np.ones(shape=(len(image_array), len(image_array[0])))
    for row in range(len(image_array) - size + 1):
        for col in range(len(image_array[0]) - size + 1):
            window = image_array[row:row+size, col:col+size]
            minm = window.min()
            maxm = window.max()
            threshold = minm+((maxm-minm)*ratio)
            if window[0, 0] < threshold:
                new_array[row, col] = 0
            else:
                new_array[row, col] = 1

    img = plt.imshow(new_array, interpolation='nearest')
    img.set_cmap('gray')
    plt.axis('off')
    plt.savefig("./output/output.jpeg", bbox_inches='tight')
    plt.clf()


# Q8 transform to gray scale

def gray_scale():
    img = cv2.imread('./input/input.jpeg')

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    cv2.imwrite("./output/output.jpeg", imgGray)
