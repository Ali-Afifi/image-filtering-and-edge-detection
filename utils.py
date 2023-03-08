import numpy as np
import cv2


# normalize the image

def normalize():
    img = cv2.imread("./input/input.jpeg")
    min = float(img.min())
    max = float(img.max())
    normalized_img =np.floor((img-min)/(max-min)*255.0)
    cv2.imwrite("./output/output.jpeg", normalized_img)
    



# transform to gray scale

def gray_scale():
    img = cv2.imread('./input/input.jpeg')

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    cv2.imwrite("./output/output.jpeg", imgGray)


# global threshold

def global_threshold(thresh = 100):
    image = cv2.imread('./input/input.jpeg', 0)
    modified_image = ((image > thresh) * 255).astype("uint8")
    cv2.imwrite("./output/output.jpeg", modified_image)



# Convolution function

def convolve2D(image_data, kernel, padding = 0, strides = 1):
    padding_width = kernel.shape[0] // 2   # in our case 3x3 it will be 1px in all sides

    # to add the padding to our image
    x = image_data.shape[0] + padding_width * 2
    y = image_data.shape[1] + padding_width * 2

    # array = np.array([padding] * (x * y))
    # new_arr = array.reshape(x, y)
    padding_img = np.zeros((x, y)) 

    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    padding_img[padding_width:-padding_width, padding_width:-padding_width] = image_data


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
    G_x = convolve2D(image_data, horizontal, padding = 0, strides = 1)
    G_y = convolve2D(image_data, vertical, padding = 0, strides = 1)
    # Gy = sc.signal.convolve2d(vertical, image_data)
    G_combined = np.sqrt(np.square(G_x) + np.square(G_y))
    # G_combined *= 255.0 / G_combined.max()
    
    cv2.imwrite("./output/output.jpeg", G_combined)
    
