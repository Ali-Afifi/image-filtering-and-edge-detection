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
