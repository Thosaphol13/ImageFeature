import cv2
import numpy as np

def gethog(img_data):
    size=(128,128)
    new_img =cv2.resize(img_data,size,interpolation=cv2.INTER_AREA)
    # img_gray = cv2.imread("img", 0)
    win_size = new_img.shape
    # win_size = image_data.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    # Set the parameters of the HOG descriptor using the variablesdefined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bins)
    # Compute the HOG Descriptor for the gray scale image
    hog_descriptor = hog.compute(new_img)
    return hog_descriptor