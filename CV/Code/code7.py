# Thresholding image

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv.imread("Image/colourful_butterfly.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)
""" 


# Simple Thresholding


# Depth Greysbolding image



# simple thresh
# threshold, thresh = cv.threshold(gray, 100,255, cv.THRESH_BINARY)
# cv.imshow("Binary Threshold", thresh)
threshold, inv_thresh = cv.threshold(gray, 100,  255, cv.THRESH_BINARY_INV)

# here 100 is thresh, gray is image
cv.imshow("Simple Threshold Inverse", inv_thresh) 

# adoptive thresholding


blocksize = 15
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize,1)
cv.imshow("Adaptive Threshold", adaptive_thresh)


# it do by best for each block ,

# we can change this for THRESH_BINARY_INV

# we can do  by mean -> ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C
 """

# laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

cv.imshow("Laplacian", lap)
# it looks like pencil drawn, also we do absolute (image itself cant have negative pixel)


# sabel
sabelx = cv.Sobel(gray, cv.CV_64F ,1, 0)
sabely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sabel = cv.bitwise_or(sabelx, sabely)


# cv.imshow("Sobel X", sabelx)
# cv.imshow("Sobel Y", sabely)
# cv.imshow("Combined sobel", combined_sabel)


# both are from different so end look totally different

canny = cv.Canny(gray, 150 , 175 )
cv.imshow("Canny", canny)

# canny is little cleaned , it is advanced, 
# so canny used lot , however for more advanced sobel used not necssary laplacian but yes sobel

cv.waitKey(0)
cv.destroyAllWindows()
